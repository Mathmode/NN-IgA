"""Train the ``helmholtz`` experiment (1D Helmholtz transmission) for one (p, seed).

Parameter = wavenumber contrast c = k1/k2; global knot mesh with a C^0 interface
knot at x=1/2 and the Nyquist h_max safeguard; a short high-contrast warm phase
precedes each continuation.

Usage:
    python 1D/scripts/train_helmholtz.py --protocol val --p 3 --seed 0
    python 1D/scripts/train_helmholtz.py --p 3 --seed 0 --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; PROJECT = ROOT.parent
for _p in (str(PROJECT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    ap = argparse.ArgumentParser(description="helmholtz (contrast Helmholtz, global mesh) training.")
    ap.add_argument("--p", type=int, default=3, choices=[2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root (default: data_results/helmholtz/p<p>/; the "
             "--init-mode uniform variant writes to .../helmholtz_theta0/).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-c", type=int, default=None)
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", "--epochs-per-level", dest="epochs", type=int,
                    default=None, help="Override the per-level epoch cap.")
    ap.add_argument(
        "--init-mode", choices=["warmstart", "uniform"], default="warmstart",
        help="'warmstart' (default): inherit weights across continuation levels. "
             "'uniform': re-initialise at every level (ablation).",
    )
    ap.add_argument("--warm-epochs", type=int, default=None)
    ap.add_argument("--no-hmax", action="store_true")
    ap.add_argument("--protocol", choices=["fixed", "val"], default="fixed",
                    help="'val' (paper): fixed 70/15/15 contrast split shared across "
                         "seeds, validation early stopping, best-val checkpoint. "
                         "'fixed' (default): legacy fixed-epoch recipe.")
    ap.add_argument("--loss-denom", choices=["uniform", "exact", "robust"], default="uniform",
                    help="Loss denominator: 'uniform' (default, eq. 28) = "
                         "eta^2 on the uniform mesh at the current level; 'exact' = "
                         "analytic |u*_c|^2_H1; 'robust' = uniform clipped at the "
                         "train p90 (study variants).")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu" if args.device == "cpu" else "gpu,cpu")

    import numpy as np
    from src.config import HELMHOLTZ, VAL, SPLIT_SEED, val_split_seeds
    from src.helmholtz.mesh_helmholtz_global import sample_contrasts
    from src.helmholtz.network_helmholtz import params_to_flat_dict
    from src.helmholtz.training_helmholtz import train_with_continuation_contrast

    t0 = time.perf_counter()
    PROBLEM = "helmholtz"
    if args.output_dir is not None:
        out = Path(args.output_dir).resolve()
    else:
        _base = PROJECT / "data_results" / PROBLEM / f"p{int(args.p)}"
        out = (_base / f"{PROBLEM}_theta0"
               if args.init_mode == "uniform" else _base)
    ckpt_dir = out / "checkpoints" / f"p{args.p}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_c = int(args.n_c) if args.n_c else (8 if args.smoke else int(HELMHOLTZ["n_c"]))
    cs = sample_contrasts(float(HELMHOLTZ["c_min"]), float(HELMHOLTZ["c_max"]), n_c, float(HELMHOLTZ["resonance_tol_c"]))

    # 'val': fixed SPLIT_SEED 70/15/15 shared across seeds; else per-seed 70/30.
    val_protocol = (args.protocol == "val")
    if val_protocol:
        perm = np.random.default_rng(val_split_seeds()["val"]).permutation(cs.size)
        n_train = max(1, int(round(float(VAL["train_frac"]) * cs.size)))
        n_val = max(1, int(round(float(VAL["val_frac"]) * cs.size)))
        train_cs = np.sort(cs[perm[:n_train]])
        val_cs = np.sort(cs[perm[n_train:n_train + n_val]])
        test_cs = np.sort(cs[perm[n_train + n_val:]])
    else:
        perm = np.random.default_rng(int(args.seed)).permutation(cs.size)
        n_train = max(1, int(round(float(HELMHOLTZ["train_frac"]) * cs.size)))
        train_cs = np.sort(cs[perm[:n_train]]); test_cs = np.sort(cs[perm[n_train:]])
        val_cs = None

    if args.smoke:
        levels = tuple(args.N_levels) if args.N_levels else (32, 48)
        epochs = {N: (int(args.epochs) if args.epochs else 12) for N in levels}
        warm = int(args.warm_epochs) if args.warm_epochs is not None else 15
    else:
        levels = tuple(args.N_levels) if args.N_levels else tuple(HELMHOLTZ["levels_c"])
        epochs = dict(HELMHOLTZ["epochs_per_level_c"])
        if args.epochs is not None:
            epochs = {N: int(args.epochs) for N in levels}
        warm = int(args.warm_epochs) if args.warm_epochs is not None else int(HELMHOLTZ["warm_epochs"])

    # val protocol: epochs_per_level is only the safety cap (early stopping fires first).
    if val_protocol:
        cap = int(args.epochs) if args.epochs else int(VAL["max_epochs"])
        epochs = {N: cap for N in levels}
        if args.smoke:
            VAL["min_epochs"] = 6; VAL["patience"] = 6   # smoke: let early stopping fire fast

    print(f"=== train_helmholtz p={args.p} seed={args.seed} smoke={args.smoke} "
          f"protocol={args.protocol} ===", flush=True)
    _vtxt = f" / {val_cs.size} val" if val_protocol else ""
    print(f"  contrasts: {cs.size} safe ({train_cs.size} train{_vtxt} / {test_cs.size} test), "
          f"range [{cs.min():.3f},{cs.max():.3f}]", flush=True)
    print(f"  levels={levels} epochs={epochs} warm_epochs={warm} hmax={not args.no_hmax}", flush=True)

    res = train_with_continuation_contrast(
        p=int(args.p), seed=int(args.seed), train_cs=train_cs,
        levels=levels, epochs_per_level=epochs, warm_epochs=warm,
        c_high_frac=float(HELMHOLTZ["c_high_frac"]), hidden_dims=tuple(HELMHOLTZ["hidden_dims"]),
        input_dim=int(HELMHOLTZ["input_dim_c"]), T=float(HELMHOLTZ["T_cap"]), h_min=float(HELMHOLTZ["h_min"]),
        lr=float(HELMHOLTZ["lr"]), lr_explore=float(HELMHOLTZ["lr_explore"]),
        weight_decay=float(HELMHOLTZ["weight_decay"]), batch_size=int(HELMHOLTZ["batch_size"]),
        use_hmax=not args.no_hmax,
        val_cs=val_cs, protocol=str(args.protocol),
        init_mode=str(args.init_mode),
        loss_denom=str(args.loss_denom),
        ckpt_dir=ckpt_dir,
    )

    import numpy as _np
    _np.savez(ckpt_dir / "checkpoint_final.npz", **params_to_flat_dict(res.params_final))
    _np.save(out / f"test_cs_p{args.p}_seed{args.seed}.npy", test_cs)
    meta = {"experiment": "helmholtz_contrast", "p": int(args.p), "seed": int(args.seed),
            "smoke": bool(args.smoke), "protocol": str(args.protocol),
            "param": "contrast", "mesh": "global", "loss_denom": str(args.loss_denom),
            "levels": list(levels), "epochs_per_level": {str(k): int(v) for k, v in epochs.items()},
            "warm_epochs": int(warm), "n_train": int(train_cs.size), "n_test": int(test_cs.size),
            "hidden_dims": list(HELMHOLTZ["hidden_dims"]), "g_min_target": float(HELMHOLTZ["g_min_target"]),
            "use_hmax": (not args.no_hmax), "final_loss": float(res.per_level[-1].final_loss),
            "elapsed_sec": float(time.perf_counter() - t0)}
    if val_protocol:
        meta["split_seed"] = int(SPLIT_SEED)
        meta["n_val"] = int(val_cs.size)
        bv = {}
        for lvl in res.per_level:
            Nlvl = int(lvl.history[0]["level_N"])
            vv = [h["val_eta2"] for h in lvl.history if np.isfinite(h["val_eta2"])]
            bv[str(Nlvl)] = (float(min(vv)) if vv else float("nan"))
        meta["best_val_eta2"] = bv
    (out / f"meta_train_p5c_p{args.p}_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))

    # Per-epoch training history (warm phase + levels on one cumulative axis).
    hist_dir = out / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    recs = [h for lvl in res.per_level for h in lvl.history]
    hist_path = hist_dir / f"history_p{args.p}_seed{args.seed}.csv"
    with open(hist_path, "w") as f:
        if val_protocol:
            f.write(f"# loss_denom={args.loss_denom}; loss=eq26 uniform_same_level eps=1e-12; "
                    "h1_rel_val_median=median over FULL val set vs analytic u*_c (h1_rel, host)\n")
            f.write("level_N,iteration,train_loss,val_eta2,h1_rel_val_median\n")
            for i, h in enumerate(recs, start=1):
                f.write(f"{int(h['level_N'])},{i},{float(h['train_loss']):.6e},"
                        f"{float(h['val_eta2']):.6e},"
                        f"{float(h.get('h1_rel_val_median', float('nan'))):.6e}\n")
        else:
            f.write("level_N,iteration,loss\n")
            for i, h in enumerate(recs, start=1):
                f.write(f"{int(h['level_N'])},{i},{float(h['train_loss']):.6e}\n")

    print(f"  saved checkpoint -> {ckpt_dir/'checkpoint_final.npz'}", flush=True)
    print(f"  saved history    -> {hist_path}  ({len(recs)} epochs)", flush=True)
    print(f"=== DONE in {meta['elapsed_sec']:.1f}s  final_loss={meta['final_loss']:.4e} ===", flush=True)


if __name__ == "__main__":
    main()
