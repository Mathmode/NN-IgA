"""Arm B of the contrast-EXTRAPOLATION study (exploratory; NOT production).

Trains the helmholtz positional density network FLAT (no contrast curriculum) on
LOW contrasts only, c in [c_min, c_split], and saves a checkpoint to be evaluated
later on the HIGH-contrast set c >= c_split (the extrapolation regime). The
coarse-to-fine continuation in N is kept (it is orthogonal to the contrast); only
the contrast-exploration warm phase A is removed -- via ``warm_epochs=0``, which
(verified by reading the code) makes Phase A a no-op (its epoch loop runs zero
times), so NO production code is edited.

This reuses the production ``train_with_continuation_contrast`` unchanged; the only
differences from production training are: (1) ``train_cs``/``val_cs`` restricted to
the low-c sub-range, (2) ``warm_epochs=0`` (flat, no contrast curriculum), and
(3) output routed to ``data_results/helmholtz_extrap/`` (production untouched).

Usage:
    python scripts/train_helmholtz_extrap.py --p 2 --seed 0
    python scripts/train_helmholtz_extrap.py --p 3 --seed 0 --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; PROJECT = ROOT.parent
for _p in (str(PROJECT), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Operational split (documented in REPORT_extrap_study.md / Phase 0). Nyquist is
# never violated in [1.5,6.0] at N>=64; the high-c difficulty is the element/wave
# (pollution) term, not the interface. c_split=3.0 is the operational lower-half
# boundary (the smooth uniform-H1 degradation crosses ~2.5; 3.0 is the documented
# default, giving a clean low-c "safe" pool and a 2x extrapolation reach).
C_SPLIT_DEFAULT = 3.0
EXTRAP_SPLIT_SEED = 20240601   # deterministic low-c train/val partition (fixed)


def main() -> None:
    ap = argparse.ArgumentParser(description="helmholtz contrast-extrapolation arm B (flat low-c training).")
    ap.add_argument("--p", type=int, default=2, choices=[2, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-split", type=float, default=C_SPLIT_DEFAULT,
                    help=f"Train on c < c_split, evaluate on c >= c_split (default {C_SPLIT_DEFAULT}).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Default: data_results/helmholtz_extrap/p<degree>/.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override per-level epoch cap")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu" if args.device == "cpu" else "gpu,cpu")

    import numpy as np
    from src.config import HELMHOLTZ, VAL
    from src.helmholtz.mesh_helmholtz_global import sample_contrasts
    from src.helmholtz.network_helmholtz import params_to_flat_dict
    from src.helmholtz.training_helmholtz import train_with_continuation_contrast

    t0 = time.perf_counter()
    out = (Path(args.output_dir).resolve() if args.output_dir is not None
           else PROJECT / "data_results" / "helmholtz_extrap" / f"p{int(args.p)}")
    ckpt_dir = out / "checkpoints" / f"p{args.p}_seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Same deterministic, resonance-pruned contrast sampling as production, then
    # partition by c_split. LOW = training pool; HIGH = the (shared) eval set.
    n_c = 8 if args.smoke else int(HELMHOLTZ["n_c"])
    cs = np.sort(sample_contrasts(float(HELMHOLTZ["c_min"]), float(HELMHOLTZ["c_max"]),
                                  n_c, float(HELMHOLTZ["resonance_tol_c"])))
    c_split = float(args.c_split)
    low = cs[cs < c_split]; high = cs[cs >= c_split]
    assert low.size >= 2 and high.size >= 1, f"degenerate split at c_split={c_split} (low={low.size}, high={high.size})"

    # Train/val partition WITHIN the low-c pool (val is in-distribution low-c, used
    # only for the early-stopping monitor). Fixed seed -> reproducible.
    perm = np.random.default_rng(EXTRAP_SPLIT_SEED).permutation(low.size)
    n_val = max(1, int(round(0.2 * low.size)))   # 20% of the low-c pool held out for val
    val_cs = np.sort(low[perm[:n_val]])
    train_cs = np.sort(low[perm[n_val:]])
    test_cs_high = high  # the extrapolation eval set (shared file is the canonical one)

    if args.smoke:
        levels = tuple(args.N_levels) if args.N_levels else (32, 48)
        cap = int(args.epochs) if args.epochs else 12
        VAL["min_epochs"] = 6; VAL["patience"] = 6
    else:
        levels = tuple(args.N_levels) if args.N_levels else tuple(HELMHOLTZ["levels_c"])
        cap = int(args.epochs) if args.epochs else int(VAL["max_epochs"])
    epochs = {N: cap for N in levels}

    print(f"=== train_helmholtz_extrap (arm B) p={args.p} seed={args.seed} smoke={args.smoke} ===", flush=True)
    print(f"  c_split={c_split}: train on LOW c<{c_split} ({train_cs.size} train / {val_cs.size} val, "
          f"range [{low.min():.3f},{low.max():.3f}]); eval later on HIGH c>={c_split} "
          f"({high.size} contrasts [{high.min():.3f},{high.max():.3f}])", flush=True)
    print(f"  levels={levels} epochs(cap)={cap} warm_epochs=0 (NO contrast curriculum) loss_denom=uniform", flush=True)

    # Arm B: flat (warm_epochs=0 -> Phase A skipped), coarse-to-fine in N kept, val
    # protocol early-stopping on low-c val, production uniform denominator.
    res = train_with_continuation_contrast(
        p=int(args.p), seed=int(args.seed), train_cs=train_cs,
        levels=levels, epochs_per_level=epochs, warm_epochs=0,
        c_high_frac=float(HELMHOLTZ["c_high_frac"]), hidden_dims=tuple(HELMHOLTZ["hidden_dims"]),
        input_dim=int(HELMHOLTZ["input_dim_c"]), T=float(HELMHOLTZ["T_cap"]), h_min=float(HELMHOLTZ["h_min"]),
        lr=float(HELMHOLTZ["lr"]), lr_explore=float(HELMHOLTZ["lr_explore"]),
        weight_decay=float(HELMHOLTZ["weight_decay"]), batch_size=int(HELMHOLTZ["batch_size"]),
        use_hmax=True, val_cs=val_cs, protocol="val",
        init_mode="warmstart", loss_denom="uniform",
        ckpt_dir=ckpt_dir,
    )

    import numpy as _np
    _np.savez(ckpt_dir / "checkpoint_final.npz", **params_to_flat_dict(res.params_final))
    _np.save(out / f"test_cs_high_p{args.p}_seed{args.seed}.npy", test_cs_high)
    _np.save(out / f"train_cs_low_p{args.p}_seed{args.seed}.npy", train_cs)
    meta = {"experiment": "helmholtz_contrast_extrap_armB", "arm": "B_extrapolation",
            "p": int(args.p), "seed": int(args.seed), "smoke": bool(args.smoke),
            "protocol": "val", "loss_denom": "uniform", "warm_epochs": 0,
            "contrast_curriculum": False, "c_split": c_split,
            "c_train_range": [float(low.min()), float(low.max())],
            "c_eval_range": [float(high.min()), float(high.max())],
            "levels": list(levels), "epochs_per_level_cap": {str(k): int(v) for k, v in epochs.items()},
            "n_train": int(train_cs.size), "n_val": int(val_cs.size), "n_eval_high": int(high.size),
            "split_seed": int(EXTRAP_SPLIT_SEED),
            "final_loss": float(res.per_level[-1].final_loss),
            "elapsed_sec": float(time.perf_counter() - t0)}
    (out / f"meta_train_extrap_p{args.p}_seed{args.seed}.json").write_text(json.dumps(meta, indent=2))

    hist_dir = out / "history"; hist_dir.mkdir(parents=True, exist_ok=True)
    recs = [h for lvl in res.per_level for h in lvl.history]
    with open(hist_dir / f"history_p{args.p}_seed{args.seed}.csv", "w") as f:
        f.write(f"# arm=B_extrapolation c_split={c_split} warm_epochs=0 loss_denom=uniform; "
                "train=low-c only; eval later on high-c\n")
        f.write("level_N,iteration,train_loss,val_eta2,h1_rel_val_median\n")
        for i, h in enumerate(recs, start=1):
            f.write(f"{int(h['level_N'])},{i},{float(h['train_loss']):.6e},"
                    f"{float(h['val_eta2']):.6e},{float(h.get('h1_rel_val_median', float('nan'))):.6e}\n")

    print(f"  saved checkpoint -> {ckpt_dir/'checkpoint_final.npz'}", flush=True)
    print(f"=== DONE in {meta['elapsed_sec']:.1f}s  final_loss={meta['final_loss']:.4e} ===", flush=True)


if __name__ == "__main__":
    main()
