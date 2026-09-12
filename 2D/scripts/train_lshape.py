"""Train the ``lshape`` experiment (immersed L-shape) for one (p, seed).

Usage:
    python 2D/scripts/train_lshape.py --protocol val --p 3 --seed 0
    python 2D/scripts/train_lshape.py --smoke --output-dir /tmp/smoke_lshape
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent              # <repo> (repository root)/2D
_PROJECT_ROOT = _TWO_D_ROOT.parent                                 # <repo> (repository root)
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax.numpy as jnp
import numpy as np

from src.config import (LSHAPE, TRAIN, TRAIN_STD, TRAIN_VAL, SPLIT_SEED,
                        EPSILON_DENOM, LOSS_NORM, CLIP_NORM, WARMUP_EPOCHS)
from src.parametric.continuation import train_with_continuation
from src.parametric.training import make_val_forward_lshape
from src.parametric.eq26 import make_denoms_fn, make_h1_metric_factory_lshape
from common.parameter_sampling import build_lshape_grid
from src.nonparametric.eta_estimator_2d import eta_uniform_squared_lshape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=None,
                    help="FE polynomial degree (overrides LSHAPE['p']).")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output root (default: data_results/lshape/p<p>/; the "
                         "--init-mode uniform variant writes to .../lshape_theta0/).")
    ap.add_argument("--init-mode", choices=["warmstart", "uniform"], default="warmstart",
                    help="'warmstart' (default): inherit weights across continuation "
                         "levels. 'uniform': re-initialise at every level (ablation).")
    ap.add_argument("--epochs-per-level", type=int, default=None,
                    help="Override the per-level epoch cap (default: config schedule).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--protocol", choices=["fixed", "std", "val"], default="fixed",
                    help="'val' (paper): fixed 70/15/15 split shared across seeds, "
                         "validation early stopping, best-val checkpoint. 'std': early "
                         "stopping on the training loss. 'fixed' (default): legacy "
                         "epoch schedule.")
    ap.add_argument("--es-max-epochs", type=int, default=None,
                    help="[--protocol std] per-level safety cap (default TRAIN_STD['max_epochs']).")
    ap.add_argument("--es-min-epochs", type=int, default=None,
                    help="[--protocol std] min epochs before stopping (default TRAIN_STD['min_epochs']).")
    ap.add_argument("--es-patience", type=int, default=None,
                    help="[--protocol std] patience in epochs (default TRAIN_STD['patience']).")
    ap.add_argument("--es-tol", type=float, default=None,
                    help="[--protocol std] relative-improvement tol (default TRAIN_STD['tol']).")
    ap.add_argument(
        "--reference-cache",
        type=str,
        default=None,
        help="Unused (kept for CLI back-compat); training computes denoms on the fly.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest per-level checkpoint in the output dir "
             "(for restarting after a SLURM TIMEOUT).",
    )
    ap.add_argument(
        "--uniform-init", action="store_true",
        help="Ablation: zero-init the network and train only at the finest level "
             "(no continuation).",
    )
    ap.add_argument(
        "--h1-ref-cache", type=str, default=None,
        help="[val] Degree-5 immersed self-reference .pkl for the per-epoch H1 "
             "monitor (absent -> NaN column; training unaffected).",
    )
    ap.add_argument(
        "--h1-subset", type=int, default=8,
        help="[val] Fixed val-subset size for the H1 monitor.",
    )
    ap.add_argument("--T", type=float, default=None,
                    help="Grading cap T override for this degree (default: config, 5.0).")
    ap.add_argument("--h-min", type=float, default=None, dest="h_min",
                    help="Minimum cell size override (default: config, 1e-7).")
    args = ap.parse_args()

    PROBLEM = "lshape"
    p_fe = int(args.p) if args.p is not None else int(LSHAPE.get("p", 2))

    # --T / --h-min apply as per-degree config overrides so every knot
    # construction (loss and monitors) sees them consistently.
    if args.T is not None or args.h_min is not None:
        import src.config as _cfg
        if args.T is not None:
            _cfg.T_LSHAPE_BY_P = {**_cfg.T_LSHAPE_BY_P, p_fe: float(args.T)}
        if args.h_min is not None:
            _cfg.H_MIN_LSHAPE_BY_P = {**_cfg.H_MIN_LSHAPE_BY_P, p_fe: float(args.h_min)}
        _eff_T = _cfg.T_LSHAPE_BY_P.get(p_fe, _cfg.T_LSHAPE)
        _eff_h = _cfg.H_MIN_LSHAPE_BY_P.get(p_fe, _cfg.H_MIN_LSHAPE)
        print(f"[train_lshape] mesh-policy override for p={p_fe}: "
              f"T={_eff_T} h_min={_eff_h} (default T=5.0 h_min=1e-7)")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = _PROJECT_ROOT / "data_results" / PROBLEM / f"p{p_fe}"
        out_dir = base / f"{PROBLEM}_theta0" if args.init_mode == "uniform" else base
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_lshape] seed={args.seed} smoke={args.smoke} out={out_dir}")
    t0 = time.perf_counter()

    # 'val': fixed SPLIT_SEED 70/15/15 shared across seeds; else per-seed 70/30.
    val_protocol = (args.protocol == "val")
    grid = build_lshape_grid(
        n_sigma1=int(LSHAPE["n_sigma1"]),
        n_sigma2=int(LSHAPE["n_sigma2"]),
        exp_min=float(LSHAPE["sigma_exp_min"]),
        exp_max=float(LSHAPE["sigma_exp_max"]),
        train_frac=(float(TRAIN_VAL["train_frac"]) if val_protocol else float(LSHAPE["train_frac"])),
        seed=(int(SPLIT_SEED) if val_protocol else int(args.seed)),
        val_frac=(float(TRAIN_VAL["val_frac"]) if val_protocol else None),
    )
    train_sigmas = grid["train"]
    test_sigmas = grid["test"]
    val_sigmas = grid["val"] if val_protocol else None
    if args.smoke:
        n_train = int(args.n_train or 12)
        n_test = int(args.n_test or 4)
        rng = np.random.default_rng(args.seed)
        train_sigmas = train_sigmas[rng.choice(train_sigmas.shape[0], n_train, replace=False)]
        test_sigmas = test_sigmas[rng.choice(test_sigmas.shape[0], n_test, replace=False)]
        if val_protocol:
            n_val = min(int(args.n_test or 4), val_sigmas.shape[0])
            val_sigmas = val_sigmas[rng.choice(val_sigmas.shape[0], n_val, replace=False)]

    print(f"  train sigmas: {train_sigmas.shape}")
    if val_protocol:
        print(f"  val sigmas:   {val_sigmas.shape}")
    print(f"  test sigmas:  {test_sigmas.shape}")

    if args.smoke:
        levels = tuple(args.N_levels) if args.N_levels else (10, 20)
        n_ep_default = args.n_epochs or 5
        epochs = {N: n_ep_default for N in levels}
        batch_size = args.batch_size or 4
    else:
        levels = tuple(args.N_levels) if args.N_levels else tuple(LSHAPE["levels"])
        epochs = dict(TRAIN["epochs_lshape"])
        if args.n_epochs is not None:
            epochs = {N: int(args.n_epochs) for N in levels}
        batch_size = args.batch_size or int(TRAIN["batch_size"])

    p_fe = int(args.p) if args.p is not None else int(LSHAPE.get("p", 2))
    ref_N = int(min(levels))
    q_est = int(LSHAPE.get("quad_estimator", 2))

    # ---- Protocol: 'fixed' (legacy) | 'std' (train-loss ES) | 'val' (val-η² ES) ----
    std = (args.protocol == "std")
    if std or val_protocol:
        src_cfg = TRAIN_STD if std else TRAIN_VAL
        max_epochs = int(args.es_max_epochs) if args.es_max_epochs else int(src_cfg["max_epochs"])
        epochs = {N: max_epochs for N in levels}
        lr_init_use = float(src_cfg["lr_init"])
        lr_end_use = float(src_cfg["lr_end"])
        es_kw = dict(
            early_stopping=True,
            es_patience=int(args.es_patience) if args.es_patience else int(src_cfg["patience"]),
            es_tol=float(src_cfg["tol"]) if args.es_tol is None else float(args.es_tol),
            es_min_epochs=int(args.es_min_epochs) if args.es_min_epochs else int(src_cfg["min_epochs"]),
        )
        if val_protocol:
            es_kw.update(monitor="val", best_checkpoint=True)
        print(f"  [protocol={args.protocol}] early-stop: patience={es_kw['es_patience']} "
              f"tol={es_kw['es_tol']:.3g} min_epochs={es_kw['es_min_epochs']} cap={max_epochs} | "
              f"lr {lr_init_use:.0e}->{lr_end_use:.0e} | monitor={es_kw.get('monitor', 'train')}")
    else:
        lr_init_use = float(TRAIN["lr1"])
        lr_end_use = float(TRAIN["lr2_lshape"])          # LSHAPE legacy fine tail
        es_kw = {}

    print(f"  p = {p_fe}")
    if args.epochs_per_level is not None:
        epochs = {N: int(args.epochs_per_level) for N in levels}
    print(f"  levels = {levels}")
    print(f"  epochs = {epochs}")
    print(f"  batch_size = {batch_size}")
    # Loss denominators (eq. 28): eta^2 on the uniform mesh at the current level,
    # recomputed per level via denoms_fn.
    if str(LOSS_NORM) != "uniform_same_level":
        raise ValueError(f"unsupported config.LOSS_NORM={LOSS_NORM!r}")
    denoms_fn = make_denoms_fn(
        train_sigmas, val_sigmas if val_protocol else None,
        kind="lshape", p=p_fe, q_est=q_est,
    )
    t_d = time.perf_counter()
    train_denoms, val_denoms = denoms_fn(int(levels[0]))
    print(f"  eq-26 denominators η²_unif at N={levels[0]}: "
          f"median={float(np.median(train_denoms)):.4e} "
          f"({time.perf_counter() - t_d:.2f}s; recomputed per level — "
          f"legacy ref_N={ref_N} freeze removed)")

    # H1-error history monitor (val protocol): direct sigma-weighted error vs the
    # degree-5 immersed reference on a fixed val subset.
    val_forward_factory = None
    h1_metric_factory = None
    hist_note = "loss=eq26 uniform_same_level eps=1e-12"
    if val_protocol:
        val_forward_factory = lambda NN: make_val_forward_lshape(
            p=p_fe, n_elem=int(NN), q_K=p_fe + 1, q_F=int(LSHAPE["quad_forcing"]),
            q_est=q_est, epsilon=float(EPSILON_DENOM),
        )
        h1_factory, h1_subset = make_h1_metric_factory_lshape(
            p_fe, ref_cache_path=args.h1_ref_cache,
            n_val=int(val_sigmas.shape[0]),
            subset_size=int(args.h1_subset), subset_seed=0,
        )
        h1_metric_factory = h1_factory
        hist_note += (f"; h1_rel_val_median=median over FIXED val subset "
                      f"size={len(h1_subset)} seed=0 vs immersed IGA self-reference "
                      f"({'cache=' + str(args.h1_ref_cache) if h1_factory else 'NO CACHE -> NaN'})")
        print(f"  [h1-monitor] {hist_note.split('; ', 1)[1]}")

    res = train_with_continuation(
        experiment="lshape",
        p=p_fe,
        seed=int(args.seed),
        train_sigmas=train_sigmas,
        train_denoms=train_denoms,
        levels=levels,
        epochs_per_level=epochs,
        batch_size=batch_size,
        lr_init=lr_init_use,
        lr_end=lr_end_use,
        log_every=int(TRAIN["log_every"]),
        q_K=p_fe + 1,
        q_F=int(LSHAPE["quad_forcing"]),
        q_est=q_est,
        epsilon=float(EPSILON_DENOM),
        uniform_init=bool(args.uniform_init),
        init_mode=str(args.init_mode),
        checkpoint_dir=ckpt_dir,
        resume=bool(args.resume),
        sigma_dim=int(LSHAPE["sigma_dim"]),
        hidden_dims=tuple(LSHAPE["hidden_dims"]),
        val_sigmas=val_sigmas, val_denoms=val_denoms,
        denoms_fn=denoms_fn,
        val_forward_factory=val_forward_factory,
        h1_metric_factory=h1_metric_factory,
        clip_norm=float(CLIP_NORM),
        warmup_epochs=int(WARMUP_EPOCHS),
        **es_kw,
    )

    from src.parametric.continuation import save_checkpoint
    save_checkpoint(res.params_final, ckpt_dir / "checkpoint_final.npz")

    meta = {
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "protocol": str(args.protocol),
        "levels": list(levels),
        "epochs_per_level": {str(k): int(v) for k, v in epochs.items()},
        "epochs_used": {str(res.levels_done[i]): int(res.per_level[i].epochs_used)
                        for i in range(len(res.per_level))},
        "stop_reasons": {str(res.levels_done[i]): str(res.per_level[i].stop_reason)
                         for i in range(len(res.per_level))},
        "batch_size": int(batch_size),
        "lr_init": float(lr_init_use),
        "lr_end": float(lr_end_use),
        "split_seed": (int(SPLIT_SEED) if val_protocol else int(args.seed)),
        "n_train": int(train_sigmas.shape[0]),
        "n_val": (int(val_sigmas.shape[0]) if val_protocol else 0),
        "n_test": int(test_sigmas.shape[0]),
        "elapsed_sec": float(time.perf_counter() - t0),
        "final_loss": float(res.per_level[-1].final_loss),
    }
    if val_protocol:
        meta["best_val_eta2"] = {str(res.levels_done[i]): float(res.per_level[i].best_val)
                                 for i in range(len(res.per_level))}
        meta["best_val_epoch"] = {str(res.levels_done[i]): int(res.per_level[i].best_epoch)
                                  for i in range(len(res.per_level))}
    with open(out_dir / f"meta_train_lshape_seed{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Per-epoch training history.
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    recs = [h for lvl in res.per_level for h in lvl.epoch_history]
    hist_path = hist_dir / f"history_p{p_fe}_seed{args.seed}.csv"
    with open(hist_path, "w") as f:
        if val_protocol:
            f.write(f"# {hist_note}\n")
            f.write("iteration,train_loss,val_eta2,h1_rel_val_median\n")
            for i, h in enumerate(recs, start=1):
                f.write(f"{i},{float(h['loss']):.6e},{float(h['val_eta2']):.6e},"
                        f"{float(h.get('h1_rel_val_median', float('nan'))):.6e}\n")
        else:
            f.write("iteration,loss\n")
            for i, h in enumerate(recs, start=1):
                f.write(f"{i},{float(h['loss']):.6e}\n")
    print(f"  history -> {hist_path} ({len(recs)} epochs)")

    print(f"=== train_lshape DONE in {meta['elapsed_sec']:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
