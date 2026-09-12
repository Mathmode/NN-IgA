"""Train the ``advdiff`` experiment (2D boundary layer) for one (p, seed).

Usage:
    python 2D/scripts/train_advdiff.py --protocol val --p 2 --seed 0
    python 2D/scripts/train_advdiff.py --smoke --output-dir /tmp/smoke_advdiff
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent              # <repo> (repository root)/2D
_PROJECT_ROOT = _TWO_D_ROOT.parent                                 # <repo> (repository root)
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from src.config import (ADVDIFF, TRAIN, TRAIN_STD, TRAIN_VAL, SPLIT_SEED,
                        EPSILON_DENOM, LOSS_NORM, CLIP_NORM, WARMUP_EPOCHS)
from src.parametric.training_advdiff import make_val_forward_advdiff
from src.parametric.eq26 import make_denoms_fn, make_h1_metric_factory_advdiff
from common.parameter_sampling import build_advdiff_grid
from common.h1_seminorm_2d import h1_seminorm_sq_analytic
from src.nonparametric.advdiff.pde import grad_u_exact_advdiff
from src.parametric.training_advdiff import train_with_continuation_advdiff
from src.parametric.continuation import save_checkpoint


def compute_p4_denoms(nus: np.ndarray, *, q_metric: int = 50) -> np.ndarray:
    """Analytic |u*|^2_H1 per nu (N-independent). Used only as the metric
    denominator of the per-epoch H1-error monitor; the training loss divides by
    the uniform-mesh eta^2 instead (eq. 28)."""
    denoms = np.zeros(nus.shape[0], dtype=np.float64)
    for k, nu in enumerate(nus):
        eps = float(10.0 ** float(nu[0]))
        b = float(nu[1])
        grad = lambda x, y: grad_u_exact_advdiff(x, y, eps, b)
        denoms[k] = float(h1_seminorm_sq_analytic(
            grad, quad_points_per_dim=int(q_metric), n_subdiv_per_dim=16
        ))
    return denoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=int(ADVDIFF.get("p", 2)))
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output root (default: data_results/advdiff/p<p>/; the "
                         "--init-mode uniform variant writes to .../advdiff_theta0/).")
    ap.add_argument("--init-mode", choices=["warmstart", "uniform"], default="warmstart",
                    help="'warmstart' (default): inherit weights across continuation "
                         "levels. 'uniform': re-initialise at every level (ablation).")
    ap.add_argument("--epochs-per-level", type=int, default=None,
                    help="Override the per-level epoch cap (default: config schedule).")
    ap.add_argument("--lr-init", type=float, default=None,
                    help="Override the peak learning rate (default: protocol config).")
    ap.add_argument("--lr-end", type=float, default=None,
                    help="Override the final learning rate (default: protocol config).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--q-est", type=int, default=None,
                    help="GL points per dim for the residual estimator.")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None,
                    help="Override continuation levels.")
    ap.add_argument("--n-train", type=int, default=None, help="Sub-sample training set.")
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
        "--uniform-init", action="store_true",
        help="Ablation: zero-init the network and train only at the finest level "
             "(no continuation).",
    )
    args = ap.parse_args()

    PROBLEM = "advdiff"
    p_fe = int(args.p) if args.p is not None else int(ADVDIFF.get("p", 2))
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        base = _PROJECT_ROOT / "data_results" / PROBLEM / f"p{p_fe}"
        out_dir = base / f"{PROBLEM}_theta0" if args.init_mode == "uniform" else base
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_advdiff] seed={args.seed} smoke={args.smoke} out={out_dir}")
    t0 = time.perf_counter()

    # Grid + split. 'val': fixed SPLIT_SEED 70/15/15 shared across seeds;
    # else legacy per-seed 70/30.
    val_protocol = (args.protocol == "val")
    grid = build_advdiff_grid(
        n_logeps=int(ADVDIFF["n_logeps"]),
        n_b=int(ADVDIFF["n_b"]),
        logeps_min=float(ADVDIFF["logeps_min"]),
        logeps_max=float(ADVDIFF["logeps_max"]),
        b_min=float(ADVDIFF["b_min"]),
        b_max=float(ADVDIFF["b_max"]),
        train_frac=(float(TRAIN_VAL["train_frac"]) if val_protocol else float(ADVDIFF["train_frac"])),
        seed=(int(SPLIT_SEED) if val_protocol else int(args.seed)),
        val_frac=(float(TRAIN_VAL["val_frac"]) if val_protocol else None),
    )
    train_nus = grid["train"]
    test_nus = grid["test"]
    val_nus = grid["val"] if val_protocol else None
    if args.smoke:
        n_train = int(args.n_train or 12)
        n_test = int(args.n_test or 4)
        rng = np.random.default_rng(args.seed)
        train_nus = train_nus[rng.choice(train_nus.shape[0], n_train, replace=False)]
        test_nus = test_nus[rng.choice(test_nus.shape[0], n_test, replace=False)]
        if val_protocol:
            n_val = min(int(args.n_test or 4), val_nus.shape[0])
            val_nus = val_nus[rng.choice(val_nus.shape[0], n_val, replace=False)]
    elif args.n_train is not None:
        train_nus = train_nus[: int(args.n_train)]
    if args.n_test is not None and not args.smoke:
        test_nus = test_nus[: int(args.n_test)]

    print(f"  train nus: {train_nus.shape}")
    if val_protocol:
        print(f"  val nus:   {val_nus.shape}")
    print(f"  test nus:  {test_nus.shape}")

    # ---- Continuation levels and epochs ----
    p_fe = int(args.p) if args.p else int(ADVDIFF.get("p", 2))
    if args.smoke:
        levels = tuple(args.N_levels) if args.N_levels else (4, 8)
        n_ep_default = args.n_epochs or 5
        epochs = {N: n_ep_default for N in levels}
        batch_size = args.batch_size or 4
    else:
        levels = tuple(args.N_levels) if args.N_levels else tuple(ADVDIFF["levels"])
        epochs = dict(TRAIN["epochs_advdiff"])
        if args.n_epochs is not None:
            epochs = {N: int(args.n_epochs) for N in levels}
        batch_size = args.batch_size or int(TRAIN["batch_size"])

    # ---- Protocol: 'fixed' (legacy) | 'std' (train-loss ES) | 'val' (val-η² ES) ----
    std = (args.protocol == "std")
    if std or val_protocol:
        src_cfg = TRAIN_STD if std else TRAIN_VAL
        max_epochs = int(args.es_max_epochs) if args.es_max_epochs else int(src_cfg["max_epochs"])
        epochs = {N: max_epochs for N in levels}
        lr_init_use = float(args.lr_init) if args.lr_init is not None else float(src_cfg["lr_init"])
        lr_end_use = float(args.lr_end) if args.lr_end is not None else float(src_cfg["lr_end"])
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
        lr_init_use = float(args.lr_init) if args.lr_init is not None else float(TRAIN["lr1"])
        lr_end_use = float(args.lr_end) if args.lr_end is not None else float(TRAIN["lr2"])
        es_kw = {}

    print(f"  p = {p_fe}")
    if args.epochs_per_level is not None:
        epochs = {N: int(args.epochs_per_level) for N in levels}
    print(f"  levels = {levels}")
    print(f"  epochs = {epochs}")
    print(f"  batch_size = {batch_size}")

    q_est = int(args.q_est) if args.q_est else int(ADVDIFF.get("quad_eta", 50))
    q_F_use = 20 if args.smoke else int(ADVDIFF["quad_forcing"])

    # Loss denominators (eq. 28): eta^2 on the uniform mesh at the current level,
    # recomputed per level via denoms_fn.
    if str(LOSS_NORM) != "uniform_same_level":
        raise ValueError(f"unsupported config.LOSS_NORM={LOSS_NORM!r}")
    denoms_fn = make_denoms_fn(
        train_nus, val_nus if val_protocol else None,
        kind="advdiff", p=p_fe, q_est=q_est,
    )
    t_d = time.perf_counter()
    train_denoms, val_denoms = denoms_fn(int(levels[0]))
    print(f"  eq-26 denominators η²_unif at N={levels[0]}: "
          f"median={float(np.median(train_denoms)):.4e} "
          f"({time.perf_counter() - t_d:.2f}s; recomputed per level)")

    # H1-error history monitor (val protocol); same quadrature as the loss.
    val_forward_factory = None
    h1_metric_factory = None
    hist_note = "loss=eq26 uniform_same_level eps=1e-12"
    if val_protocol:
        q_metric_denom = int(ADVDIFF.get("quad_metric", 50)) if not args.smoke else 20
        metric_dens = compute_p4_denoms(val_nus, q_metric=q_metric_denom)
        val_forward_factory = lambda NN: make_val_forward_advdiff(
            p=p_fe, n_elem=int(NN), q_K=p_fe + 1, q_F=q_F_use, q_est=q_est,
            epsilon=float(EPSILON_DENOM),
        )
        h1_metric_factory = make_h1_metric_factory_advdiff(
            p_fe, metric_denoms=metric_dens, quad_points_per_dim=20,
        )
        hist_note += ("; h1_rel_val_median=median over FULL val set vs analytic "
                      "u* (H1-seminorm, q=20)")

    # ---- Train ----
    res = train_with_continuation_advdiff(
        p=p_fe,
        seed=int(args.seed),
        train_nus=train_nus,
        train_denoms=train_denoms,
        levels=levels,
        epochs_per_level=epochs,
        batch_size=batch_size,
        lr_init=lr_init_use,
        lr_end=lr_end_use,
        log_every=int(TRAIN["log_every"]),
        q_K=p_fe + 1,
        q_F=q_F_use,
        q_est=q_est,
        epsilon=float(EPSILON_DENOM),                # decision 7-A
        uniform_init=bool(args.uniform_init),        # decision 7-E
        init_mode=str(args.init_mode),               # warmstart (default) | uniform
        checkpoint_dir=ckpt_dir,
        resume=False,
        sigma_dim=int(ADVDIFF["sigma_dim"]),
        hidden_dims=tuple(ADVDIFF["hidden_dims"]),
        val_nus=val_nus, val_denoms=val_denoms,      # protocol=val: held-out monitor
        denoms_fn=denoms_fn,                         # eq-26: per-level η²_unif
        val_forward_factory=val_forward_factory,     # one solve -> val loss + u_h
        h1_metric_factory=h1_metric_factory,         # per-epoch h1_rel_val_median
        clip_norm=float(CLIP_NORM),
        warmup_epochs=int(WARMUP_EPOCHS),
        **es_kw,                                     # std/val: early stopping (+monitor/best ckpt)
    )

    # ---- Save final checkpoint ----
    final_ckpt = ckpt_dir / "checkpoint_final.npz"
    save_checkpoint(res.params_final, final_ckpt)
    print(f"  final checkpoint -> {final_ckpt}")

    # ---- History CSV (per-level training trajectory) ----
    history_path = out_dir / f"history_p4_seed{args.seed}.csv"
    hist_rows = []
    for lvl_idx, lr in enumerate(res.per_level):
        N = res.levels_done[lvl_idx]
        for row in lr.history:
            hist_rows.append({"level_N": int(N), **{k: row[k] for k in row}})
    if hist_rows:
        fieldnames = ["level_N"] + [k for k in hist_rows[0] if k != "level_N"]
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in hist_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"  history -> {history_path} ({len(hist_rows)} rows)")

    # ---- Metadata ----
    meta = {
        "experiment": "advdiff",
        "seed": int(args.seed),
        "p": int(p_fe),
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
        "n_train": int(train_nus.shape[0]),
        "n_val": (int(val_nus.shape[0]) if val_protocol else 0),
        "n_test": int(test_nus.shape[0]),
        "elapsed_sec": float(time.perf_counter() - t0),
        "final_loss": float(res.per_level[-1].final_loss),
    }
    if val_protocol:
        meta["best_val_eta2"] = {str(res.levels_done[i]): float(res.per_level[i].best_val)
                                 for i in range(len(res.per_level))}
        meta["best_val_epoch"] = {str(res.levels_done[i]): int(res.per_level[i].best_epoch)
                                  for i in range(len(res.per_level))}
    with open(out_dir / f"meta_train_advdiff_seed{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Per-epoch training history (in addition to the per-iteration CSV above).
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    recs = [(int(res.levels_done[lvl_idx]), h)
            for lvl_idx, lvl in enumerate(res.per_level) for h in lvl.epoch_history]
    epoch_hist_path = hist_dir / f"history_p{p_fe}_seed{args.seed}.csv"
    with open(epoch_hist_path, "w") as f:
        if val_protocol:
            f.write(f"# {hist_note}\n")
            f.write("level_N,iteration,train_loss,val_eta2,h1_rel_val_median\n")
            for i, (N, h) in enumerate(recs, start=1):
                f.write(f"{N},{i},{float(h['loss']):.6e},{float(h['val_eta2']):.6e},"
                        f"{float(h.get('h1_rel_val_median', float('nan'))):.6e}\n")
        else:
            f.write("level_N,iteration,loss\n")
            for i, (N, h) in enumerate(recs, start=1):
                f.write(f"{N},{i},{float(h['loss']):.6e}\n")
    print(f"  epoch-history -> {epoch_hist_path} ({len(recs)} epochs)")

    print(f"=== train_advdiff DONE in {meta['elapsed_sec']:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
