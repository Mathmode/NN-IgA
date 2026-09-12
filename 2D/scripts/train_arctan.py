"""Train the ``arctan`` experiment (2D arctangent layer) for one (p, seed).

Usage:
    python 2D/scripts/train_arctan.py --protocol val --p 2 --seed 0
    python 2D/scripts/train_arctan.py --smoke --output-dir /tmp/smoke_arctan
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent              # <repo>/2D
_PROJECT_ROOT = _TWO_D_ROOT.parent                                 # <repo> (repository root)
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax.numpy as jnp
import numpy as np

from src.config import (ARCTAN, TRAIN, TRAIN_STD, TRAIN_VAL, SPLIT_SEED,
                        EPSILON_DENOM, LOSS_NORM, CLIP_NORM, WARMUP_EPOCHS)
from src.parametric.continuation import train_with_continuation
from src.parametric.training import make_val_forward_arctan
from src.parametric.eq26 import make_denoms_fn, make_h1_metric_factory_arctan
from common.parameter_sampling import build_arctan_grid
from common.h1_seminorm_2d import h1_seminorm_sq_analytic
from src.nonparametric.arctan.pde import grad_u_sigma


def compute_p2_denoms(sigmas: np.ndarray, *, q_metric: int = 50) -> np.ndarray:
    """Analytic |u*|^2_H1 per sigma (N-independent). Used only as the metric
    denominator of the per-epoch H1-error monitor; the training loss divides by
    the uniform-mesh eta^2 instead (eq. 28, see ``src.parametric.eq26``)."""
    denoms = np.zeros(sigmas.shape[0], dtype=np.float64)
    for i, (a, s1, s2) in enumerate(sigmas):
        a_f, s1_f, s2_f = float(a), float(s1), float(s2)
        grad = lambda x, y: grad_u_sigma(x, y, a_f, s1_f, s2_f)
        denoms[i] = h1_seminorm_sq_analytic(grad, quad_points_per_dim=int(q_metric))
    return denoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=int(ARCTAN.get("p", 3)))
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output root (default: data_results/arctan/p<p>/; the "
                         "--init-mode uniform variant writes to .../arctan_theta0/).")
    ap.add_argument("--init-mode", choices=["warmstart", "uniform"], default="warmstart",
                    help="'warmstart' (default): inherit weights across continuation "
                         "levels. 'uniform': re-initialise at every level (ablation).")
    ap.add_argument("--epochs-per-level", type=int, default=None,
                    help="Override the per-level epoch cap (default: config schedule).")
    ap.add_argument("--solver", choices=["dense", "fdm"], default="dense",
                    help="'dense' (default): dense Cholesky. 'fdm': exact O(N^3) "
                         "Kronecker fast-diagonalization (same result, faster; use a "
                         "separate --output-dir).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--q-est", type=int, default=None,
                    help="GL points/dim for the estimator (default 50; q=20 is "
                         "round-off-equivalent, perf-only).")
    ap.add_argument("--q-forcing", type=int, default=None,
                    help="GL points/dim for the forcing/Neumann assembly (default 50; "
                         "q=20 is round-off-equivalent, perf-only).")
    ap.add_argument("--es-abs-tol", type=float, default=None,
                    help="[val] Deprecated absolute plateau floor; prefer "
                         "--es-plateau-rel-tol (scale-free).")
    ap.add_argument("--es-plateau-rel-tol", type=float, default=None,
                    help="[val] Stop a level once best val-eta^2 improved by less than "
                         "this fraction over the last --es-patience epochs (default 0 = "
                         "legacy anchor).")
    ap.add_argument(
        "--N-levels",
        type=int,
        nargs="+",
        default=None,
        help="Override continuation levels.",
    )
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

    PROBLEM = "arctan"
    p_fe = int(args.p)
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

    print(f"[train_arctan] seed={args.seed} smoke={args.smoke} out={out_dir}")
    t0 = time.perf_counter()

    # Grid + split. 'val': fixed SPLIT_SEED 70/15/15 shared across seeds;
    # else legacy per-seed 70/30.
    val_protocol = (args.protocol == "val")
    grid = build_arctan_grid(
        n_alpha=int(ARCTAN["n_alpha"]),
        n_s1=int(ARCTAN["n_s1"]),
        n_s2=int(ARCTAN["n_s2"]),
        alpha_min=float(ARCTAN["alpha_min"]),
        alpha_max=float(ARCTAN["alpha_max"]),
        s_min=float(ARCTAN["s_min"]),
        s_max=float(ARCTAN["s_max"]),
        train_frac=(float(TRAIN_VAL["train_frac"]) if val_protocol else float(ARCTAN["train_frac"])),
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
    elif args.n_train is not None:
        train_sigmas = train_sigmas[: int(args.n_train)]
    if args.n_test is not None and not args.smoke:
        test_sigmas = test_sigmas[: int(args.n_test)]

    print(f"  train sigmas: {train_sigmas.shape}")
    if val_protocol:
        print(f"  val sigmas:   {val_sigmas.shape}")
    print(f"  test sigmas:  {test_sigmas.shape}")

    # ---- Continuation levels and epochs ----
    p_fe = int(args.p) if hasattr(args, "p") and args.p else int(ARCTAN.get("p", 3))
    if args.smoke:
        levels = tuple(args.N_levels) if args.N_levels else (4, 8)
        n_ep_default = args.n_epochs or 5
        epochs = {N: n_ep_default for N in levels}
        batch_size = args.batch_size or 4
    else:
        levels = tuple(args.N_levels) if args.N_levels else tuple(ARCTAN["levels"])
        epochs = dict(TRAIN["epochs_arctan"])
        if args.n_epochs is not None:
            epochs = {N: int(args.n_epochs) for N in levels}
        batch_size = args.batch_size or int(TRAIN["batch_size"])

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
            if args.es_abs_tol is not None:
                es_kw["es_abs_tol"] = float(args.es_abs_tol)
            if args.es_plateau_rel_tol is not None:
                es_kw["es_plateau_rel_tol"] = float(args.es_plateau_rel_tol)
                print(f"  [early-stop] scale-adaptive plateau: stop when best val-η² "
                      f"improves < {float(args.es_plateau_rel_tol):.3g} (relative) over "
                      f"{es_kw['es_patience']} epochs (fires at coarse AND fine N)")
        print(f"  [protocol={args.protocol}] early-stop: patience={es_kw['es_patience']} "
              f"tol={es_kw['es_tol']:.3g} min_epochs={es_kw['es_min_epochs']} cap={max_epochs} | "
              f"lr {lr_init_use:.0e}->{lr_end_use:.0e} | monitor={es_kw.get('monitor', 'train')}")
    else:
        lr_init_use = float(TRAIN["lr1"])
        lr_end_use = float(TRAIN["lr2"])
        es_kw = {}

    if args.epochs_per_level is not None:                         # override the schedule
        epochs = {N: int(args.epochs_per_level) for N in levels}
    print(f"  p = {p_fe}")
    print(f"  levels = {levels}")
    print(f"  epochs = {epochs}  (init_mode={args.init_mode})")
    print(f"  batch_size = {batch_size}")

    q_est = int(args.q_est) if hasattr(args, "q_est") and args.q_est else int(ARCTAN.get("quad_estimator", 50))

    # Loss denominators (eq. 28): eta^2 on the uniform mesh at the current level,
    # recomputed per level via denoms_fn.
    if str(LOSS_NORM) != "uniform_same_level":
        raise ValueError(f"unsupported config.LOSS_NORM={LOSS_NORM!r}")
    denoms_fn = make_denoms_fn(
        train_sigmas, val_sigmas if val_protocol else None,
        kind="arctan", p=p_fe, q_est=q_est,
    )
    t_d = time.perf_counter()
    train_denoms, val_denoms = denoms_fn(int(levels[0]))
    print(f"  eq-26 denominators η²_unif at N={levels[0]}: "
          f"median={float(np.median(train_denoms)):.4e} "
          f"({time.perf_counter() - t_d:.2f}s; recomputed per level)")

    # Forcing/Neumann GL degree: --q-forcing overrides; else smoke=20, full=config.
    if args.q_forcing is not None:
        q_F_use = int(args.q_forcing)
    else:
        q_F_use = 20 if args.smoke else int(ARCTAN["quad_forcing"])
    print(f"  quadrature: q_K={p_fe + 1}  q_forcing={q_F_use}  q_est={q_est}  "
          f"(config defaults q_forcing={int(ARCTAN['quad_forcing'])}, q_est={int(ARCTAN.get('quad_estimator', 50))})")

    # H1-error history monitor (val protocol); same quadrature as the loss.
    val_forward_factory = None
    h1_metric_factory = None
    hist_note = "loss=eq26 uniform_same_level eps=1e-12"
    if val_protocol:
        q_metric_denom = int(ARCTAN.get("quad_metric", 50)) if not args.smoke else 20
        metric_dens = compute_p2_denoms(val_sigmas, q_metric=q_metric_denom)
        val_forward_factory = lambda N: make_val_forward_arctan(
            p=p_fe, n_elem=int(N), q_K=p_fe + 1, q_F=q_F_use, q_est=q_est,
            epsilon=float(EPSILON_DENOM), solver=str(args.solver),
        )
        h1_metric_factory = make_h1_metric_factory_arctan(
            p_fe, metric_denoms=metric_dens, quad_points_per_dim=20,
        )
        hist_note += ("; h1_rel_val_median=median over FULL val set vs analytic "
                      "u* (H1-seminorm, q=20)")

    # Per-epoch history, streamed incrementally during training.
    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_path = hist_dir / f"history_p{p_fe}_seed{args.seed}.csv"

    # ---- Train ----
    res = train_with_continuation(
        experiment="arctan",
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
        q_F=q_F_use,
        q_est=q_est,
        epsilon=float(EPSILON_DENOM),
        uniform_init=bool(args.uniform_init),
        init_mode=str(args.init_mode),
        solver=str(args.solver),
        checkpoint_dir=ckpt_dir,
        resume=False,
        sigma_dim=int(ARCTAN["sigma_dim"]),
        hidden_dims=tuple(ARCTAN["hidden_dims"]),
        val_sigmas=val_sigmas, val_denoms=val_denoms,
        history_csv_path=hist_path,
        history_is_val=val_protocol,
        denoms_fn=denoms_fn,
        val_forward_factory=val_forward_factory,
        h1_metric_factory=h1_metric_factory,
        clip_norm=float(CLIP_NORM),
        warmup_epochs=int(WARMUP_EPOCHS),
        history_header_note=hist_note,
        **es_kw,
    )

    # ---- Save final checkpoint and meta ----
    final_ckpt = ckpt_dir / "checkpoint_final.npz"
    from src.parametric.continuation import save_checkpoint
    save_checkpoint(res.params_final, final_ckpt)
    print(f"  final checkpoint -> {final_ckpt}")

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
    with open(out_dir / f"meta_train_arctan_seed{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=2)

    # History was streamed during training; just confirm the file.
    try:
        n_hist = max(sum(1 for _ in open(hist_path)) - 1, 0)
    except FileNotFoundError:
        n_hist = 0
    print(f"  history -> {hist_path} ({n_hist} epochs, streamed incrementally)")

    print(f"=== train_arctan DONE in {meta['elapsed_sec']:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
