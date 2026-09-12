"""Re-evaluate a saved ``advdiff`` checkpoint on the test set.

Writes the per-seed ``summary_p4_seed{seed}.csv`` (self-contained: metrics are
computed directly from the adv-diff solver + estimator). --N-levels may include
the zero-shot N=64. Use --protocol val for the paper's held-out test subset.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_TWO_D_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _TWO_D_ROOT.parent
for p in (str(_PROJECT_ROOT), str(_TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax.numpy as jnp
import numpy as np

import hashlib

from src.config import ADVDIFF, SPLIT_SEED, TRAIN_VAL
from src.parametric.continuation import load_checkpoint
from common.csv_utils import summary_is_complete
from src.parametric.training_advdiff import knots_p4_from_network_axis
from src.parametric.corrector_advdiff import corrector_advdiff
from src.parametric.positional_density_network_2d import cell_midpoints, forward
from common.parameter_sampling import build_advdiff_grid
from common.h1_seminorm_2d import (
    h1_seminorm_sq_2d,
    h1_seminorm_sq_analytic,
    open_uniform_knots,
)
from src.nonparametric.advdiff.pde import grad_u_exact_advdiff
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.advdiff.eta_estimator_advdiff import eta_squared_advdiff

CSV_COLUMNS = [
    "experiment", "p", "seed", "N", "method",
    "logeps", "b", "eps",
    "H1_semi_abs", "H1_semi_rel", "eta", "eff_index",
    "t_solve", "t_metric",
    # Corrector diagnostics (positional_corrected only; blank otherwise).
    "corrector_n_iter", "corrector_converged",
    "corrector_loss_init", "corrector_loss_final", "t_corrector_sec",
]


def _test_set_hash(test_nus) -> str:
    """Stable, order-independent hash of the test tuples so two eval runs can be
    asserted to share the SAME held-out test set."""
    rows = sorted(tuple(round(float(v), 12) for v in row) for row in np.asarray(test_nus))
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def _metrics_for_knots(knots_x, knots_y, *, p, N, eps, b, grad, q_K, q_F, q_metric, q_est):
    t0 = time.perf_counter()
    res = galerkin_solve_advdiff(
        knots_x, knots_y, p, N, N, float(eps), float(b), q_K=q_K, q_F=q_F,
    )
    u_h = np.asarray(res.u_h)
    t1 = time.perf_counter()
    H1_semi_abs_sq = float(h1_seminorm_sq_2d(
        u_h, np.asarray(knots_x), np.asarray(knots_y), p, grad,
        quad_points_per_dim=q_metric,
    ))
    norm_sq = float(h1_seminorm_sq_analytic(
        grad, quad_points_per_dim=q_metric, n_subdiv_per_dim=16,
    ))
    H1_semi_rel = math.sqrt(H1_semi_abs_sq / norm_sq) if norm_sq > 0 else float("nan")
    eta_sq = float(eta_squared_advdiff(
        jnp.asarray(u_h), knots_x, knots_y, p, N, N, float(eps), float(b), q_est=q_est,
    ))
    eta = math.sqrt(eta_sq) if eta_sq >= 0 else float("nan")
    eff_index = (eta / math.sqrt(H1_semi_abs_sq)
                 if H1_semi_abs_sq > 0 else float("nan"))
    t2 = time.perf_counter()
    return {
        "H1_semi_abs": H1_semi_abs_sq, "H1_semi_rel": H1_semi_rel,
        "eta": eta, "eff_index": eff_index,
        "t_solve": t1 - t0, "t_metric": t2 - t1,
    }


def evaluate_advdiff(params, test_nus, *, p, levels, seed, output_dir,
                q_K=None, q_F=50, q_metric=50, q_est=50,
                corrector_max_iter=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"summary_p4_seed{seed}.csv"
    if q_K is None:
        q_K = p + 1

    rows = []
    for N in levels:
        print(f"\n=== eval ADVDIFF N={N} ===")
        for nu in test_nus:
            logeps = float(nu[0]); b = float(nu[1]); eps = float(10.0 ** logeps)
            grad = lambda x, y: grad_u_exact_advdiff(x, y, eps, b)
            base = {
                "experiment": "advdiff", "p": int(p), "seed": int(seed), "N": int(N),
                "logeps": logeps, "b": b, "eps": eps,
            }
            # uniform baseline
            ku = jnp.asarray(open_uniform_knots(int(N), p))
            m_u = _metrics_for_knots(ku, ku, p=p, N=int(N), eps=eps, b=b, grad=grad,
                                     q_K=q_K, q_F=q_F, q_metric=q_metric, q_est=q_est)
            rows.append({**base, "method": "uniform", **m_u})
            # positional prediction
            nu_j = jnp.asarray(nu, dtype=jnp.asarray(0.0).dtype)
            kx = knots_p4_from_network_axis(params, nu_j, int(N), p, axis_id=0.0)
            ky = knots_p4_from_network_axis(params, nu_j, int(N), p, axis_id=1.0)
            m_p = _metrics_for_knots(kx, ky, p=p, N=int(N), eps=eps, b=b, grad=grad,
                                     q_K=q_K, q_F=q_F, q_metric=q_metric, q_est=q_est)
            rows.append({**base, "method": "positional", **m_p})

            # positional_corrected: warm-start from the network logits, refine by
            # L-BFGS-B on the training residual loss. In the convection-dominated
            # regime eta is not norm-equivalent to the H1 error, so the corrector
            # may improve or degrade it; the result is recorded either way, and a
            # failure degrades to a NaN row.
            t_c0 = time.perf_counter()
            try:
                xi = cell_midpoints(int(N))
                zx = forward(params, nu_j, xi, axis_id=0.0)
                zy = forward(params, nu_j, xi, axis_id=1.0)
                denom = float(h1_seminorm_sq_analytic(
                    grad, quad_points_per_dim=q_metric, n_subdiv_per_dim=16))
                cres = corrector_advdiff(
                    zx, zy, nu_j, denom,
                    p=p, n_elem=int(N), q_K=q_K, q_F=q_F, q_est=q_est,
                    max_iter=int(corrector_max_iter),
                )
                m_c = _metrics_for_knots(
                    cres.knots_x, cres.knots_y, p=p, N=int(N), eps=eps, b=b,
                    grad=grad, q_K=q_K, q_F=q_F, q_metric=q_metric, q_est=q_est)
                t_corr = time.perf_counter() - t_c0
                rows.append({
                    **base, "method": "positional_corrected", **m_c,
                    "corrector_n_iter": int(cres.n_iter),
                    "corrector_converged": int(bool(cres.converged)),
                    "corrector_loss_init": float(cres.loss_init),
                    "corrector_loss_final": float(cres.loss_final),
                    "t_corrector_sec": float(t_corr),
                })
                corr_h1 = m_c["H1_semi_rel"]
                corr_note = (f"corrected H1_rel={corr_h1:.3e} "
                             f"(n_iter={cres.n_iter}, conv={cres.converged})")
            except Exception as exc:  # noqa: BLE001 — never let the corrector kill the eval
                t_corr = time.perf_counter() - t_c0
                nan = float("nan")
                rows.append({
                    **base, "method": "positional_corrected",
                    "H1_semi_abs": nan, "H1_semi_rel": nan, "eta": nan,
                    "eff_index": nan, "t_solve": nan, "t_metric": nan,
                    "corrector_n_iter": -1, "corrector_converged": 0,
                    "corrector_loss_init": nan, "corrector_loss_final": nan,
                    "t_corrector_sec": float(t_corr),
                })
                corr_note = f"corrector FAILED: {type(exc).__name__}: {exc}"
                print(f"    [WARN] {corr_note}")

            print(f"  logeps={logeps:+.3f} b={b:.3f}  "
                  f"uniform H1_rel={m_u['H1_semi_rel']:.3e}  "
                  f"positional H1_rel={m_p['H1_semi_rel']:.3e} eff={m_p['eff_index']:.2f}  "
                  f"| {corr_note}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    print(f"\nP4 eval CSV -> {out_csv} ({len(rows)} rows)")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N-levels", type=int, nargs="+", default=None)
    ap.add_argument("--n-test-nus", type=int, default=None)
    ap.add_argument("--corrector-max-iter", type=int, default=30,
                    help="L-BFGS-B iteration cap for the positional_corrected method.")
    ap.add_argument("--protocol", choices=["fixed", "val"], default="fixed",
                    help="'val' (paper): the fixed 70/15/15 held-out test set, identical "
                         "across seeds. 'fixed' (default): legacy per-seed 70/30 split.")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if a complete summary CSV for this "
                         "seed already exists (default: skip complete seeds).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_advdiff] seed={args.seed} smoke={args.smoke} -> {out_dir}")

    # 'val': fixed SPLIT_SEED 70/15/15 shared across seeds; else per-seed 70/30.
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
    test_nus = grid["test"]
    print(f"  protocol={args.protocol}  test_set_hash={_test_set_hash(test_nus)}  "
          f"(val mode -> hash is seed-independent)")
    if args.smoke:
        n_test = int(args.n_test_nus or 4)
        # val mode subsamples with SPLIT_SEED so the smoke set is seed-independent.
        rng = np.random.default_rng(int(SPLIT_SEED) if val_protocol else args.seed)
        test_nus = test_nus[rng.choice(test_nus.shape[0], n_test, replace=False)]
    elif args.n_test_nus is not None:
        test_nus = test_nus[: int(args.n_test_nus)]
    print(f"  test nus: {test_nus.shape}")

    levels = tuple(args.N_levels) if args.N_levels else (
        (4,) if args.smoke else tuple(ADVDIFF["levels"])
    )
    print(f"  levels: {levels}")

    summary_path = out_dir / f"summary_p4_seed{int(args.seed)}.csv"
    if not args.force and summary_is_complete(summary_path, levels):
        print(f"[skip] {summary_path} already complete for N={list(levels)}; "
              f"pass --force to re-evaluate")
        return 0

    params = load_checkpoint(Path(args.checkpoint))
    out_csv = evaluate_advdiff(
        params, test_nus,
        p=int(args.p), levels=levels, seed=int(args.seed), output_dir=out_dir,
        q_K=int(args.p) + 1,
        q_F=40 if args.smoke else int(ADVDIFF["quad_forcing"]),
        q_metric=40 if args.smoke else int(ADVDIFF["quad_metric"]),
        q_est=40 if args.smoke else int(ADVDIFF["quad_eta"]),
        corrector_max_iter=int(args.corrector_max_iter),
    )
    print(f"=== eval_advdiff DONE -> {out_csv} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
