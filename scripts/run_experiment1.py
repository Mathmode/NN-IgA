from __future__ import annotations

"""Experiment 1 (1D): Power manufactured solution (JAX).

This script is meant to be a *training exercise* for the full r-adapt pipeline:
  1) parametrise the mesh with unconstrained `theta`,
  2) assemble + solve the IGA Galerkin system (JAX, differentiable),
  3) compute a residual estimator (analytic + Gauss–Legendre),
  4) optimise `theta` with Adam (Optax),
  5) export a reproducible CSV layout.

Reproducibility:
  - writes `run_metadata.json` in the experiment output root.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# JAX is optional for Experiment 2, but required here
from jax import config as jcfg

jcfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Make `1D/` importable (repo-local)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ONE_D = REPO_ROOT / "1D"
if str(ONE_D) not in sys.path:
    sys.path.insert(0, str(ONE_D))

from csv_utils import write_csv_dicts
from results_io import StandardResults

from power_assembly import assemble_system_analytic
from power_knots import uniform_knots
from power_metrics import compute_errors_power_analytic, right_flux
from power_pde import BETA, du_exact, u_exact
from power_quadrature import estimator_eta2_analytic_power, estimator_loss_analytic_power
from power_r_adapt import r_adapt_optimize_estimator
from power_solver import solve_system
from export_utils_1d import save_full_state, save_history_csv

from experiment_utils import parse_int_list, write_run_metadata


def run_one_case(
    *,
    io: StandardResults,
    p: int,
    N: int,
    beta: float,
    qK: int,
    iters: int,
    lr0: float,
    lr1: float,
    log_every: int,
    h_min: float,
    n_eval: int,
    seed: int,
):
    # ---------- uniform baseline ----------
    knots_u = uniform_knots(int(N), int(p), a=0.0, b=1.0, dtype=jnp.float64)
    K_u, F_u = assemble_system_analytic(knots_u, int(p), float(beta), qK=int(qK))
    U_u = solve_system(K_u, F_u)

    H1_u, L2_u = compute_errors_power_analytic(
        knots_u, int(p), U_u, beta=float(beta), relative=True
    )
    flux_u = right_flux(knots_u, int(p), U_u)
    eta2_u = estimator_eta2_analytic_power(knots_u, int(p), U_u, beta=float(beta))
    loss_u = float(0.5 * eta2_u)

    # Save uniform artifacts
    out_u = io.case_dir(int(p), int(N), "uniform")
    save_full_state(
        out_u,
        knots=knots_u,
        p=int(p),
        u_coeffs=U_u,
        u_exact=u_exact,
        du_exact=du_exact,
        n_eval=int(n_eval),
    )

    row_u = {
        "N": int(N),
        "p": int(p),
        "method": "uniform",
        "iters": 0,
        "dofs": int(U_u.shape[0]),
        "h_min": float(np.min(np.diff(np.asarray(knots_u)[int(p):-int(p)]))),
        "H1_rel": float(H1_u),
        "L2_rel": float(L2_u),
        "eta2_total": float(eta2_u),
        "loss_val": float(loss_u),
        "flux_right": float(flux_u),
        "elapsed_sec": 0.0,
    }

    # ---------- r-adapt ----------
    theta0 = jnp.zeros((int(N),), dtype=jnp.float64)
    key = jax.random.PRNGKey(int(seed))
    keyN = jax.random.fold_in(key, int(N))

    t0 = time.perf_counter()
    theta_opt, knots_opt, U_opt, history, summary = r_adapt_optimize_estimator(
        theta0,
        degree=int(p),
        beta=float(beta),
        qK=int(qK),
        iters=int(iters),
        lr0=float(lr0),
        lr1=float(lr1),
        key=keyN,
        log_every=int(log_every),
        h_min=float(h_min),
    )
    train_dt = time.perf_counter() - t0

    H1_a, L2_a = compute_errors_power_analytic(
        knots_opt, int(p), U_opt, beta=float(beta), relative=True
    )
    flux_a = right_flux(knots_opt, int(p), U_opt)
    eta2_a = estimator_eta2_analytic_power(knots_opt, int(p), U_opt, beta=float(beta))
    loss_a = float(estimator_loss_analytic_power(knots_opt, int(p), U_opt, beta=float(beta)))

    # Save r-adapt artifacts
    out_a = io.case_dir(int(p), int(N), "r_adapt")
    save_full_state(
        out_a,
        knots=knots_opt,
        p=int(p),
        u_coeffs=U_opt,
        u_exact=u_exact,
        du_exact=du_exact,
        n_eval=int(n_eval),
    )
    save_history_csv(out_a / "history.csv", history)

    row_a = {
        "N": int(N),
        "p": int(p),
        "method": "r_adapt",
        "iters": int(iters),
        "dofs": int(U_opt.shape[0]),
        "h_min": float(np.min(np.diff(np.asarray(knots_opt)[int(p):-int(p)]))),
        "H1_rel": float(H1_a),
        "L2_rel": float(L2_a),
        "eta2_total": float(eta2_a),
        "loss_val": float(loss_a),
        "flux_right": float(flux_a),
        "elapsed_sec": float(train_dt),
        **{k: v for k, v in summary.items() if k not in {"elapsed_sec"}},
    }

    # ---------- per-N summary.csv ----------
    case_root = out_u.parent
    fieldnames = sorted(set(row_u.keys()) | set(row_a.keys()))
    write_csv_dicts(case_root / "summary.csv", [row_u, row_a], fieldnames)

    return row_u, row_a


def main():
    ap = argparse.ArgumentParser(description="Experiment 1: power solution")
    ap.add_argument("--out", type=str, default="results_1D")
    ap.add_argument("--p-list", type=str, default="2,3")
    ap.add_argument("--n-list", type=str, default="2,4,8,16,32,64,128,256")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--lr0", type=float, default=1e-2)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--h-min", type=float, default=1e-6)
    ap.add_argument("--n-eval", type=int, default=2001)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    out_root = Path(args.out) / "power_solution"
    out_root.mkdir(parents=True, exist_ok=True)

    write_run_metadata(
        out_root,
        repo_root=REPO_ROOT,
        args=vars(args),
        extra={
            "experiment": 1,
            "problem": "power_solution",
            "beta": float(BETA),
            "jax_backend": str(jax.default_backend()),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jcfg.read("jax_enable_x64")),
        },
    )

    p_list = parse_int_list(args.p_list)
    n_list = parse_int_list(args.n_list)

    for p in p_list:
        # Standard layout is <out>/p{p}/N{N}/{uniform|r_adapt}/...
        io = StandardResults(root=out_root, tag="power")
        rows_p = []

        qK = int(p) + 1  # robust GL order for stiffness

        # heuristic: scale iters with N if user kept default
        for N in n_list:
            iters = int(args.iters)
            if "iters" not in vars(args) or args.iters == 40000:
                if N <= 64:
                    iters = 40_000
                elif N == 128:
                    iters = 80_000
                else:
                    iters = 300_000

            row_u, row_a = run_one_case(
                io=io,
                p=int(p),
                N=int(N),
                beta=float(BETA),
                qK=qK,
                iters=iters,
                lr0=float(args.lr0),
                lr1=float(args.lr1),
                log_every=int(args.log_every),
                h_min=float(args.h_min),
                n_eval=int(args.n_eval),
                seed=int(args.seed),
            )
            rows_p.extend([row_u, row_a])

        # Per-degree series CSV at <out>/power_solution/power_p{p}.csv
        fieldnames = sorted(set().union(*[r.keys() for r in rows_p])) if rows_p else ["N", "p", "method"]
        io.write_series(p=int(p), rows=rows_p, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
