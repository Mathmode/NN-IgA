from __future__ import annotations

"""Experiment 4 (2D): Laplace on an L-shape multipatch domain (JAX).

Training exercise goals:
  1) softmax mesh parametrisation,
  2) multipatch assembly + solve (CG or dense),
  3) residual estimator-driven r-adapt optimisation,
  4) reproducible CSV layout + run metadata.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from jax import config as jcfg

jcfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODS = ROOT / "2D"
if str(MODS) not in sys.path:
    sys.path.insert(0, str(MODS))

from results_io import StandardResults
from export_utils_lshape_jax import export_lshape_case
from mesh_param_jax import build_breakpoints_softmax
from lshape_dof import build_global_dof_maps, dirichlet_dofs_lshape
from lshape_poisson_jax import solve_lshape
from lshape_estimator_jax import estimator_eta2
from lshape_metrics_jax import compute_error_metrics
from lshape_optimize_jax import TrainConfig, train_estimator_adam

from csv_utils import write_csv_dicts

from experiment_utils import parse_int_list, write_run_metadata


def choose_precond(n: int, mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode == "jacobi":
        return "jacobi"
    if mode in ("block_jacobi", "block-jacobi", "block"):
        return "block_jacobi"
    if mode == "none":
        return "none"
    if mode == "auto":
        # As meshes get finer, r-adapt can produce severe anisotropy near the
        # re-entrant corner. Block-Jacobi is more expensive per application
        # than Jacobi, but can substantially reduce CG iterations on large N.
        if int(n) >= 32:
            return "block_jacobi"
        return "jacobi" if int(n) >= 16 else "none"
    raise ValueError("precond must be one of: auto, block_jacobi, jacobi, none")


def choose_solver_by_n(n: int, mode: str) -> str:
    """Exercise 4 policy: dense for N<=16, CG for N>=32 (independent of DOF threshold)."""
    mode = str(mode).lower().strip()
    if mode in ("dense", "cg"):
        return mode
    if mode == "auto":
        return "dense" if int(n) <= 16 else "cg"
    raise ValueError("solver must be one of: auto, dense, cg")


def resolve_quadrature(p: int, q_order: int, q_order_val: int, q_order_err: int):
    p = int(p)
    q = (p + 1) if int(q_order) <= 0 else int(q_order)
    qv = (2 * (p + 1)) if int(q_order_val) <= 0 else int(q_order_val)
    qe = 20 if int(q_order_err) <= 0 else int(q_order_err)
    return q, qv, qe


def midpoint_refine_breakpoints(bp: np.ndarray) -> np.ndarray:
    bp = np.asarray(bp, dtype=float)
    mids = 0.5 * (bp[:-1] + bp[1:])
    out = np.empty((bp.size + mids.size,), dtype=float)
    out[0::2] = bp
    out[1::2] = mids
    return out


def theta_from_breakpoints(bp: np.ndarray, h_min: float) -> jnp.ndarray:
    """Invert softmax-of-sizes to an (unnormalized) theta (NumPy -> JAX array)."""
    bp = np.asarray(bp, dtype=float)
    h = bp[1:] - bp[:-1]
    L = float(bp[-1] - bp[0])
    n = int(h.size)
    denom = L - n * float(h_min)
    if denom <= 0.0:
        denom = 1e-12
    delta = (h - float(h_min)) / denom
    eps = 1e-32
    delta = np.clip(delta, eps, None)
    delta = delta / float(delta.sum())
    theta = np.log(delta)
    theta = theta - float(theta.mean())
    return jnp.asarray(theta, dtype=jnp.float64)


def main():
    ap = argparse.ArgumentParser(description="Experiment 4: L-shape Laplace")
    ap.add_argument("--n-list", type=str, default="4,8,16,32", help="Comma-separated list of n (elements per interval)")
    ap.add_argument("--p", type=int, default=2, help="Spline degree")
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--q-order", type=int, default=0)
    ap.add_argument("--q-order-val", type=int, default=0)
    ap.add_argument("--q-order-err", type=int, default=20)
    ap.add_argument("--cg-tol-train", type=float, default=1e-5)
    ap.add_argument("--cg-maxiter", type=int, default=4000)
    ap.add_argument("--h-min", type=float, default=1e-5)
    ap.add_argument("--mesh-reg", type=float, default=1e-5, help="Smoothness regularization weight (0 disables)")
    ap.add_argument("--mesh-reg-hmin", type=float, default=0.0, help="Penalty for very small elements (0 disables)")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip (0 disables)")
    ap.add_argument("--warmstart", type=str, default="uniform", help="Initialisation: uniform|midpoint")
    ap.add_argument("--precond", type=str, default="auto", help="Preconditioner: auto|block_jacobi|jacobi|none")
    ap.add_argument("--solver-eval", type=str, default="auto", help="Eval solver: auto|cg|dense")
    ap.add_argument(
        "--solver-train",
        type=str,
        default="auto",
        help="Train solver: cg|dense|auto (default auto; dense only recommended for small N)",
    )
    ap.add_argument(
        "--dense-max-dofs",
        type=int,
        default=1500,
        help="Max global DOFs for dense solve when enabled (auto switches to CG above this)",
    )
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=str, default="results_exp4_lshape")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    write_run_metadata(
        out_root,
        repo_root=ROOT,
        args=vars(args),
        extra={
            "experiment": 4,
            "problem": "lshape_poisson",
            "jax_backend": str(jax.default_backend()),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jcfg.read("jax_enable_x64")),
        },
    )

    res = StandardResults(out_root, tag="exp4_lshape")

    n_list = parse_int_list(args.n_list)
    p = int(args.p)

    q_order, q_order_val, q_order_err = resolve_quadrature(p, args.q_order, args.q_order_val, args.q_order_err)

    series_rows = []

    prev_n: int | None = None
    prev_bp_adapt: dict[str, np.ndarray] | None = None

    for n in n_list:
        n = int(n)
        precond = choose_precond(n, args.precond)
        solver_train = choose_solver_by_n(n, args.solver_train)
        solver_eval = choose_solver_by_n(n, args.solver_eval)

        nBxL = n + p
        nBxR = n + p
        nByB = n + p
        nByT = n + p
        g0_np, g1_np, g2_np, n_dofs = build_global_dof_maps(nBxL, nBxR, nByB, nByT)
        dir_np = dirichlet_dofs_lshape(g0_np, g1_np, g2_np)
        free_np = np.setdiff1d(np.arange(int(n_dofs), dtype=np.int32), np.asarray(dir_np, dtype=np.int32))

        g0 = jnp.asarray(g0_np, dtype=jnp.int32)
        g1 = jnp.asarray(g1_np, dtype=jnp.int32)
        g2 = jnp.asarray(g2_np, dtype=jnp.int32)
        dirichlet = jnp.asarray(dir_np, dtype=jnp.int32)
        free_idx = jnp.asarray(free_np, dtype=jnp.int32)

        # ---------------------------
        # Uniform baseline
        # ---------------------------
        theta0 = jnp.zeros((n,), dtype=jnp.float64)
        bp_xL = build_breakpoints_softmax(theta0, -1.0, 0.0, h_min=float(args.h_min))
        bp_xR = build_breakpoints_softmax(theta0, 0.0, 1.0, h_min=float(args.h_min))
        bp_yB = build_breakpoints_softmax(theta0, -1.0, 0.0, h_min=float(args.h_min))
        bp_yT = build_breakpoints_softmax(theta0, 0.0, 1.0, h_min=float(args.h_min))

        u_u, sys_u = solve_lshape(
            bp_xL,
            bp_xR,
            bp_yB,
            bp_yT,
            p=p,
            q_order=int(q_order),
            solver=str(solver_eval),
            dense_max_dofs=int(args.dense_max_dofs),
            cg_tol=1e-10,
            cg_maxiter=int(args.cg_maxiter),
            precond=str(precond),
            g0=g0,
            g1=g1,
            g2=g2,
            dirichlet=dirichlet,
            free_idx=free_idx,
            n_dofs=int(n_dofs),
        )

        eta2_u, _ = estimator_eta2(
            u_u,
            bp_xL=bp_xL,
            bp_xR=bp_xR,
            bp_yB=bp_yB,
            bp_yT=bp_yT,
            knots_xL=sys_u.patches["P0"].knots_x,
            knots_xR=sys_u.patches["P2"].knots_x,
            knots_yB=sys_u.patches["P0"].knots_y,
            knots_yT=sys_u.patches["P1"].knots_y,
            g0=sys_u.g0,
            g1=sys_u.g1,
            g2=sys_u.g2,
            p=p,
            q_order=int(q_order_val),
            include_jump=True,
        )
        eta2_u_f = float(jax.device_get(eta2_u))
        metrics_u = compute_error_metrics(
            u_u,
            bp_xL=bp_xL,
            bp_xR=bp_xR,
            bp_yB=bp_yB,
            bp_yT=bp_yT,
            g0=sys_u.g0,
            g1=sys_u.g1,
            g2=sys_u.g2,
            p=p,
            q_order=int(q_order_err),
        )
        Ieff_u = float(np.sqrt(float(eta2_u_f)) / (metrics_u["dG_err"] + 1e-30))

        export_lshape_case(res.method_dir(p, n, "uniform"), method="uniform", u_global=u_u, sys=sys_u)

        row_u = {
            "N": n,
            "p": p,
            "method": "uniform",
            "solver": str(solver_eval),
            "solver_train": str(solver_train),
            "solver_eval": str(solver_eval),
            "dense_max_dofs": int(args.dense_max_dofs),
            "precond": str(precond),
            "iters": 0,
            "warmstart": "none",
            **metrics_u,
            "eta2_total": float(eta2_u_f),
            "loss_val": float(0.5 * eta2_u_f),
            "reliability_index": Ieff_u,
            "elapsed_sec": 0.0,
        }
        series_rows.append(row_u)

        # ---------------------------
        # r-adapt training
        # ---------------------------
        init_theta = None
        warmstart_tag = "uniform"
        if str(args.warmstart).lower().strip() == "midpoint":
            if prev_bp_adapt is not None and prev_n is not None and n == 2 * prev_n:
                bp_init = {
                    "xL": midpoint_refine_breakpoints(prev_bp_adapt["xL"]),
                    "xR": midpoint_refine_breakpoints(prev_bp_adapt["xR"]),
                    "yB": midpoint_refine_breakpoints(prev_bp_adapt["yB"]),
                    "yT": midpoint_refine_breakpoints(prev_bp_adapt["yT"]),
                }
                init_theta = {
                    "xL": theta_from_breakpoints(bp_init["xL"], h_min=float(args.h_min)),
                    "xR": theta_from_breakpoints(bp_init["xR"], h_min=float(args.h_min)),
                    "yB": theta_from_breakpoints(bp_init["yB"], h_min=float(args.h_min)),
                    "yT": theta_from_breakpoints(bp_init["yT"], h_min=float(args.h_min)),
                }
                warmstart_tag = "midpoint"
        elif str(args.warmstart).lower().strip() != "uniform":
            raise ValueError("warmstart must be one of: uniform, midpoint")

        cfg = TrainConfig(
            p=p,
            iters=int(args.iters),
            lr1=float(args.lr1),
            lr2=float(args.lr2),
            q_order=q_order,
            q_order_val=q_order_val,
            q_order_err=q_order_err,
            h_min=float(args.h_min),
            include_jump=True,
            log_every=int(args.log_every),
            cg_tol=float(args.cg_tol_train),
            cg_maxiter=int(args.cg_maxiter),
            precond=str(precond),
            solver=str(solver_train),
            dense_max_dofs=int(args.dense_max_dofs),
            mesh_reg=float(args.mesh_reg),
            mesh_reg_hmin=float(args.mesh_reg_hmin),
            grad_clip=float(args.grad_clip),
        )

        train_out = train_estimator_adam(
            n_xL=n,
            n_xR=n,
            n_yB=n,
            n_yT=n,
            cfg=cfg,
            init_theta=init_theta,
        )

        bp = train_out["breakpoints"]
        prev_bp_adapt = bp
        prev_n = n

        bp_xL_a = jnp.asarray(bp["xL"], dtype=jnp.float64)
        bp_xR_a = jnp.asarray(bp["xR"], dtype=jnp.float64)
        bp_yB_a = jnp.asarray(bp["yB"], dtype=jnp.float64)
        bp_yT_a = jnp.asarray(bp["yT"], dtype=jnp.float64)

        u_a, sys_a = solve_lshape(
            bp_xL_a,
            bp_xR_a,
            bp_yB_a,
            bp_yT_a,
            p=p,
            q_order=int(q_order),
            solver=str(solver_eval),
            dense_max_dofs=int(args.dense_max_dofs),
            cg_tol=1e-10,
            cg_maxiter=int(args.cg_maxiter),
            precond=str(precond),
            g0=g0,
            g1=g1,
            g2=g2,
            dirichlet=dirichlet,
            free_idx=free_idx,
            n_dofs=int(n_dofs),
        )

        eta2_a, _ = estimator_eta2(
            u_a,
            bp_xL=bp_xL_a,
            bp_xR=bp_xR_a,
            bp_yB=bp_yB_a,
            bp_yT=bp_yT_a,
            knots_xL=sys_a.patches["P0"].knots_x,
            knots_xR=sys_a.patches["P2"].knots_x,
            knots_yB=sys_a.patches["P0"].knots_y,
            knots_yT=sys_a.patches["P1"].knots_y,
            g0=sys_a.g0,
            g1=sys_a.g1,
            g2=sys_a.g2,
            p=p,
            q_order=int(q_order_val),
            include_jump=True,
        )
        eta2_a_f = float(jax.device_get(eta2_a))
        metrics_a = compute_error_metrics(
            u_a,
            bp_xL=bp_xL_a,
            bp_xR=bp_xR_a,
            bp_yB=bp_yB_a,
            bp_yT=bp_yT_a,
            g0=sys_a.g0,
            g1=sys_a.g1,
            g2=sys_a.g2,
            p=p,
            q_order=int(q_order_err),
        )
        Ieff_a = float(np.sqrt(float(eta2_a_f)) / (metrics_a["dG_err"] + 1e-30))

        export_lshape_case(
            res.method_dir(p, n, "r_adapt"),
            method="r_adapt",
            u_global=u_a,
            sys=sys_a,
            history=train_out["history"],
        )

        row_a = {
            "N": n,
            "p": p,
            "method": "r_adapt",
            "solver": str(solver_eval),
            "solver_train": str(solver_train),
            "solver_eval": str(solver_eval),
            "dense_max_dofs": int(args.dense_max_dofs),
            "precond": str(precond),
            "iters": int(cfg.iters),
            "warmstart": warmstart_tag,
            **metrics_a,
            "eta2_total": float(eta2_a_f),
            "loss_val": float(0.5 * eta2_a_f),
            "reliability_index": Ieff_a,
            "elapsed_sec": float(train_out["elapsed_sec"]),
        }
        series_rows.append(row_a)

        # Case summary.csv
        fieldnames = sorted(set(row_u.keys()) | set(row_a.keys()))
        write_csv_dicts(res.summary_path(p, n), [row_u, row_a], fieldnames)

    # Per-degree series CSV
    if series_rows:
        fieldnames = [
            "N",
            "p",
            "method",
            "solver",
            "solver_train",
            "solver_eval",
            "dense_max_dofs",
            "precond",
            "iters",
            "warmstart",
            "dofs",
            "h_min",
            "h_max",
            "h_ratio",
            "L2_abs",
            "L2_rel",
            "H1_semi_abs",
            "H1_semi_rel",
            "H1_abs",
            "H1_rel",
            "dG_err",
            "eta2_total",
            "loss_val",
            "reliability_index",
            "elapsed_sec",
        ]
        res.write_series(p, series_rows, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
