from __future__ import annotations

"""Experiment 3 (2D): Reaction–diffusion on the unit square (JAX).

Training exercise goals:
  1) softmax mesh parametrisation in each coordinate,
  2) tensor-product IGA assembly + solve (JAX),
  3) residual estimator-driven r-adapt optimisation,
  4) reproducible CSV layout + run metadata.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from jax import config as jcfg

jcfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

# Make `2D/` importable (modules are imported by filename, not as a package)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODS = ROOT / "2D"
if str(MODS) not in sys.path:
    sys.path.insert(0, str(MODS))

from results_io import StandardResults
from export_utils_jax import export_square_case
from mesh_param_jax import build_breakpoints_softmax
from square_reactdiff_jax import solve_square_reactdiff
from exp3_estimator_jax import estimator_eta2_exp3
from exp3_metrics_jax import compute_error_metrics_exp3
from exp3_optimize_jax import Exp3TrainConfig, train_exp3_estimator_adam

from csv_utils import write_csv_dicts

from experiment_utils import parse_int_list, write_run_metadata


def main():
    ap = argparse.ArgumentParser(description="Experiment 3: unit-square reaction--diffusion")
    ap.add_argument("--n-list", type=str, default="4,8,16,32,64,128", help="Comma-separated list of N (elements per axis)")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--lr1", type=float, default=1e-2)
    ap.add_argument("--lr2", type=float, default=1e-3)
    ap.add_argument("--q-order", type=int, default=12)
    ap.add_argument("--q-order-val", type=int, default=20)
    ap.add_argument("--q-order-err", type=int, default=20)
    ap.add_argument("--q-order-bc", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--h-min", type=float, default=1e-6)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=str, default="results_exp3_square")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    n_list = parse_int_list(args.n_list)
    p = int(args.p)

    write_run_metadata(
        out_root,
        repo_root=ROOT,
        args=vars(args),
        extra={
            "experiment": 3,
            "problem": "square_reactdiff",
            "jax_backend": str(jax.default_backend()),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jcfg.read("jax_enable_x64")),
        },
    )

    res = StandardResults(out_root, tag="exp3_square")

    series_rows = []

    for N in n_list:
        N = int(N)
        case_dir = res.case_dir(p, N)

        # -------------------------
        # Uniform baseline
        # -------------------------
        theta0 = jnp.zeros((int(N),), dtype=jnp.float64)
        bp_x = build_breakpoints_softmax(theta0, 0.0, 1.0, h_min=float(args.h_min))
        bp_y = build_breakpoints_softmax(theta0, 0.0, 1.0, h_min=float(args.h_min))

        u_u, sys_u = solve_square_reactdiff(
            bp_x,
            bp_y,
            p,
            eps=float(args.eps),
            sigma=float(args.sigma),
            q_order=int(args.q_order),
            q_order_bc=int(args.q_order_bc),
        )
        eta2_u, _ = estimator_eta2_exp3(
            u_u,
            bp_x,
            bp_y,
            sys_u.knots_x,
            sys_u.knots_y,
            p,
            eps=float(args.eps),
            sigma=float(args.sigma),
            q_order=int(args.q_order_val),
        )
        metrics_u = compute_error_metrics_exp3(
            u_u,
            bp_x,
            bp_y,
            sys_u.knots_x,
            sys_u.knots_y,
            p,
            q_order=int(args.q_order_err),
            eps=float(args.eps),
            sigma=float(args.sigma),
        )

        export_square_case(res.method_dir(p, N, "uniform"), method="uniform", u_global=u_u, sys=sys_u)

        row_u = {
            "N": N,
            "p": p,
            "method": "uniform",
            "iters": 0,
            "dofs": int(u_u.shape[0]),
            **metrics_u,
            "eta2_total": float(jax.device_get(eta2_u)),
            "loss_val": float(0.5 * float(jax.device_get(eta2_u))),
            "elapsed_sec": 0.0,
        }
        series_rows.append(row_u)

        # -------------------------
        # r-adapt training
        # -------------------------
        cfg = Exp3TrainConfig(
            p=p,
            iters=int(args.iters),
            lr1=float(args.lr1),
            lr2=float(args.lr2),
            q_order=int(args.q_order),
            q_order_val=int(args.q_order_val),
            q_order_err=int(args.q_order_err),
            q_order_bc=int(args.q_order_bc),
            log_every=int(args.log_every),
            eps=float(args.eps),
            sigma=float(args.sigma),
            h_min=float(args.h_min),
        )
        train_out = train_exp3_estimator_adam(N=N, cfg=cfg)

        bp_x_a = jnp.asarray(train_out["breakpoints"]["x"], dtype=jnp.float64)
        bp_y_a = jnp.asarray(train_out["breakpoints"]["y"], dtype=jnp.float64)

        u_a, sys_a = solve_square_reactdiff(
            bp_x_a,
            bp_y_a,
            p,
            eps=float(args.eps),
            sigma=float(args.sigma),
            q_order=int(args.q_order),
            q_order_bc=int(args.q_order_bc),
        )
        eta2_a, _ = estimator_eta2_exp3(
            u_a,
            bp_x_a,
            bp_y_a,
            sys_a.knots_x,
            sys_a.knots_y,
            p,
            eps=float(args.eps),
            sigma=float(args.sigma),
            q_order=int(args.q_order_val),
        )
        metrics_a = compute_error_metrics_exp3(
            u_a,
            bp_x_a,
            bp_y_a,
            sys_a.knots_x,
            sys_a.knots_y,
            p,
            q_order=int(args.q_order_err),
            eps=float(args.eps),
            sigma=float(args.sigma),
        )

        export_square_case(
            res.method_dir(p, N, "r_adapt"),
            method="r_adapt",
            u_global=u_a,
            sys=sys_a,
            history=train_out["history"],
        )

        row_a = {
            "N": N,
            "p": p,
            "method": "r_adapt",
            "iters": int(args.iters),
            "dofs": int(u_a.shape[0]),
            **metrics_a,
            "eta2_total": float(jax.device_get(eta2_a)),
            "loss_val": float(0.5 * float(jax.device_get(eta2_a))),
            "elapsed_sec": float(train_out["elapsed_sec"]),
        }
        series_rows.append(row_a)
        # Case summary.csv
        fieldnames = sorted(set(row_u.keys()) | set(row_a.keys()))
        write_csv_dicts(res.summary_path(p, N), [row_u, row_a], fieldnames)

    # Per-degree series CSV
    fieldnames = ["N", "p", "method", "iters", "dofs", "L2_abs", "L2_rel", "H1_abs", "H1_rel", "h_min", "h_max", "h_ratio", "eta2_total", "loss_val", "elapsed_sec"]
    res.write_series(p, series_rows, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
