from __future__ import annotations

"""Experiment 2 (1D): Helmholtz with a C0 interface (JAX + JIT).

Training exercise goals:
  1) fixed-split mesh parametrisation (JIT-friendly static shapes),
  2) vectorised IGA assembly + solve (dense),
  3) residual estimator with jump + boundary terms,
  4) end-to-end differentiable r-adapt optimisation (Optax),
  5) reproducible CSV + metadata exports.

Reproducibility:
  - writes `run_metadata.json` in the experiment output root.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from jax import config as jcfg

jcfg.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# Make `1D/` importable (repo-local, no installation required)
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ONE_D = REPO_ROOT / "1D"
if str(ONE_D) not in sys.path:
    sys.path.insert(0, str(ONE_D))

from csv_utils import write_csv_dicts
from results_io import StandardResults

from helmholtz_problem_jax import OneInterfaceHelmholtzPaper
from helmholtz_mesh_param_jax import allocate_by_phase, build_piecewise_uniform_breakpoints
from helmholtz_iga_jax import solve_helmholtz
from helmholtz_estimator_jax import estimator_eta2
from helmholtz_metrics_jax import compute_error_metrics
from helmholtz_optimize_jax import TrainConfig, train_estimator_adam
from export_utils_1d import save_full_state, save_history_csv

from experiment_utils import parse_int_list, write_run_metadata


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def run_one_case(
    *,
    io: StandardResults,
    problem: OneInterfaceHelmholtzPaper,
    n_elem: int,
    p: int,
    iters: int,
    lr1: float,
    lr2: float,
    q_order: int,
    q_order_val: int,
    q_order_err: int,
    h_min: float,
    log_every: int,
    n_eval: int,
) -> tuple[dict[str, object], dict[str, object]]:
    xI = float(problem.xI)

    # Initial split (heuristic) for r-adapt initialisation
    n_left_init, n_right_init = allocate_by_phase(n_elem, xI, problem.k_L, problem.k_R, p=p)
    bp_init = build_piecewise_uniform_breakpoints(n_elem, xI, n_left_init)

    # Uniform baseline: choose n_left so xI is a breakpoint
    n_left_uniform = _round_half_up(n_elem * xI)
    n_left_uniform = max(1, min(n_elem - 1, n_left_uniform))
    bp_uniform = build_piecewise_uniform_breakpoints(n_elem, xI, n_left_uniform)

    # -----------------------
    # Uniform solve
    # -----------------------
    bp_u = jnp.asarray(bp_uniform, dtype=jnp.float64)
    u_u, knots_u = solve_helmholtz(
        bp_u,
        int(p),
        xI=float(problem.xI),
        sigma_L=float(problem.sigma_L),
        sigma_R=float(problem.sigma_R),
        alpha_L=float(problem.alpha_L),
        alpha_R=float(problem.alpha_R),
        n_left=int(n_left_uniform),
        q_order=int(q_order),
        bc_neumann=True,
        u0=float(problem.u0()),
        u1=float(problem.u1()),
        gN=float(problem.gN_right()),
    )

    metrics_u = compute_error_metrics(
        u_coeffs=u_u,
        breakpoints=bp_u,
        knots=knots_u,
        p=int(p),
        xI=float(problem.xI),
        sigma_L=float(problem.sigma_L),
        sigma_R=float(problem.sigma_R),
        alpha_L=float(problem.alpha_L),
        alpha_R=float(problem.alpha_R),
        k_L=float(problem.k_L),
        k_R=float(problem.k_R),
        A_L=float(problem.A_L),
        A_R=float(problem.A_R),
        B_R=float(problem.B_R),
        gN=float(problem.gN_right()),
        n_left=int(n_left_uniform),
        q_order=int(q_order_err),
    )

    eta2_u, _terms_u = estimator_eta2(
        u_u,
        bp_u,
        knots_u,
        int(p),
        xI=float(problem.xI),
        sigma_L=float(problem.sigma_L),
        sigma_R=float(problem.sigma_R),
        alpha_L=float(problem.alpha_L),
        alpha_R=float(problem.alpha_R),
        gN=float(problem.gN_right()),
        n_left=int(n_left_uniform),
        q_order=int(q_order_val),
        bc_neumann=True,
        include_jump=True,
    )

    eta_u = float(np.sqrt(float(eta2_u)))
    dG_u = float(metrics_u["dG_err"])
    Ieff_u = float(eta_u / (dG_u + 1e-30))

    row_u: dict[str, object] = {
        "N": int(n_elem),
        "p": int(p),
        "method": "uniform",
        "n_left": int(n_left_uniform),
        "dofs": int(u_u.size),
        **{k: float(metrics_u[k]) for k in metrics_u.keys()},
        "eta2_total": float(eta2_u),
        "loss_val": float(0.5 * eta2_u),
        "iters": 0,
        "reliability_index": float(Ieff_u),
        "elapsed_sec": 0.0,
    }

    # Export uniform state
    out_u = io.case_dir(p, n_elem, "uniform")
    save_full_state(
        out_u,
        breakpoints=np.asarray(bp_u),
        knots=np.asarray(knots_u),
        p=p,
        u_coeffs=np.asarray(u_u),
        u_exact=problem.u_exact,
        du_exact=problem.du_exact,
        n_eval=n_eval,
    )

    # -----------------------
    # r-adapt training
    # -----------------------
    cfg = TrainConfig(
        p=p,
        iters=iters,
        lr1=lr1,
        lr2=lr2,
        lr_switch=0.5,
        q_order=q_order,
        q_order_val=q_order_val,
        q_order_err=q_order_err,
        h_min=h_min,
        bc_right="neumann",
        include_jump=True,
        log_every=log_every,
    )

    train_out = train_estimator_adam(
        problem,
        n_left=n_left_init,
        n_right=n_right_init,
        cfg=cfg,
        init_breakpoints=bp_init,
    )

    bp_a = jnp.asarray(train_out["breakpoints"], dtype=jnp.float64)
    u_a = jnp.asarray(train_out["u_coeffs"], dtype=jnp.float64)
    knots_a = jnp.asarray(train_out["knots"], dtype=jnp.float64)
    n_left_a = int(train_out["n_left_final"])

    metrics_a = compute_error_metrics(
        u_coeffs=u_a,
        breakpoints=bp_a,
        knots=knots_a,
        p=int(p),
        xI=float(problem.xI),
        sigma_L=float(problem.sigma_L),
        sigma_R=float(problem.sigma_R),
        alpha_L=float(problem.alpha_L),
        alpha_R=float(problem.alpha_R),
        k_L=float(problem.k_L),
        k_R=float(problem.k_R),
        A_L=float(problem.A_L),
        A_R=float(problem.A_R),
        B_R=float(problem.B_R),
        gN=float(problem.gN_right()),
        n_left=int(n_left_a),
        q_order=int(q_order_err),
    )

    eta2_a, _terms_a = estimator_eta2(
        u_a,
        bp_a,
        knots_a,
        int(p),
        xI=float(problem.xI),
        sigma_L=float(problem.sigma_L),
        sigma_R=float(problem.sigma_R),
        alpha_L=float(problem.alpha_L),
        alpha_R=float(problem.alpha_R),
        gN=float(problem.gN_right()),
        n_left=int(n_left_a),
        q_order=int(q_order_val),
        bc_neumann=True,
        include_jump=True,
    )

    eta_a = float(np.sqrt(float(eta2_a)))
    dG_a = float(metrics_a["dG_err"])
    Ieff_a = float(eta_a / (dG_a + 1e-30))

    row_a: dict[str, object] = {
        "N": int(n_elem),
        "p": int(p),
        "method": "r_adapt",
        "n_left": int(n_left_a),
        "dofs": int(u_a.size),
        **{k: float(metrics_a[k]) for k in metrics_a.keys()},
        "eta2_total": float(eta2_a),
        "loss_val": float(0.5 * eta2_a),
        "iters": int(iters),
        "reliability_index": float(Ieff_a),
        "elapsed_sec": float(train_out["elapsed_sec"]),
    }

    out_a = io.case_dir(p, n_elem, "r_adapt")
    save_full_state(
        out_a,
        breakpoints=np.asarray(bp_a),
        knots=np.asarray(knots_a),
        p=p,
        u_coeffs=np.asarray(u_a),
        u_exact=problem.u_exact,
        du_exact=problem.du_exact,
        n_eval=n_eval,
    )
    save_history_csv(out_a / "history.csv", train_out["history"])

    # summary.csv at <...>/p{p}/N{N}/summary.csv
    case_root = out_u.parent
    write_csv_dicts(
        case_root / "summary.csv",
        rows=[row_u, row_a],
        fieldnames=sorted(set(row_u.keys()) | set(row_a.keys())),
    )

    return row_u, row_a


def main():
    ap = argparse.ArgumentParser(description="Experiment 2: Helmholtz 1D C0 interface")
    ap.add_argument("--out", type=str, default="results_1D")
    ap.add_argument("--n-list", type=str, default="24,32,46,64,93,128,186,256")
    ap.add_argument("--p-list", type=str, default="2,3")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--q-order", type=int, default=20)
    ap.add_argument("--q-order-val", type=int, default=40)
    ap.add_argument("--q-order-err", type=int, default=60)
    ap.add_argument("--h-min", type=float, default=1e-6)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu", help="Ignored (kept for backwards compatibility)")
    ap.add_argument("--n-eval", type=int, default=2001, help="points for solution.csv")

    args = ap.parse_args()

    out_root = Path(args.out) / "helmholtz"
    io = StandardResults(root=out_root, tag="helmholtz")

    write_run_metadata(
        out_root,
        repo_root=REPO_ROOT,
        args=vars(args),
        extra={
            "experiment": 2,
            "problem": "helmholtz_interface",
            "xI": 0.5,
            "jax_backend": str(jax.default_backend()),
            "jax_devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jcfg.read("jax_enable_x64")),
        },
    )

    n_list = parse_int_list(args.n_list)
    p_list = parse_int_list(args.p_list)

    for p in p_list:
        rows_p: list[dict[str, object]] = []

        for n_elem in n_list:

            problem = OneInterfaceHelmholtzPaper(xI=0.5)

            row_u, row_a = run_one_case(
                io=io,
                problem=problem,
                n_elem=int(n_elem),
                p=int(p),
                iters=int(args.iters),
                lr1=float(args.lr1),
                lr2=float(args.lr2),
                q_order=int(args.q_order),
                q_order_val=int(args.q_order_val),
                q_order_err=int(args.q_order_err),
                h_min=float(args.h_min),
                log_every=int(args.log_every),
                n_eval=int(args.n_eval),
            )

            rows_p.extend([row_u, row_a])

        # Per-degree series CSV at <out>/helmholtz/helmholtz_p{p}.csv
        fieldnames = sorted(set().union(*[r.keys() for r in rows_p])) if rows_p else ["N", "p", "method"]
        io.write_series(p=int(p), rows=rows_p, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
