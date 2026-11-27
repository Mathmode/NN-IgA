"""Algorithm 1 in 2D with the continuation ladder (revision v2, items 3.1, 3.3, 3.4).

Per-instance r-adaptivity (L-BFGS-B on the block logits, prolongation N -> 2N by
splitting every element in two) with the estimator loss and/or the Ritz energy,
compared with the uniform mesh and, optionally, the trained networks at the same
levels and instances.

  arctan   Experiment 3 test tuples (alpha, s1, s2); error = relative H1 seminorm
  lshape   Experiment 4 test pairs (sigma1, sigma2); error = energy identity vs NGSolve F(u)
  oblique  rotated arctangent layer u = xy atan(t(x cos a + y sin a - s)), a in {0, pi/8, pi/4}

Usage:
    python 2D/scripts/alg1_2d.py --experiment oblique --p 2 --levels 4 8 16 32
    python 2D/scripts/alg1_2d.py --experiment lshape --p 2 --instances 4 --networks eta ritz
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import jax.numpy as jnp
import numpy as np
import pandas as pd

from common.h1_seminorm_2d import h1_seminorm_sq_2d, h1_seminorm_sq_analytic
from common.parameter_sampling import build_arctan_grid, build_lshape_grid
from src.config import ARCTAN, LSHAPE, SPLIT_SEED, TRAIN_VAL
from src.nonparametric import alg1_ladder_2d as A
from src.nonparametric.arctan.pde import grad_u_sigma
from src.nonparametric.oblique import grad_exact

ANGLES = (0.0, math.pi / 8, math.pi / 4)


def test_instances(experiment, k):
    if experiment == "oblique":
        return [np.array([a]) for a in ANGLES]
    if experiment == "arctan":
        grid = build_arctan_grid(n_alpha=int(ARCTAN["n_alpha"]), n_s1=int(ARCTAN["n_s1"]),
                                 n_s2=int(ARCTAN["n_s2"]), alpha_min=float(ARCTAN["alpha_min"]),
                                 alpha_max=float(ARCTAN["alpha_max"]), s_min=float(ARCTAN["s_min"]),
                                 s_max=float(ARCTAN["s_max"]), train_frac=float(TRAIN_VAL["train_frac"]),
                                 seed=int(SPLIT_SEED), val_frac=float(TRAIN_VAL["val_frac"]))
    else:
        grid = build_lshape_grid(n_sigma1=int(LSHAPE["n_sigma1"]), n_sigma2=int(LSHAPE["n_sigma2"]),
                                 exp_min=float(LSHAPE["sigma_exp_min"]), exp_max=float(LSHAPE["sigma_exp_max"]),
                                 train_frac=float(TRAIN_VAL["train_frac"]), seed=int(SPLIT_SEED),
                                 val_frac=float(TRAIN_VAL["val_frac"]))
    test = np.asarray(grid["test"])
    order = np.argsort(test[:, 0] + 1e-3 * test[:, -1])
    return list(test[order][np.linspace(0, len(test) - 1, int(k)).round().astype(int)])


def problem_for(experiment, nu):
    if experiment == "arctan":
        return A.arctan_problem(nu)
    if experiment == "lshape":
        return A.lshape_problem(nu)
    return A.oblique_problem(float(nu[0]))


def error_fn(experiment, nu, reference=None, split="test"):
    if experiment == "lshape":
        if reference is not None:
            from scripts.alg1_lshape_Tscan import reference_energy
            from src.nonparametric.lshape.variational_energy import energy_and_load
            F = reference_energy(reference, nu, split)
            def error(r, kx, ky, p):
                strain, load = energy_and_load(r.u_h, kx, ky, p, *nu)
                gap = float((F + strain - 2*load) / F)
                if gap < 0 or not np.isfinite(gap):
                    raise ValueError(f"unresolved signed energy gap {gap} at {nu}")
                return math.sqrt(gap)
            return error
        E = pd.read_csv(REPO / "references_v2" / "lshape_energy_ngsolve.csv")
        row = E.iloc[np.argmin(np.abs(E.sigma1 - nu[0]) + np.abs(E.sigma2 - nu[1]))]
        if abs(row.sigma1 - nu[0]) + abs(row.sigma2 - nu[1]) > 1e-9:
            raise SystemExit(f"no NGSolve energy for sigma={nu}")
        F = float(row.F_order14)
        return lambda r, kx, ky, p: math.sqrt(max(1.0 + 2.0 * float(r.ritz_energy) / F, 0.0))
    grad = ((lambda x, y: grad_u_sigma(x, y, *nu)) if experiment == "arctan"
            else (lambda x, y: grad_exact(x, y, float(nu[0]))))
    ref = h1_seminorm_sq_analytic(grad, quad_points_per_dim=20)
    return lambda r, kx, ky, p: math.sqrt(
        h1_seminorm_sq_2d(r.u_h, np.asarray(kx), np.asarray(ky), p, grad, quad_points_per_dim=20) / ref)


def network_knots(experiment, kind, p, seed, model_root=None):
    from src.parametric.continuation import load_checkpoint
    from src.parametric.positional_density_network_2d import (
        knots_p2_from_network_axis, knots_p3_from_network_axis)
    root = REPO / ("data_results" if kind == "eta" else "data_results_v2")
    name = experiment if kind == "eta" else f"{experiment}_ritz"
    ck = (model_root or root / name) / f"p{p}" / "checkpoints" / f"seed{seed}" / "checkpoint_final.npz"
    if not ck.exists():
        if model_root is not None:
            raise FileNotFoundError(ck)
        return None
    params = load_checkpoint(ck)
    build = knots_p2_from_network_axis if experiment == "arctan" else knots_p3_from_network_axis

    def knots(nu, N):
        s = jnp.asarray(nu, jnp.float64)
        return build(params, s, N, p, 0.0), build(params, s, N, p, 1.0)
    return knots


def row(experiment, p, nu, N, method, r, kx, ky, eta2, err, time_s, nit=0, stop="", n_active=0):
    loc, ar = A.mesh_quality(kx, ky, p)
    return dict(experiment=experiment, p=p, nu=" ".join(f"{v:.6g}" for v in np.atleast_1d(nu)), N=N,
                method=method, err=err, eta=math.sqrt(max(eta2, 0.0)), ritz=float(r.ritz_energy),
                time_s=time_s, nit=nit, stop=stop, n_active=n_active, gamma_loc=loc, aspect=ar)


def paired_lshape_ladders(args, instances, reference, p, levels):
    from common.resources import hardware_info
    from lshape_phase_e_eval import condition_numbers, energy_metrics
    from mesh_diagnostics_2d import aspect
    from scripts.alg1_lshape_Tscan import realized_hmin, reference_energy
    from src.nonparametric.lshape.schedule import read_schedule, policy_at
    from src.nonparametric.solver_2d import free_mask_lshape

    schedule = read_schedule(args.mesh_schedule, p)
    hw = hardware_info()
    target = REPO / args.out_dir / f"lshape_p{p}_{args.tag}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(target, float_precision="round_trip") if target.exists() else pd.DataFrame()
    rows = previous.to_dict("records")
    for nu in instances:
        F = reference_energy(reference, nu, args.split)
        prob = A.lshape_problem(nu, mesh_schedule=schedule)
        for loss in args.losses:
            existing = [r for r in rows if (r["method"],r["sigma1"],r["sigma2"]) == (loss,*nu)]
            if existing:
                if sorted(r["N"] for r in existing) != levels or any(r["split"] != args.split for r in existing):
                    raise ValueError(f"incomplete ladder in {target}")
                continue
            for N, result in A.run_ladder(prob, loss, p, levels, args.maxiter).items():
                sol,x,y,_,e2 = A.solution(prob,result["theta"],p,N)
                free = np.asarray(free_mask_lshape(x,y,p),dtype=bool).ravel()
                cond,jac = condition_numbers(sol.K,free)
                T,floor = policy_at(schedule,N)
                rows.append(dict(method=loss, p=p, N=N, sigma1=nu[0], sigma2=nu[1], split=args.split,
                                 **energy_metrics(sol.u_h,x,y,p,nu,F,e2), ritz=float(sol.ritz_energy),
                                 gamma_loc=A.mesh_quality(x,y,p)[0], aspect=aspect(x,y,"lshape"),
                                 h_min=realized_hmin(x,y,N), T=T, h_min_floor=floor, cond=cond, cond_jacobi=jac,
                                 free_dofs=int(free.sum()), time_s=result["time_s"], nit=result["nit"],
                                 n_active=result["n_active"], stop=result["stop"], cpu=hw["cpu_model"], cpus=hw["slurm_cpus"], host=hw["host"]))
            pd.DataFrame(rows).to_csv(target.with_suffix(".partial"),index=False)
            os.replace(target.with_suffix(".partial"),target)
            print(f"{args.split} {nu} p={p} {loss} complete", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["arctan", "lshape", "oblique"], required=True)
    ap.add_argument("--p", type=int, default=2, choices=[2, 3])
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--instance-range", type=int, nargs=2, default=None)
    ap.add_argument("--losses", nargs="+", default=["eta", "ritz"])
    ap.add_argument("--networks", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="data_results_v2/alg1_2d")
    ap.add_argument("--mesh-schedule", type=Path)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--lshape-reference", type=Path)
    ap.add_argument("--eta-root", type=Path)
    ap.add_argument("--ritz-root", type=Path)
    ap.add_argument("--alg-only", action="store_true")
    args = ap.parse_args()
    if (args.mesh_schedule or args.lshape_reference or args.split == "val") and args.experiment != "lshape":
        ap.error("the split, reference and mesh schedule options apply to lshape")
    if args.split == "val" and args.lshape_reference is None:
        ap.error("supply the validation reference via --lshape-reference")
    p, levels = int(args.p), [int(n) for n in args.levels]

    rows, t_all = [], time.perf_counter()
    roots = {"eta": args.eta_root, "ritz": args.ritz_root}
    nets = {k: network_knots(args.experiment, k, p, args.seed, roots[k]) for k in args.networks}
    reference = (pd.read_csv(args.lshape_reference, float_precision="round_trip")
                 if args.lshape_reference is not None else None)
    if args.experiment == "lshape":
        from scripts.alg1_lshape_Tscan import split_instances
        instances = split_instances(args.split, args.instances)
    else:
        instances = test_instances(args.experiment, args.instances)
    if args.instance_range:
        instances = instances[args.instance_range[0]:args.instance_range[1]]
    if args.alg_only:
        if args.experiment != "lshape" or reference is None or not args.mesh_schedule or args.networks:
            ap.error("--alg-only requires a L-shape reference and mesh schedule")
        return paired_lshape_ladders(args, instances, reference, p, levels)
    for nu in instances:
        prob, err = problem_for(args.experiment, nu), error_fn(args.experiment, nu, reference, args.split)
        if args.mesh_schedule:
            from src.nonparametric.lshape.schedule import read_schedule
            prob = A.lshape_problem(nu, mesh_schedule=read_schedule(args.mesh_schedule, p))
        for N in levels:
            theta0 = jnp.zeros(prob.n_blocks * prob.block_size(N))
            r, kx, ky, n, e2 = A.solution(prob, theta0, p, N)
            rows.append(row(args.experiment, p, nu, N, "uniform", r, kx, ky, e2, err(r, kx, ky, p), 0.0))
            for kind, kn in nets.items():
                if kn is None:
                    continue
                t0 = time.perf_counter()
                kx, ky = kn(nu, N)
                kx.block_until_ready()
                t = time.perf_counter() - t0
                n = kx.shape[0] - 2 * p - 1
                r = prob.solve(kx, ky, p, n)
                rows.append(row(args.experiment, p, nu, N, f"net-{kind}", r, kx, ky,
                                float(prob.eta2(r, kx, ky, p, n)), err(r, kx, ky, p), t))
        for loss in args.losses:
            for N, res in A.run_ladder(prob, loss, p, levels, args.maxiter).items():
                r, kx, ky, n, e2 = A.solution(prob, res["theta"], p, N)
                rows.append(row(args.experiment, p, nu, N, f"alg1-{loss}", r, kx, ky, e2,
                                err(r, kx, ky, p), res["time_s"], res["nit"], res["stop"], res["n_active"]))
        print(pd.DataFrame(rows).query("N == @levels[-1]").tail(2 + len(args.losses) + len(nets))
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"), flush=True)

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if args.experiment == "lshape":
        df["split"] = args.split
    path = out / f"{args.experiment}_p{p}{('_' + args.tag) if args.tag else ''}.csv"
    df.to_csv(path, index=False)
    print(df.groupby(["N", "method"]).err.median().unstack().to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {path} ({time.perf_counter() - t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
