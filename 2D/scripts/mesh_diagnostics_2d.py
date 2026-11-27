"""Mesh diagnostics of the trained 2D networks (revision v2, items 3.4 and 3.5).

For Experiments 3 (arctan), 4 (lshape) and 5 (advdiff), all seeds, a spread
subset of test parameters and every evaluation level:
  realized local quasi-uniformity gamma_loc (max size ratio over 2p+1 consecutive
  elements, per direction), global size ratio, maximum element aspect ratio
  (L-shape: elements of the L only), and the a priori bound e^{2T};
plus, for seed 0 and a few parameters, the condition number of the stiffness
matrix on the free dofs (raw and Jacobi-scaled), uniform versus adapted, and the
online time (network forward pass + Galerkin solve).

Usage:
    python 2D/scripts/mesh_diagnostics_2d.py --out-dir data_results_v2/mesh_diagnostics
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

from common.h1_seminorm_2d import open_uniform_knots
from common.parameter_sampling import build_advdiff_grid, build_arctan_grid, build_lshape_grid
from src import config as C
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network_2d import (
    knots_p2_from_network_axis,
    knots_p3_from_network_axis,
)

LEVELS = (4, 8, 16, 32, 64)


def test_params(exp, k):
    kw = dict(train_frac=float(C.TRAIN_VAL["train_frac"]), seed=int(C.SPLIT_SEED),
              val_frac=float(C.TRAIN_VAL["val_frac"]))
    if exp == "arctan":
        A = C.ARCTAN
        g = build_arctan_grid(n_alpha=int(A["n_alpha"]), n_s1=int(A["n_s1"]), n_s2=int(A["n_s2"]),
                              alpha_min=float(A["alpha_min"]), alpha_max=float(A["alpha_max"]),
                              s_min=float(A["s_min"]), s_max=float(A["s_max"]), **kw)
    elif exp == "lshape":
        L = C.LSHAPE
        g = build_lshape_grid(n_sigma1=int(L["n_sigma1"]), n_sigma2=int(L["n_sigma2"]),
                              exp_min=float(L["sigma_exp_min"]), exp_max=float(L["sigma_exp_max"]), **kw)
    else:
        D = C.ADVDIFF
        g = build_advdiff_grid(n_logeps=int(D["n_logeps"]), n_b=int(D["n_b"]), logeps_min=float(D["logeps_min"]),
                               logeps_max=float(D["logeps_max"]), b_min=float(D["b_min"]), b_max=float(D["b_max"]), **kw)
    t = np.asarray(g["test"])
    return t[np.linspace(0, len(t) - 1, int(k)).round().astype(int)]


def t_cap(exp):
    return float(C.T_LSHAPE if exp == "lshape" else C.T_ARCTAN)


def network(exp, p, seed):
    ck = REPO / "data_results" / exp / f"p{p}" / "checkpoints" / f"seed{seed}" / "checkpoint_final.npz"
    params = load_checkpoint(ck)
    build = knots_p3_from_network_axis if exp == "lshape" else knots_p2_from_network_axis

    def knots(nu, N):
        s = jnp.asarray(nu, jnp.float64)
        return build(params, s, N, p, 0.0), build(params, s, N, p, 1.0)
    return knots


def uniform_knots(exp, p, N):
    if exp == "lshape":
        z = jnp.zeros((N // 2,))
        k = theta_to_knots_lshape(z, z, p)
    else:
        k = jnp.asarray(open_uniform_knots(N, p))
    return k, k


def solve(exp, kx, ky, p, N, nu):
    if exp == "arctan":
        return galerkin_solve_arctan(kx, ky, p, N, N, nu[0], nu[1], nu[2], q_K=p + 1, q_F=int(C.ARCTAN["quad_forcing"]))
    if exp == "lshape":
        n = p3_effective_n_elem(N, p)
        return galerkin_solve_lshape(kx, ky, p, n, n, nu[0], nu[1], q_K=p + 1, q_F=int(C.LSHAPE["quad_forcing"]))
    return galerkin_solve_advdiff(kx, ky, p, N, N, 10.0 ** nu[0], nu[1], q_K=p + 1, q_F=int(C.ADVDIFF["quad_forcing"]))


def sizes(k):
    return np.diff(np.unique(np.asarray(k)))


def gamma_loc(h, p):
    w = min(2 * p + 1, h.size)
    win = np.lib.stride_tricks.sliding_window_view(h, w)
    return float(np.max(win.max(1) / win.min(1)))


def aspect(kx, ky, exp):
    bx, by = np.unique(np.asarray(kx)), np.unique(np.asarray(ky))
    hx, hy = np.diff(bx), np.diff(by)
    ar = np.maximum(hx[:, None] / hy[None, :], hy[None, :] / hx[:, None])
    if exp == "lshape":
        cx, cy = 0.5 * (bx[:-1] + bx[1:]), 0.5 * (by[:-1] + by[1:])
        ar = ar[~((cx[:, None] > 0.5) & (cy[None, :] < 0.5))]
    return float(np.max(ar))


def conditioning(res):
    K = np.asarray(res.K)
    free = ~np.isclose(np.abs(K).sum(1) - np.abs(np.diag(K)), 0.0) | ~np.isclose(np.diag(K), 1.0)
    Kf = K[np.ix_(free, free)]
    d = 1.0 / np.sqrt(np.abs(np.diag(Kf)))
    return float(np.linalg.cond(Kf)), float(np.linalg.cond(d[:, None] * Kf * d[None, :]))


def part_constants(exps, k, seeds):
    rows = []
    for exp in exps:
        for p in (2, 3):
            for seed in seeds:
                kn = network(exp, p, seed)
                for N in LEVELS:
                    for nu in test_params(exp, k):
                        kx, ky = kn(nu, N)
                        hx, hy = sizes(kx), sizes(ky)
                        rows.append(dict(experiment=exp, p=p, seed=seed, N=N,
                                         nu=" ".join(f"{v:.6g}" for v in nu),
                                         gamma_loc=max(gamma_loc(hx, p), gamma_loc(hy, p)),
                                         global_ratio=max(hx.max() / hx.min(), hy.max() / hy.min()),
                                         aspect=aspect(kx, ky, exp), e2T=math.exp(2 * t_cap(exp))))
            print(f"  constants {exp} p={p} done", flush=True)
    return pd.DataFrame(rows)


def part_cond_online(exps, k, levels):
    rows = []
    for exp in exps:
        for p in (2, 3):
            kn = network(exp, p, 0)
            for N in levels:
                for nu in test_params(exp, k):
                    nu_j = jnp.asarray(nu, jnp.float64)
                    for method in ("uniform", "adapted"):
                        get = (lambda: uniform_knots(exp, p, N)) if method == "uniform" else (lambda: kn(nu, N))
                        kx, ky = get()
                        res = solve(exp, kx, ky, p, N, nu_j)
                        res.u_h.block_until_ready()
                        times = []
                        for _ in range(3):
                            t0 = time.perf_counter()
                            kx, ky = get()
                            solve(exp, kx, ky, p, N, nu_j).u_h.block_until_ready()
                            times.append(time.perf_counter() - t0)
                        c_raw, c_jac = conditioning(res)
                        rows.append(dict(experiment=exp, p=p, N=N, nu=" ".join(f"{v:.6g}" for v in nu),
                                         method=method, cond=c_raw, cond_jacobi=c_jac,
                                         online_s=float(np.median(times)), dofs=int(np.asarray(res.u_h).size)))
            print(f"  cond/online {exp} p={p} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["arctan", "lshape", "advdiff"])
    ap.add_argument("--n-params", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cond-levels", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--cond-params", type=int, default=3)
    ap.add_argument("--parts", nargs="+", default=["constants", "cond"])
    ap.add_argument("--out-dir", default="data_results_v2/mesh_diagnostics")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    if "constants" in args.parts:
        df = part_constants(args.experiments, args.n_params, args.seeds)
        df.to_csv(out / "constants_2d.csv", index=False)
        summ = df.groupby(["experiment", "p", "N"]).agg(
            gamma_loc_median=("gamma_loc", "median"), gamma_loc_max=("gamma_loc", "max"),
            aspect_max=("aspect", "max"), global_ratio_max=("global_ratio", "max"), e2T=("e2T", "first"))
        summ.to_csv(out / "constants_2d_summary.csv")
        print(summ.to_string(float_format=lambda v: f"{v:.3g}"))
    if "cond" in args.parts:
        df = part_cond_online(args.experiments, args.cond_params, args.cond_levels)
        df.to_csv(out / "conditioning_online_2d.csv", index=False)
        print(df.groupby(["experiment", "p", "N", "method"])[["cond", "cond_jacobi", "online_s"]]
              .median().to_string(float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
