"""Mesh diagnostics of the trained 1D networks (revision v2, items 3.4 and 3.5).

Experiments 1 (singular), 2 (Helmholtz) and 6 (advection-diffusion-reaction),
all seeds and test parameters: realized gamma_loc (max size ratio over 2p+1
consecutive elements), global size ratio and the a priori bound e^{2T}; condition
numbers of the stiffness matrix on the free dofs (raw and Jacobi-scaled, uniform
versus adapted) for the coercive Experiments 1 and 6; online time (forward pass
+ Galerkin solve).

Usage:
    python 1D/scripts/mesh_diagnostics_1d.py --out-dir data_results_v2/mesh_diagnostics
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
sys.path[:0] = [str(REPO), str(REPO / "1D")]

import jax.numpy as jnp
import numpy as np
import pandas as pd

import src.helmholtz.mesh_helmholtz_global as G
from src.adr import discretization as AD
from src.adr import problem as AP
from src.adr.network import from_flat, logits as adr_logits
from src.config import EVAL_LEVELS, H_MIN_SCHEDULE, HELMHOLTZ, T_SCHEDULE
from src.helmholtz.network_helmholtz import knots_from_network_global, params_from_flat_dict
from src.lhs_sampling import generate_splits_val
from src.nonparametric.discretization import (
    assemble_reduced_system_from_cache, build_eval_cache, solve_state, stiffness_quadrature_order)
from src.nonparametric.knots import theta_to_sizes
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network import logits_at_level

DATA = REPO / "data_results"
HELM_LEVELS = (32, 48, 64, 96, 128, 256)


def gamma_loc(h, p):
    h = np.asarray(h, float)
    w = min(2 * p + 1, h.size)
    win = np.lib.stride_tricks.sliding_window_view(h, w)
    return float(np.max(win.max(1) / win.min(1)))


def cond_pair(K):
    K = np.asarray(K)
    d = 1.0 / np.sqrt(np.abs(np.diag(K)))
    return float(np.linalg.cond(K)), float(np.linalg.cond(d[:, None] * K * d[None, :]))


def singular_rows(seeds, levels):
    betas = np.sort(generate_splits_val()[2])
    rows = []
    for p in (2, 3):
        for seed in seeds:
            params = load_checkpoint(DATA / "singular" / f"p{p}" / "checkpoints" / f"p{p}_seed{seed}" / "checkpoint_final.npz")
            for N in levels:
                for b in betas:
                    h = np.asarray(theta_to_sizes(logits_at_level(params, float(b), p=p, N=N), H_MIN_SCHEDULE[p][N]))
                    rows.append(dict(experiment="singular", p=p, seed=seed, N=N, nu=float(b),
                                     gamma_loc=gamma_loc(h, p), global_ratio=float(h.max() / h.min()),
                                     e2T=math.exp(2 * T_SCHEDULE[p][N])))
    return rows


def helmholtz_rows(seeds, levels):
    T, hmin = float(HELMHOLTZ["T_cap"]), float(HELMHOLTZ["h_min"])
    rows = []
    for p in (2, 3):
        for seed in seeds:
            root = DATA / "helmholtz" / f"p{p}"
            params = params_from_flat_dict(np.load(root / "checkpoints" / f"p{p}_seed{seed}" / "checkpoint_final.npz"))
            for N in levels:
                for c in np.load(root / f"test_cs_p{p}_seed{seed}.npy"):
                    kn = np.asarray(knots_from_network_global(params, jnp.asarray(float(c)), N, p, T=T, h_min=hmin, use_hmax=True))
                    h = np.diff(np.unique(kn))
                    rows.append(dict(experiment="helmholtz", p=p, seed=seed, N=N, nu=float(c),
                                     gamma_loc=gamma_loc(h, p), global_ratio=float(h.max() / h.min()),
                                     e2T=math.exp(2 * T)))
    return rows


def adr_rows():
    rows = []
    for f in sorted((REPO / "data_results_v2" / "adr").glob("p*/summary_p*_seed*.csv")):
        d = pd.read_csv(f)
        d = d[d.method == "adapted"]
        for r in d.itertuples():
            rows.append(dict(experiment="adr", p=r.p, seed=r.seed, N=r.N, nu=f"{r.log10_eps:.4f} {r.b:.4f}",
                             gamma_loc=r.gamma_loc, global_ratio=r.h_ratio, e2T=math.exp(2 * AP.T_CAP)))
    return rows


def timed(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cond_online_rows(levels):
    rows = []
    b = float(np.median(generate_splits_val()[2]))
    for p in (2, 3):
        q = stiffness_quadrature_order(p)
        params = load_checkpoint(DATA / "singular" / f"p{p}" / "checkpoints" / f"p{p}_seed0" / "checkpoint_final.npz")
        for N in levels:
            for method in ("uniform", "adapted"):
                th_fn = (lambda: jnp.zeros(N)) if method == "uniform" else (lambda: logits_at_level(params, b, p=p, N=N))
                K, _ = assemble_reduced_system_from_cache(build_eval_cache(th_fn(), p, q, H_MIN_SCHEDULE[p][N]), p, b)
                c, cj = cond_pair(K)
                t = timed(lambda: solve_state(th_fn(), p, q, H_MIN_SCHEDULE[p][N], b)[0].block_until_ready())
                rows.append(dict(experiment="singular", p=p, N=N, nu=b, method=method, cond=c, cond_jacobi=cj, online_s=t))
    nu = np.array([-1.5, 1.25])
    eps, bb = AP.unpack(nu)
    for p in (2, 3):
        ck = REPO / "data_results_v2" / "adr" / f"p{p}" / "checkpoints" / f"p{p}_seed0" / "checkpoint_final.npz"
        if not ck.exists():
            continue
        params = from_flat(np.load(ck))
        for N in (16, 64, 128):
            for method in ("uniform", "adapted"):
                th_fn = (lambda: jnp.zeros(N)) if method == "uniform" else (lambda: adr_logits(params, jnp.asarray(nu), N, AP.T_CAP))
                kn = AD.knots_from_theta(th_fn(), p, AP.H_MIN)
                t = timed(lambda: AD.solve(AD.knots_from_theta(th_fn(), p, AP.H_MIN), p, eps, bb, AD.quad_order(p)).block_until_ready())
                rows.append(dict(experiment="adr", p=p, N=N, nu="-1.5 1.25", method=method, cond=np.nan, cond_jacobi=np.nan, online_s=t))
    for p in (2, 3):
        root = DATA / "helmholtz" / f"p{p}"
        params = params_from_flat_dict(np.load(root / "checkpoints" / f"p{p}_seed0" / "checkpoint_final.npz"))
        c = 3.1
        for N in (64, 128, 256):
            for method in ("uniform", "adapted"):
                kfn = ((lambda: G.uniform_two_patch_knots(N, p)) if method == "uniform" else
                       (lambda: knots_from_network_global(params, jnp.asarray(c), N, p, T=float(HELMHOLTZ["T_cap"]),
                                                          h_min=float(HELMHOLTZ["h_min"]), use_hmax=True)))
                t = timed(lambda: G.solve_u(kfn(), p, jnp.asarray(G.rho1_of_c(c)), G.quad_order(p)).block_until_ready())
                rows.append(dict(experiment="helmholtz", p=p, N=N, nu=c, method=method, cond=np.nan, cond_jacobi=np.nan, online_s=t))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--cond-levels", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--out-dir", default="data_results_v2/mesh_diagnostics")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    df = pd.DataFrame(singular_rows(args.seeds, EVAL_LEVELS) + helmholtz_rows(args.seeds, HELM_LEVELS) + adr_rows())
    df.to_csv(out / "constants_1d.csv", index=False)
    summ = df.groupby(["experiment", "p", "N"]).agg(
        gamma_loc_median=("gamma_loc", "median"), gamma_loc_max=("gamma_loc", "max"),
        global_ratio_max=("global_ratio", "max"), e2T=("e2T", "first"))
    summ.to_csv(out / "constants_1d_summary.csv")
    print(summ.to_string(float_format=lambda v: f"{v:.3g}"))
    co = pd.DataFrame(cond_online_rows(args.cond_levels))
    co.to_csv(out / "conditioning_online_1d.csv", index=False)
    print(co.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
