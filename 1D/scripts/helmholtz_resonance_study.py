"""Experiment 2 resonance study (revision v2, item 2.2).

Produces, from the trained checkpoints (no retraining):
  errors_N30_32_34.csv / table_R2.csv   medians over 4 seeds x 15 test contrasts
  scan.csv                              error vs c in [4.5, 5.5], seed 0, N in {32, 64}
  table_R1.csv                          discrete resonance shifts (uniform vs adapted)
  adapted_resonance_hits.csv            test cases where the adapted mesh sits on its own resonance
  fig6_*.csv                            exact / uniform / adapted solutions at c = 5.0005
Every error is computed with the repository solver and cross-checked with the
independent scipy solver of ``src.helmholtz.resonance``.

Usage:
    python 1D/scripts/helmholtz_resonance_study.py --out-dir data_results_v2/helmholtz
"""
from __future__ import annotations

import argparse
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
import src.helmholtz.resonance as R
from src.config import HELMHOLTZ
from src.helmholtz.network_helmholtz import knots_from_network_global, params_from_flat_dict

T_CAP = float(HELMHOLTZ["T_cap"])
H_MIN = float(HELMHOLTZ["h_min"])
DATA = REPO / "data_results" / "helmholtz"
C_FIG = 5.0005


def load_params(p, seed):
    return params_from_flat_dict(np.load(DATA / f"p{p}" / "checkpoints" / f"p{p}_seed{seed}" / "checkpoint_final.npz"))


def test_contrasts(p, seed):
    return np.load(DATA / f"p{p}" / f"test_cs_p{p}_seed{seed}.npy")


def mesh(method, params, c, N, p):
    if method == "uniform":
        return np.asarray(G.uniform_two_patch_knots(N, p))
    return np.asarray(knots_from_network_global(params, jnp.asarray(float(c)), N, p,
                                                T=T_CAP, h_min=H_MIN, use_hmax=True))


def errors(kn, p, c):
    u = G.solve_u(jnp.asarray(kn), p, jnp.asarray(G.rho1_of_c(c)), G.quad_order(p))
    err = float(G.h1_rel(kn, p, u, G.ExactContrast(c)))
    err_indep = R.h1_rel(kn, p, R.solve(kn, p, c), c)
    return err, err_indep


def level_table(seeds, levels):
    rows = []
    for p in (2, 3):
        for seed in seeds:
            params = load_params(p, seed)
            for N in levels:
                for c in test_contrasts(p, seed):
                    for method in ("uniform", "adapted"):
                        kn = mesh(method, params, c, N, p)
                        err, err_i = errors(kn, p, float(c))
                        rows.append(dict(p=p, seed=seed, N=N, c=float(c), method=method,
                                         err=err, err_indep=err_i))
    df = pd.DataFrame(rows)
    med = df.groupby(["p", "N", "method"]).err.median().unstack("method")
    med["ratio"] = med["uniform"] / med["adapted"]
    return df, med


def scan(cgrid, levels, seed=0):
    rows = []
    for p in (2, 3):
        params = load_params(p, seed)
        for N in levels:
            for c in cgrid:
                for method in ("uniform", "adapted"):
                    kn = mesh(method, params, c, N, p)
                    err, err_i = errors(kn, p, float(c))
                    rows.append(dict(p=p, N=N, c=float(c), method=method, err=err, err_indep=err_i))
    return pd.DataFrame(rows)


def resonance_shifts(levels, targets=(4.8, 5.2), seed=0):
    rows = []
    for p in (2, 3):
        params = load_params(p, seed)
        for N in levels:
            cu = R.discrete_resonances(mesh("uniform", params, 0.0, N, p), p)
            for cr in targets:
                ca = R.discrete_resonances(mesh("adapted", params, cr, N, p), p)
                rows.append(dict(
                    p=p, N=N, c=cr,
                    c_h_uniform=R.associated_by_index(cu, cr),
                    c_h_adapted=R.associated_by_index(ca, cr),
                    c_h_uniform_window=R.associated_in_window(cu, cr),
                    c_h_adapted_window=R.associated_in_window(ca, cr),
                ))
    df = pd.DataFrame(rows)
    df["shift_uniform"] = df.c_h_uniform - df.c
    df["shift_adapted"] = df.c_h_adapted - df.c
    df["reduction"] = df.shift_uniform / df.shift_adapted
    return df


def resonance_hits(df, threshold=5.0):
    piv = df.pivot_table(index=["p", "seed", "N", "c"], columns="method", values="err").reset_index()
    hits = piv[piv.adapted > threshold * piv.uniform].copy()
    near = []
    for r in hits.itertuples():
        params = load_params(r.p, r.seed)
        ca = R.discrete_resonances(mesh("adapted", params, r.c, r.N, r.p), r.p)
        cu = R.discrete_resonances(mesh("uniform", params, r.c, r.N, r.p), r.p)
        near.append((R.nearest(ca, r.c), R.nearest(cu, r.c)))
    hits["nearest_c_h_adapted"] = [a for a, _ in near]
    hits["nearest_c_h_uniform"] = [b for _, b in near]
    hits["dist_adapted"] = (hits.nearest_c_h_adapted - hits.c).abs()
    hits["dist_uniform"] = (hits.nearest_c_h_uniform - hits.c).abs()
    return hits


def solution_profiles(c, levels, seed=0, n=1201):
    x = np.linspace(0.0, 1.0, n)
    out, errs = {}, []
    for p in (2, 3):
        params = load_params(p, seed)
        for N in levels:
            cols = {"x": x, "exact": R.exact_solution(c)(x)}
            for method in ("uniform", "adapted"):
                kn = mesh(method, params, c, N, p)
                u = R.solve(kn, p, c)
                cols[method] = R.evaluate(kn, p, u, np.clip(x, 0.0, 1.0 - 1e-15))
                errs.append(dict(p=p, N=N, c=c, method=method, err=R.h1_rel(kn, p, u, c),
                                 split_left=G.split_counts(kn, p)[0]))
            out[(p, N)] = pd.DataFrame(cols)
    return out, pd.DataFrame(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data_results_v2/helmholtz")
    ap.add_argument("--n-scan", type=int, default=401)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    t0 = time.time()

    df, med = level_table(args.seeds, (30, 32, 34))
    df.to_csv(out / "errors_N30_32_34.csv", index=False)
    med.to_csv(out / "table_R2.csv")
    print("Table R2 (medians):\n", med.to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"  max |err - err_indep| / err = {((df.err - df.err_indep).abs() / df.err).max():.1e}  ({time.time() - t0:.0f}s)")

    hits = resonance_hits(df)
    hits.to_csv(out / "adapted_resonance_hits.csv", index=False)
    print("Adapted-mesh resonance hits (adapted > 5x uniform):\n", hits.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("p=3 max adapted error:", df[(df.p == 3) & (df.method == "adapted")].err.max())

    r1 = resonance_shifts((32, 64))
    r1.to_csv(out / "table_R1.csv", index=False)
    print("Table R1:\n", r1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sc = scan(np.linspace(4.5, 5.5, args.n_scan), (32, 64))
    sc.to_csv(out / "scan.csv", index=False)
    print(f"  scan rows {len(sc)}; max |err - err_indep| / err = {((sc.err - sc.err_indep).abs() / sc.err).max():.1e}  ({time.time() - t0:.0f}s)")

    prof, perr = solution_profiles(C_FIG, (32, 34))
    for (p, N), d in prof.items():
        d.to_csv(out / f"fig6_profile_p{p}_N{N}.csv", index=False)
    perr.to_csv(out / "fig6_errors.csv", index=False)
    print("Fig. 6 errors at c = 5.0005:\n", perr.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
