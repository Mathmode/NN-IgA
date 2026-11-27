from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import pandas as pd

from common.parameter_sampling import build_lshape_grid
from src.config import LSHAPE, SPLIT_SEED, TRAIN_VAL
from src.nonparametric import alg1_ladder_2d as A
from src.nonparametric.lshape.variational_energy import energy_and_load


def split_instances(split, k):
    g = build_lshape_grid(n_sigma1=int(LSHAPE["n_sigma1"]), n_sigma2=int(LSHAPE["n_sigma2"]),
                          exp_min=float(LSHAPE["sigma_exp_min"]), exp_max=float(LSHAPE["sigma_exp_max"]),
                          train_frac=float(TRAIN_VAL["train_frac"]), seed=int(SPLIT_SEED),
                          val_frac=float(TRAIN_VAL["val_frac"]))
    pts = np.asarray(g[split])
    order = np.argsort(pts[:, 0] + 1e-3 * pts[:, -1])
    if not 1 <= k <= len(pts):
        raise ValueError("invalid number of split instances")
    return [tuple(float(v) for v in x) for x in pts[order][np.linspace(0, len(pts) - 1, int(k)).round().astype(int)]]


def reference_energy(energies, sigma, split):
    col = max((c for c in energies if c.startswith("F_order") and c[7:].isdigit()),
              key=lambda c: int(c[7:]))
    mask = ((energies.sigma1 - sigma[0]).abs() < 1e-12) & ((energies.sigma2 - sigma[1]).abs() < 1e-12)
    matches = energies.loc[mask]
    if len(matches) != 1 or matches.iloc[0]["split"] != split:
        raise ValueError(f"expected one {split} reference for {sigma}")
    value = float(matches.iloc[0][col])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid reference for {sigma}")
    return value


def realized_hmin(kx, ky, N):
    spans = [np.diff(np.unique(axis)) for axis in (kx, ky)]
    if any(h.size != N for h in spans):
        raise ValueError(f"physical mesh lost a span at level {N}")
    return float(min(h.min() for h in spans))


def with_bound(problem, T):
    return dataclasses.replace(problem, bound=lambda p: float(T))


def rows_for(sigma, split, p, T, levels, maxiter, F):
    prob = with_bound(A.lshape_problem(sigma), T)
    rows = []

    def row(N, method, theta, time_s=0.0, nit=0, stop="", n_active=0):
        r, kx, ky, n, e2 = A.solution(prob, theta, p, N)
        loc, ar = A.mesh_quality(kx, ky, p)
        coeffs = r.u_h.ravel()
        J_assembled = float(0.5 * (coeffs @ r.K @ coeffs) - coeffs @ r.F)
        strain, load = energy_and_load(r.u_h, kx, ky, p, *sigma)
        J = strain / 2 - load
        F_load = float(np.asarray(r.F) @ np.asarray(r.u_h).ravel())
        gap = float((np.longdouble(F) + 2 * J) / F)
        if not np.isfinite([gap, F_load, e2, loc, ar]).all() or e2 < 0:
            raise ValueError(f"invalid solve for {sigma}, p={p}, T={T}, N={N}")
        return dict(split=split, p=p, T=float(T), sigma1=sigma[0], sigma2=sigma[1], N=N, method=method,
                    gap=gap, err=math.sqrt(gap) if gap >= 0 else np.nan, eta=math.sqrt(e2),
                    ritz=float(J), ritz_assembled=J_assembled, strain_quadrature=float(strain),
                    load_quadrature=float(load), F_ref=F, F_load=F_load, time_s=time_s,
                    nit=nit, stop=stop, n_active=n_active, gamma_loc=loc, aspect=ar, h_min=realized_hmin(kx, ky, N))

    for N in levels:
        rows.append(row(N, "uniform", np.zeros(prob.n_blocks * prob.block_size(N))))
    for N, res in A.run_ladder(prob, "eta", p, levels, maxiter).items():
        rows.append(row(N, "alg1-eta", res["theta"], res["time_s"], res["nit"], res["stop"], res["n_active"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--p", type=int, choices=[2, 3], required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--instances", type=int, default=16)
    ap.add_argument("--instance-range", type=int, nargs=2, default=None)
    ap.add_argument("--reference", default="references_v3/lshape_energy_ngsolve.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="data_results_v3/lshape_alg1_Tscan")
    args = ap.parse_args()

    inst = split_instances(args.split, args.instances)
    if args.instance_range:
        lo, hi = args.instance_range
        if not 0 <= lo < hi <= len(inst):
            ap.error("invalid instance range")
        inst = inst[lo:hi]
    if any(n < 4 or n % 2 for n in args.levels) or any(b != 2 * a for a, b in zip(args.levels, args.levels[1:])):
        ap.error("levels must be consecutive even doublings")
    if not np.isfinite(args.T) or args.T <= 0 or args.maxiter < 1:
        ap.error("positive T and maxiter required")
    energies = pd.read_csv(REPO / args.reference, float_precision="round_trip")
    rows, t0 = [], time.perf_counter()
    for sigma in inst:
        F = reference_energy(energies, sigma, args.split)
        rows += rows_for(sigma, args.split, args.p, args.T, [int(n) for n in args.levels], args.maxiter, F)
        last = rows[-1]
        print(f"sigma=({sigma[0]:.4g},{sigma[1]:.4g}) N={last['N']} err={last['err']:.3e} nit={last['nit']} "
              f"stop={last['stop']} active={last['n_active']} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    T = f"{args.T:g}".replace(".", "p")
    path = out / f"{args.split}_p{args.p}_T{T}{('_' + args.tag) if args.tag else ''}.csv"
    partial = path.with_suffix(".partial.csv")
    data = pd.DataFrame(rows)
    if args.split == "test":
        from common.resources import hardware_info
        hardware = hardware_info()
        data = data.assign(cpu=hardware["cpu_model"], cpus=hardware["slurm_cpus"], host=hardware["host"])
    data.to_csv(partial, index=False)
    partial.replace(path)
    print(f"wrote {path} ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
