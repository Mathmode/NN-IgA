"""Loss ablation for Experiment 1 (revision v2, item 3.1).

Per test exponent of the shared val-protocol split, compares at equal DOFs:
  uniform        theta = 0
  alg1-eta       Algorithm 1, Adam on 1/2 eta^2, continuation ladder 2 -> 64
  alg1-ritz      Algorithm 1, Adam on the Ritz energy J, same ladder
  net-eta        trained network, eq-26 loss (data_results/singular)
  net-ritz       trained network, Ritz loss (data_results_v2/singular_ritz)
and, to reproduce the pilot, Algorithm 1 cold-started at N=16.

Usage:
    python 1D/scripts/loss_ablation_1d.py --part alg1 --iters 1000
    python 1D/scripts/loss_ablation_1d.py --part alg1 --beta-range 0 9 --tag chunk0
    python 1D/scripts/loss_ablation_1d.py --part networks
    python 1D/scripts/loss_ablation_1d.py --part pilot --iters 3000
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

import numpy as np
import pandas as pd

from src.config import CONTINUATION_LEVELS, EVAL_LEVELS
from src.lhs_sampling import generate_splits_val
from src.nonparametric.alg1_ladder import h_min_for, run_cold, run_ladder
from src.nonparametric.discretization import stiffness_quadrature_order
from src.nonparametric.r_adapt import evaluate_case_detailed
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network import logits_at_level

NETWORKS = {"net-eta": REPO / "data_results" / "singular",
            "net-ritz": REPO / "data_results_v2" / "singular_ritz",
            "net-nolog": REPO / "data_results_v2" / "singular_nolog"}


def metrics(theta, p, N, beta):
    d = evaluate_case_detailed(theta, int(p), stiffness_quadrature_order(int(p)), h_min_for(p, N), float(beta))
    return float(d["err_energy_rel"]), float(d["eta_residual"])


def test_betas(limit=None):
    tb = np.sort(generate_splits_val()[2])
    return tb if limit is None else tb[np.linspace(0, len(tb) - 1, int(limit)).round().astype(int)]


def part_alg1(ps, betas, iters, levels):
    rows = []
    for p in ps:
        for b in betas:
            for kind in ("eta", "ritz"):
                for N, (theta, t) in run_ladder(kind, p, b, levels, iters).items():
                    e, eta = metrics(theta, p, N, b)
                    rows.append(dict(p=p, beta=float(b), N=N, method=f"alg1-{kind}", err=e, eta=eta, time_s=t))
            for N in levels:
                e, eta = metrics(np.zeros(N), p, N, b)
                rows.append(dict(p=p, beta=float(b), N=N, method="uniform", err=e, eta=eta, time_s=0.0))
            print(pd.DataFrame(rows[-3 * len(levels):]).query("N == @levels[-1]").to_string(index=False), flush=True)
    return pd.DataFrame(rows)


def part_pilot(ps, betas, iters, N):
    rows = []
    for p in ps:
        for b in betas:
            for kind in ("eta", "ritz"):
                theta, t = run_cold(kind, p, b, N, iters)
                e, eta = metrics(theta, p, N, b)
                rows.append(dict(p=p, beta=float(b), N=N, method=f"alg1-cold-{kind}", err=e, eta=eta, time_s=t))
    return pd.DataFrame(rows)


def part_networks(ps, betas, levels, variants):
    rows = []
    for name in variants:
        root = NETWORKS[name]
        for p in ps:
            for ck in sorted((root / f"p{p}" / "checkpoints").glob(f"p{p}_seed*/checkpoint_final.npz")):
                seed = int(ck.parent.name.split("seed")[-1])
                params = load_checkpoint(ck)
                for N in levels:
                    for b in betas:
                        t0 = time.perf_counter()
                        theta = logits_at_level(params, float(b), p=p, N=N).block_until_ready()
                        t = time.perf_counter() - t0
                        e, eta = metrics(theta, p, N, b)
                        rows.append(dict(p=p, seed=seed, beta=float(b), N=N, method=name, err=e, eta=eta, time_s=t))
                print(f"  {name} p={p} seed={seed} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["alg1", "pilot", "networks"], required=True)
    ap.add_argument("--p", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--n-betas", type=int, default=None)
    ap.add_argument("--beta-range", type=int, nargs=2, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--variants", nargs="+", default=["net-eta", "net-ritz"])
    ap.add_argument("--out-dir", default="data_results_v2/loss_ablation")
    args = ap.parse_args()
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.part == "alg1":
        betas = test_betas(args.n_betas)
        if args.beta_range:
            betas = betas[args.beta_range[0]:args.beta_range[1]]
        df = part_alg1(args.p, betas, args.iters, tuple(CONTINUATION_LEVELS))
        name = f"exp1_alg1_ladder{('_' + args.tag) if args.tag else ''}.csv"
    elif args.part == "pilot":
        df = part_pilot(args.p, test_betas(args.n_betas or 3), args.iters, 16)
        name = "exp1_alg1_cold_N16.csv"
    else:
        df = part_networks(args.p, test_betas(args.n_betas), tuple(EVAL_LEVELS), args.variants)
        name = "exp1_networks_" + "_".join(v.replace("net-", "") for v in args.variants) + ".csv"
    df.to_csv(out / name, index=False)
    print(df.groupby(["p", "N", "method"]).err.median().unstack("method").to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
