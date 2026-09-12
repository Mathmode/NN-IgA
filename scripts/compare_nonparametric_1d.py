"""Non-parametric (Algorithm 1) vs parametric (Algorithm 2) — 1D singular problem.

For a mesh of N elements (default N=100, i.e. 101 breakpoints) and each test
exponent beta, compares three meshes at identical DOF count:

  uniform      theta = 0 (the baseline).
  algorithm-1  per-instance r-adaptivity: Adam on the reduced residual loss
               L(theta) = 1/2 eta^2(theta) starting FROM THE UNIFORM MESH
               (theta0 = 0, no warm start, no continuation) — the paper's
               Algorithm 1 run cold at this single level.
  parametric   the trained positional density network (Algorithm 2 checkpoint),
               evaluated at this level in ONE forward pass. N=100 was never a
               training level (continuation trained N <= 64), so this is a
               zero-shot use of the density property: the same weights are
               collocated at the 100 cell midpoints.

Reported per (p, beta): relative H1-seminorm error, estimator eta, and the
mesh-construction cost (Algorithm 1: full optimization wall time; parametric:
one forward pass). Both then need the same single Galerkin solve to produce u.

Notes on the parametric mesh policy at an off-schedule level: T and h_min are
per-(p, N) config schedules keyed on the eval levels; for N=100 we take the
schedule value of the nearest level in log2 (N=128: T=5 for p=2, T=6 for p=3;
h_min=1e-8), which is how the network would be deployed at an unseen level.

Usage:
    python scripts/compare_nonparametric_1d.py                 # p=2 and p=3, 3 betas
    python scripts/compare_nonparametric_1d.py --p 3 --N 100 --iters 20000
    python scripts/compare_nonparametric_1d.py --smoke         # fast validity check
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

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "1D")]

import numpy as np
import jax.numpy as jnp

from src.config import T_SCHEDULE
from src.lhs_sampling import generate_splits_val
from src.nonparametric.r_adapt import (
    evaluate_case_detailed,
    iterations_for_level,
    run_r_adapt_case,
)
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network import (
    cell_xi,
    forward_scalar,
    gauge_fix,
    saturate,
)


def _T_for(p: int, N: int) -> float:
    """T at the nearest scheduled level in log2 (deployment rule for unseen N)."""
    sched = T_SCHEDULE[int(p)]
    key = min(sched, key=lambda k: abs(math.log2(k) - math.log2(N)))
    return float(sched[key])


def _h_min_for(p: int, N: int) -> float:
    """Config h_min rule: p=2 -> 1e-7 (N<=32) else 1e-8; p=3 -> 1e-8."""
    if int(p) == 2:
        return 1e-7 if N <= 32 else 1e-8
    return 1e-8


def _network_theta(params, beta: float, *, p: int, N: int) -> jnp.ndarray:
    z = forward_scalar(params, jnp.asarray(beta), cell_xi(N), N)
    return saturate(gauge_fix(z), _T_for(p, N))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--iters", type=int, default=None,
                    help="Adam iterations for Algorithm 1 (default: the repo's "
                         "per-level budget, 20000 for N=100).")
    ap.add_argument("--n-betas", type=int, default=3,
                    help="Number of test exponents (min/median/max of the fixed "
                         "held-out test subset).")
    ap.add_argument("--seed", type=int, default=0, help="Checkpoint seed.")
    ap.add_argument("--smoke", action="store_true",
                    help="1 beta, 200 iterations — validity check only.")
    args = ap.parse_args()

    N = int(args.N)
    iters = int(args.iters) if args.iters is not None else iterations_for_level(N)
    _, _, test_betas = generate_splits_val()
    test_betas = np.sort(test_betas)
    if args.smoke:
        betas, iters = [float(np.median(test_betas))], 200
    else:
        idx = np.linspace(0, test_betas.size - 1, int(args.n_betas)).round().astype(int)
        betas = [float(b) for b in test_betas[np.unique(idx)]]

    print(f"# Non-parametric (Algorithm 1, cold from uniform) vs parametric "
          f"(trained network, zero-shot)\n"
          f"# N={N} elements, Adam iters={iters}, checkpoint seed={args.seed}, "
          f"betas={[round(b, 4) for b in betas]}\n")
    header = (f"{'p':>2} {'beta':>8} {'method':>12} {'H1_semi_rel':>12} "
              f"{'eta':>12} {'mesh cost':>12}")
    print(header)
    print("-" * len(header))

    rows = []
    for p in args.p:
        q = p + 1
        h_min = _h_min_for(p, N)
        ck = REPO / f"data_results/singular/p{p}/checkpoints/p{p}_seed{args.seed}/checkpoint_final.npz"
        params = load_checkpoint(ck)
        for beta in betas:
            # --- uniform baseline ---
            ev_u = evaluate_case_detailed(jnp.zeros(N), p, q, h_min, beta)

            # --- Algorithm 1: cold start from the uniform mesh ---
            t0 = time.perf_counter()
            _, ev_a1, _, metrics_a1, _ = run_r_adapt_case(
                jnp.zeros(N), N, p, q, h_min,
                history_every=max(iters, 1), iters_override=iters, beta=beta,
            )
            t_a1 = time.perf_counter() - t0

            # --- parametric: one forward pass at the (unseen) level N ---
            _network_theta(params, beta, p=p, N=N)          # jit warm-up
            t0 = time.perf_counter()
            theta_net = _network_theta(params, beta, p=p, N=N)
            theta_net.block_until_ready()
            t_net = time.perf_counter() - t0
            ev_net = evaluate_case_detailed(theta_net, p, q, h_min, beta)

            for tag, ev, cost in (("uniform", ev_u, "--"),
                                  ("algorithm-1", ev_a1, f"{t_a1:8.1f} s"),
                                  ("parametric", ev_net, f"{t_net*1e3:8.2f} ms")):
                print(f"{p:>2} {beta:>8.4f} {tag:>12} "
                      f"{ev['err_energy_rel']:>12.3e} {ev['eta_residual']:>12.3e} "
                      f"{cost:>12}")
                rows.append(dict(p=p, beta=beta, N=N, method=tag,
                                 H1_semi_rel=float(ev["err_energy_rel"]),
                                 eta=float(ev["eta_residual"]),
                                 cost=cost.strip()))
            r_a1 = ev_u["err_energy_rel"] / ev_a1["err_energy_rel"]
            r_nn = ev_u["err_energy_rel"] / ev_net["err_energy_rel"]
            print(f"   -> improvement over uniform: algorithm-1 {r_a1:.1f}x, "
                  f"parametric {r_nn:.1f}x\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
