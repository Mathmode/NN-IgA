"""Non-parametric (Algorithm 1) vs parametric (Algorithm 2) — 2D L-shape problem.

For an N x N immersed L-shape mesh (default N=40 per axis; the C0 interface knot
at 0.5 splits each axis into two blocks of N/2 elements) and each test pair
(sigma1, sigma2), compares three meshes at identical DOF count:

  uniform      all four block logit vectors = 0.
  algorithm-1  per-instance r-adaptivity: L-BFGS-B on eta^2(theta) over the
               FOUR block logit vectors simultaneously (x-left, x-right,
               y-left, y-right; 4*(N/2) parameters), starting FROM THE UNIFORM
               MESH (theta0 = 0, no warm start) — the paper's Algorithm 1 run
               cold at this single level. Gradients via jax.value_and_grad
               through the custom-VJP Cholesky solve.
  parametric   the trained positional density network (Algorithm 2 checkpoint)
               evaluated at this level in ONE forward pass per block. N=40 was
               never a training level (continuation trained N <= 32), so this
               is a zero-shot use of the density property.

Error metric: relative sigma-weighted H1 seminorm against the degree-5 graded
immersed reference (references/reference_lshape.pkl), exactly as in Table 4.

Caveats (documented, intentional): the non-parametric 2D map is the raw
per-block softmax of section 2.3 — no T saturation cap and no h_min floor
(audit R3) — so Algorithm 1 explores a slightly LARGER mesh family than the
network's policy (T=5, h_min=1e-7).

Usage:
    python scripts/compare_nonparametric_2d.py                 # p=2, N=40, 2 sigmas
    python scripts/compare_nonparametric_2d.py --p 3 --maxiter 200
    python scripts/compare_nonparametric_2d.py --smoke         # fast validity check
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from src.config import LSHAPE, SPLIT_SEED, TRAIN_VAL
from common.parameter_sampling import build_lshape_grid
from src.nonparametric.r_adapt import p3_effective_n_elem, theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_lshape
from src.nonparametric.eta_estimator_2d import eta_squared_lshape
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network_2d import knots_p3_from_network_axis
from src.parametric.evaluation_v2 import _direct_h1_error_lshape, _lookup_ref_entry

Q_F = int(LSHAPE["quad_forcing"])       # 4 (exact for f=1)
Q_EST = int(LSHAPE["quad_estimator"])   # 4


def _solve_and_eta(kx, ky, p, n_eff, s1, s2):
    res = galerkin_solve_lshape(kx, ky, p, n_eff, n_eff, s1, s2, q_K=p + 1, q_F=Q_F)
    eta_sq = eta_squared_lshape(res.u_h, kx, ky, p, n_eff, n_eff, s1, s2, q_est=Q_EST)
    return res.u_h, eta_sq


def _metrics(u_h, kx, ky, p, s1, s2, ref_entry, eta_sq):
    rel, _abs, _rn, _cn = _direct_h1_error_lshape(
        np.asarray(u_h), np.asarray(kx), np.asarray(ky), p, s1, s2, ref_entry)
    return float(rel), float(np.sqrt(max(float(eta_sq), 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2, choices=[2, 3])
    ap.add_argument("--N", type=int, default=40, help="Elements per axis (even).")
    ap.add_argument("--maxiter", type=int, default=200,
                    help="L-BFGS-B iteration budget for Algorithm 1.")
    ap.add_argument("--n-sigmas", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="Checkpoint seed.")
    ap.add_argument("--smoke", action="store_true",
                    help="1 sigma pair, maxiter=10 — validity check only.")
    args = ap.parse_args()

    p, N = int(args.p), int(args.N)
    half = N // 2
    assert 2 * half == N, "L-shape needs even N"
    n_eff = p3_effective_n_elem(N, p)
    maxiter = 10 if args.smoke else int(args.maxiter)

    # Held-out test sigmas (shared val-protocol split), spread across the range.
    grid = build_lshape_grid(
        n_sigma1=int(LSHAPE["n_sigma1"]), n_sigma2=int(LSHAPE["n_sigma2"]),
        exp_min=float(LSHAPE["sigma_exp_min"]), exp_max=float(LSHAPE["sigma_exp_max"]),
        train_frac=float(TRAIN_VAL["train_frac"]), seed=int(SPLIT_SEED),
        val_frac=float(TRAIN_VAL["val_frac"]),
    )
    test = grid["test"]
    contrast = np.abs(np.log10(test[:, 0]) - np.log10(test[:, 1]))
    picks = [test[np.argmin(contrast)]]                 # near-homogeneous pair
    if not args.smoke and int(args.n_sigmas) > 1:
        picks.append(test[np.argmax(contrast)])         # high-contrast pair
    picks = picks[: (1 if args.smoke else int(args.n_sigmas))]

    ref_cache = pickle.load(open(REPO / "references" / "reference_lshape.pkl", "rb"))
    params = load_checkpoint(
        REPO / f"data_results/lshape/p{p}/checkpoints/seed{args.seed}/checkpoint_final.npz")

    print(f"# Non-parametric (Algorithm 1, cold from uniform) vs parametric "
          f"(trained network, zero-shot)\n"
          f"# L-shape, p={p}, N={N}/axis (4 blocks x {half} logits), "
          f"L-BFGS-B maxiter={maxiter}, checkpoint seed={args.seed}\n")
    header = (f"{'sigma1':>8} {'sigma2':>8} {'method':>12} {'H1_rel(ref)':>12} "
              f"{'eta':>12} {'mesh cost':>14}")
    print(header)
    print("-" * len(header))

    for sig in picks:
        s1, s2 = float(sig[0]), float(sig[1])
        ref_entry = _lookup_ref_entry(ref_cache, s1, s2)
        assert ref_entry is not None, f"no reference entry for ({s1}, {s2})"
        s1j, s2j = jnp.asarray(s1), jnp.asarray(s2)

        # --- uniform ---
        z = jnp.zeros(half)
        kx_u = theta_to_knots_lshape(z, z, p)
        u_u, eta2_u = _solve_and_eta(kx_u, kx_u, p, n_eff, s1j, s2j)
        rel_u, eta_u = _metrics(u_u, kx_u, kx_u, p, s1, s2, ref_entry, eta2_u)

        # --- Algorithm 1: L-BFGS-B over the 4 block logit vectors, cold ---
        def loss(theta_flat):
            xl, xr, yl, yr = jnp.split(theta_flat, 4)
            kx = theta_to_knots_lshape(xl, xr, p)
            ky = theta_to_knots_lshape(yl, yr, p)
            _, eta_sq = _solve_and_eta(kx, ky, p, n_eff, s1j, s2j)
            return eta_sq

        vg = jax.jit(jax.value_and_grad(loss))
        nit = [0]

        def fun(th):
            v, g = vg(jnp.asarray(th))
            return float(v), np.asarray(g, dtype=np.float64)

        t0 = time.perf_counter()
        res = minimize(fun, np.zeros(4 * half), jac=True, method="L-BFGS-B",
                       callback=lambda *_: nit.__setitem__(0, nit[0] + 1),
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
        t_a1 = time.perf_counter() - t0
        xl, xr, yl, yr = np.split(res.x, 4)
        kx_a = theta_to_knots_lshape(jnp.asarray(xl), jnp.asarray(xr), p)
        ky_a = theta_to_knots_lshape(jnp.asarray(yl), jnp.asarray(yr), p)
        u_a, eta2_a = _solve_and_eta(kx_a, ky_a, p, n_eff, s1j, s2j)
        rel_a, eta_a = _metrics(u_a, kx_a, ky_a, p, s1, s2, ref_entry, eta2_a)

        # --- parametric: one forward pass per block at the (unseen) level N ---
        sig_j = jnp.asarray([s1, s2])
        knots_p3_from_network_axis(params, sig_j, N, p, 0.0)     # jit warm-up
        t0 = time.perf_counter()
        kx_n = knots_p3_from_network_axis(params, sig_j, N, p, 0.0)
        ky_n = knots_p3_from_network_axis(params, sig_j, N, p, 1.0)
        ky_n.block_until_ready()
        t_net = time.perf_counter() - t0
        u_n, eta2_n = _solve_and_eta(kx_n, ky_n, p, n_eff, s1j, s2j)
        rel_n, eta_n = _metrics(u_n, kx_n, ky_n, p, s1, s2, ref_entry, eta2_n)

        for tag, rel, eta, cost in (
                ("uniform", rel_u, eta_u, "--"),
                ("algorithm-1", rel_a, eta_a, f"{t_a1:8.1f} s ({res.nit} it)"),
                ("parametric", rel_n, eta_n, f"{t_net*1e3:8.2f} ms")):
            print(f"{s1:>8.4f} {s2:>8.4f} {tag:>12} {rel:>12.3e} {eta:>12.3e} "
                  f"{cost:>14}")
        print(f"   -> improvement over uniform: algorithm-1 {rel_u/rel_a:.1f}x, "
              f"parametric {rel_u/rel_n:.1f}x\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
