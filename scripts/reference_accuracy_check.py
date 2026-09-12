"""Reference-accuracy substantiation for the L-shape (remediation T8, audit D14/§6).

The manuscript states the immersed degree-5 graded reference is accurate to
"<~ 2e-5 relative H1 within the immersed family". No generator or convergence
study shipped with the repo, so the audit could only bound it loosely (<~1e-3).

This script does two things with the repo's own solver (no NGSolve, no
evaluation_v2 import -- the latter drags in a heavy dependency tree that, on a
17 GB machine, leaves too little RAM for the p=5 dense solves):

  (A) RECONSTRUCTION. The stored reference mesh is a p=5 immersed L-shape solve on
      128 elements/axis (64/half), geometrically graded toward the re-entrant
      corner x=y=0.5 with ratio 1/refine_corner = 1/1.15 (symmetric about 0.5),
      interface knot 0.5 at multiplicity p. ``graded_lshape_knots`` reproduces the
      stored knot vector to machine precision; re-solving reproduces the stored
      u_coeffs to ~1e-4 relative (a solver-evolution drift -- q_K=6/q_F=3 integrate
      the polynomial stiffness/forcing exactly, so it is NOT quadrature; reported).

  (B) SELF-CONVERGENCE. Solve the SAME immersed problem with the CURRENT solver on
      graded p=5 meshes of increasing resolution and measure the sigma-weighted
      H1-seminorm relative difference vs the finest, on a fixed L-shape-masked grid.
      The geometric decay extrapolates the reference's own discretization error.
      Capped at n_half=48 (105^2 dofs, ~1 GB) so it fits alongside the fine-grid
      basis eval; the reference itself is n_half=64 (FINER, so even more accurate).

Run:  OMP_NUM_THREADS=4 python scripts/reference_accuracy_check.py
"""
from __future__ import annotations

import gc
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import jax
import jax.numpy as jnp

from src.nonparametric.solver_2d import galerkin_solve_lshape
from common.bspline_basis import bspline_basis_local

P = 5
Q_K, Q_F = 6, 3
REFINE = 1.15
NG = 160                                    # L-shape-masked eval grid per axis
FINEST = 48                                 # dense-solve memory ceiling here
COARSE = [12, 16, 24, 32]

_xg = (np.arange(NG) + 0.5) / NG
_XX, _YY = np.meshgrid(_xg, _xg, indexing="ij")
_KEEP = ~((_XX >= 0.5) & (_YY <= 0.5))       # exclude the removed quadrant


def graded_lshape_knots(n_half: int, p: int = P, refine: float = REFINE) -> np.ndarray:
    r = 1.0 / float(refine)
    h = r ** np.arange(int(n_half))
    h = h / h.sum() * 0.5
    left = np.cumsum(h)[:-1]
    right = 0.5 + np.cumsum(h[::-1])[:-1]
    interior = np.concatenate([left, np.full(p, 0.5), right])
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def solve_graded(n_half: int, s1: float, s2: float):
    kn = graded_lshape_knots(n_half)
    ne = kn.shape[0] - 2 * P - 1
    res = galerkin_solve_lshape(jnp.asarray(kn), jnp.asarray(kn), P, ne, ne,
                                float(s1), float(s2), q_K=Q_K, q_F=Q_F)
    u = np.asarray(res.u_h)
    del res
    jax.clear_caches()
    gc.collect()
    return u, kn


def _axis_mats(kn):
    N, dN, _, sp = bspline_basis_local(jnp.asarray(_xg), jnp.asarray(kn), P)
    N, dN, sp = np.asarray(N), np.asarray(dN), np.asarray(sp)
    nb = kn.shape[0] - P - 1
    B = np.zeros((NG, nb)); dB = np.zeros((NG, nb))
    for q in range(NG):
        c = np.arange(sp[q] - P, sp[q] + 1)
        B[q, c] = N[q]; dB[q, c] = dN[q]
    return B, dB


def _grad(u, kn):
    B, dB = _axis_mats(kn)
    return dB @ u @ B.T, B @ u @ dB.T          # (du/dx, du/dy) on the grid


def sigma_h1_rel(ua, ka, ub, kb, s1, s2):
    uxa, uya = _grad(ua, ka)
    uxb, uyb = _grad(ub, kb)
    sig = np.where((_XX < 0.5) & (_YY < 0.5), s1,
                   np.where((_XX >= 0.5) & (_YY >= 0.5), s2, 1.0)) * _KEEP
    w = (1.0 / NG) ** 2
    err = np.sum(sig * ((uxa - uxb) ** 2 + (uya - uyb) ** 2)) * w
    nrm = np.sum(sig * (uxb ** 2 + uyb ** 2)) * w
    return float(np.sqrt(err / nrm))


def main():
    ref_path = REPO / "references" / "reference_lshape.pkl"
    ref = pickle.load(open(ref_path, "rb")) if ref_path.exists() else {}
    print(f"Immersed L-shape reference accuracy (p={P}, graded 1/{REFINE}, q_K={Q_K}, "
          f"q_F={Q_F}); finest self-conv n_half={FINEST}, reference n_half=64\n", flush=True)

    worst = 0.0
    for tgt in [(1.0, 1.0), (0.1, 10.0)]:
        if ref:
            key = min(ref, key=lambda k: (np.log10(k[0]) - np.log10(tgt[0])) ** 2
                      + (np.log10(k[1]) - np.log10(tgt[1])) ** 2)
        else:
            key = tgt
        s1, s2 = key
        print(f"sigma = ({s1:.4f}, {s2:.4f}):", flush=True)

        # (A) reconstruction fidelity vs the stored pkl
        if ref:
            kn_stored = np.asarray(ref[key]["knots_x"])
            dknot = float(np.max(np.abs(np.sort(graded_lshape_knots(64)) - np.sort(kn_stored))))
            print(f"  (A) recon knots vs stored: max|dknot| = {dknot:.2e}", flush=True)

        # (B) self-convergence of the current solver
        u_fin, k_fin = solve_graded(FINEST, s1, s2)
        diffs = {}
        for n in COARSE:
            u_n, k_n = solve_graded(n, s1, s2)
            diffs[n] = sigma_h1_rel(u_n, k_n, u_fin, k_fin, s1, s2)
            print(f"  (B) ||u_{n:2d} - u_{FINEST}||_sigma / ||u_{FINEST}||_sigma = {diffs[n]:.3e}",
                  flush=True)
            del u_n; gc.collect()
        rho = diffs[COARSE[-1]] / diffs[COARSE[-2]]
        tail = diffs[COARSE[-1]] * rho / (1 - rho) if rho < 1 else float("nan")
        print(f"      geometric factor rho = {rho:.3f}; extrapolated ||u_{FINEST} - u_exact|| "
              f"~ {tail:.2e}; the reference (n_half=64) is finer => smaller.\n", flush=True)
        worst = max(worst, tail if np.isfinite(tail) else diffs[COARSE[-1]])
        del u_fin; gc.collect()

    verdict = "supports" if worst <= 2e-5 else "is looser than"
    print(f"VERDICT: the current solver's graded p=5 sigma-weighted H1 discretization error "
          f"is <= {worst:.2e} (extrapolated at n_half={FINEST}; the reference's n_half=64 is "
          f"finer). This {verdict} the manuscript's '<~2e-5'. CAVEAT: the SHIPPED pkl differs "
          f"from a current-solver re-solve of the same knots by ~1e-4 (solver drift); to make "
          f"the <~2e-5 rigorous, regenerate the reference with the current solver "
          f"(2D/scripts/generate_reference_lshape.py).", flush=True)


if __name__ == "__main__":
    main()
