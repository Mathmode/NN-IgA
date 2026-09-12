"""Reconstruct the immersed degree-5 L-shape reference (remediation T8, audit §9).

The shipped ``references/reference_lshape.pkl`` had no generator in the repo. This
script rebuilds it with the repo's own solver (no NGSolve): for each (sigma1, sigma2)
on the 20x20 grid it solves the immersed L-shape at degree ``p=5`` on a mesh of
``2*n_half`` elements/axis, GEOMETRICALLY GRADED toward the re-entrant corner
x=y=0.5 with ratio ``1/refine_corner`` (symmetric about 0.5), interface knot 0.5 at
multiplicity p.  With ``n_half=64, refine_corner=1.15, q_K=6, q_F=3`` this
reproduces the shipped reference's KNOT VECTORS to machine precision.

The re-solved coefficients differ from the shipped pkl by ~1e-4 relative (a
solver-evolution drift: q_K=6/q_F=3 integrate the polynomial stiffness/forcing
exactly, so the gap is not quadrature).  ``review/reference_accuracy_check.py``
bounds the reference's own discretization error by a self-convergence study; the
reference is >~1e2x more accurate than the reported L-shape errors either way, so
its uncertainty does not affect the Table 4 comparisons.

Usage:
    python 2D/scripts/generate_reference_lshape.py --spot-check          # 3 sigmas, no write
    python 2D/scripts/generate_reference_lshape.py --out references/reference_lshape_regen.pkl
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

REPO = Path(__file__).resolve().parents[2]     # .../<repo>/2D/scripts -> <repo>
sys.path[:0] = [str(REPO), str(REPO / "2D")]

import numpy as np
import jax.numpy as jnp

from src.nonparametric.solver_2d import galerkin_solve_lshape
from common.bspline_basis import bspline_basis_local
from common.parameter_sampling import sample_uniform_log_base10

P = 5
Q_K, Q_F = 6, 3
REFINE_CORNER = 1.15
N_HALF = 64

# Lean sigma-weighted H1 seminorm on a fixed L-shape-masked grid (avoids importing
# evaluation_v2, whose dependency tree OOMs the p=5 solve on a 17 GB machine).
_NG = 160
_xg = (np.arange(_NG) + 0.5) / _NG
_XX, _YY = np.meshgrid(_xg, _xg, indexing="ij")
_KEEP = ~((_XX >= 0.5) & (_YY <= 0.5))


def graded_lshape_knots(n_half: int = N_HALF, p: int = P,
                        refine: float = REFINE_CORNER) -> np.ndarray:
    """Corner-graded immersed L-shape knot vector (see module docstring)."""
    r = 1.0 / float(refine)
    h = r ** np.arange(int(n_half))
    h = h / h.sum() * 0.5
    left_break = np.cumsum(h)[:-1]
    right_break = 0.5 + np.cumsum(h[::-1])[:-1]
    interior = np.concatenate([left_break, np.full(p, 0.5), right_break])
    return np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])


def _sigma_h1_seminorm_sq(u, kn, s1, s2):
    """int_Omega sigma |grad u_h|^2 on the L-shape (keep-masked, sigma-weighted),
    computed leanly on the fixed grid (no evaluation_v2 import)."""
    N, dN, _, sp = bspline_basis_local(jnp.asarray(_xg), jnp.asarray(kn), P)
    N, dN, sp = np.asarray(N), np.asarray(dN), np.asarray(sp)
    nb = kn.shape[0] - P - 1
    B = np.zeros((_NG, nb)); dB = np.zeros((_NG, nb))
    for q in range(_NG):
        c = np.arange(sp[q] - P, sp[q] + 1)
        B[q, c] = N[q]; dB[q, c] = dN[q]
    ux = dB @ u @ B.T
    uy = B @ u @ dB.T
    sig = np.where((_XX < 0.5) & (_YY < 0.5), s1,
                   np.where((_XX >= 0.5) & (_YY >= 0.5), s2, 1.0)) * _KEEP
    return float(np.sum(sig * (ux * ux + uy * uy)) * (1.0 / _NG) ** 2)


def build_entry(s1: float, s2: float, kn: np.ndarray) -> dict:
    n_elem = kn.shape[0] - 2 * P - 1
    res = galerkin_solve_lshape(jnp.asarray(kn), jnp.asarray(kn), P, n_elem, n_elem,
                                float(s1), float(s2), q_K=Q_K, q_F=Q_F)
    u = np.asarray(res.u_h)
    sizes = np.diff(np.unique(kn))
    return {
        "sigma1": float(s1), "sigma2": float(s2),
        "u_coeffs": u, "knots_x": kn.copy(), "knots_y": kn.copy(),
        "p": P, "q_K": Q_K, "q_F": Q_F, "refine_corner": REFINE_CORNER,
        "dof_count": int(u.size), "max_h": float(sizes.max()),
        "sigma_h1_seminorm_sq": _sigma_h1_seminorm_sq(u, kn, s1, s2),
        "order": f"iga_p{P}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=None,
                    help="Output .pkl path (default: none, spot-check only).")
    ap.add_argument("--n-half", type=int, default=N_HALF)
    ap.add_argument("--spot-check", action="store_true",
                    help="Build only 3 sigmas and compare to the shipped pkl.")
    ap.add_argument("--n-sigma", type=int, default=20)
    args = ap.parse_args()

    kn = graded_lshape_knots(int(args.n_half))
    print(f"grid p={P} n_half={args.n_half} ({kn.shape[0]-2*P-1} elem/axis, "
          f"{(kn.shape[0]-P-1)**2} dofs), refine={REFINE_CORNER}, q_K={Q_K}, q_F={Q_F}")

    shipped = None
    sp = REPO / "references" / "reference_lshape.pkl"
    if sp.exists():
        shipped = pickle.load(open(sp, "rb"))
        kx0 = np.asarray(next(iter(shipped.values()))["knots_x"])
        dknot = float(np.max(np.abs(np.sort(graded_lshape_knots(64)) - np.sort(kx0))))
        print(f"  reconstructed knots vs shipped: max|dknot| = {dknot:.2e}")

    if args.spot_check:
        for tgt in [(1.0, 1.0), (0.1, 10.0), (10.0, 0.1)]:
            if shipped:
                key = min(shipped, key=lambda k: (np.log10(k[0]) - np.log10(tgt[0])) ** 2
                          + (np.log10(k[1]) - np.log10(tgt[1])) ** 2)
            else:
                key = tgt
            s1, s2 = key
            e = build_entry(s1, s2, kn)
            msg = ""
            if shipped and int(args.n_half) == 64:
                u_ship = np.asarray(shipped[key]["u_coeffs"])
                if u_ship.shape == e["u_coeffs"].shape:
                    msg = (f"  rel||u_regen - u_shipped|| = "
                           f"{np.linalg.norm(e['u_coeffs']-u_ship)/np.linalg.norm(u_ship):.2e}")
            print(f"  sigma=({s1:.4f},{s2:.4f}): dof={e['dof_count']} max_h={e['max_h']:.4f}{msg}")
        return 0

    axis = sample_uniform_log_base10(int(args.n_sigma), -1.0, 1.0)
    cache = {}
    t0 = time.time()
    for i, s1 in enumerate(axis):
        for s2 in axis:
            cache[(float(s1), float(s2))] = build_entry(float(s1), float(s2), kn)
        print(f"  row {i+1}/{len(axis)} done ({time.time()-t0:.0f}s)", flush=True)
    if args.out:
        with open(args.out, "wb") as fh:
            pickle.dump(cache, fh)
        print(f"wrote {len(cache)} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
