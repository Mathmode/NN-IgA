#!/usr/bin/env python
"""Regenerate the L-shape (Experiment 4) gradient-magnitude ZOOM figure data from
the CURRENT trained checkpoints, into ``TiKz_overleaf/exp4_lshape/gradzoom/``.

Three panels, all at the same problem instance sigma = param1 = (0.89, 0.89):
  * ``P3_grad_uniform_p2.csv``   -- |grad u_h| on a uniform   p=2 mesh
  * ``P3_grad_radapted_p2.csv``  -- |grad u_h| on the network p=2 r-adapted mesh
  * ``P3_grad_radapted_p3.csv``  -- |grad u_h| on the network p=3 r-adapted mesh
each an 81x81 grid over the corner zoom [0.25,0.75]^2 (columns x,y,z, z=|grad u_h|),
plus ``P3_grad_knots_<method>_p<p>.dat`` -- the mesh (knot) lines clipped to the zoom
with the removed quadrant cut out.

The recipe is the documented one (memory: lshape-figure-data-generator): N=20, seed0,
``checkpoint_final``; r-adapt knots via ``knots_p3_from_network_axis`` (the lshape builder
parametrised by p -- the "p3" name is legacy), uniform via ``theta_to_knots_lshape``;
solve ``galerkin_solve_lshape(..., n_eff=p3_effective_n_elem(N,p), q_K=p+1, q_F=4)``.
Importing from the live 2D/src means the current calibrated T/h_min and the current
checkpoints are used -- i.e. it matches the current convergence/loss figures.

Run from the repo root:
    python 2D/scripts/generate_lshape_gradient_zoom.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "2D"))
sys.path.insert(0, _ROOT)

import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network_2d import knots_p3_from_network_axis
from src.nonparametric.r_adapt import theta_to_knots_lshape, p3_effective_n_elem
from src.nonparametric.solver_2d import galerkin_solve_lshape
from src.parametric.evaluation_v2 import _p3_basis_on_grid, _grad_on_grid

OUT = os.path.join(_ROOT, "TiKz_overleaf", "exp4_lshape", "gradzoom")
N = 16   # a trained continuation level (training used 4,8,16,32) — matches arctan/advdiff
SIGMA = (0.89, 0.89)            # param1 -- the documented gradient-zoom instance
Q_F = 4


def solve(params, sigma, p, method):
    n_eff = p3_effective_n_elem(N, p)
    if method == "uniform":
        half = N // 2
        kx = ky = theta_to_knots_lshape(jnp.zeros(half), jnp.zeros(half), p)
    else:
        sj = jnp.asarray(sigma, dtype=jnp.float64)
        kx = knots_p3_from_network_axis(params, sj, N, p, axis_id=0.0)
        ky = knots_p3_from_network_axis(params, sj, N, p, axis_id=1.0)
    res = galerkin_solve_lshape(kx, ky, p, n_eff, n_eff,
                                jnp.asarray(sigma[0]), jnp.asarray(sigma[1]),
                                q_K=p + 1, q_F=Q_F)
    return np.asarray(kx), np.asarray(ky), np.asarray(res.u_h)


def gradmag_on_grid(C, kx, ky, p, n=81, lo=0.25, hi=0.75):
    g = np.linspace(lo, hi, n)
    Bx, dBx = _p3_basis_on_grid(np.asarray(kx), p, g)
    By, dBy = _p3_basis_on_grid(np.asarray(ky), p, g)
    ux, uy = _grad_on_grid(C, np.asarray(Bx), np.asarray(dBx), np.asarray(By), np.asarray(dBy))
    return g, np.sqrt(np.asarray(ux) ** 2 + np.asarray(uy) ** 2)   # [ix, iy]


def write_heat(path, g, Z):
    # z = physical |grad u_h| (kept for reuse); logz = log10(z) is the column the
    # figure colour-maps -- the corner singularity makes |grad u_h| span ~64x
    # (bulk ~0.09, corner peak up to 5.74) so a log colour scale is the only one
    # that shows both the field decay AND the per-mesh corner peak.
    with open(path, "w") as f:
        f.write("x,y,z,logz\n")
        for iy, y in enumerate(g):
            for ix, x in enumerate(g):
                z = float(Z[ix, iy])
                f.write(f"{x:.6f},{y:.6f},{z:.6e},{np.log10(max(z, 1e-12)):.6e}\n")


def unique_knots(kv, tol=1e-7):
    out = []
    for v in np.sort(np.asarray(kv)):
        if not out or abs(v - out[-1]) > tol:
            out.append(float(v))
    return out


def mesh_segments(xk, yk, lo=0.25, hi=0.75):
    """Knot grid lines with the L-shape cutout (removed quadrant x>0.5 & y<0.5),
    clipped to [lo,hi]. Vertical lines first, then horizontal."""
    segs = []
    for x in xk:
        if x < lo - 1e-9 or x > hi + 1e-9:
            continue
        y0, y1 = (max(0.5, lo), hi) if x > 0.5 + 1e-9 else (lo, hi)
        segs.append(((x, y0), (x, y1)))
    for y in yk:
        if y < lo - 1e-9 or y > hi + 1e-9:
            continue
        x0, x1 = (lo, min(0.5, hi)) if y < 0.5 - 1e-9 else (lo, hi)
        segs.append(((x0, y), (x1, y)))
    return segs


def write_mesh(path, segs):
    with open(path, "w") as f:
        f.write("\n\n".join(f"{a[0]:.6f} {a[1]:.6f}\n{b[0]:.6f} {b[1]:.6f}" for a, b in segs) + "\n\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = {p: load_checkpoint(os.path.join(
        _ROOT, f"data_results/lshape/p{p}/checkpoints/seed0/checkpoint_final.npz")) for p in (2, 3)}
    peaks = {}
    for method, p in (("uniform", 2), ("radapted", 2), ("radapted", 3)):
        kx, ky, C = solve(ck[p], SIGMA, p, "uniform" if method == "uniform" else "radapt")
        g, G = gradmag_on_grid(C, kx, ky, p)
        write_heat(os.path.join(OUT, f"P3_grad_{method}_p{p}.csv"), g, G)
        write_mesh(os.path.join(OUT, f"P3_grad_knots_{method}_p{p}.dat"),
                   mesh_segments(unique_knots(kx), unique_knots(ky)))
        peaks[f"{method}_p{p}"] = float(G.max())
        print(f"  {method:9} p{p}: zoom |grad u_h| peak = {G.max():.4f}  "
              f"(n_unique_knots x={len(unique_knots(kx))})")
    vmax = max(peaks.values())
    print(f"\n  point meta max (vmax) = {vmax:.4f}")
    print(f"  wrote 3 CSV + 3 knot .dat to {OUT}")


if __name__ == "__main__":
    main()
