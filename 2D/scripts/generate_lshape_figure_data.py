#!/usr/bin/env python
"""Regenerate the L-shape (Experiment 4) SOLUTION + MESH figure data from the CURRENT
checkpoints, into ``TiKz_overleaf/exp4_lshape/``:
  * ``fielddata/P3_heat_p{2,3}_param{1,2,3}.csv`` -- discrete solution u_h on an 81x81
    grid over [0,1]^2 (r-adapted mesh, N=16, seed0/checkpoint_final), columns x,y,z.
  * ``meshdata/P3_mesh_p{2,3}_param{1,2,3}_{uniform,radapted}.dat`` -- knot grid lines
    with the L-shape cutout (removed quadrant x>0.5 & y<0.5).

Heatmap = discrete r-adapted solution (same solve that feeds the gradient panels).
The gradient ZOOM is generated separately (generate_lshape_gradient_zoom.py, log scale).
Recipe (memory: lshape-figure-data-generator): N=16, seed0; r-adapt knots via
``knots_p3_from_network_axis``, uniform via ``theta_to_knots_lshape``; solve
``galerkin_solve_lshape(..., n_eff=p3_effective_n_elem(N,p), q_K=p+1, q_F=4)``.

Run from the repo root:  python 2D/scripts/generate_lshape_figure_data.py
"""
import argparse
import json
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
from src.parametric.evaluation_v2 import _p3_basis_on_grid

OUT = os.path.join(_ROOT, "TiKz_overleaf", "exp4_lshape")
N = 16   # a trained continuation level (training used 4,8,16,32) — matches arctan/advdiff
PARAMS = {"param1": (0.89, 0.89), "param2": (0.1, 10.0), "param3": (10.0, 0.1)}
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


def u_on_grid(C, kx, ky, p, n=81, lo=0.0, hi=1.0):
    g = np.linspace(lo, hi, n)
    Bx, _ = _p3_basis_on_grid(np.asarray(kx), p, g)
    By, _ = _p3_basis_on_grid(np.asarray(ky), p, g)
    return g, np.asarray(Bx) @ C @ np.asarray(By).T            # [ix, iy]


def write_heat(path, g, Z):
    with open(path, "w") as f:
        f.write("x,y,z\n")
        for iy, y in enumerate(g):
            for ix, x in enumerate(g):
                f.write(f"{x:.6f},{y:.6f},{Z[ix, iy]:.6e}\n")


def mesh_segments(xk, yk, lo=0.0, hi=1.0):
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
        f.write("\n\n".join(f"{a[0]:.17g} {a[1]:.17g}\n{b[0]:.17g} {b[1]:.17g}" for a, b in segs) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-root", default=os.path.join(_ROOT, "data_results/lshape"))
    ap.add_argument("--output-dir", default=OUT)
    args = ap.parse_args()
    out = args.output_dir
    os.makedirs(os.path.join(out, "fielddata"), exist_ok=True)
    os.makedirs(os.path.join(out, "meshdata"), exist_ok=True)
    with open(os.path.join(out, "params_used.json"), "w") as f:
        json.dump(dict(N=N, seed=0, sigma_pairs=PARAMS, methods=["uniform", "radapted"]), f, indent=2)
        f.write("\n")
    for p in (2, 3):
        params = load_checkpoint(os.path.join(
            args.checkpoint_root, f"p{p}/checkpoints/seed0/checkpoint_final.npz"))
        for name, sigma in PARAMS.items():
            kxr, kyr, Cr = solve(params, sigma, p, "radapt")
            kxu, kyu, Cu = solve(params, sigma, p, "uniform")
            g, Z = u_on_grid(Cr, kxr, kyr, p)
            write_heat(os.path.join(out, f"fielddata/P3_heat_p{p}_{name}.csv"), g, Z)
            _, Zu = u_on_grid(Cu, kxu, kyu, p)
            write_heat(os.path.join(out, f"fielddata/P3_heat_p{p}_{name}_uniform.csv"), g, Zu)
            write_mesh(os.path.join(out, f"meshdata/P3_mesh_p{p}_{name}_radapted.dat"),
                       mesh_segments(np.unique(kxr), np.unique(kyr)))
            write_mesh(os.path.join(out, f"meshdata/P3_mesh_p{p}_{name}_uniform.dat"),
                       mesh_segments(np.unique(kxu), np.unique(kyu)))
            print(f"  p{p} {name} sigma={sigma}: u_h peak={Z.max():.4e}")
    print(f"DONE -> {out}/fielddata + meshdata")


if __name__ == "__main__":
    main()
