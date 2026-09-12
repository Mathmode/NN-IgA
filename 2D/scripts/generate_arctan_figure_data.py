#!/usr/bin/env python
"""Regenerate the ARCTAN (Experiment 3, internal code P2) SOLUTION + MESH figure data
from the CURRENT checkpoints, into ``TiKz_overleaf/exp3_arctan/``:
  * ``fielddata/P2_heat_param{1,2,3}.csv`` -- discrete solution u_h on an 80x80 grid over
    [0,1]^2 (p2 network / r-adapted mesh, N=16, seed0/checkpoint_final), columns x,y,z.
  * ``meshdata/P2_mesh_p{2,3}_param{1,2,3}_{uniform,positional}.dat`` -- knot grid lines.

Representative parameter spectrum (originals were not recorded; the old .tex comments only
kept alpha=3.9 / 19.1). sigma = (alpha, s1, s2) with u^sigma = u_axis(x)*u_axis(y),
u_axis(t)=arctan(alpha*(t-s))+arctan(alpha*s): alpha = front steepness, s = front location.
Centered fronts s1=s2=0.5, alpha in {4,10,19} (mild / medium / steep).

Recipe (matches evaluation_v2._eval_p2_*): uniform via ``open_uniform_knots(N,p)``,
positional via ``knots_p2_from_network_axis`` (the smooth-[0,1] builder parametrised by p),
solve ``galerkin_solve_arctan(..., n_elem=N, q_K=p+1, q_F)``.

Run from the repo root:  python 2D/scripts/generate_arctan_figure_data.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "2D"))
sys.path.insert(0, _ROOT)

import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.config import ARCTAN
from src.parametric.continuation import load_checkpoint
from src.parametric.positional_density_network_2d import knots_p2_from_network_axis
from src.nonparametric.solver_2d import galerkin_solve_arctan
from src.parametric.evaluation_v2 import _p3_basis_on_grid
from common.h1_seminorm_2d import open_uniform_knots

OUT = os.path.join(_ROOT, "TiKz_overleaf", "exp3_arctan")
N = 16
Q_F = int(ARCTAN["quad_forcing"])
PARAMS = {"param1": (4.0, 0.5, 0.5),      # mild centered front
          "param2": (10.0, 0.5, 0.5),     # medium
          "param3": (19.0, 0.5, 0.5)}     # steep


def solve(params, sigma, p, method):
    if method == "uniform":
        k = jnp.asarray(open_uniform_knots(N, p))
        kx = ky = k
    else:
        sj = jnp.asarray(sigma, dtype=jnp.float64)
        kx = knots_p2_from_network_axis(params, sj, N, p, axis_id=0.0)
        ky = knots_p2_from_network_axis(params, sj, N, p, axis_id=1.0)
    res = galerkin_solve_arctan(kx, ky, p, N, N,
                                jnp.asarray(sigma[0]), jnp.asarray(sigma[1]), jnp.asarray(sigma[2]),
                                q_K=p + 1, q_F=Q_F)
    return np.asarray(kx), np.asarray(ky), np.asarray(res.u_h)


def u_on_grid(C, kx, ky, p, n=80, lo=0.0, hi=1.0):
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


def unique_knots(kv, tol=1e-7):
    out = []
    for v in np.sort(np.asarray(kv)):
        if not out or abs(v - out[-1]) > tol:
            out.append(float(v))
    return out


def mesh_segments(xk, yk, lo=0.0, hi=1.0):
    """Full-square [0,1]^2 knot grid lines (no cutout). Vertical then horizontal."""
    segs = [((x, lo), (x, hi)) for x in xk if lo - 1e-9 <= x <= hi + 1e-9]
    segs += [((lo, y), (hi, y)) for y in yk if lo - 1e-9 <= y <= hi + 1e-9]
    return segs


def write_mesh(path, segs):
    with open(path, "w") as f:
        f.write("\n\n".join(f"{a[0]:.6f} {a[1]:.6f}\n{b[0]:.6f} {b[1]:.6f}" for a, b in segs) + "\n\n")


def main():
    os.makedirs(os.path.join(OUT, "fielddata"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "meshdata"), exist_ok=True)
    ck = {p: load_checkpoint(os.path.join(
        _ROOT, f"data_results/arctan/p{p}/checkpoints/seed0/checkpoint_final.npz")) for p in (2, 3)}
    for name, sigma in PARAMS.items():
        # heatmap: discrete p2 network (r-adapted) solution -- degree-independent visually
        kx2, ky2, C2 = solve(ck[2], sigma, 2, "positional")
        g, Z = u_on_grid(C2, kx2, ky2, 2)
        write_heat(os.path.join(OUT, f"fielddata/P2_heat_{name}.csv"), g, Z)
        # meshes: uniform + positional, both degrees
        for p in (2, 3):
            kxp, kyp, _ = solve(ck[p], sigma, p, "positional")
            kxu, kyu, _ = solve(ck[p], sigma, p, "uniform")
            write_mesh(os.path.join(OUT, f"meshdata/P2_mesh_p{p}_{name}_positional.dat"),
                       mesh_segments(unique_knots(kxp), unique_knots(kyp)))
            write_mesh(os.path.join(OUT, f"meshdata/P2_mesh_p{p}_{name}_uniform.dat"),
                       mesh_segments(unique_knots(kxu), unique_knots(kyu)))
        print(f"  {name} sigma={sigma}: u_h peak={Z.max():.4e}, grid {len(g)}x{len(g)}")
    print(f"DONE -> {OUT}/fielddata + meshdata")


if __name__ == "__main__":
    main()
