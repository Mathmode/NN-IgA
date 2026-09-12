#!/usr/bin/env python
"""Regenerate the ADV-DIFF (Experiment 5, internal code P4) SOLUTION + MESH + CROSS-SECTION
figure data from the CURRENT checkpoints, into ``TiKz_overleaf/exp5_advdiff/``:
  * ``fielddata/P4_heat_param{1,2,3}.csv``  -- u_h on an 81x81 grid over [0,1]^2
    (p2 network / r-adapted, N=16, seed0/checkpoint_final), columns x,y,z.
  * ``meshdata/P4_mesh_p{2,3}_param{1,2,3}_{uniform,positional}.dat`` -- knot grid lines.
  * ``fielddata/P4_xsection_param{i}.csv`` -- the y=0.5 cross-section, columns
    x,u_exact,u_uniform,u_radapted (400 dense x), and
    ``P4_xsection_markers_param{i}.csv`` / ``..._markers_radapt_param{i}.csv`` -- u_h at the
    greville abscissae of the uniform / r-adapted p2 mesh (marker positions = the DOFs; the
    r-adapted ones cluster in the x=1 boundary layer).

Representative parameter spectrum (originals not recorded). nu = (logeps, b): the network
gets logeps directly, the solver uses eps=10^logeps; boundary layer width delta=eps/b at
x=1. param1 (-1.5,0.5) delta~0.063 thick, param2 (-1.75,1.0) ~0.018, param3 (-2.0,2.0)
~0.005 thin. Recipe = eval_advdiff: uniform via open_uniform_knots, positional via
knots_p4_from_network_axis (= the smooth-[0,1] builder), galerkin_solve_advdiff, q_F=50.

Run from the repo root:  python 2D/scripts/generate_advdiff_figure_data.py
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
from src.parametric.training_advdiff import knots_p4_from_network_axis   # = knots_p2_from_network_axis
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.advdiff.pde import u_exact_advdiff
from src.nonparametric.solver_2d import greville_abscissae
from src.parametric.evaluation_v2 import _p3_basis_on_grid
from common.h1_seminorm_2d import open_uniform_knots

OUT = os.path.join(_ROOT, "TiKz_overleaf", "exp5_advdiff")
N = 16
Q_F = 50
PARAMS = {"param1": (-1.5, 0.5),      # eps=10^-1.5=0.032, delta~0.063 (thick layer)
          "param2": (-1.75, 1.0),     # eps=0.018,          delta~0.018
          "param3": (-2.0, 2.0)}      # eps=0.01,           delta~0.005 (thin layer)


def solve(params, nu, p, method):
    logeps, b = float(nu[0]), float(nu[1]); eps = 10.0 ** logeps
    if method == "uniform":
        k = jnp.asarray(open_uniform_knots(N, p)); kx = ky = k
    else:
        nu_j = jnp.asarray(nu, dtype=jnp.float64)
        kx = knots_p4_from_network_axis(params, nu_j, N, p, axis_id=0.0)
        ky = knots_p4_from_network_axis(params, nu_j, N, p, axis_id=1.0)
    res = galerkin_solve_advdiff(kx, ky, p, N, N, jnp.asarray(eps), jnp.asarray(b),
                                 q_K=p + 1, q_F=Q_F)
    return np.asarray(kx), np.asarray(ky), np.asarray(res.u_h), eps, b


def u_on_grid(C, kx, ky, p, n=81, lo=0.0, hi=1.0):
    g = np.linspace(lo, hi, n)
    Bx, _ = _p3_basis_on_grid(np.asarray(kx), p, g)
    By, _ = _p3_basis_on_grid(np.asarray(ky), p, g)
    return g, np.asarray(Bx) @ C @ np.asarray(By).T            # [ix, iy]


def u_on_line(C, kx, ky, p, xline, y0=0.5):
    """u_h(x, y0) for x in xline (tensor-product B-spline eval)."""
    Bx, _ = _p3_basis_on_grid(np.asarray(kx), p, np.asarray(xline))
    By0, _ = _p3_basis_on_grid(np.asarray(ky), p, np.array([y0]))
    return (np.asarray(Bx) @ C @ np.asarray(By0).T).ravel()


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
    """Full-square [0,1]^2 knot grid lines (no cutout)."""
    segs = [((x, lo), (x, hi)) for x in xk if lo - 1e-9 <= x <= hi + 1e-9]
    segs += [((lo, y), (hi, y)) for y in yk if lo - 1e-9 <= y <= hi + 1e-9]
    return segs


def write_mesh(path, segs):
    with open(path, "w") as f:
        f.write("\n\n".join(f"{a[0]:.6f} {a[1]:.6f}\n{b[0]:.6f} {b[1]:.6f}" for a, b in segs) + "\n\n")


def write_cols(path, header, cols):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in zip(*cols):
            f.write(",".join(f"{v:.8e}" for v in row) + "\n")


def main():
    os.makedirs(os.path.join(OUT, "fielddata"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "meshdata"), exist_ok=True)
    ck = {p: load_checkpoint(os.path.join(
        _ROOT, f"data_results/advdiff/p{p}/checkpoints/seed0/checkpoint_final.npz")) for p in (2, 3)}

    # ---- solution heatmaps (110x110, p2 network) + meshes (both degrees) ----
    for name, nu in PARAMS.items():
        kx2, ky2, C2, _, _ = solve(ck[2], nu, 2, "positional")
        g, Z = u_on_grid(C2, kx2, ky2, 2)
        write_heat(os.path.join(OUT, f"fielddata/P4_heat_{name}.csv"), g, Z)
        for p in (2, 3):
            kxp, kyp, _, _, _ = solve(ck[p], nu, p, "positional")
            kxu, kyu, _, _, _ = solve(ck[p], nu, p, "uniform")
            write_mesh(os.path.join(OUT, f"meshdata/P4_mesh_p{p}_{name}_positional.dat"),
                       mesh_segments(unique_knots(kxp), unique_knots(kyp)))
            write_mesh(os.path.join(OUT, f"meshdata/P4_mesh_p{p}_{name}_uniform.dat"),
                       mesh_segments(unique_knots(kxu), unique_knots(kyu)))
        print(f"  heat/mesh {name} nu={nu}: u_h peak={Z.max():.4e}")

    # ---- y=0.5 cross-section (p2): dense curves + greville-DOF markers ----
    xd = np.linspace(0.0, 1.0, 400)
    for name, nu in PARAMS.items():
        kxu, kyu, Cu, eps, b = solve(ck[2], nu, 2, "uniform")
        kxr, kyr, Cr, _, _ = solve(ck[2], nu, 2, "positional")
        ue = np.asarray(u_exact_advdiff(jnp.asarray(xd), jnp.full_like(jnp.asarray(xd), 0.5),
                                        jnp.asarray(eps), jnp.asarray(b)))
        uu = u_on_line(Cu, kxu, kyu, 2, xd)
        ur = u_on_line(Cr, kxr, kyr, 2, xd)
        write_cols(os.path.join(OUT, f"fielddata/P4_xsection_{name}.csv"),
                   ["x", "u_exact", "u_uniform", "u_radapted"], [xd, ue, uu, ur])
        gu = np.asarray(greville_abscissae(jnp.asarray(kxu), 2))
        write_cols(os.path.join(OUT, f"fielddata/P4_xsection_markers_{name}.csv"),
                   ["x", "u_uniform"], [gu, u_on_line(Cu, kxu, kyu, 2, gu)])
        gr = np.asarray(greville_abscissae(jnp.asarray(kxr), 2))
        write_cols(os.path.join(OUT, f"fielddata/P4_xsection_markers_radapt_{name}.csv"),
                   ["x", "u_radapted"], [gr, u_on_line(Cr, kxr, kyr, 2, gr)])
        print(f"  xsection {name}: delta={eps/b:.4f}, {len(gu)} unif / {len(gr)} radapt greville DOFs, "
              f"max|u_exact|={np.abs(ue).max():.3f}")
    print(f"DONE -> {OUT}/fielddata + meshdata")


if __name__ == "__main__":
    main()
