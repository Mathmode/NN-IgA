from __future__ import annotations

"""CSV export helpers for the JAX L-shape experiment (multipatch).

Torch-free variant of ``2D/export_utils.py`` focused on:
  - breakpoints.csv
  - knots.csv
  - greville.csv
  - solution.csv
  - history.csv (optional)

Note: this exports *control coefficients* u (global DOF vector) with associated
Greville coordinates per patch. It also writes `gradient.csv` with evaluated
field gradients (ux, uy) on the Greville tensor grid.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import jax

from csv_utils import ensure_dir, write_csv_dicts
from bspline_basis import bspline_basis_local
from iga2d_jax import greville_abscissae


def _to_numpy(x) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def _eval_patch_grad_on_greville(
    C: np.ndarray,
    *,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ux,uy) evaluated on the Greville tensor grid for one patch."""
    p = int(p)
    dtype = jax.numpy.float64

    Cj = jax.numpy.asarray(C, dtype=dtype)
    kx = jax.numpy.asarray(knots_x, dtype=dtype).reshape(-1)
    ky = jax.numpy.asarray(knots_y, dtype=dtype).reshape(-1)

    gx = greville_abscissae(kx, p)
    gy = greville_abscissae(ky, p)

    Nx, dNx, _d2Nx, spans_x = bspline_basis_local(gx, kx, p)
    Ny, dNy, _d2Ny, spans_y = bspline_basis_local(gy, ky, p)

    idx_x = spans_x[:, None] - p + jax.numpy.arange(p + 1, dtype=jax.numpy.int32)[None, :]
    idx_y = spans_y[:, None] - p + jax.numpy.arange(p + 1, dtype=jax.numpy.int32)[None, :]

    # (nBy,nBx,p+1,p+1): local coefficient blocks for each (y,x) Greville point
    C_blocks = Cj[idx_y[:, None, :, None], idx_x[None, :, None, :]]

    ux = jax.numpy.einsum("ya,yxab,xb->yx", Ny, C_blocks, dNx)
    uy = jax.numpy.einsum("ya,yxab,xb->yx", dNy, C_blocks, Nx)

    ux_np, uy_np = jax.device_get((ux, uy))
    return np.asarray(ux_np, dtype=float), np.asarray(uy_np, dtype=float)


def save_history_csv(path: str | Path, history: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    if len(history) == 0:
        write_csv_dicts(path, [], ["iter"])
        return

    fieldnames: list[str] = []
    seen = set()
    for rec in history:
        for k in rec.keys():
            if k not in seen:
                fieldnames.append(str(k))
                seen.add(k)

    write_csv_dicts(path, list(history), fieldnames)


def export_lshape_case(
    out_dir: str | Path,
    *,
    method: str,
    u_global,
    sys: Any,
    history: Sequence[Mapping[str, object]] | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)
    p = int(sys.p)
    u_np = _to_numpy(u_global).reshape(-1)

    rows_breakpoints: list[dict] = []
    rows_knots: list[dict] = []
    rows_greville: list[dict] = []
    rows_solution: list[dict] = []
    rows_grad: list[dict] = []

    for patch_name, P in sys.patches.items():
        bp_x = _to_numpy(P.breakpoints_x).astype(float)
        bp_y = _to_numpy(P.breakpoints_y).astype(float)
        kn_x = _to_numpy(P.knots_x).astype(float)
        kn_y = _to_numpy(P.knots_y).astype(float)

        for axis, bp in (("x", bp_x), ("y", bp_y)):
            for i, val in enumerate(bp.tolist()):
                rows_breakpoints.append(
                    {"method": method, "patch": patch_name, "axis": axis, "i": int(i), "value": float(val)}
                )

        for axis, kn in (("x", kn_x), ("y", kn_y)):
            for i, val in enumerate(kn.tolist()):
                rows_knots.append({"method": method, "patch": patch_name, "axis": axis, "i": int(i), "value": float(val)})

        gx = _to_numpy(greville_abscissae(jax.device_put(kn_x), p)).astype(float)
        gy = _to_numpy(greville_abscissae(jax.device_put(kn_y), p)).astype(float)

        gmap = _to_numpy(getattr(sys, f"g{patch_name[-1]}")).astype(int)
        nBy, nBx = gmap.shape

        # Evaluate ∇u on the Greville tensor grid (for post-processing/flux plots).
        C_patch = u_np[gmap.reshape(-1)].reshape(nBy, nBx)
        ux_patch, uy_patch = _eval_patch_grad_on_greville(C_patch, knots_x=kn_x, knots_y=kn_y, p=p)

        for iy in range(nBy):
            y = float(gy[iy])
            for ix in range(nBx):
                dof = int(gmap[iy, ix])
                x = float(gx[ix])
                rows_greville.append(
                    {"method": method, "patch": patch_name, "ix": int(ix), "iy": int(iy), "dof": dof, "x": x, "y": y}
                )
                rows_solution.append(
                    {
                        "method": method,
                        "patch": patch_name,
                        "ix": int(ix),
                        "iy": int(iy),
                        "dof": dof,
                        "x": x,
                        "y": y,
                        "u": float(u_np[dof]),
                    }
                )
                ux = float(ux_patch[iy, ix])
                uy = float(uy_patch[iy, ix])
                rows_grad.append(
                    {
                        "method": method,
                        "patch": patch_name,
                        "ix": int(ix),
                        "iy": int(iy),
                        "dof": dof,
                        "x": x,
                        "y": y,
                        "ux": ux,
                        "uy": uy,
                        "grad_mag": float(np.hypot(ux, uy)),
                    }
                )

    write_csv_dicts(out_dir / "breakpoints.csv", rows_breakpoints, ["method", "patch", "axis", "i", "value"])
    write_csv_dicts(out_dir / "knots.csv", rows_knots, ["method", "patch", "axis", "i", "value"])
    write_csv_dicts(out_dir / "greville.csv", rows_greville, ["method", "patch", "ix", "iy", "dof", "x", "y"])
    write_csv_dicts(out_dir / "solution.csv", rows_solution, ["method", "patch", "ix", "iy", "dof", "x", "y", "u"])
    write_csv_dicts(out_dir / "gradient.csv", rows_grad, ["method", "patch", "ix", "iy", "dof", "x", "y", "ux", "uy", "grad_mag"])

    if history is not None:
        save_history_csv(out_dir / "history.csv", list(history))


__all__ = [
    "export_lshape_case",
    "save_history_csv",
]
