from __future__ import annotations

"""CSV export helpers for the JAX 2D experiments.

This is a torch-free variant of ``2D/export_utils.py`` focused on the square
single-patch case. It writes the standard artifacts used by the repo:
  - breakpoints.csv
  - knots.csv
  - greville.csv
  - solution.csv
  - history.csv (optional)
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import jax

from csv_utils import ensure_dir, write_csv_dicts
from iga2d_jax import greville_abscissae


def _to_numpy(x) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float)


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


def export_square_case(
    out_dir: str | Path,
    *,
    method: str,
    u_global,
    sys: Any,
    history: Sequence[Mapping[str, object]] | None = None,
) -> None:
    out_dir = ensure_dir(out_dir)
    p = int(sys.p)

    bp_x = _to_numpy(sys.breakpoints_x)
    bp_y = _to_numpy(sys.breakpoints_y)
    knots_x = _to_numpy(sys.knots_x)
    knots_y = _to_numpy(sys.knots_y)
    u_np = np.asarray(jax.device_get(u_global)).reshape(-1)

    rows_breakpoints: list[dict] = []
    for axis, bp in (("x", bp_x), ("y", bp_y)):
        for i, val in enumerate(bp.tolist()):
            rows_breakpoints.append({"method": method, "axis": axis, "i": int(i), "value": float(val)})

    rows_knots: list[dict] = []
    for axis, kn in (("x", knots_x), ("y", knots_y)):
        for i, val in enumerate(kn.tolist()):
            rows_knots.append({"method": method, "axis": axis, "i": int(i), "value": float(val)})

    gx = _to_numpy(greville_abscissae(jax.device_put(knots_x), p))
    gy = _to_numpy(greville_abscissae(jax.device_put(knots_y), p))
    nBx = int(sys.nBx)
    nBy = int(sys.nBy)

    rows_greville: list[dict] = []
    rows_solution: list[dict] = []
    for iy in range(nBy):
        y = float(gy[iy])
        for ix in range(nBx):
            dof = int(iy * nBx + ix)
            x = float(gx[ix])
            rows_greville.append({"method": method, "ix": int(ix), "iy": int(iy), "dof": dof, "x": x, "y": y})
            rows_solution.append(
                {"method": method, "ix": int(ix), "iy": int(iy), "dof": dof, "x": x, "y": y, "u": float(u_np[dof])}
            )

    write_csv_dicts(out_dir / "breakpoints.csv", rows_breakpoints, ["method", "axis", "i", "value"])
    write_csv_dicts(out_dir / "knots.csv", rows_knots, ["method", "axis", "i", "value"])
    write_csv_dicts(out_dir / "greville.csv", rows_greville, ["method", "ix", "iy", "dof", "x", "y"])
    write_csv_dicts(out_dir / "solution.csv", rows_solution, ["method", "ix", "iy", "dof", "x", "y", "u"])

    if history is not None:
        save_history_csv(out_dir / "history.csv", list(history))


__all__ = [
    "export_square_case",
    "save_history_csv",
]

