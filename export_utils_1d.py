from __future__ import annotations

"""CSV export helpers for the 1D experiments (JAX).

This module is shared by:
  - Experiment 1: power manufactured solution
  - Experiment 2: Helmholtz with a C0 interface

It writes the standard per-case artifacts used across the repo:
  - breakpoints.csv
  - knots.csv
  - greville.csv
  - solution.csv
  - history.csv (optional)

All computations (basis evaluation) are performed in JAX and converted to NumPy
for I/O. Exact solutions may be NumPy or JAX functions; both are supported.
"""

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from bspline_basis import bspline_basis_local
from csv_utils import ensure_dir, write_csv_rows

Array = jnp.ndarray


def greville_abscissae(knots: Array, p: int) -> Array:
    """Greville points for an (open) B-spline knot vector (works with repeated interior knots)."""
    p = int(p)
    knots = jnp.asarray(knots).reshape(-1)
    n_ctrl = int(knots.shape[0] - p - 1)
    if p <= 0 or n_ctrl <= 0:
        return jnp.empty((0,), dtype=knots.dtype)

    parts = [knots[1 + i : 1 + i + n_ctrl] for i in range(p)]
    sums = jnp.stack(parts, axis=0).sum(axis=0)
    return sums / float(p)


def _breakpoints_from_knots(knots: np.ndarray, p: int) -> np.ndarray:
    """Best-effort breakpoints extraction for open-clamped knots (Exp. 1)."""
    kv = np.asarray(knots, dtype=float).ravel()
    p = int(p)
    if kv.size < 2 * p + 2:
        return np.asarray([], dtype=float)
    return kv[p:-p].astype(float)


def save_breakpoints_csv(out_dir: str | Path, *, breakpoints: np.ndarray) -> None:
    out_dir = ensure_dir(out_dir)
    bp = np.asarray(breakpoints, dtype=float).ravel()
    rows = [(int(i), float(bp[i])) for i in range(bp.size)]
    write_csv_rows(Path(out_dir) / "breakpoints.csv", ["i", "a"], rows)


def save_knots_csv(out_dir: str | Path, *, knots: np.ndarray) -> None:
    out_dir = ensure_dir(out_dir)
    kv = np.asarray(knots, dtype=float).ravel()
    rows = [(int(i), float(kv[i])) for i in range(kv.size)]
    write_csv_rows(Path(out_dir) / "knots.csv", ["knot_index", "x"], rows)


def save_greville_csv(out_dir: str | Path, *, knots: np.ndarray, p: int, u_coeffs: np.ndarray) -> None:
    """Save greville.csv with columns i, xg, uh."""
    out_dir = ensure_dir(out_dir)
    p = int(p)

    kv = jnp.asarray(knots, dtype=jnp.float64).reshape(-1)
    u = jnp.asarray(u_coeffs, dtype=kv.dtype).reshape(-1)

    G = greville_abscissae(kv, p)
    if int(G.shape[0]) == 0:
        write_csv_rows(Path(out_dir) / "greville.csv", ["i", "xg", "uh"], [])
        return

    Ng, _dNg, _d2Ng, spans_g = bspline_basis_local(G, kv, p)
    cols_g = spans_g[:, None] - p + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    u_loc = u[cols_g]
    uh_g = jnp.sum(Ng * u_loc, axis=1)

    G_np = np.asarray(jax.device_get(G), dtype=float)
    uh_np = np.asarray(jax.device_get(uh_g), dtype=float)
    rows = [(int(i), float(G_np[i]), float(uh_np[i])) for i in range(G_np.size)]
    write_csv_rows(Path(out_dir) / "greville.csv", ["i", "xg", "uh"], rows)


def save_solution_csv(
    out_dir: str | Path,
    *,
    knots: np.ndarray,
    p: int,
    u_coeffs: np.ndarray,
    u_exact,
    du_exact=None,
    n_eval: int = 2001,
) -> None:
    """Save solution.csv with columns x, uh, duh, ue, due."""
    out_dir = ensure_dir(out_dir)
    p = int(p)
    n_eval = int(n_eval)

    kv = jnp.asarray(knots, dtype=jnp.float64).reshape(-1)
    u = jnp.asarray(u_coeffs, dtype=kv.dtype).reshape(-1)

    x_np = np.linspace(float(kv[0]), float(kv[-1]), int(n_eval), dtype=float)
    x = jnp.asarray(x_np, dtype=kv.dtype)

    N, dN, _d2N, spans = bspline_basis_local(x, kv, p)
    cols = spans[:, None] - p + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    u_loc = u[cols]

    uh = jnp.sum(N * u_loc, axis=1)
    duh = jnp.sum(dN * u_loc, axis=1) if p >= 1 else jnp.zeros_like(uh)

    uh_np = np.asarray(jax.device_get(uh), dtype=float)
    duh_np = np.asarray(jax.device_get(duh), dtype=float)
    ue_np = np.asarray(u_exact(x_np), dtype=float)
    due_np = np.asarray(du_exact(x_np), dtype=float) if du_exact is not None else np.zeros_like(ue_np)

    rows = [
        (float(xx), float(uu), float(du_), float(ue_), float(due_))
        for xx, uu, du_, ue_, due_ in zip(x_np, uh_np, duh_np, ue_np, due_np)
    ]
    write_csv_rows(Path(out_dir) / "solution.csv", ["x", "uh", "duh", "ue", "due"], rows)


def _to_scalar_or_list(v: Any):
    if isinstance(v, (list, tuple)):
        return list(v)
    if hasattr(v, "shape"):
        arr = np.asarray(v)
        if arr.ndim == 0 or arr.size == 1:
            return arr.reshape(()).item()
        return arr.tolist()
    return v


def save_history_csv(path: str | Path, history: Any) -> None:
    """Write history.csv.

    Supported forms:
      - list[dict]: each entry is one record
      - dict[str, list]: dict-of-lists style
    """
    path = Path(path)
    ensure_dir(path.parent)

    # list-of-dicts
    if isinstance(history, (list, tuple)) and (len(history) == 0 or isinstance(history[0], Mapping)):
        rows = list(history)
        if len(rows) == 0:
            write_csv_rows(path, ["iter"], [])
            return
        fieldnames: list[str] = []
        seen = set()
        for rec in rows:
            for k in rec.keys():
                if k not in seen:
                    fieldnames.append(str(k))
                    seen.add(k)

        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for rec in rows:
                w.writerow({k: rec.get(k, "") for k in fieldnames})
        return

    # dict-of-lists
    if isinstance(history, Mapping):
        seq: dict[str, list] = {}
        scal: dict[str, Any] = {}

        for k, v in history.items():
            vv = _to_scalar_or_list(v)
            if isinstance(vv, list):
                seq[str(k)] = vv
            else:
                scal[str(k)] = vv

        keys_seq = sorted(seq.keys())
        if "iter" in keys_seq:
            keys_seq = ["iter"] + [k for k in keys_seq if k != "iter"]
        keys_scal = sorted(scal.keys())
        header = keys_seq + keys_scal

        L = max((len(v) for v in seq.values()), default=1)
        rows_out: list[list[Any]] = []
        for i in range(L):
            row: list[Any] = []
            for k in keys_seq:
                vlist = seq.get(k, [])
                row.append(vlist[i] if i < len(vlist) else "")
            for k in keys_scal:
                row.append(scal[k])
            rows_out.append(row)

        write_csv_rows(path, header, rows_out)
        return

    raise TypeError("history must be a list-of-dicts or a dict-of-lists")


def save_full_state(
    out_dir: str | Path,
    *,
    knots: np.ndarray,
    p: int,
    u_coeffs: np.ndarray,
    u_exact,
    du_exact=None,
    breakpoints: np.ndarray | None = None,
    n_eval: int = 2001,
) -> None:
    """Save all per-case CSV artifacts in *out_dir* (method folder)."""
    out_dir = ensure_dir(out_dir)
    p = int(p)

    bp = np.asarray(breakpoints, dtype=float).ravel() if breakpoints is not None else _breakpoints_from_knots(knots, p)
    save_breakpoints_csv(out_dir, breakpoints=bp)
    save_knots_csv(out_dir, knots=np.asarray(knots, dtype=float))
    save_greville_csv(out_dir, knots=np.asarray(knots, dtype=float), p=int(p), u_coeffs=np.asarray(u_coeffs, dtype=float))
    save_solution_csv(
        out_dir,
        knots=np.asarray(knots, dtype=float),
        p=int(p),
        u_coeffs=np.asarray(u_coeffs, dtype=float),
        u_exact=u_exact,
        du_exact=du_exact,
        n_eval=int(n_eval),
    )


__all__ = [
    "greville_abscissae",
    "save_breakpoints_csv",
    "save_knots_csv",
    "save_greville_csv",
    "save_solution_csv",
    "save_history_csv",
    "save_full_state",
]

