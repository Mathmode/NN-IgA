from __future__ import annotations

"""Global DOF maps for the 3-patch L-shape discretization (square patches).

Patch layout (parametric == physical coordinates):
  - P0: [-1,0] x [-1,0]
  - P1: [-1,0] x [ 0,1]
  - P2: [ 0,1] x [ 0,1]

Interfaces:
  - Gamma01: y=0, x in [-1,0]  (P0 top)  <-> (P1 bottom)
  - Gamma12: x=0, y in [ 0,1]  (P1 right) <-> (P2 left)

The global mapping identifies corresponding tensor-product basis functions
along these interfaces (conforming coupling).

This module is intentionally NumPy/Python only (static mesh topology); it is
used to generate integer maps that are then passed into JAX code.
"""

from typing import Dict, List, Tuple

import numpy as np


def _uf_find(parent: List[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: List[int], a: int, b: int) -> None:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


def build_global_dof_maps(
    nBxL: int,
    nBxR: int,
    nByB: int,
    nByT: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (g0,g1,g2,n_dofs) for the 3-patch L-shape coupling."""

    nBxL = int(nBxL)
    nBxR = int(nBxR)
    nByB = int(nByB)
    nByT = int(nByT)

    def lid(ix: int, iy: int, nBx: int) -> int:
        return int(iy) * int(nBx) + int(ix)

    n0 = nBxL * nByB
    n1 = nBxL * nByT
    n2 = nBxR * nByT
    off0 = 0
    off1 = n0
    off2 = n0 + n1
    total = n0 + n1 + n2

    parent = list(range(total))

    # Gamma01: P0 (iy=nByB-1) <-> P1 (iy=0)
    for ix in range(nBxL):
        _uf_union(parent, off0 + lid(ix, nByB - 1, nBxL), off1 + lid(ix, 0, nBxL))

    # Gamma12: P1 (ix=nBxL-1) <-> P2 (ix=0)
    for iy in range(nByT):
        _uf_union(parent, off1 + lid(nBxL - 1, iy, nBxL), off2 + lid(0, iy, nBxR))

    root_to_gid: Dict[int, int] = {}
    gid = 0
    gmap = [0] * total
    for i in range(total):
        r = _uf_find(parent, i)
        if r not in root_to_gid:
            root_to_gid[r] = gid
            gid += 1
        gmap[i] = root_to_gid[r]

    gmap_np = np.asarray(gmap, dtype=np.int32)
    g0 = gmap_np[off0:off1].reshape(nByB, nBxL)
    g1 = gmap_np[off1:off2].reshape(nByT, nBxL)
    g2 = gmap_np[off2:].reshape(nByT, nBxR)
    return g0, g1, g2, int(gid)


def dirichlet_dofs_lshape(g0: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Sorted array of global Dirichlet dofs for the L-shape boundary."""
    dir_set = set()

    # P0 boundaries: x=-1, y=-1, x=0 (cutout)
    dir_set.update(g0[:, 0].tolist())
    dir_set.update(g0[0, :].tolist())
    dir_set.update(g0[:, -1].tolist())

    # P1 boundaries: x=-1, y=1
    dir_set.update(g1[:, 0].tolist())
    dir_set.update(g1[-1, :].tolist())

    # P2 boundaries: x=1, y=1, y=0 (cutout)
    dir_set.update(g2[:, -1].tolist())
    dir_set.update(g2[-1, :].tolist())
    dir_set.update(g2[0, :].tolist())

    return np.asarray(sorted(dir_set), dtype=np.int32)


__all__ = [
    "build_global_dof_maps",
    "dirichlet_dofs_lshape",
]

