"""High-resolution NGSolve reference for lshape (Aballay 4.3.2 L-shape).

The L-shape problem has no analytic solution. To report H¹ relative error
we compute a high-resolution FEM reference using NGSolve on an unstructured
triangular mesh with adaptive refinement near the re-entrant corner.

This module is **import-guarded**: if NGSolve is not available, the module
imports cleanly but every function raises ``ImportError`` with installation
instructions when called. The rest of the pipeline (tests, training, eval
without NGSolve cache) keeps working.

The reference is computed ONCE per (sigma1, sigma2) and cached to disk via
pickle. The cache is loaded by ``evaluation_v2`` to compute the H¹ relative
error column ``h1_rel_error_ngsolve``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import ngsolve as _ngsolve
    from netgen import occ as _netgen_occ
    NGSOLVE_AVAILABLE = True
    _NGSOLVE_IMPORT_ERROR: Optional[Exception] = None
except Exception as _err:                                                # pragma: no cover
    NGSOLVE_AVAILABLE = False
    _NGSOLVE_IMPORT_ERROR = _err


_INSTALL_HINT = (
    "NGSolve is required for the lshape high-resolution reference but is not\n"
    "installed in this environment.\n\n"
    "Install with:\n"
    "    pip install ngsolve netgen-occ\n"
    "or on the cluster:\n"
    "    python -m pip install ngsolve netgen-occ\n"
)


def _ensure_ngsolve() -> None:
    if not NGSOLVE_AVAILABLE:
        raise ImportError(_INSTALL_HINT) from _NGSOLVE_IMPORT_ERROR


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _build_lshape_geometry():
    """Build the L-shape domain Omega = (0, 1)^2 \\ (0.5, 1] x [0, 0.5)
    as a SINGLE face whose boundary is the closed polygon

        (0, 0) -> (0.5, 0) -> (0.5, 0.5) -> (1, 0.5) -> (1, 1) -> (0, 1) -> (0, 0)

    i.e. exactly 6 external edges. All 6 edges receive the same boundary
    tag (``default``) so that ``dirichlet=".*"`` applies u = 0 on the
    *external* boundary only.

    Previous bug (audit_p3_diagnostic.md): the earlier ``Glue`` of three
    0.5 x 0.5 squares produced a mesh with 10 boundary edges (6 external
    + 4 duplicated internal interfaces between the three faces). The
    regex ``dirichlet=".*"`` then matched the interior interfaces too,
    decoupling the L-shape into 3 independent squares and reducing the
    reference ``sigma-H1^2(sigma=(1,1))`` from the correct 1.338e-02
    down to the buggy 6.59e-03 = 3 * (0.5)^4 * 0.035144.

    Since the new single-face mesh has only one material, piecewise sigma
    is now expressed as a coordinate-dependent ``CoefficientFunction``
    inside ``compute_p3_ngsolve_reference`` (using ``ngs.IfPos`` on x and
    y), not via material names.
    """
    _ensure_ngsolve()
    from netgen.occ import OCCGeometry, WorkPlane

    wp = (
        WorkPlane()
        .MoveTo(0.0, 0.0)
        .LineTo(0.5, 0.0)
        .LineTo(0.5, 0.5)
        .LineTo(1.0, 0.5)
        .LineTo(1.0, 1.0)
        .LineTo(0.0, 1.0)
        .Close()
    )
    face = wp.Face()
    face.faces.name = "lshape"
    return OCCGeometry(face, dim=2)


# --------------------------------------------------------------------------
# Reference solve
# --------------------------------------------------------------------------


def compute_p3_ngsolve_reference(
    sigma1: float,
    sigma2: float,
    *,
    max_h: float = 0.01,
    order: int = 3,
    refine_corner: int = 3,
) -> dict:
    """High-resolution NGSolve solve of lshape at one (sigma1, sigma2).

    PDE: ``- div(sigma(x) grad u) = 1`` in Omega, ``u = 0`` on boundary(Omega).

    The mesh uses ``MeshingParameters.maxh = max_h`` globally, with extra
    local refinement around the re-entrant corner (0.5, 0.5) by a factor
    of ``2 ** refine_corner``. Polynomial degree ``order`` (default 3 ~=
    Aballay's quadratic-piecewise convention).

    Returns a dict with the post-solve diagnostics:

        - ``energy``                  : 0.5 * int sigma|grad u|^2 - int u
                                        (= -ritz, on this finer mesh)
        - ``h1_seminorm_sq``          : int |grad u|^2
        - ``sigma_h1_seminorm_sq``    : int sigma |grad u|^2  (= 2 * int u)
        - ``int_u``                   : int u dx
        - plus mesh metadata.
    """
    _ensure_ngsolve()
    import ngsolve as ngs
    from netgen.meshing import MeshingParameters

    geo = _build_lshape_geometry()
    mp = MeshingParameters(maxh=float(max_h))
    # Local refinement near the re-entrant corner (sphere/disc in 2D).
    mp.RestrictH(
        x=0.5, y=0.5, z=0.0,
        h=float(max_h) / (2.0 ** int(refine_corner)),
    )
    mesh = ngs.Mesh(geo.GenerateMesh(mp=mp))

    fes = ngs.H1(mesh, order=int(order), dirichlet=".*")
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # Piecewise sigma by COORDINATE (since the single-face mesh has only
    # one material "lshape"). Regions:
    #   top_left   = {x < 0.5, y > 0.5} -> sigma = 1
    #   bot_left   = {x < 0.5, y < 0.5} -> sigma = sigma1
    #   top_right  = {x > 0.5, y > 0.5} -> sigma = sigma2
    # The removed quadrant {x > 0.5, y < 0.5} is NOT in the mesh, so its
    # sigma value never enters integration; we fall through to sigma2
    # for that branch (irrelevant, but defined for completeness).
    # IfPos(a, b, c) returns b if a > 0, c if a <= 0.
    sigma_cf = ngs.IfPos(
        0.5 - ngs.x,                                          # x < 0.5
        ngs.IfPos(0.5 - ngs.y, float(sigma1), 1.0),           #   y < 0.5: bot_left;  y > 0.5: top_left
        ngs.IfPos(0.5 - ngs.y, float(sigma2), float(sigma2)), # x > 0.5: top_right (the {x>0.5, y<0.5} branch is unreachable)
    )

    a = ngs.BilinearForm(fes, symmetric=True)
    a += sigma_cf * ngs.grad(u) * ngs.grad(v) * ngs.dx
    a.Assemble()

    f = ngs.LinearForm(fes)
    f += 1.0 * v * ngs.dx
    f.Assemble()

    gfu = ngs.GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    sigma_h1_sq = float(
        ngs.Integrate(sigma_cf * ngs.grad(gfu) * ngs.grad(gfu), mesh)
    )
    h1_sq = float(ngs.Integrate(ngs.grad(gfu) * ngs.grad(gfu), mesh))
    int_u = float(ngs.Integrate(gfu, mesh))
    energy = 0.5 * sigma_h1_sq - int_u

    return {
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "energy": energy,
        "h1_seminorm_sq": h1_sq,
        "sigma_h1_seminorm_sq": sigma_h1_sq,
        "int_u": int_u,
        "dof_count": int(fes.ndof),
        "max_h": float(max_h),
        "order": int(order),
        "refine_corner": int(refine_corner),
    }


# --------------------------------------------------------------------------
# Cache I/O
# --------------------------------------------------------------------------


def build_p3_reference_cache(
    sigma_tuples: np.ndarray,
    cache_path: Path,
    *,
    max_h: float = 0.01,
    order: int = 3,
    refine_corner: int = 3,
    force_recompute: bool = False,
    save_every: int = 10,
    verbose: bool = True,
) -> dict:
    """Compute (or extend) the NGSolve reference cache for the given σ list.

    Cache layout (pickled dict):
        {(sigma1, sigma2): {energy, h1_seminorm_sq, sigma_h1_seminorm_sq,
                            int_u, dof_count, max_h, order, refine_corner}}

    Missing entries are computed and added; existing entries are kept
    unless ``force_recompute=True``.
    """
    _ensure_ngsolve()

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if cache_path.exists() and not force_recompute:
        with cache_path.open("rb") as fh:
            cache = pickle.load(fh)
        if verbose:
            print(f"  loaded {len(cache)} pre-existing entries from {cache_path}")

    sigma_tuples = np.asarray(sigma_tuples, dtype=np.float64).reshape(-1, 2)
    n_total = sigma_tuples.shape[0]
    n_computed = 0

    for i, (s1, s2) in enumerate(sigma_tuples):
        key = (float(s1), float(s2))
        if key in cache and not force_recompute:
            continue

        if verbose:
            print(
                f"  [{i + 1:>4}/{n_total}] sigma=({s1:.4g}, {s2:.4g}) ... ",
                end="", flush=True,
            )
        cache[key] = compute_p3_ngsolve_reference(
            s1, s2,
            max_h=max_h, order=order, refine_corner=refine_corner,
        )
        n_computed += 1
        if verbose:
            entry = cache[key]
            print(
                f"DOFs={entry['dof_count']:>7d}  "
                f"σ-H1²={entry['sigma_h1_seminorm_sq']:.4e}  "
                f"E={entry['energy']:.4e}",
                flush=True,
            )

        if n_computed % save_every == 0:
            with cache_path.open("wb") as fh:
                pickle.dump(cache, fh)

    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)

    if verbose:
        print(f"  wrote {len(cache)} total entries to {cache_path}")
    return cache


def load_p3_reference_cache(cache_path: Path) -> dict:
    """Load an existing NGSolve reference cache from disk."""
    cache_path = Path(cache_path)
    with cache_path.open("rb") as fh:
        return pickle.load(fh)


__all__ = [
    "NGSOLVE_AVAILABLE",
    "compute_p3_ngsolve_reference",
    "build_p3_reference_cache",
    "load_p3_reference_cache",
]
