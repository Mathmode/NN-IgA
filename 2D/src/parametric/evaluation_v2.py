"""Evaluation pipeline producing CSV summaries for ARCTAN and LSHAPE.

Per (test sigma, N, method) one row is appended to the CSV. The fixed
column schema (~50 cols) is shared across ARCTAN and LSHAPE so the same downstream
analysis works on both; experiment-specific columns are filled with empty
strings when not applicable.

Methods evaluated per sigma and N:
  * 'uniform'              — uniform mesh baseline.
  * 'positional'           — direct network prediction (no corrector).
  * 'positional_corrected' — network prediction + L-BFGS-B local correction.
"""
from __future__ import annotations

import functools
import math
import time
from pathlib import Path
from typing import Iterable, List, Optional

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp
import numpy as np

from src.parametric.corrector_v2 import corrector_arctan, corrector_lshape
from src.parametric.positional_density_network_2d import PDN2DParams, cell_midpoints, forward, knots_p2_from_network_axis, knots_p3_from_network_axis
from src.nonparametric.arctan.pde import grad_u_sigma
from src.nonparametric.r_adapt import theta_to_knots_lshape
from src.nonparametric.solver_2d import galerkin_solve_arctan, galerkin_solve_lshape
from src.nonparametric.eta_estimator_2d import (
    eta_squared_arctan,
    eta_squared_lshape,
)
from src.config import ARCTAN, LSHAPE
from common.csv_utils import append_csv_dicts, write_csv_dicts
from common.h1_seminorm_2d import (
    h1_seminorm_sq_2d,
    h1_seminorm_sq_analytic,
)

# Estimator quadrature: pinned to the value used at training time so the
# effectivity index I_eff = eta / |u - u_h|_{H^1} is consistent with the
# residual used by the loss. Do NOT hard-code; read from config.
_Q_EST_P2 = int(ARCTAN["quad_estimator"])
_Q_EST_P3 = int(LSHAPE["quad_estimator"])

Array = jnp.ndarray


# --------------------------------------------------------------------------
# CSV schema (~50 columns)
# --------------------------------------------------------------------------


CSV_COLUMNS = [
    # Experiment/run metadata
    "experiment", "p", "seed", "N", "method",
    # Parameter columns (filled per experiment)
    "alpha", "s1", "s2",                                # ARCTAN
    "sigma1", "sigma2",                                 # LSHAPE
    # Metric columns
    "H1_seminorm_abs_sq", "H1_seminorm_rel", "ritz_energy", "ritz_reference",
    "ritz_rel_error",
    # LSHAPE H1-seminorm error. CURRENT metric: DIRECT sigma-weighted seminorm
    # difference vs. an IGA self-reference (``h1_rel_error_iga_ref`` =
    # ||grad(u_h - u_ref)||_b / ||u_ref||_b, ``h1_seminorm_abs_iga_ref`` =
    # ||grad(u_h - u_ref)||_b). ``h1_rel_error_ngsolve`` is kept as an ALIAS of
    # ``h1_rel_error_iga_ref`` for backward compatibility with existing readers;
    # with an NGSolve scalar-only cache it falls back to the DEPRECATED
    # energy-subtraction metric. ``h1_seminorm_ngsolve_ref`` is the reference
    # sigma-H1 seminorm^2 (||u_ref||_b^2); ``sigma_h1_seminorm_sq_fem`` is the
    # candidate sigma-H1 seminorm^2 (||u_h||_b^2), both Omega-masked.
    "h1_rel_error_ngsolve",
    "h1_rel_error_iga_ref",
    "h1_seminorm_abs_iga_ref",
    "h1_seminorm_ngsolve_ref",
    "sigma_h1_seminorm_sq_fem",
    # Residual a-posteriori estimator (added 2026-05-18 — needed for the
    # effectivity-index analysis). ``eta_squared`` is the per-row estimator
    # eta^2 (same q_est as training); ``eta`` is its sqrt (convenience);
    # ``eta_uniform_squared`` is eta^2 evaluated on the UNIFORM mesh at the
    # same (sigma, N) — identical across the three methods of a single
    # (sigma, N), but stored per row so the CSV is self-contained.
    "eta_squared", "eta", "eta_uniform_squared",
    # Decision 7-F: precomputed per-row effectivity index
    # I_eff = eta / sqrt(H1_seminorm_abs_sq).
    "eff_index",
    # Decision 7-B/7-D: online gate + certification (per (σ, N) row).
    # ``tau`` is the global threshold (calibrated once via
    # calibrate_tau_global). Both raw (positional alone) and corrected
    # rows record certified status; uniform rows pass vacuously.
    "tau", "kappa",
    "residual_gate_pass", "shape_gate_pass",
    "accepted_raw", "accepted_corrected", "certified",
    # Correction diagnostics
    "corrector_n_iter", "corrector_converged", "corrector_energy_init",
    "corrector_energy_final", "corrector_max_iter",
    # Timings
    "t_solve_sec", "t_metric_sec", "t_corrector_sec",
    # Provenance
    "checkpoint_path", "n_train", "n_test", "batch_size",
    "lr_init", "lr_end", "epochs_at_level",
    "quad_K", "quad_F", "quad_metric",
    "anchor_N", "hidden_dims", "sigma_dim",
    # Extras (for compat with the future cluster pipeline). Note:
    # `multistart_K`, `gate_layer`, `gate_AR_max` were placeholder
    # columns from an older schema; the first is removed per the
    # unification (multi-start dropped), the latter two never had a
    # producer and are dropped here too. `T_cap` is the bounded-logit
    # cap T used in the policy (decision 7-C) — informational.
    "split_seed", "h_min", "T_cap",
    "free_dofs", "total_dofs", "constraint_count",
    "norm_denominator", "experiment_version", "elapsed_sec",
]


# --------------------------------------------------------------------------
# Per-sigma evaluation primitives (ARCTAN and LSHAPE)
# --------------------------------------------------------------------------


def _eval_p2_uniform(sigma, *, p, n_elem, q_K, q_F, q_metric):
    from common.h1_seminorm_2d import open_uniform_knots
    knots = open_uniform_knots(n_elem, p)
    t0 = time.perf_counter()
    res = galerkin_solve_arctan(
        jnp.asarray(knots), jnp.asarray(knots), p, n_elem, n_elem,
        jnp.asarray(sigma[0]), jnp.asarray(sigma[1]), jnp.asarray(sigma[2]),
        q_K=q_K, q_F=q_F,
    )
    t1 = time.perf_counter()
    grad = lambda x, y: grad_u_sigma(x, y, sigma[0], sigma[1], sigma[2])
    num = h1_seminorm_sq_2d(np.asarray(res.u_h), knots, knots, p, grad, quad_points_per_dim=q_metric)
    den = h1_seminorm_sq_analytic(grad, quad_points_per_dim=q_metric, n_subdiv_per_dim=16)
    h1_rel = math.sqrt(num / den)
    t2 = time.perf_counter()
    return {
        "knots_x": knots, "knots_y": knots,
        "u_h": np.asarray(res.u_h),
        "ritz": float(res.ritz_energy),
        "H1_abs_sq": float(num),
        "H1_rel": float(h1_rel),
        "norm_denominator": float(den),
        "t_solve": t1 - t0, "t_metric": t2 - t1,
    }


def _eval_p2_network(params: PDN2DParams, sigma, *, p, n_elem, q_K, q_F, q_metric):
    sigma_jx = jnp.asarray(sigma, dtype=DEFAULT_DTYPE)
    t0 = time.perf_counter()
    knots_x = knots_p2_from_network_axis(params, sigma_jx, n_elem, p, axis_id=0.0)
    knots_y = knots_p2_from_network_axis(params, sigma_jx, n_elem, p, axis_id=1.0)
    res = galerkin_solve_arctan(
        knots_x, knots_y, p, n_elem, n_elem,
        sigma_jx[0], sigma_jx[1], sigma_jx[2],
        q_K=q_K, q_F=q_F,
    )
    t1 = time.perf_counter()
    grad = lambda x, y: grad_u_sigma(x, y, sigma[0], sigma[1], sigma[2])
    num = h1_seminorm_sq_2d(np.asarray(res.u_h), np.asarray(knots_x), np.asarray(knots_y), p, grad, quad_points_per_dim=q_metric)
    den = h1_seminorm_sq_analytic(grad, quad_points_per_dim=q_metric, n_subdiv_per_dim=16)
    h1_rel = math.sqrt(num / den)
    t2 = time.perf_counter()
    return {
        "knots_x": np.asarray(knots_x), "knots_y": np.asarray(knots_y),
        "u_h": np.asarray(res.u_h),
        "ritz": float(res.ritz_energy),
        "H1_abs_sq": float(num),
        "H1_rel": float(h1_rel),
        "norm_denominator": float(den),
        "t_solve": t1 - t0, "t_metric": t2 - t1,
    }


def _eval_p3_uniform(sigma, *, p, n_elem, q_K, q_F):
    from src.nonparametric.r_adapt import p3_effective_n_elem
    half = n_elem // 2
    knots = theta_to_knots_lshape(jnp.zeros((half,)), jnp.zeros((half,)), p)
    n_eff = p3_effective_n_elem(int(n_elem), int(p))
    t0 = time.perf_counter()
    res = galerkin_solve_lshape(
        knots, knots, p, n_eff, n_eff,
        jnp.asarray(sigma[0]), jnp.asarray(sigma[1]),
        q_K=q_K, q_F=q_F,
    )
    t1 = time.perf_counter()
    return {
        "knots_x": np.asarray(knots), "knots_y": np.asarray(knots),
        "u_h": np.asarray(res.u_h),
        "ritz": float(res.ritz_energy),
        "t_solve": t1 - t0,
        "n_elem_eff": n_eff,
    }


def _eval_p3_network(params: PDN2DParams, sigma, *, p, n_elem, q_K, q_F):
    from src.nonparametric.r_adapt import p3_effective_n_elem
    sigma_jx = jnp.asarray(sigma, dtype=DEFAULT_DTYPE)
    t0 = time.perf_counter()
    knots_x = knots_p3_from_network_axis(params, sigma_jx, n_elem, p, axis_id=0.0)
    knots_y = knots_p3_from_network_axis(params, sigma_jx, n_elem, p, axis_id=1.0)
    n_eff = p3_effective_n_elem(int(n_elem), int(p))
    res = galerkin_solve_lshape(
        knots_x, knots_y, p, n_eff, n_eff,
        sigma_jx[0], sigma_jx[1],
        q_K=q_K, q_F=q_F,
    )
    t1 = time.perf_counter()
    return {
        "knots_x": np.asarray(knots_x), "knots_y": np.asarray(knots_y),
        "u_h": np.asarray(res.u_h),
        "ritz": float(res.ritz_energy),
        "t_solve": t1 - t0,
        "n_elem_eff": n_eff,
    }


# --------------------------------------------------------------------------
# σ-weighted H¹ seminorm on the FEM mesh — used for LSHAPE H¹ relative error.
#
# DEPRECATED (kept for the NGSolve scalar-cache fallback only). This computes
# ``int_Omega sigma|grad u_h|^2`` for the candidate; the OLD metric subtracted
# it from a cached reference seminorm (Galerkin identity). That identity FAILS
# for the non-conforming masked-IGA LSHAPE solver — the masked seminorm overshoots
# and is non-monotone under refinement, so the subtraction clips to 0. The
# CURRENT metric is the direct difference ``_direct_h1_error_lshape`` below.
# --------------------------------------------------------------------------


def _sigma_h1_seminorm_sq_fem_p3(
    u_coeffs: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    sigma1: float,
    sigma2: float,
    *,
    q: int = 8,
) -> float:
    """Compute ``int_{L-shape} sigma(x) |grad u_h|^2 dx`` on our tensor-product
    B-spline mesh.

    The integrand is piecewise-polynomial of degree 2(p-1) (per element)
    times a piecewise-constant ``sigma``; GL ``q^2`` per element with ``q
    >= p`` is exact. We default to ``q = 8`` for safety.

    Elements whose centroid lies in the closed removed quadrant
    ``{x>=0.5, y<=0.5}`` are excluded (the FE DOFs there are zero by
    masking; including them would still give zero but we skip for clarity).
    """
    from common.bspline_basis import basis_batch_for_degree
    from common.quadrature import rule_gl_on_elements
    from src.nonparametric.lshape.pde import sigma_field
    from src.nonparametric.solver_2d import _element_data

    knots_x_j = jnp.asarray(knots_x, dtype=DEFAULT_DTYPE)
    knots_y_j = jnp.asarray(knots_y, dtype=DEFAULT_DTYPE)
    n_elem_x = int(knots_x_j.shape[0] - 2 * p - 1)
    n_elem_y = int(knots_y_j.shape[0] - 2 * p - 1)
    a_x, b_x, U_local_x, idx_local_x = _element_data(knots_x_j, int(p), n_elem_x)
    a_y, b_y, U_local_y, idx_local_y = _element_data(knots_y_j, int(p), n_elem_y)

    xq, wq_x = rule_gl_on_elements(a_x, b_x, int(q))
    yq, wq_y = rule_gl_on_elements(a_y, b_y, int(q))
    Nx, dNx, _ = basis_batch_for_degree(int(p), xq, U_local_x)
    Ny, dNy, _ = basis_batch_for_degree(int(p), yq, U_local_y)

    C = jnp.asarray(u_coeffs, dtype=DEFAULT_DTYPE)            # (n_x, n_y)
    Cx = C[idx_local_x[:, :, None, None], idx_local_y[None, None, :, :]]
    C_loc = Cx.transpose(0, 2, 1, 3)                          # (n_ex, n_ey, p+1, p+1)

    # partial_x u_h and partial_y u_h on the 4D tensor-product GL grid.
    ux = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, dNx, Ny)
    uy = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, Nx, dNy)
    grad_sq = ux * ux + uy * uy

    XX = jnp.broadcast_to(xq[:, None, :, None], grad_sq.shape)
    YY = jnp.broadcast_to(yq[None, :, None, :], grad_sq.shape)
    sigma_at = sigma_field(XX, YY, jnp.asarray(sigma1, dtype=DEFAULT_DTYPE),
                                  jnp.asarray(sigma2, dtype=DEFAULT_DTYPE))
    integrand = sigma_at * grad_sq

    # Weights and element-centroid mask (drop closed removed quadrant).
    w_2d = wq_x[:, None, :, None] * wq_y[None, :, None, :]
    per_elem = jnp.sum(integrand * w_2d, axis=(2, 3))         # (n_ex, n_ey)
    mid_x = 0.5 * (a_x + b_x)
    mid_y = 0.5 * (a_y + b_y)
    in_removed_x = jnp.heaviside(mid_x - 0.5, 1.0)
    in_removed_y = jnp.heaviside(0.5 - mid_y, 1.0)
    keep = 1.0 - in_removed_x[:, None] * in_removed_y[None, :]
    return float(jnp.sum(keep * per_elem))


def _h1_rel_ngsolve_p3(
    u_coeffs: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    sigma1: float,
    sigma2: float,
    ngsolve_cache: Optional[dict],
) -> tuple[float, float, float]:
    """DEPRECATED energy-subtraction metric (kept for the NGSolve scalar cache).

    Superseded by ``_direct_h1_error_lshape`` (dispatched via ``_h1_rel_p3``). Use
    this ONLY when the reference cache stores scalars without solution fields
    (no ``u_coeffs``), i.e. a legacy NGSolve cache. For the masked-IGA solver
    the underlying Galerkin identity is invalid (see header note on the
    non-monotone masked seminorm) and the numerator clips to 0.

    Returns ``(h1_rel_error_ngsolve, sigma_h1_seminorm_ngsolve_ref,
    sigma_h1_seminorm_sq_fem)``. Each can be ``float('nan')`` if the
    cache does not contain the (sigma1, sigma2) entry.

    Identity used (Galerkin projection in the energy inner product):
        ``||u - u_h||_b^2 = ||u||_b^2 - ||u_h||_b^2``
    where ``||v||_b^2 := int sigma |grad v|^2``. ``||u||_b^2`` is read
    from the NGSolve cache; ``||u_h||_b^2`` is computed on our FE mesh.
    """
    if ngsolve_cache is None:
        return float("nan"), float("nan"), float("nan")
    key = (round(float(sigma1), 12), round(float(sigma2), 12))
    if key not in ngsolve_cache:
        # Tolerant match — find the closest cached σ within tight tolerance.
        for k in ngsolve_cache:
            if abs(k[0] - float(sigma1)) < 1e-9 and abs(k[1] - float(sigma2)) < 1e-9:
                key = k
                break
        else:
            return float("nan"), float("nan"), float("nan")
    ref = ngsolve_cache[key]
    ref_norm_sq = float(ref["sigma_h1_seminorm_sq"])
    our_norm_sq = _sigma_h1_seminorm_sq_fem_p3(
        u_coeffs, knots_x, knots_y, p, sigma1, sigma2
    )
    err_sq = ref_norm_sq - our_norm_sq
    err_sq = max(err_sq, 0.0)
    rel_err = math.sqrt(err_sq / ref_norm_sq) if ref_norm_sq > 0 else float("nan")
    return rel_err, ref_norm_sq, our_norm_sq


# --------------------------------------------------------------------------
# DIRECT σ-weighted H¹ seminorm DIFFERENCE metric for LSHAPE (CURRENT).
#
# Computes  err² = ∫_Ω σ |∇u_h − ∇u_ref|²  (≥ 0 by construction) against a
# stored high-resolution IGA self-reference solution (knots + u_coeffs in the
# cache entry). Both B-spline solutions are sampled on ONE fixed,
# candidate-independent Gauss–Legendre grid; the reference is sampled once per
# σ. This replaces the energy-subtraction identity above, which is invalid for
# the non-conforming masked-IGA LSHAPE solver (non-monotone masked seminorm).
# --------------------------------------------------------------------------

# Fixed quadrature grid: M cells/axis (EVEN ⇒ the material interface / re-entrant
# corner x=y=0.5 and the removed-quadrant boundary lines x=0.5, y=0.5 are all
# CELL BOUNDARIES, so no cell straddles a σ-jump / C⁰ kink and every GL point is
# unambiguously inside or outside Ω) × Q Gauss points/cell.
_P3_DIRECT_M = 200          # cells per axis
_P3_DIRECT_Q = 4            # GL points per cell  (⇒ 800 sample points/axis)

# Bounded cache of dense (B, dB) basis matrices on the fixed grid, keyed by
# (knots bytes, p, n_grid): the reference knots are identical for every σ (1
# key, reused ~N_σ×3×4 times) and the uniform-candidate knots repeat across σ
# at fixed N (a few keys). Positional/corrected knots are σ-unique and churn,
# so we cap the cache and evict FIFO.
_P3_BASIS_CACHE: dict = {}
_P3_BASIS_CACHE_MAX = 32


@functools.lru_cache(maxsize=4)
def _p3_direct_grid(m: int, q: int):
    """Fixed tensor GL grid on [0, 1]²: 1D nodes/weights + Ω keep-mask.

    Returns ``(xq, wq, keep)`` with ``xq, wq`` shape ``(m*q,)`` and ``keep`` a
    ``(m*q, m*q)`` float array (1.0 inside the L-shape, 0.0 in the removed
    quadrant ``{x>0.5, y<0.5}``). Both axes share ``xq``.
    """
    gl_x, gl_w = np.polynomial.legendre.leggauss(int(q))     # nodes/weights on [-1, 1]
    edges = np.linspace(0.0, 1.0, int(m) + 1)
    a = edges[:-1]
    b = edges[1:]
    half = 0.5 * (b - a)                                     # (m,)
    mid = 0.5 * (a + b)
    xq = (mid[:, None] + half[:, None] * gl_x[None, :]).reshape(-1)   # (m*q,)
    wq = (half[:, None] * gl_w[None, :]).reshape(-1)                  # (m*q,)
    removed = (xq[:, None] > 0.5) & (xq[None, :] < 0.5)              # (x>0.5 & y<0.5)
    keep = (~removed).astype(np.float64)
    return xq.astype(np.float64), wq.astype(np.float64), keep


def _p3_basis_on_grid(knots: np.ndarray, p: int, xq: np.ndarray):
    """Dense ``(B, dB)`` of shape ``(n_q, n_basis)``: every B-spline basis
    function and its first derivative evaluated at the grid nodes ``xq``.

    Scatters the ``(p+1)`` locally-active values returned by
    ``bspline_basis_local`` into the global columns. Cached by knot bytes.
    """
    knots = np.asarray(knots, dtype=np.float64)
    key = (knots.tobytes(), int(p), int(xq.shape[0]))
    hit = _P3_BASIS_CACHE.get(key)
    if hit is not None:
        return hit
    from common.bspline_basis import bspline_basis_local
    N_loc, dN_loc, _d2, spans = bspline_basis_local(
        jnp.asarray(xq, dtype=DEFAULT_DTYPE), jnp.asarray(knots, dtype=DEFAULT_DTYPE), int(p)
    )
    N_loc = np.asarray(N_loc, dtype=np.float64)
    dN_loc = np.asarray(dN_loc, dtype=np.float64)
    spans = np.asarray(spans)
    n_q = int(xq.shape[0])
    n_basis = int(knots.shape[0] - p - 1)
    cols = spans[:, None] - int(p) + np.arange(int(p) + 1)[None, :]   # (n_q, p+1) distinct/row
    rows = np.arange(n_q)[:, None]
    B = np.zeros((n_q, n_basis), dtype=np.float64)
    dB = np.zeros((n_q, n_basis), dtype=np.float64)
    B[rows, cols] = N_loc
    dB[rows, cols] = dN_loc
    if len(_P3_BASIS_CACHE) >= _P3_BASIS_CACHE_MAX:
        _P3_BASIS_CACHE.pop(next(iter(_P3_BASIS_CACHE)))             # FIFO evict
    _P3_BASIS_CACHE[key] = (B, dB)
    return B, dB


def _grad_on_grid(u_coeffs: np.ndarray, Bx, dBx, By, dBy):
    """Tensor-product B-spline gradient on the grid.

    ``u_h(x,y) = Σ_ij C[i,j] Bx_i(x) By_j(y)`` ⇒
    ``∂x u = dBx · C · Byᵀ``, ``∂y u = Bx · C · dByᵀ`` (both ``(n_qx, n_qy)``).
    """
    C = np.asarray(u_coeffs, dtype=np.float64)                  # (n_x, n_y)
    ux = dBx @ C @ By.T
    uy = Bx @ C @ dBy.T
    return ux, uy


def _direct_h1_error_lshape(
    u_coeffs: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    sigma1: float,
    sigma2: float,
    ref_entry: dict,
) -> tuple[float, float, float, float]:
    """DIRECT σ-weighted H¹ seminorm error vs. a stored IGA self-reference.

    Returns ``(h1_rel_error, h1_seminorm_abs, ref_norm_sq, cand_norm_sq)`` where

        err²        = ∫_Ω σ |∇u_h − ∇u_ref|²
        ref_norm_sq = ∫_Ω σ |∇u_ref|²          (denominator, ||u_ref||_b²)
        cand_norm_sq= ∫_Ω σ |∇u_h|²            (candidate ||u_h||_b²)
        h1_rel      = sqrt(err² / ref_norm_sq)
        h1_abs      = sqrt(err²)

    all integrated on the fixed grid with the Ω keep-mask, so err² ≥ 0 always.
    """
    from src.nonparametric.lshape.pde import sigma_field

    xq, wq, keep = _p3_direct_grid(int(_P3_DIRECT_M), int(_P3_DIRECT_Q))

    # σ(x,y)·keep·w_x·w_y on the full 2D grid (σ_field broadcasts over (n,1),(1,n)).
    sig = np.asarray(
        sigma_field(jnp.asarray(xq)[:, None], jnp.asarray(xq)[None, :],
                    float(sigma1), float(sigma2)),
        dtype=np.float64,
    )
    w2d = (wq[:, None] * wq[None, :]) * keep * sig              # (n_q, n_q)

    # Reference gradient (knots identical for every σ ⇒ basis matrices cached).
    p_ref = int(ref_entry.get("p", p))
    Bxr, dBxr = _p3_basis_on_grid(ref_entry["knots_x"], p_ref, xq)
    Byr, dByr = _p3_basis_on_grid(ref_entry["knots_y"], p_ref, xq)
    uxr, uyr = _grad_on_grid(ref_entry["u_coeffs"], Bxr, dBxr, Byr, dByr)

    # Candidate gradient on the SAME grid.
    Bxc, dBxc = _p3_basis_on_grid(knots_x, int(p), xq)
    Byc, dByc = _p3_basis_on_grid(knots_y, int(p), xq)
    uxc, uyc = _grad_on_grid(u_coeffs, Bxc, dBxc, Byc, dByc)

    dux = uxc - uxr
    duy = uyc - uyr
    err_sq = float(np.sum(w2d * (dux * dux + duy * duy)))
    ref_norm_sq = float(np.sum(w2d * (uxr * uxr + uyr * uyr)))
    cand_norm_sq = float(np.sum(w2d * (uxc * uxc + uyc * uyc)))
    err_sq = max(err_sq, 0.0)
    rel = math.sqrt(err_sq / ref_norm_sq) if ref_norm_sq > 0.0 else float("nan")
    return rel, math.sqrt(err_sq), ref_norm_sq, cand_norm_sq


# --------------------------------------------------------------------------
# ADDITIVE (option a1): DIRECT metric against an NGSolve-AMR reference whose
# gradient is PRECOMPUTED on the same fixed xq grid (grad_uxr/grad_uyr) by
# generate_reference_lshape_amr.py. Identical to ``_direct_h1_error_lshape`` except the
# reference gradient is READ (not recomputed from B-spline coeffs); the candidate
# side, grid, keep-mask, σ-weighting, and err² formula are byte-identical, so the
# AMR result is directly comparable to the IGA result. The IGA path above is
# untouched. Dispatched by the new ``grad_uxr`` branch in ``_h1_rel_p3``.
# --------------------------------------------------------------------------
def _direct_h1_error_p3_amr(
    u_coeffs: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    sigma1: float,
    sigma2: float,
    ref_entry: dict,
) -> tuple[float, float, float, float]:
    """σ-weighted H¹ seminorm error vs. a stored AMR reference gradient.

    Returns ``(h1_rel_error, h1_seminorm_abs, ref_norm_sq, cand_norm_sq)`` with
    the SAME definitions as ``_direct_h1_error_lshape`` (only the reference gradient
    source differs)."""
    from src.nonparametric.lshape.pde import sigma_field

    xq, wq, keep = _p3_direct_grid(int(_P3_DIRECT_M), int(_P3_DIRECT_Q))

    sig = np.asarray(
        sigma_field(jnp.asarray(xq)[:, None], jnp.asarray(xq)[None, :],
                    float(sigma1), float(sigma2)),
        dtype=np.float64,
    )
    w2d = (wq[:, None] * wq[None, :]) * keep * sig              # (n_q, n_q)

    # Reference gradient: precomputed on the SAME fixed xq grid (AMR entry).
    uxr = np.asarray(ref_entry["grad_uxr"], dtype=np.float64)
    uyr = np.asarray(ref_entry["grad_uyr"], dtype=np.float64)

    # Candidate gradient on the SAME grid (identical to the IGA path).
    Bxc, dBxc = _p3_basis_on_grid(knots_x, int(p), xq)
    Byc, dByc = _p3_basis_on_grid(knots_y, int(p), xq)
    uxc, uyc = _grad_on_grid(u_coeffs, Bxc, dBxc, Byc, dByc)

    dux = uxc - uxr
    duy = uyc - uyr
    err_sq = float(np.sum(w2d * (dux * dux + duy * duy)))
    ref_norm_sq = float(np.sum(w2d * (uxr * uxr + uyr * uyr)))
    cand_norm_sq = float(np.sum(w2d * (uxc * uxc + uyc * uyc)))
    err_sq = max(err_sq, 0.0)
    rel = math.sqrt(err_sq / ref_norm_sq) if ref_norm_sq > 0.0 else float("nan")
    return rel, math.sqrt(err_sq), ref_norm_sq, cand_norm_sq


def _lookup_ref_entry(ref_cache: Optional[dict], sigma1: float, sigma2: float):
    """(σ1, σ2) lookup in a reference cache, tolerant to float rounding."""
    if ref_cache is None:
        return None
    key = (round(float(sigma1), 12), round(float(sigma2), 12))
    if key in ref_cache:
        return ref_cache[key]
    for k in ref_cache:
        if abs(k[0] - float(sigma1)) < 1e-9 and abs(k[1] - float(sigma2)) < 1e-9:
            return ref_cache[k]
    return None


def _h1_rel_p3(
    u_coeffs: np.ndarray,
    knots_x: np.ndarray,
    knots_y: np.ndarray,
    p: int,
    sigma1: float,
    sigma2: float,
    ref_cache: Optional[dict],
) -> tuple[float, float, float, float]:
    """Dispatcher for the LSHAPE H¹-seminorm error.

    * IGA self-reference (cache entry has ``u_coeffs``) → DIRECT difference
      metric ``_direct_h1_error_lshape`` (current).
    * Legacy NGSolve scalar cache (no ``u_coeffs``) → DEPRECATED
      energy-subtraction ``_h1_rel_ngsolve_p3``.

    Returns ``(h1_rel_error, h1_seminorm_abs, ref_norm_sq, cand_norm_sq)``.
    """
    entry = _lookup_ref_entry(ref_cache, sigma1, sigma2)
    if entry is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if "u_coeffs" in entry and "knots_x" in entry and "knots_y" in entry:
        return _direct_h1_error_lshape(
            u_coeffs, knots_x, knots_y, p, sigma1, sigma2, entry
        )
    # --- ADDITIVE: NGSolve-AMR reference (precomputed grad on the fixed grid) ---
    if "grad_uxr" in entry and "grad_uyr" in entry:
        return _direct_h1_error_p3_amr(
            u_coeffs, knots_x, knots_y, p, sigma1, sigma2, entry
        )
    # --- deprecated fallback: NGSolve scalar cache (Galerkin subtraction) ---
    rel, ref_norm_sq, cand_norm_sq = _h1_rel_ngsolve_p3(
        u_coeffs, knots_x, knots_y, p, sigma1, sigma2, ref_cache
    )
    if math.isfinite(ref_norm_sq) and math.isfinite(cand_norm_sq):
        h1_abs = math.sqrt(max(ref_norm_sq - cand_norm_sq, 0.0))
    else:
        h1_abs = float("nan")
    return rel, h1_abs, ref_norm_sq, cand_norm_sq


# --------------------------------------------------------------------------
# Top-level evaluators (ARCTAN and LSHAPE): produce one CSV row per (sigma, N, method).
# --------------------------------------------------------------------------


def _build_row(base: dict, **overrides) -> dict:
    out = {k: "" for k in CSV_COLUMNS}
    out.update(base)
    out.update(overrides)
    return out


# --------------------------------------------------------------------------
# Decision 7-D: per-axis box shape gate.  AR_axis = max(h_axis) / min(h_axis);
# axis-wise to avoid penalising x-only or y-only concentration.
# --------------------------------------------------------------------------


def _per_axis_max_ratio(knots: np.ndarray, p: int) -> float:
    """Max element-size ratio along one axis (drop zero-length elements
    introduced by the multiplicity-p knot at 0.5 in LSHAPE)."""
    knots = np.asarray(knots, dtype=np.float64)
    # element sizes = consecutive differences of unique knots
    sizes = np.diff(knots)
    # drop zero-length elements (multiplicity-p mult knot in LSHAPE produces them)
    sizes = sizes[sizes > 1e-15]
    if sizes.size == 0:
        return float("inf")
    return float(np.max(sizes) / np.min(sizes))


def _shape_gate_pass(knots_x: np.ndarray, knots_y: np.ndarray, p: int,
                      ar_max: float) -> bool:
    """Decision 7-D: per-axis box; pass iff every axis's max h-ratio ≤ ar_max."""
    ratio_x = _per_axis_max_ratio(knots_x, p)
    ratio_y = _per_axis_max_ratio(knots_y, p)
    return bool(ratio_x <= ar_max and ratio_y <= ar_max)


def _eff_index(eta: float, h1_abs_sq: float) -> float:
    """Decision 7-F: I_eff = eta / sqrt(H1_seminorm_abs_sq); NaN if H1 is
    non-positive or non-finite."""
    if not (math.isfinite(h1_abs_sq) and h1_abs_sq > 0.0):
        return float("nan")
    if not math.isfinite(eta):
        return float("nan")
    return float(eta / math.sqrt(h1_abs_sq))


# --------------------------------------------------------------------------
# Decision 7-B: τ = κ · median r_η  over a held-out validation subset.
# Calibration helper for both ARCTAN and LSHAPE.
# --------------------------------------------------------------------------


def calibrate_tau_global_arctan(
    params: PDN2DParams,
    calibration_sigmas: np.ndarray,
    *,
    p: int,
    N: int,
    q_K: int = None,
    q_F: int = None,
) -> float:
    """Decision 7-B: calibrate the global residual-gate threshold τ for
    Experiment ARCTAN. Computes ``r_η(σ) = η(θ_φ(σ); σ) / η(θ_unif; σ)`` on
    the held-out ``calibration_sigmas`` at refinement level ``N``, then
    returns ``κ · median(r_η)`` with ``κ = CORRECTOR["gate_kappa"]``."""
    from src.config import CORRECTOR
    if q_K is None:
        q_K = int(p) + 1
    if q_F is None:
        q_F = int(ARCTAN.get("quad_forcing", 50))
    kappa = float(CORRECTOR["gate_kappa"])
    ratios: List[float] = []
    for sigma in np.asarray(calibration_sigmas, dtype=np.float64):
        u = _eval_p2_uniform(sigma, p=int(p), n_elem=int(N),
                              q_K=int(q_K), q_F=int(q_F), q_metric=int(q_F))
        pos = _eval_p2_network(params, sigma, p=int(p), n_elem=int(N),
                                q_K=int(q_K), q_F=int(q_F), q_metric=int(q_F))
        sigma_a = jnp.asarray(sigma[0], dtype=DEFAULT_DTYPE)
        sigma_s1 = jnp.asarray(sigma[1], dtype=DEFAULT_DTYPE)
        sigma_s2 = jnp.asarray(sigma[2], dtype=DEFAULT_DTYPE)
        eta_unif = math.sqrt(max(0.0, float(eta_squared_arctan(
            jnp.asarray(u["u_h"]), jnp.asarray(u["knots_x"]), jnp.asarray(u["knots_y"]),
            int(p), int(N), int(N), sigma_a, sigma_s1, sigma_s2, q_est=_Q_EST_P2,
        ))))
        eta_pos = math.sqrt(max(0.0, float(eta_squared_arctan(
            jnp.asarray(pos["u_h"]), jnp.asarray(pos["knots_x"]), jnp.asarray(pos["knots_y"]),
            int(p), int(N), int(N), sigma_a, sigma_s1, sigma_s2, q_est=_Q_EST_P2,
        ))))
        if eta_unif > 0.0:
            ratios.append(eta_pos / eta_unif)
    if not ratios:
        return float("nan")
    return float(kappa * float(np.median(np.asarray(ratios))))


def calibrate_tau_global_lshape(
    params: PDN2DParams,
    calibration_sigmas: np.ndarray,
    *,
    p: int,
    N: int,
    q_K: int = None,
    q_F: int = None,
) -> float:
    """Decision 7-B: calibrate the global residual-gate threshold τ for
    Experiment LSHAPE (L-shape). Same logic as ARCTAN but uses ``eta_squared_lshape``
    on the effective LSHAPE mesh ``n_eff = n_elem + (p - 1)``."""
    from src.config import CORRECTOR
    if q_K is None:
        q_K = int(p) + 1
    if q_F is None:
        q_F = int(LSHAPE.get("quad_forcing", 6))
    kappa = float(CORRECTOR["gate_kappa"])
    ratios: List[float] = []
    for sigma in np.asarray(calibration_sigmas, dtype=np.float64):
        u = _eval_p3_uniform(sigma, p=int(p), n_elem=int(N),
                              q_K=int(q_K), q_F=int(q_F))
        pos = _eval_p3_network(params, sigma, p=int(p), n_elem=int(N),
                                q_K=int(q_K), q_F=int(q_F))
        n_eff = int(u["n_elem_eff"])
        sigma1 = jnp.asarray(sigma[0], dtype=DEFAULT_DTYPE)
        sigma2 = jnp.asarray(sigma[1], dtype=DEFAULT_DTYPE)
        eta_unif = math.sqrt(max(0.0, float(eta_squared_lshape(
            jnp.asarray(u["u_h"]), jnp.asarray(u["knots_x"]), jnp.asarray(u["knots_y"]),
            int(p), n_eff, n_eff, sigma1, sigma2, q_est=_Q_EST_P3,
        ))))
        eta_pos = math.sqrt(max(0.0, float(eta_squared_lshape(
            jnp.asarray(pos["u_h"]), jnp.asarray(pos["knots_x"]), jnp.asarray(pos["knots_y"]),
            int(p), int(pos["n_elem_eff"]), int(pos["n_elem_eff"]),
            sigma1, sigma2, q_est=_Q_EST_P3,
        ))))
        if eta_unif > 0.0:
            ratios.append(eta_pos / eta_unif)
    if not ratios:
        return float("nan")
    return float(kappa * float(np.median(np.asarray(ratios))))


def evaluate_arctan(
    params: PDN2DParams,
    test_sigmas: np.ndarray,
    *,
    p: int,
    levels: Iterable[int],
    seed: int,
    output_dir: Path,
    q_K: int = 4,
    q_F: int = 50,
    q_metric: int = 40,
    corrector_max_iter: int = 30,
    checkpoint_path: str = "",
    train_meta: Optional[dict] = None,
    tau_global: Optional[float] = None,
) -> Path:
    """Evaluate uniform / positional / positional_corrected at each (sigma, N).

    Returns the path to the written CSV (one file per seed).
    """
    from src.config import CORRECTOR as _CORR_CFG, T_ARCTAN as _T_P2_CFG
    kappa_const = float(_CORR_CFG["gate_kappa"])
    ar_max = float(_CORR_CFG["shape_gate_AR_max"])
    tau_const = float(tau_global) if tau_global is not None else float("nan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"summary_p2_seed{seed}.csv"
    write_csv_dicts(out_csv, [], CSV_COLUMNS)   # initialise header
    n_rows_total = 0

    for N in levels:
        print(f"\n=== eval ARCTAN N={N} ===")
        rows: List[dict] = []                     # per-level batch (flushed after this N)
        for sigma in test_sigmas:
            base = {
                "experiment": "arctan",
                "p": int(p),
                "seed": int(seed),
                "N": int(N),
                "alpha": float(sigma[0]),
                "s1": float(sigma[1]),
                "s2": float(sigma[2]),
                "anchor_N": max(levels),
                "sigma_dim": 3,
                "quad_K": int(q_K),
                "quad_F": int(q_F),
                "quad_metric": int(q_metric),
                "experiment_version": "v2",
                "checkpoint_path": str(checkpoint_path),
                "T_cap": float(_T_P2_CFG),
            }
            if train_meta is not None:
                base.update({
                    "n_train": int(train_meta.get("n_train", 0)),
                    "n_test": int(train_meta.get("n_test", 0)),
                    "batch_size": int(train_meta.get("batch_size", 0)),
                    "lr_init": float(train_meta.get("lr_init", 0.0)),
                    "lr_end": float(train_meta.get("lr_end", 0.0)),
                    "hidden_dims": str(train_meta.get("hidden_dims", "")),
                    "epochs_at_level": int(train_meta.get("epochs_at_level", {}).get(N, 0))
                                          if isinstance(train_meta.get("epochs_at_level", {}), dict) else 0,
                })

            # Parameter tuple used by both the solver and the estimator.
            sigma_a_j = jnp.asarray(sigma[0], dtype=DEFAULT_DTYPE)
            sigma_s1_j = jnp.asarray(sigma[1], dtype=DEFAULT_DTYPE)
            sigma_s2_j = jnp.asarray(sigma[2], dtype=DEFAULT_DTYPE)

            # ------- Compute all three method results up front so we can
            # back-fill accepted_raw / accepted_corrected consistently
            # across the three rows. ----------------------------------

            # --- uniform ---
            t0_u = time.perf_counter()
            u = _eval_p2_uniform(
                sigma, p=p, n_elem=int(N), q_K=q_K, q_F=q_F, q_metric=q_metric
            )
            # Residual estimator on the uniform-mesh solve. Reused below as
            # `eta_uniform_squared` for every method of this (sigma, N).
            eta_sq_unif = float(eta_squared_arctan(
                jnp.asarray(u["u_h"]), jnp.asarray(u["knots_x"]), jnp.asarray(u["knots_y"]),
                int(p), int(N), int(N),
                sigma_a_j, sigma_s1_j, sigma_s2_j, q_est=_Q_EST_P2,
            ))
            eta_unif = math.sqrt(max(eta_sq_unif, 0.0))
            elapsed_u = time.perf_counter() - t0_u

            # --- positional (no corrector) ---
            t0_p = time.perf_counter()
            pos = _eval_p2_network(
                params, sigma, p=p, n_elem=int(N), q_K=q_K, q_F=q_F, q_metric=q_metric
            )
            eta_sq_pos = float(eta_squared_arctan(
                jnp.asarray(pos["u_h"]), jnp.asarray(pos["knots_x"]), jnp.asarray(pos["knots_y"]),
                int(p), int(N), int(N),
                sigma_a_j, sigma_s1_j, sigma_s2_j, q_est=_Q_EST_P2,
            ))
            eta_pos = math.sqrt(max(eta_sq_pos, 0.0))
            elapsed_p = time.perf_counter() - t0_p

            # --- positional_corrected (skipped when corrector_max_iter <= 0) ---
            run_corrector = int(corrector_max_iter) > 0
            if run_corrector:
                t0_c = time.perf_counter()
                sigma_jx = jnp.asarray(sigma, dtype=DEFAULT_DTYPE)
                xi = cell_midpoints(int(N))
                zx = forward(params, sigma_jx, xi, axis_id=0.0)
                zy = forward(params, sigma_jx, xi, axis_id=1.0)
                result, knots_x_opt, knots_y_opt = corrector_arctan(
                    zx, zy, sigma_jx,
                    p=p, n_elem=int(N), q_K=q_K, q_F=q_F,
                    max_iter=int(corrector_max_iter),
                )
                res_opt = galerkin_solve_arctan(
                    knots_x_opt, knots_y_opt, p, int(N), int(N),
                    sigma_jx[0], sigma_jx[1], sigma_jx[2],
                    q_K=q_K, q_F=q_F,
                )
                grad = lambda x, y: grad_u_sigma(x, y, sigma[0], sigma[1], sigma[2])
                num = h1_seminorm_sq_2d(
                    np.asarray(res_opt.u_h), np.asarray(knots_x_opt), np.asarray(knots_y_opt), p,
                    grad, quad_points_per_dim=q_metric,
                )
                h1_rel_c = math.sqrt(num / pos["norm_denominator"])
                eta_sq_corr = float(eta_squared_arctan(
                    res_opt.u_h, knots_x_opt, knots_y_opt,
                    int(p), int(N), int(N),
                    sigma_a_j, sigma_s1_j, sigma_s2_j, q_est=_Q_EST_P2,
                ))
                eta_corr = math.sqrt(max(eta_sq_corr, 0.0))
                elapsed_c = time.perf_counter() - t0_c
            else:
                # Corrector excluded (corrector_max_iter <= 0): safe dummies so the
                # uniform/positional rows still build. No corrected row is emitted.
                result = None
                knots_x_opt = pos["knots_x"]
                knots_y_opt = pos["knots_y"]
                res_opt = None
                num = float("nan")
                h1_rel_c = float("nan")
                eta_sq_corr = float("nan")
                eta_corr = float("nan")
                elapsed_c = 0.0

            # ------- Decisions 7-B / 7-D / 7-F per-row diagnostics ----

            # uniform is the baseline: shape AR=1 (uniform mesh), residual
            # gate is vacuous (it is the reference against which everything
            # else is judged). Certified=True so it never counts as a
            # failure case in downstream analysis.
            unif_residual_pass = True
            unif_shape_pass = True
            unif_certified = True

            # positional: actual gate evaluation on the positional mesh.
            pos_kx_np = np.asarray(pos["knots_x"])
            pos_ky_np = np.asarray(pos["knots_y"])
            pos_residual_pass = (
                bool(eta_pos <= tau_const)
                if math.isfinite(tau_const) else True
            )
            pos_shape_pass = _shape_gate_pass(pos_kx_np, pos_ky_np, int(p), ar_max)
            pos_certified = bool(pos_residual_pass and pos_shape_pass)

            # positional_corrected: same on the corrected mesh.
            if run_corrector:
                corr_kx_np = np.asarray(knots_x_opt)
                corr_ky_np = np.asarray(knots_y_opt)
                corr_residual_pass = (
                    bool(eta_corr <= tau_const)
                    if math.isfinite(tau_const) else True
                )
                corr_shape_pass = _shape_gate_pass(corr_kx_np, corr_ky_np, int(p), ar_max)
                corr_certified = bool(corr_residual_pass and corr_shape_pass)
            else:
                corr_residual_pass = False
                corr_shape_pass = False
                corr_certified = False

            # Cross-row fields back-filled the same on every row of this (σ,N).
            accepted_raw = bool(pos_certified)
            accepted_corrected = bool(corr_certified)

            # ------- Emit three rows ----------------------------------

            rows.append(_build_row(base, method="uniform",
                H1_seminorm_abs_sq=u["H1_abs_sq"],
                H1_seminorm_rel=u["H1_rel"],
                ritz_energy=u["ritz"],
                eta_squared=eta_sq_unif,
                eta=eta_unif,
                eta_uniform_squared=eta_sq_unif,
                eff_index=_eff_index(eta_unif, u["H1_abs_sq"]),
                tau=tau_const if math.isfinite(tau_const) else "",
                kappa=kappa_const,
                residual_gate_pass=int(unif_residual_pass),
                shape_gate_pass=int(unif_shape_pass),
                accepted_raw=int(accepted_raw),
                accepted_corrected=int(accepted_corrected),
                certified=int(unif_certified),
                t_solve_sec=u["t_solve"], t_metric_sec=u["t_metric"],
                norm_denominator=u["norm_denominator"],
                elapsed_sec=elapsed_u,
            ))

            rows.append(_build_row(base, method="positional",
                H1_seminorm_abs_sq=pos["H1_abs_sq"],
                H1_seminorm_rel=pos["H1_rel"],
                ritz_energy=pos["ritz"],
                eta_squared=eta_sq_pos,
                eta=eta_pos,
                eta_uniform_squared=eta_sq_unif,
                eff_index=_eff_index(eta_pos, pos["H1_abs_sq"]),
                tau=tau_const if math.isfinite(tau_const) else "",
                kappa=kappa_const,
                residual_gate_pass=int(pos_residual_pass),
                shape_gate_pass=int(pos_shape_pass),
                accepted_raw=int(accepted_raw),
                accepted_corrected=int(accepted_corrected),
                certified=int(pos_certified),
                t_solve_sec=pos["t_solve"], t_metric_sec=pos["t_metric"],
                norm_denominator=pos["norm_denominator"],
                elapsed_sec=elapsed_p,
            ))

            if run_corrector:
                rows.append(_build_row(base, method="positional_corrected",
                    H1_seminorm_abs_sq=float(num),
                    H1_seminorm_rel=float(h1_rel_c),
                    ritz_energy=float(res_opt.ritz_energy),
                    eta_squared=eta_sq_corr,
                    eta=eta_corr,
                    eta_uniform_squared=eta_sq_unif,
                    eff_index=_eff_index(eta_corr, float(num)),
                    tau=tau_const if math.isfinite(tau_const) else "",
                    kappa=kappa_const,
                    residual_gate_pass=int(corr_residual_pass),
                    shape_gate_pass=int(corr_shape_pass),
                    accepted_raw=int(accepted_raw),
                    accepted_corrected=int(accepted_corrected),
                    certified=int(corr_certified),
                    corrector_n_iter=int(result.n_iter),
                    corrector_converged=int(result.converged),
                    corrector_energy_init=float(pos["ritz"]),
                    corrector_energy_final=float(result.energy_final),
                    corrector_max_iter=int(corrector_max_iter),
                    t_corrector_sec=elapsed_c,
                    norm_denominator=pos["norm_denominator"],
                    elapsed_sec=elapsed_c,
                ))

        # Per-level flush: persist this N's rows immediately so a later
        # timeout / kill leaves the earlier N values intact on disk.
        append_csv_dicts(out_csv, rows, CSV_COLUMNS, fsync=True)
        n_rows_total += len(rows)
        print(f"  ARCTAN N={N}: {len(rows)} rows flushed to {out_csv}")

    print(f"\nP2 evaluation CSV written: {out_csv} ({n_rows_total} rows total)")
    return out_csv


def evaluate_lshape(
    params: PDN2DParams,
    test_sigmas: np.ndarray,
    j_ref_lookup: dict,                # {(sigma1, sigma2): j_ref}
    *,
    p: int,
    levels: Iterable[int],
    seed: int,
    output_dir: Path,
    q_K: int = 4,
    q_F: int = 2,
    corrector_max_iter: int = 30,
    checkpoint_path: str = "",
    train_meta: Optional[dict] = None,
    ngsolve_cache: Optional[dict] = None,
    iga_ref_cache: Optional[dict] = None,
    tau_global: Optional[float] = None,
) -> Path:
    """Evaluate uniform / positional / positional_corrected at each (sigma, N).

    Per decision 7-B the residual gate uses a global τ calibrated once
    via ``calibrate_tau_global_lshape``; pass it in via ``tau_global``.

    Reference for the H¹-seminorm true error (decision: DIRECT difference, not
    energy subtraction). ``iga_ref_cache`` is the high-resolution IGA self-
    reference cache whose entries carry the solution field (``u_coeffs`` +
    ``knots_x/_y``); when supplied, ``_h1_rel_p3`` takes the DIRECT
    ``‖∇(u_h − u_ref)‖_σ / ‖∇u_ref‖_σ`` branch (no ``max(ref²−our²,0)`` clip,
    so the error keeps converging at fine N). If ``iga_ref_cache`` is None we
    fall back to ``ngsolve_cache`` (the legacy SCALAR cache → energy-subtraction
    metric, which clips/flattens at fine N — kept only for back-compat).
    """
    from src.config import CORRECTOR as _CORR_CFG, T_LSHAPE as _T_P3_CFG
    kappa_const = float(_CORR_CFG["gate_kappa"])
    ar_max = float(_CORR_CFG["shape_gate_AR_max"])
    tau_const = float(tau_global) if tau_global is not None else float("nan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"summary_p3_seed{seed}.csv"
    write_csv_dicts(out_csv, [], CSV_COLUMNS)
    n_rows_total = 0

    # Reference cache that drives the reported H¹ true error. Prefer the IGA
    # self-reference (entries carry u_coeffs -> _h1_rel_p3 uses the DIRECT,
    # non-clipping metric); fall back to the legacy NGSolve scalar cache
    # (energy subtraction) only if no IGA cache was supplied.
    h1_ref_cache = iga_ref_cache if iga_ref_cache is not None else ngsolve_cache

    def lookup_jref(s1, s2):
        k = (round(float(s1), 12), round(float(s2), 12))
        return float(j_ref_lookup.get(k, float("nan")))

    for N in levels:
        print(f"\n=== eval LSHAPE N={N} ===")
        rows: List[dict] = []                     # per-level batch (flushed after this N)
        for sigma in test_sigmas:
            base = {
                "experiment": "lshape",
                "p": int(p),
                "seed": int(seed),
                "N": int(N),
                "sigma1": float(sigma[0]),
                "sigma2": float(sigma[1]),
                "anchor_N": max(levels),
                "sigma_dim": 2,
                "quad_K": int(q_K),
                "quad_F": int(q_F),
                "experiment_version": "v2",
                "checkpoint_path": str(checkpoint_path),
                "T_cap": float(_T_P3_CFG),
            }
            j_ref = lookup_jref(sigma[0], sigma[1])

            sigma1_j = jnp.asarray(sigma[0], dtype=DEFAULT_DTYPE)
            sigma2_j = jnp.asarray(sigma[1], dtype=DEFAULT_DTYPE)

            # ------- Compute all three method results up front so we can
            # back-fill accepted_raw / accepted_corrected consistently
            # across the three rows. ----------------------------------

            # --- uniform ---
            t0_u = time.perf_counter()
            u = _eval_p3_uniform(sigma, p=p, n_elem=int(N), q_K=q_K, q_F=q_F)
            n_elem_eff = int(u["n_elem_eff"])
            from src.nonparametric.ritz_energy import ritz_relative_error
            rel = ritz_relative_error(u["ritz"], j_ref) if not math.isnan(j_ref) else float("nan")
            h1_rel_ng, h1_abs_u, ng_ref_h1, fem_h1 = _h1_rel_p3(
                u["u_h"], u["knots_x"], u["knots_y"], p,
                float(sigma[0]), float(sigma[1]), h1_ref_cache,
            )
            # Residual estimator on the uniform-mesh solve. Same value is reused
            # below as `eta_uniform_squared` for every method of this (sigma, N).
            eta_sq_unif = float(eta_squared_lshape(
                jnp.asarray(u["u_h"]), jnp.asarray(u["knots_x"]), jnp.asarray(u["knots_y"]),
                int(p), n_elem_eff, n_elem_eff,
                sigma1_j, sigma2_j, q_est=_Q_EST_P3,
            ))
            eta_unif = math.sqrt(max(eta_sq_unif, 0.0))
            elapsed_u = time.perf_counter() - t0_u

            # --- positional ---
            t0_p = time.perf_counter()
            pos = _eval_p3_network(params, sigma, p=p, n_elem=int(N), q_K=q_K, q_F=q_F)
            rel_pos = ritz_relative_error(pos["ritz"], j_ref) if not math.isnan(j_ref) else float("nan")
            h1_rel_ng_pos, h1_abs_pos, ng_ref_h1_pos, fem_h1_pos = _h1_rel_p3(
                pos["u_h"], pos["knots_x"], pos["knots_y"], p,
                float(sigma[0]), float(sigma[1]), h1_ref_cache,
            )
            eta_sq_pos = float(eta_squared_lshape(
                jnp.asarray(pos["u_h"]), jnp.asarray(pos["knots_x"]), jnp.asarray(pos["knots_y"]),
                int(p), int(pos["n_elem_eff"]), int(pos["n_elem_eff"]),
                sigma1_j, sigma2_j, q_est=_Q_EST_P3,
            ))
            eta_pos = math.sqrt(max(eta_sq_pos, 0.0))
            elapsed_p = time.perf_counter() - t0_p

            # --- positional_corrected (skipped when corrector_max_iter <= 0) ---
            run_corrector = int(corrector_max_iter) > 0
            if run_corrector:
                t0_c = time.perf_counter()
                sigma_jx = jnp.asarray(sigma, dtype=DEFAULT_DTYPE)
                half = int(N) // 2
                xi = cell_midpoints(half)
                zx_l = forward(params, sigma_jx, xi, axis_id=0.0)
                zx_r = forward(params, sigma_jx, xi, axis_id=0.5)
                zy_l = forward(params, sigma_jx, xi, axis_id=1.0)
                zy_r = forward(params, sigma_jx, xi, axis_id=1.5)
                result, knots_x_opt, knots_y_opt = corrector_lshape(
                    zx_l, zx_r, zy_l, zy_r, sigma_jx,
                    p=p, n_elem=int(N), q_K=q_K, q_F=q_F,
                    max_iter=int(corrector_max_iter),
                )
                from src.nonparametric.r_adapt import p3_effective_n_elem
                n_eff = p3_effective_n_elem(int(N), int(p))
                res_opt = galerkin_solve_lshape(
                    knots_x_opt, knots_y_opt, p, n_eff, n_eff,
                    sigma_jx[0], sigma_jx[1], q_K=q_K, q_F=q_F,
                )
                rel_c = ritz_relative_error(float(res_opt.ritz_energy), j_ref) if not math.isnan(j_ref) else float("nan")
                h1_rel_ng_c, h1_abs_c, ng_ref_h1_c, fem_h1_c = _h1_rel_p3(
                    np.asarray(res_opt.u_h),
                    np.asarray(knots_x_opt), np.asarray(knots_y_opt),
                    p, float(sigma[0]), float(sigma[1]), h1_ref_cache,
                )
                eta_sq_corr = float(eta_squared_lshape(
                    res_opt.u_h, knots_x_opt, knots_y_opt,
                    int(p), int(n_eff), int(n_eff),
                    sigma1_j, sigma2_j, q_est=_Q_EST_P3,
                ))
                eta_corr = math.sqrt(max(eta_sq_corr, 0.0))
                elapsed_c = time.perf_counter() - t0_c
            else:
                # Corrector excluded (corrector_max_iter <= 0): safe dummies so the
                # uniform/positional rows still build. No corrected row is emitted.
                result = None
                knots_x_opt = pos["knots_x"]
                knots_y_opt = pos["knots_y"]
                res_opt = None
                rel_c = float("nan")
                h1_rel_ng_c = float("nan")
                h1_abs_c = float("nan")
                ng_ref_h1_c = float("nan")
                fem_h1_c = float("nan")
                eta_sq_corr = float("nan")
                eta_corr = float("nan")
                elapsed_c = 0.0

            # ------- Decisions 7-B / 7-D / 7-F per-row diagnostics ----

            # uniform: baseline, trivially certified.
            unif_residual_pass = True
            unif_shape_pass = True
            unif_certified = True

            # positional: actual gate evaluation on the positional mesh.
            pos_kx_np = np.asarray(pos["knots_x"])
            pos_ky_np = np.asarray(pos["knots_y"])
            pos_residual_pass = (
                bool(eta_pos <= tau_const)
                if math.isfinite(tau_const) else True
            )
            pos_shape_pass = _shape_gate_pass(pos_kx_np, pos_ky_np, int(p), ar_max)
            pos_certified = bool(pos_residual_pass and pos_shape_pass)

            # positional_corrected: same on the corrected mesh.
            if run_corrector:
                corr_kx_np = np.asarray(knots_x_opt)
                corr_ky_np = np.asarray(knots_y_opt)
                corr_residual_pass = (
                    bool(eta_corr <= tau_const)
                    if math.isfinite(tau_const) else True
                )
                corr_shape_pass = _shape_gate_pass(corr_kx_np, corr_ky_np, int(p), ar_max)
                corr_certified = bool(corr_residual_pass and corr_shape_pass)
            else:
                corr_residual_pass = False
                corr_shape_pass = False
                corr_certified = False

            # Cross-row fields back-filled the same on every row of this (σ,N).
            accepted_raw = bool(pos_certified)
            accepted_corrected = bool(corr_certified)

            # Effectivity denominator (audit D1 fix / remediation decision 2):
            # I_eff = eta / |u - u_ref|_{sigma-H1}, i.e. eta divided by the
            # sigma-weighted H1-seminorm ERROR against the reference, NOT by the
            # reference NORM.  ``_h1_rel_p3`` returns the absolute error as its
            # second element (``h1_abs_*`` = sqrt(err^2)); ``_eff_index`` takes a
            # SQUARED denominator (it divides by sqrt), so we pass err^2 =
            # h1_abs_*^2.  Previously this fed ``ng_ref_h1`` = int sigma|grad u_ref|^2
            # (the reference norm squared), which produced the incorrect
            # 0.86..0.00 column of Table 4.  When no reference error is available
            # (h1_abs is NaN) the index is NaN.
            def _err_sq_for_eff(h1_abs_err: float) -> float:
                if math.isfinite(h1_abs_err) and h1_abs_err > 0.0:
                    return float(h1_abs_err) ** 2
                return float("nan")

            h1_eff_unif = _err_sq_for_eff(h1_abs_u)
            h1_eff_pos = _err_sq_for_eff(h1_abs_pos)
            h1_eff_corr = _err_sq_for_eff(h1_abs_c)

            # ------- Emit three rows ----------------------------------

            rows.append(_build_row(base, method="uniform",
                H1_seminorm_abs_sq=h1_eff_unif if math.isfinite(h1_eff_unif) else "",
                ritz_energy=u["ritz"],
                ritz_reference=j_ref,
                ritz_rel_error=rel,
                h1_rel_error_ngsolve=h1_rel_ng,
                h1_rel_error_iga_ref=h1_rel_ng,
                h1_seminorm_abs_iga_ref=h1_abs_u,
                h1_seminorm_ngsolve_ref=ng_ref_h1,
                sigma_h1_seminorm_sq_fem=fem_h1,
                eta_squared=eta_sq_unif,
                eta=eta_unif,
                eta_uniform_squared=eta_sq_unif,
                eff_index=_eff_index(eta_unif, h1_eff_unif),
                tau=tau_const if math.isfinite(tau_const) else "",
                kappa=kappa_const,
                residual_gate_pass=int(unif_residual_pass),
                shape_gate_pass=int(unif_shape_pass),
                accepted_raw=int(accepted_raw),
                accepted_corrected=int(accepted_corrected),
                certified=int(unif_certified),
                t_solve_sec=u["t_solve"],
                elapsed_sec=elapsed_u,
            ))

            rows.append(_build_row(base, method="positional",
                H1_seminorm_abs_sq=h1_eff_pos if math.isfinite(h1_eff_pos) else "",
                ritz_energy=pos["ritz"],
                ritz_reference=j_ref,
                ritz_rel_error=rel_pos,
                h1_rel_error_ngsolve=h1_rel_ng_pos,
                h1_rel_error_iga_ref=h1_rel_ng_pos,
                h1_seminorm_abs_iga_ref=h1_abs_pos,
                h1_seminorm_ngsolve_ref=ng_ref_h1_pos,
                sigma_h1_seminorm_sq_fem=fem_h1_pos,
                eta_squared=eta_sq_pos,
                eta=eta_pos,
                eta_uniform_squared=eta_sq_unif,
                eff_index=_eff_index(eta_pos, h1_eff_pos),
                tau=tau_const if math.isfinite(tau_const) else "",
                kappa=kappa_const,
                residual_gate_pass=int(pos_residual_pass),
                shape_gate_pass=int(pos_shape_pass),
                accepted_raw=int(accepted_raw),
                accepted_corrected=int(accepted_corrected),
                certified=int(pos_certified),
                t_solve_sec=pos["t_solve"],
                elapsed_sec=elapsed_p,
            ))

            if run_corrector:
                rows.append(_build_row(base, method="positional_corrected",
                    H1_seminorm_abs_sq=h1_eff_corr if math.isfinite(h1_eff_corr) else "",
                    ritz_energy=float(res_opt.ritz_energy),
                    ritz_reference=j_ref,
                    ritz_rel_error=rel_c,
                    h1_rel_error_ngsolve=h1_rel_ng_c,
                    h1_rel_error_iga_ref=h1_rel_ng_c,
                    h1_seminorm_abs_iga_ref=h1_abs_c,
                    h1_seminorm_ngsolve_ref=ng_ref_h1_c,
                    sigma_h1_seminorm_sq_fem=fem_h1_c,
                    eta_squared=eta_sq_corr,
                    eta=eta_corr,
                    eta_uniform_squared=eta_sq_unif,
                    eff_index=_eff_index(eta_corr, h1_eff_corr),
                    tau=tau_const if math.isfinite(tau_const) else "",
                    kappa=kappa_const,
                    residual_gate_pass=int(corr_residual_pass),
                    shape_gate_pass=int(corr_shape_pass),
                    accepted_raw=int(accepted_raw),
                    accepted_corrected=int(accepted_corrected),
                    certified=int(corr_certified),
                    corrector_n_iter=int(result.n_iter),
                    corrector_converged=int(result.converged),
                    corrector_energy_init=float(pos["ritz"]),
                    corrector_energy_final=float(result.energy_final),
                    corrector_max_iter=int(corrector_max_iter),
                    t_corrector_sec=elapsed_c,
                    elapsed_sec=elapsed_c,
                ))

        # Per-level flush: persist this N's rows immediately so a later
        # timeout / kill leaves the earlier N values intact on disk.
        append_csv_dicts(out_csv, rows, CSV_COLUMNS, fsync=True)
        n_rows_total += len(rows)
        print(f"  LSHAPE N={N}: {len(rows)} rows flushed to {out_csv}")

    print(f"\nP3 evaluation CSV written: {out_csv} ({n_rows_total} rows total)")
    return out_csv


__all__ = [
    "CSV_COLUMNS",
    "evaluate_arctan",
    "evaluate_lshape",
    "calibrate_tau_global_arctan",
    "calibrate_tau_global_lshape",
]
