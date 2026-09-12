from __future__ import annotations

"""Extended effectivity diagnostics for the singular experiment."""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.nonparametric.discretization import assemble_reduced_system_from_cache, basis_poly_coeffs_in_x
# NOTE: STABLE_ERROR_* constants were used by the old analytic-cancellation
# fallback path in metrics.py.  That path is gone; the v3 H¹ metric never
# falls back analytically.  We keep these constants locally for
# back-compatibility of `_stable_error_fallback_used_from_state` (which
# now always returns False) without introducing a circular import.
STABLE_ERROR_NEG_TOL = 1.0e-24
STABLE_ERROR_TOTAL_THRESHOLD = 1.0e-14
from src.nonparametric.pde import BETA
from src.nonparametric.quadrature_analytic import loss_terms_from_state
from common.quadrature import monomial_integral

Array = jnp.ndarray
RATIO_DENOM_THRESHOLD = 1.0e-14


def _safe_ratio(num: float, den: float) -> float:
    if abs(float(den)) <= RATIO_DENOM_THRESHOLD:
        return float("nan")
    return float(num) / float(den)


@functools.partial(jax.jit, static_argnames=("degree", "beta"))
def _projected_volume_terms_from_state(
    u_full: Array,
    cache,
    degree: int,
    beta: float = BETA,
) -> tuple[Array, Array, Array, Array]:
    degree = int(degree)
    dtype = cache.knots.dtype
    beta_t = jnp.asarray(float(beta), dtype=dtype)
    alpha_t = beta_t - jnp.asarray(2.0, dtype=dtype)
    c0_t = beta_t * (jnp.asarray(1.0, dtype=dtype) - beta_t)

    a = cache.a
    b = cache.b
    h = cache.sizes

    coeffs_x = basis_poly_coeffs_in_x(cache, degree)
    u_loc = u_full[cache.active_ids]

    idx = jnp.arange(degree - 1, dtype=dtype)  # 0 .. p-2
    factor = (idx + 1.0) * (idx + 2.0)
    bcoef = jnp.einsum("emj,ej->em", factor[None, :, None] * coeffs_x[:, 2:, :], u_loc)

    gram = monomial_integral(a[:, None, None], b[:, None, None], idx[None, :, None] + idx[None, None, :])
    rhs = c0_t * monomial_integral(a[:, None], b[:, None], alpha_t + idx[None, :])
    cproj = jnp.linalg.solve(gram, rhs[..., None])[..., 0]

    exact_poly = jnp.einsum("em,emn,en->e", bcoef, gram, bcoef)
    exact_cross = 2.0 * jnp.sum(bcoef * rhs, axis=1)
    f_sq = (c0_t * c0_t) * monomial_integral(a, b, 2.0 * alpha_t)
    residual_exact_sq = exact_poly + exact_cross + f_sq

    proj_coef = bcoef + cproj
    residual_proj_sq = jnp.einsum("em,emn,en->e", proj_coef, gram, proj_coef)
    osc_sq_elem = jnp.maximum(f_sq - jnp.sum(cproj * rhs, axis=1), jnp.asarray(0.0, dtype=dtype))

    weighted_exact_elem = (h * h) * residual_exact_sq
    weighted_proj_elem = (h * h) * residual_proj_sq
    weighted_osc_elem = (h * h) * osc_sq_elem
    identity_elem = weighted_exact_elem - weighted_proj_elem - weighted_osc_elem

    return (
        jnp.sum(weighted_proj_elem),
        jnp.sum(weighted_osc_elem),
        jnp.max(jnp.abs(identity_elem)),
        jnp.sum(jnp.abs(identity_elem)),
    )


@functools.partial(jax.jit, static_argnames=("degree", "beta"))
def _stable_error_fallback_used_from_state(
    u_full: Array,
    cache,
    degree: int,
    beta: float = BETA,
) -> Array:
    """DEPRECATED.  The old metric had an analytic-then-GL-fallback path;
    this returned True when the analytic totals looked suspicious.  The
    v3 metric does not have an analytic path at all (it always uses GL
    + geometric subdivision), so the question is moot.

    Returns ``False`` unconditionally.  The "fallback_used" key is kept
    in the effectivity-diagnostics dict for back-compatibility with
    plotting and reporting code that consumes it.
    """
    return jnp.asarray(False)


def effectivity_diagnostics_from_state(
    u_full: Array,
    cache,
    degree: int,
    err_h1_semi_abs: float,
    *,
    beta: float = BETA,
    loss_terms=None,
) -> dict[str, Any]:
    degree = int(degree)
    if degree < 2:
        raise ValueError("singular effectivity diagnostics require degree >= 2.")

    if loss_terms is None:
        loss_terms = loss_terms_from_state(u_full, cache, degree, beta)

    eta2_exact = float(2.0 * jax.device_get(loss_terms.total_loss))
    eta2_n = float(2.0 * jax.device_get(loss_terms.neumann_loss))
    eta2_j = 0.0

    eta2_proj_vol, osc_sq, identity_elem_max_abs, identity_elem_l1 = jax.device_get(
        _projected_volume_terms_from_state(u_full, cache, degree, beta)
    )
    eta2_proj = float(eta2_proj_vol) + eta2_n + eta2_j
    osc_sq = float(osc_sq)

    eta_exact = math.sqrt(max(eta2_exact, 0.0))
    eta_proj = math.sqrt(max(eta2_proj, 0.0))
    eta_n = math.sqrt(max(eta2_n, 0.0))
    eta_j = 0.0
    osc_abs = math.sqrt(max(osc_sq, 0.0))

    k_ff, f_f = assemble_reduced_system_from_cache(cache, degree, beta)
    k_ff_np = np.asarray(jax.device_get(k_ff), dtype=np.float64)
    f_f_np = np.asarray(jax.device_get(f_f), dtype=np.float64)
    u_free_np = np.asarray(jax.device_get(u_full[1:]), dtype=np.float64)
    solve_residual_rel = _safe_ratio(np.linalg.norm(k_ff_np @ u_free_np - f_f_np), np.linalg.norm(f_f_np))
    cond_k = float(np.linalg.cond(k_ff_np))

    identity_eta2_abs = abs(eta2_exact - eta2_proj - osc_sq)
    identity_eta2_rel = _safe_ratio(identity_eta2_abs, eta2_exact)
    fallback_used = bool(jax.device_get(_stable_error_fallback_used_from_state(u_full, cache, degree, beta)))

    return {
        "err_h1_semi_abs": float(err_h1_semi_abs),
        "eta_exact": float(eta_exact),
        "eta2_exact": float(eta2_exact),
        "eta_proj": float(eta_proj),
        "eta2_proj": float(eta2_proj),
        "osc": float(osc_abs),
        "osc_sq": float(osc_sq),
        "eta_N": float(eta_n),
        "eta2_N": float(eta2_n),
        "eta_J": float(eta_j),
        "eta2_J": float(eta2_j),
        "I_eff_exact": _safe_ratio(eta_exact, err_h1_semi_abs),
        "I_eff_proj": _safe_ratio(eta_proj, err_h1_semi_abs),
        "I_osc": _safe_ratio(osc_abs, err_h1_semi_abs),
        "frac_proj": _safe_ratio(eta2_proj, eta2_exact),
        "frac_osc": _safe_ratio(osc_sq, eta2_exact),
        "frac_N": _safe_ratio(eta2_n, eta2_exact),
        "frac_J": _safe_ratio(eta2_j, eta2_exact),
        "solve_residual_rel": float(solve_residual_rel),
        "cond_K": float(cond_k),
        "identity_eta2_abs": float(identity_eta2_abs),
        "identity_eta2_rel": float(identity_eta2_rel),
        "identity_elem_max_abs": float(identity_elem_max_abs),
        "identity_elem_l1": float(identity_elem_l1),
        "used_gl64_error_fallback": int(fallback_used),
    }


__all__ = ["effectivity_diagnostics_from_state"]
