from __future__ import annotations

"""Exact error metrics and data oscillation diagnostics for the singular experiment.

H¹ metric — v3 (post-cancellation-bug fix)
==========================================

Prior versions of ``diagnostic_metrics_from_state`` computed the
``||u_h - x^β||²`` integrals analytically by expanding the square as
``∫ u_h^2 - 2 ∫ u_h x^β + ∫ x^{2β}``.  When ``u_h ≈ x^β`` (a typical
"accurate solution") the three terms cancel down 4-5 decimal digits and
the reported H¹_rel was inflated by 100x-2000x.  See
``src/shared/h1_metric.py`` for the failure analysis and the fix.

This module now delegates the H¹ piece to
``src.shared.h1_metric.h1_error_v3_from_state`` (GL quadrature +
geometric subdivision of the first element).  The oscillation term
``osc_sq`` is still computed by the original analytic projection in
``oscillation_element_L2sq_analytic_power``; that integral is over the
data ``f`` only and does not have a cancellation pathology.

The ``PowerDiagnostics`` named tuple and the public API of
``diagnostic_metrics_from_state`` are PRESERVED bit-for-bit (same
fields, same shapes, same JIT signature), so all downstream callers
keep working unchanged.  Only the numerical *values* of the H¹ fields
change — they are now correct.

The original analytic-cancellation code path is removed.  It is not
preserved as a fallback because the cond-fallback in the previous
implementation never fired for the actual bug case (the cancellation
inflated the total, while the guard checked for *suspiciously small*
totals).  If a future caller needs the exact analytic numbers for
reproducibility, consult git history.
"""

import functools
from typing import NamedTuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from src.h1_metric import h1_error_v3_from_state
from src.nonparametric.pde import BETA
from src.nonparametric.quadrature_analytic import (
    oscillation_element_L2sq_analytic_power,
)

Array = jnp.ndarray


class PowerDiagnostics(NamedTuple):
    err_l2_abs: Array
    err_l2_rel: Array
    err_energy_abs: Array
    err_energy_rel: Array
    err_full_h1_abs: Array
    err_full_h1_rel: Array
    osc_sq: Array
    osc_abs: Array


@functools.partial(jax.jit, static_argnames=("degree", "beta"))
def data_oscillation_sq_from_state(
    cache,
    degree: int,
    beta: float = BETA,
) -> Array:
    beta_t = jnp.asarray(beta, dtype=cache.a.dtype)
    elem_osc_sq = jax.vmap(
        lambda a_i, b_i: oscillation_element_L2sq_analytic_power(a_i, b_i, p=degree, beta=beta_t)
    )(cache.a, cache.b)
    return jnp.sum((cache.sizes * cache.sizes) * elem_osc_sq)


@functools.partial(jax.jit, static_argnames=("degree", "beta"))
def diagnostic_metrics_from_state(
    u_full: Array,
    cache,
    degree: int,
    beta: float = BETA,
) -> PowerDiagnostics:
    """Compute exact-error diagnostics for the IGA solution ``u_full``.

    Uses ``h1_error_v3_from_state`` (pointwise GL quadrature + geometric
    subdivision of the first element) for the H¹ error norms, and the
    analytic projection in ``oscillation_element_L2sq_analytic_power``
    for the data-oscillation term ``osc_sq``.

    Returns a ``PowerDiagnostics`` namedtuple with the same fields as
    every previous version of this function.

    The previous analytic-subtraction implementation produced inflated
    H¹_rel values when ``u_h ≈ x^β`` (catastrophic cancellation).
    Verified bug: 110 v1 "bad samples" (p=3, N∈{8,16}, H¹_rel > 1e-3
    under the old metric) all collapse to H¹_rel < 1e-3 under the v3
    metric, with inflation factor median 323x (range 102x–1939x).
    """
    errs = h1_error_v3_from_state(u_full, cache, degree=int(degree),
                                    beta=float(beta))
    osc_sq = jnp.maximum(
        data_oscillation_sq_from_state(cache, degree, beta),
        jnp.asarray(0.0, dtype=u_full.dtype))
    osc_abs = jnp.sqrt(osc_sq)
    return PowerDiagnostics(
        err_l2_abs=errs.l2_abs,
        err_l2_rel=errs.l2_rel,
        err_energy_abs=errs.energy_abs,
        err_energy_rel=errs.energy_rel,
        err_full_h1_abs=errs.h1_abs,
        err_full_h1_rel=errs.h1_rel,
        osc_sq=osc_sq,
        osc_abs=osc_abs,
    )


exact_error_metrics_from_state = diagnostic_metrics_from_state


@functools.partial(jax.jit, static_argnames=("degree",))
def right_flux_from_state(u_full: Array, cache, degree: int) -> Array:
    u_coeffs_t = element_solution_coeffs_t(u_full, cache, degree)
    du_coeffs_t = differentiate_poly_t_wrt_x(u_coeffs_t, cache.sizes, order=1)
    return jnp.sum(du_coeffs_t[-1, :])


def right_flux(knots: Array, p: int, u: Array) -> float:
    knots_j = jnp.asarray(knots, dtype=DEFAULT_DTYPE).reshape(-1)
    u_j = jnp.asarray(u, dtype=DEFAULT_DTYPE).reshape(-1)
    x = jnp.asarray([1.0], dtype=DEFAULT_DTYPE)
    _n, d_n, _d2_n, spans = bspline_basis_local(x, knots_j, int(p))
    idx = spans[:, None] - int(p) + jnp.arange(int(p) + 1, dtype=jnp.int32)[None, :]
    u_loc = u_j[idx[0]]
    return float(jax.device_get(jnp.sum(d_n[0] * u_loc)))


__all__ = [
    "PowerDiagnostics",
    "data_oscillation_sq_from_state",
    "diagnostic_metrics_from_state",
    "exact_error_metrics_from_state",
    "right_flux_from_state",
    "right_flux",
]
