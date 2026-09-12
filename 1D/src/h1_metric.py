"""H¹ error metric with GL quadrature + geometric subdivision (v3).

Replaces the analytic-subtraction approach that suffered catastrophic
cancellation when ``u_h ≈ x^β``.

Background
==========

The legacy implementation in ``nonparametric/exp1/metrics.py`` computed

    ||u_h - x^β||²_H¹ = ∫ u_h^2 - 2 ∫ u_h · x^β + ∫ x^{2β}
                       + (∫ u_h'^2 - 2β ∫ u_h' · x^{β-1} + β² ∫ x^{2β-2})

by analytic monomial integration of each term separately, then summed.
When ``u_h`` is an accurate approximation of ``x^β`` the three terms in
each square cancel down 4-5 decimal digits, leaving the result with
catastrophic precision loss — often inflated by 100x-2000x relative
to the true value.  The old "stable fallback" guard only triggered
when the analytic total was ``<= 1e-14``, but the cancellation
actually INFLATES the total (does not collapse it), so the guard
never fired.

We verified the bug on the 110 v1 bad samples (p=3, N∈{8,16}, H¹_rel
> 1e-3 under legacy): all 110 fall below the 1e-3 threshold under
the v3 metric.  Inflation factor median = 323x, range 102x–1939x.

Implementation
==============

For each element ``E_i`` we evaluate ``(u_h - x^β)²`` and
``(u_h' - β x^{β-1})²`` pointwise at Gauss–Legendre quadrature points,
then sum with the quadrature weights:

  - For interior elements ``[x_i, x_{i+1}]`` with ``x_i > 0`` we use
    a single GL rule of ``q = 2p + 2`` points (default).  This is
    exact for polynomials of degree up to ``2(p+2) - 1 = 2p + 3``
    and is more than enough since ``x^β`` is analytic on every
    interval bounded away from 0.

  - For the first element ``E_0 = [0, x_1]`` (which touches the
    singularity at x=0) we subdivide geometrically: with
    ``n_sub = 20`` (default) and ratio ``r = 0.5`` we form ``n_sub``
    sub-cells covering ``[x_1 * r^{k+1}, x_1 * r^k]`` for
    ``k = 0..n_sub - 1``, plus a final tip cell ``[0, x_1 * r^{n_sub}]``
    that contains the singularity.  GL on each sub-cell with
    ``q = 2p + 2`` points yields the converged value to ~1e-10
    relative precision.

The whole pipeline is pure JAX so it runs inside ``jax.jit`` (in
particular inside ``solve_state_and_metrics``, which is JIT'd).

JIT note
========
``n_sub`` and ``q`` must be **static** (compile-time constants) for
the JIT trace.  ``degree``, ``beta``, and ``cache.knots.shape`` are
also effectively static.  Run time after JIT cache-hit: ~5 ms per
sample at p=3, N=8.

API
===

    h1_error_v3_from_state(u_full, cache, *, degree, beta,
                            n_sub=20, q=None) -> H1Result

Returns a dataclass with absolute and relative L², energy
(H¹-seminorm), and full H¹ error norms, alongside the exact
denominators (closed form).

The companion ``diagnostic_metrics_from_state`` in
``nonparametric/exp1/metrics.py`` is a thin wrapper that adds the
data-oscillation term and packages everything into the existing
``PowerDiagnostics`` namedtuple, preserving the API consumed by the
evaluator pipelines.
"""
from __future__ import annotations

import functools
from typing import NamedTuple, Optional

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.quadrature import rule_gl_on_elements
from src.nonparametric.discretization import basis_batch
from src.nonparametric.pde import (
    BETA,
    exact_energy_norm_power,
    exact_h1_norm_power,
    exact_l2_norm_power,
)


Array = jnp.ndarray


class H1Errors(NamedTuple):
    """Absolute and relative L², energy (H¹-seminorm) and full H¹
    errors.  All values are non-negative."""
    l2_abs: Array
    l2_rel: Array
    energy_abs: Array       # || u_h' - β x^{β-1} ||_{L²}
    energy_rel: Array
    h1_abs: Array           # sqrt(l2_abs² + energy_abs²)
    h1_rel: Array


# --------------------------------------------------------------------------
# Pure JAX helper: build the per-cell (a, b, window-index) arrays for the
# first element's geometric subdivision plus the remaining standard elements.
# --------------------------------------------------------------------------


def _geometric_subdivision_first_elem(b0: Array, *,
                                        n_sub: int,
                                        ratio: float = 0.5,
                                        dtype) -> tuple[Array, Array]:
    """Return ``(a_arr, b_arr)`` of shape ``(n_sub + 1,)`` covering
    ``[0, b0]`` with ``n_sub`` geometric sub-cells plus a tip cell.

    Sub-cell ``k`` (``k = 0..n_sub - 1``) covers
        [b0 * ratio^(k+1), b0 * ratio^k]
    The tip cell (last entry) covers
        [0, b0 * ratio^n_sub]

    All shapes are static for ``jax.jit``.
    """
    n_sub_i = int(n_sub)
    k_arr = jnp.arange(n_sub_i, dtype=dtype)
    powers = jnp.power(jnp.asarray(ratio, dtype=dtype), k_arr)
    sub_b = b0 * powers                                    # shape (n_sub,)
    sub_a = b0 * jnp.power(jnp.asarray(ratio, dtype=dtype),
                              k_arr + jnp.asarray(1.0, dtype=dtype))
    tip_b = b0 * jnp.power(jnp.asarray(ratio, dtype=dtype),
                              jnp.asarray(float(n_sub_i), dtype=dtype))
    tip_a = jnp.asarray(0.0, dtype=dtype)
    a_arr = jnp.concatenate([sub_a, tip_a[None]])           # (n_sub + 1,)
    b_arr = jnp.concatenate([sub_b, tip_b[None]])
    return a_arr, b_arr


# --------------------------------------------------------------------------
# Core JIT'd metric — operates on already-solved state.
# --------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    # NOTE (eq-26 release): ``beta`` is DYNAMIC — it is only used numerically,
    # so tracing it (a) lets the in-graph val H¹ monitor vmap over beta and
    # (b) removes the per-beta recompilation the old static marking caused.
    static_argnames=("degree", "n_sub", "q"),
)
def _h1_error_squared(u_full: Array, cache, degree: int, beta: float,
                       *, n_sub: int = 20, q: Optional[int] = None
                       ) -> tuple[Array, Array]:
    """Return ``(||u_h - u*||²_L², ||u_h' - (u*)'||²_L²)``."""
    p_i = int(degree)
    if q is None:
        q_i = 2 * p_i + 2
    else:
        q_i = int(q)
    n_sub_i = int(n_sub)

    dtype = u_full.dtype
    # NOTE: jnp.asarray (not float()) so the kernel is traceable in beta —
    # the eq-26 release's in-graph val H¹ monitor vmaps over beta. Concrete
    # (float) callers are unaffected.
    beta_t = jnp.asarray(beta, dtype=dtype)

    # ------------------------------------------------------------------
    # First element: n_sub subcells (geometric) + tip cell.
    # The first element of the mesh has a[0] = 0 by construction of
    # theta_to_knots.  We unconditionally use the geometric subdivision.
    # ------------------------------------------------------------------
    h0 = cache.b[0]   # = b[0] - a[0] = b[0] since a[0] = 0
    a_first, b_first = _geometric_subdivision_first_elem(
        h0, n_sub=n_sub_i, dtype=dtype)
    # All these subcells belong to element 0, so they share window[0]
    # and active_ids[0].
    win_first = jnp.broadcast_to(cache.windows[0][None, :],
                                   (n_sub_i + 1, cache.windows.shape[1]))
    actives_first = jnp.broadcast_to(cache.active_ids[0][None, :],
                                       (n_sub_i + 1, cache.active_ids.shape[1]))

    # ------------------------------------------------------------------
    # Remaining elements: use their existing windows/active_ids unchanged.
    # ------------------------------------------------------------------
    a_rest = cache.a[1:]
    b_rest = cache.b[1:]
    win_rest = cache.windows[1:]
    actives_rest = cache.active_ids[1:]

    # ------------------------------------------------------------------
    # Concatenate: (n_sub + 1 + n_elem - 1, ...) total cells.
    # ------------------------------------------------------------------
    a_all = jnp.concatenate([a_first, a_rest])
    b_all = jnp.concatenate([b_first, b_rest])
    win_all = jnp.concatenate([win_first, win_rest], axis=0)
    actives_all = jnp.concatenate([actives_first, actives_rest], axis=0)

    # ------------------------------------------------------------------
    # GL quadrature on every cell at the same q.
    # ------------------------------------------------------------------
    xq, wq = rule_gl_on_elements(a_all, b_all, q_i)
    # xq, wq shape (n_total, q)
    basis, dbasis, _d2 = basis_batch(xq, win_all, p_i)
    # basis shape: (n_total, q, p+1)

    u_loc = u_full[actives_all]                               # (n_total, p+1)
    uh = jnp.einsum("eqi,ei->eq", basis, u_loc)
    duh = jnp.einsum("eqi,ei->eq", dbasis, u_loc)
    ue = jnp.power(xq, beta_t)
    due = beta_t * jnp.power(xq, beta_t - jnp.asarray(1.0, dtype=dtype))

    l2_sq = jnp.sum(wq * jnp.square(uh - ue))
    en_sq = jnp.sum(wq * jnp.square(duh - due))
    zero = jnp.asarray(0.0, dtype=dtype)
    return jnp.maximum(l2_sq, zero), jnp.maximum(en_sq, zero)


# --------------------------------------------------------------------------
# Public API (non-JIT; thin wrapper that packages results + denominators).
# --------------------------------------------------------------------------


def h1_error_v3_from_state(u_full: Array, cache, *, degree: int,
                            beta: float = BETA,
                            n_sub: int = 20,
                            q: Optional[int] = None) -> H1Errors:
    """v3 metric.  Pure-JAX, JIT-traceable through the inner kernel.

    Parameters
    ----------
    u_full : ``Array`` of shape ``(n_basis,)``
        IGA solution coefficients (already includes the Dirichlet DOF).
    cache : EvalCache-like
        Must provide ``knots, sizes, a, b, windows, active_ids``.
    degree, beta : standard.
    n_sub : int, default 20
        Geometric sub-cells in the first element.  ``20`` is the
        converged setting for the worst-case β=1.55 stress test
        (1e-10 relative precision).  ``10`` is acceptable for
        production network meshes (1e-7) but conservative-defaults
        ship with ``20``.
    q : int, optional
        GL points per sub-cell.  Default ``2*degree + 2`` (exact for
        polynomials of degree up to ``2 degree + 3``).
    """
    l2_sq, en_sq = _h1_error_squared(u_full, cache, int(degree),
                                       float(beta), n_sub=int(n_sub),
                                       q=q)
    l2_abs = jnp.sqrt(l2_sq)
    en_abs = jnp.sqrt(en_sq)
    h1_abs = jnp.sqrt(l2_sq + en_sq)

    dtype = u_full.dtype
    l2_norm = jnp.asarray(exact_l2_norm_power(float(beta)), dtype=dtype)
    en_norm = jnp.asarray(exact_energy_norm_power(float(beta)), dtype=dtype)
    h1_norm = jnp.asarray(exact_h1_norm_power(float(beta)), dtype=dtype)
    return H1Errors(
        l2_abs=l2_abs,
        l2_rel=l2_abs / l2_norm,
        energy_abs=en_abs,
        energy_rel=en_abs / en_norm,
        h1_abs=h1_abs,
        h1_rel=h1_abs / h1_norm,
    )


# --------------------------------------------------------------------------
# Deprecated legacy stub — kept for API compatibility, redirects to v3.
# --------------------------------------------------------------------------


def h1_error_legacy_from_state(u_full: Array, cache, *, degree: int,
                                 beta: float = BETA) -> H1Errors:
    """DEPRECATED.  Old analytic-subtraction H¹ metric.

    The original implementation suffered catastrophic cancellation
    when ``u_h ≈ u^*``, inflating ``H¹_rel`` by 100x–2000x for
    accurate IGA solutions.  This function now redirects to
    ``h1_error_v3_from_state``.

    Old numerical values are NOT reproducible from this function;
    consult git history (commits before this fix) if you need to
    reproduce a pre-fix v1/v2 CSV.
    """
    import warnings
    warnings.warn(
        "h1_error_legacy_from_state is deprecated; the cancellation bug "
        "made its results unreliable.  Now redirects to "
        "h1_error_v3_from_state.",
        DeprecationWarning,
        stacklevel=2,
    )
    return h1_error_v3_from_state(u_full, cache, degree=degree, beta=beta)


__all__ = [
    "H1Errors",
    "h1_error_v3_from_state",
    "h1_error_legacy_from_state",
]
