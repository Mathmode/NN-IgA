from __future__ import annotations

"""Analytic integrals, assembly quadrature, and residual estimator for the singular experiment.

The manufactured solution is u(x)=x^beta on [0,1], with RHS f(x)=beta(1-beta)x^{beta-2}
(and sigma=1, alpha=0). Many quantities (load, estimator terms, exact errors) are computed
analytically via monomial integrals.

Note: stiffness matrix is assembled with element-wise Gauss-Legendre quadrature (robust
and fast), while the load and estimator pieces use analytic integrals.
"""

import functools
import math
from typing import NamedTuple, Tuple

import numpy as np

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
from jax import lax

from common.bspline_basis import bspline_basis_local
from common.optimization import Eta2Terms, residual_loss_from_eta2
from common.quadrature import as_dtype as _as_dtype
from common.quadrature import monomial_integral as _I
from common.quadrature import rule_gl_on_element, rule_gl_on_elements
from src.h1_metric import _geometric_subdivision_first_elem
from src.nonparametric.discretization import differentiate_poly_t_wrt_x, element_solution_coeffs_t, solve_state
from src.nonparametric.pde import BETA

Array = jnp.ndarray


def rule_on_element(a: float, b: float, n: int) -> Tuple[Array, Array]:
    return rule_gl_on_element(a, b, int(n))


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def stiffness_global_gl(knots: Array, p: int, nq: int = 8) -> Array:
    """Assemble the stiffness matrix using elementwise Gauss–Legendre quadrature."""
    p = int(p)
    dtype = knots.dtype
    n_ctrl = knots.shape[0] - p - 1
    n_elem = int(knots.shape[0] - 2 * p - 1)

    a = knots[p : p + n_elem]
    b = knots[p + 1 : p + n_elem + 1]
    x, w = rule_gl_on_elements(a, b, int(nq))

    x_flat = x.reshape(-1)
    _, dN_flat, _, _ = bspline_basis_local(x_flat, knots, p)
    dN = dN_flat.reshape(n_elem, int(nq), p + 1)

    Kloc = jnp.einsum("eqi,eq,eqj->eij", dN, w, dN)

    cols = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    K = jnp.zeros((n_ctrl, n_ctrl), dtype=dtype)
    return K.at[cols[:, :, None], cols[:, None, :]].add(Kloc)


@functools.partial(jax.jit, static_argnames=("p", "nq"))
def stiffness_global_gl_coo(knots: Array, p: int, nq: int = 8) -> tuple[Array, Array, Array]:
    """Assemble stiffness entries in COO triplet form (rows, cols, values)."""
    p = int(p)
    n_elem = int(knots.shape[0] - 2 * p - 1)

    a = knots[p : p + n_elem]
    b = knots[p + 1 : p + n_elem + 1]
    x, w = rule_gl_on_elements(a, b, int(nq))

    x_flat = x.reshape(-1)
    _, dN_flat, _, _ = bspline_basis_local(x_flat, knots, p)
    dN = dN_flat.reshape(n_elem, int(nq), p + 1)
    Kloc = jnp.einsum("eqi,eq,eqj->eij", dN, w, dN)

    cols = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    rows = jnp.broadcast_to(cols[:, :, None], Kloc.shape).reshape(-1)
    cols = jnp.broadcast_to(cols[:, None, :], Kloc.shape).reshape(-1)
    vals = Kloc.reshape(-1)
    return rows, cols, vals


# -----------------------------------------------------------------------------
# Local polynomial reconstruction (for analytic load & estimator)
# -----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p",))
def _poly_coeffs_in_x_on_span(knots: Array, p: int, a: float, b: float) -> Array:
    """Element polynomial coefficients for the active B-splines on one element.

    Returns C of shape (p+1, p+1) such that on [a,b], for the (p+1) active basis functions:
        N_j(x) = sum_{k=0}^p C[k,j] x^k.

    Implementation: evaluate basis at (p+1) Chebyshev-like points and solve Vandermonde.
    """
    p = int(p)
    h = b - a
    r = jnp.arange(p + 1, dtype=knots.dtype)
    t = 0.5 * (1.0 + jnp.cos((2.0 * r + 1.0) * jnp.pi / (2.0 * (p + 1.0))))
    x = a + h * t
    N, _, _, _ = bspline_basis_local(x, knots, p)
    V = jnp.vander(x, N=p + 1, increasing=True)
    return jnp.linalg.solve(V, N)


@functools.partial(jax.jit, static_argnames=("p",))
def _poly_coeffs_in_x_on_spans(knots: Array, p: int, a: Array, b: Array) -> Array:
    """Vectorised polynomial coefficients for active B-splines on many elements.

    Returns C of shape (n_elem, p+1, p+1) where:
        N_{e,j}(x) = sum_{k=0}^p C[e,k,j] x^k  on element e.
    """
    p = int(p)
    dtype = knots.dtype

    a = jnp.asarray(a, dtype=dtype).reshape(-1)
    b = jnp.asarray(b, dtype=dtype).reshape(-1)

    h = b - a
    r = jnp.arange(p + 1, dtype=dtype)
    t = 0.5 * (1.0 + jnp.cos((2.0 * r + 1.0) * jnp.pi / (2.0 * (p + 1.0))))

    x = a[:, None] + h[:, None] * t[None, :]
    x_flat = x.reshape(-1)
    N_flat, _, _, _ = bspline_basis_local(x_flat, knots, p)
    N = N_flat.reshape(a.shape[0], p + 1, p + 1)

    k = jnp.arange(p + 1, dtype=dtype)
    V = jnp.power(x[:, :, None], k[None, None, :])
    return jnp.linalg.solve(V, N)


@functools.partial(jax.jit, static_argnames=("p",))
def load_element_analytic_power(
    knots: Array,
    p: int,
    a: float,
    b: float,
    c0: float,
    alpha: float,
) -> Array:
    """Local load vector for f(x)=c0 x^alpha."""
    p = int(p)
    C = _poly_coeffs_in_x_on_span(knots, p, a, b)
    k = jnp.arange(p + 1, dtype=knots.dtype)
    Jk = _I(a, b, _as_dtype(alpha, knots.dtype) + k)
    return _as_dtype(c0, C.dtype) * (C.T @ Jk)


@functools.partial(jax.jit, static_argnames=("p",))
def load_global_analytic_power(knots: Array, p: int, beta: float) -> Array:
    """Assemble global RHS for the power-load problem analytically."""
    p = int(p)
    dtype = knots.dtype
    beta = _as_dtype(beta, dtype)
    c0 = beta * (1.0 - beta)
    alpha = beta - 2.0

    n_ctrl = knots.shape[0] - p - 1
    n_elem = int(knots.shape[0] - 2 * p - 1)

    a = knots[p : p + n_elem]
    b = knots[p + 1 : p + n_elem + 1]

    C = _poly_coeffs_in_x_on_spans(knots, p, a, b)
    k = jnp.arange(p + 1, dtype=dtype)
    Jk = _I(a[:, None], b[:, None], _as_dtype(alpha, dtype) + k[None, :])
    Floc = _as_dtype(c0, dtype) * jnp.einsum("ekj,ek->ej", C, Jk)

    cols = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    F = jnp.zeros((n_ctrl,), dtype=dtype)
    return F.at[cols.reshape(-1)].add(Floc.reshape(-1))


# -----------------------------------------------------------------------------
# Analytic residual estimator for the power problem
# -----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p",))
def _d2_coeffs_of_uh(knots: Array, p: int, a: float, b: float, Uloc: Array) -> Array:
    """Coefficients of u_h'' on one element.

    Returns bcoef[m] such that u_h''(x) = sum_{m=0}^{p-2} bcoef[m] x^m on [a,b].
    """
    p = int(p)
    C = _poly_coeffs_in_x_on_span(knots, p, a, b)  # (p+1,p+1)
    m = jnp.arange(max(p - 1, 0), dtype=knots.dtype)  # 0..p-2
    if m.size == 0:
        return C[:0, :] @ Uloc
    factor = (m + 1.0) * (m + 2.0)
    B = factor[:, None] * C[2:, :]
    return B @ Uloc


# -----------------------------------------------------------------------------
# Cancellation-aware residual-squared integral.
#
# For the singular-power problem the per-element strong residual is
#     R_E(x) = u_h''(x) + c0 * x^alpha,
# with c0 = beta*(1 - beta) and alpha = beta - 2 < 0. The expand-the-square
# path
#     ∫ R_E^2 dx = ∫ u_h''^2 dx + 2 c0 ∫ u_h'' x^alpha dx + c0^2 ∫ x^{2 alpha} dx
# computes three O(1) integrals; their sum is the (tiny) ||R_E||^2. When
# the network learns u_h'' ≈ -f very well — Experiment singular, p = 3, large
# N — the cancellation drives the floating-point sum slightly negative;
# the old ``max(R2, 0)`` guard then clipped it to exactly 0 in ~10 % of
# the (p = 3, N ∈ {8..64}) rows, spuriously zeroing eta.
#
# Fix (this module): keep the analytic-monomial expansion for elements
# where it is reliable (its three terms don't yet cancel below the FP
# floor — verified by ``R2_analytic > 0``), and fall back to a
# cancellation-free pointwise-quadrature value on elements where the
# cancellation has fired. Concretely, per element
#     R2_e = R2_analytic_e  if R2_analytic_e > 0
#          = R2_quadrature_e  otherwise.
#
# Rationale:
# * **Non-cancelling regime** (uniform mesh or weakly trained network):
#   the analytic value is exact and dominated by the singular
#   c0^2 ∫x^{2α} dx; the quadrature path, while non-negative, loses
#   accuracy on the strongly-singular tip of the first element
#   (e.g. ~22 % at β=1.55, p=2, N=4 uniform). Use the analytic value.
# * **Cancellation regime** (well-trained, fine N): the analytic value
#   underflows to 0 or below; the quadrature value is a small positive
#   number computed from R_E(x) — itself well-defined pointwise without
#   cancellation. The integrand R_E^2 is bounded and small everywhere on
#   the well-trained mesh, so a single GL rule per cell (with geometric
#   subdivision on the first element only for safety) is accurate.
#
# Both branches are JIT/grad-compatible; the per-element ``jnp.where``
# selects whichever value is the better estimate of the (non-negative)
# integral.
# -----------------------------------------------------------------------------


def _bcoef_x_per_element(knots: Array, p: int, U: Array) -> Array:
    """Return ``bcoef[e, m]`` such that ``u_h''(x) = Σ_m bcoef[e, m] x^m``
    on element ``e``. Shape ``(n_elem, max(p - 1, 0))``.

    Shared between the analytic and quadrature R^2 paths so both speak
    the same per-element coefficient representation.
    """
    p_i = int(p)
    dtype = knots.dtype
    n_elem = int(knots.shape[0] - 2 * p_i - 1)
    a = knots[p_i : p_i + n_elem]
    b = knots[p_i + 1 : p_i + n_elem + 1]
    cols = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(p_i + 1, dtype=jnp.int32)[None, :]
    Uloc = U[cols]
    C = _poly_coeffs_in_x_on_spans(knots, p_i, a, b)        # (n_elem, p+1, p+1)
    if p_i < 2:
        return jnp.zeros((n_elem, 0), dtype=dtype)
    m_arr = jnp.arange(p_i - 1, dtype=dtype)
    factor = (m_arr + 1.0) * (m_arr + 2.0)
    B = factor[None, :, None] * C[:, 2:, :]
    return jnp.einsum("emj,ej->em", B, Uloc)                # (n_elem, p-1)


@functools.partial(jax.jit, static_argnames=("p",))
def _residual_R2_analytic_unclipped(
    knots: Array,
    p: int,
    U: Array,
    beta: Array,
) -> Array:
    """Per-element ``R^2`` via the analytic expand-the-square sum.

    Identical to the original formula used by
    ``estimator_eta2_terms_analytic_power``/``loss_terms_from_state``
    *but without the* ``max(R2, 0)`` *guard* — we want the genuine
    (possibly slightly-negative) value so the caller can detect
    cancellation and fall back to the quadrature path.
    """
    p_i = int(p)
    dtype = knots.dtype
    n_elem = int(knots.shape[0] - 2 * p_i - 1)
    a = knots[p_i : p_i + n_elem]
    b = knots[p_i + 1 : p_i + n_elem + 1]

    beta_t = jnp.asarray(beta, dtype=dtype)
    alpha = beta_t - jnp.asarray(2.0, dtype=dtype)
    c0 = beta_t * (jnp.asarray(1.0, dtype=dtype) - beta_t)

    if p_i < 2:
        # u_h'' ≡ 0 ⇒ a single non-negative term; no cancellation possible.
        return (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)

    bcoef = _bcoef_x_per_element(knots, p_i, U)             # (n_elem, p-1)
    m_arr = jnp.arange(p_i - 1, dtype=dtype)
    M = m_arr[:, None] + m_arr[None, :]
    I_M = _I(a[:, None, None], b[:, None, None], M[None, :, :])
    I_d2 = jnp.einsum("em,en,emn->e", bcoef, bcoef, I_M)
    I_ma = _I(a[:, None], b[:, None], m_arr[None, :] + _as_dtype(alpha, dtype))
    I_cross = 2.0 * c0 * jnp.sum(bcoef * I_ma, axis=1)
    I_f2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
    return I_d2 + I_cross + I_f2


@functools.partial(jax.jit, static_argnames=("p", "q", "n_sub"))
def _residual_R2_quadrature_only(
    knots: Array,
    p: int,
    U: Array,
    beta: Array,
    *,
    q: int = 0,
    n_sub: int = 20,
) -> Array:
    """Per-element ``R^2`` via pointwise GL quadrature of ``(u_h'' + c0 x^alpha)^2``.

    Manifestly non-negative: the integrand is squared *before* the sum,
    so cancellation between the polynomial and singular parts of the
    residual is avoided. Used as the fallback branch of the hybrid
    estimator when the analytic expansion has lost all accuracy.

    The first element ``[0, b_0]`` is geometrically subdivided to keep
    the ``x^{2 alpha}`` divergence at the origin under control. Other
    elements (``a_e > 0``) use a single GL rule; the integrand is smooth
    there and GL converges rapidly.
    """
    p_i = int(p)
    dtype = knots.dtype
    n_elem = int(knots.shape[0] - 2 * p_i - 1)
    a = knots[p_i : p_i + n_elem]
    b = knots[p_i + 1 : p_i + n_elem + 1]

    beta_t = jnp.asarray(beta, dtype=dtype)
    alpha = beta_t - jnp.asarray(2.0, dtype=dtype)
    c0 = beta_t * (jnp.asarray(1.0, dtype=dtype) - beta_t)

    if p_i < 2:
        return (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)

    bcoef_x = _bcoef_x_per_element(knots, p_i, U)            # (n_elem, p-1)

    q_i = int(q) if q else (2 * p_i + 2)
    n_sub_i = int(n_sub)
    a_first, b_first = _geometric_subdivision_first_elem(
        b[0], n_sub=n_sub_i, dtype=dtype)
    xq_first, wq_first = rule_gl_on_elements(a_first, b_first, q_i)
    xq_rest, wq_rest = rule_gl_on_elements(a[1:], b[1:], q_i)
    xq_all = jnp.concatenate([xq_first, xq_rest], axis=0)    # (n_total, q)
    wq_all = jnp.concatenate([wq_first, wq_rest], axis=0)
    elem_idx = jnp.concatenate(
        [
            jnp.zeros((n_sub_i + 1,), dtype=jnp.int32),
            jnp.arange(1, n_elem, dtype=jnp.int32),
        ],
        axis=0,
    )

    m_arr = jnp.arange(p_i - 1, dtype=dtype)
    x_powers = jnp.power(xq_all[:, :, None], m_arr[None, None, :])  # (n_total, q, p-1)
    bcoef_per_subcell = bcoef_x[elem_idx]                            # (n_total, p-1)
    uh_d2_at_q = jnp.einsum("nqm,nm->nq", x_powers, bcoef_per_subcell)

    f_q = c0 * jnp.power(xq_all, alpha)
    R_q = uh_d2_at_q + f_q
    R2_subcell = jnp.sum(wq_all * R_q * R_q, axis=-1)                # (n_total,) ≥ 0

    R2_per_elem = jnp.zeros((n_elem,), dtype=dtype)
    R2_per_elem = R2_per_elem.at[elem_idx].add(R2_subcell)
    return R2_per_elem


@functools.partial(jax.jit, static_argnames=("p", "q", "n_sub"))
def _residual_R2_per_element_quadrature(
    knots: Array,
    p: int,
    U: Array,
    beta: Array,
    *,
    q: int = 0,
    n_sub: int = 20,
) -> Array:
    """Per-element ``∫_E (u_h''(x) + c0 x^alpha)^2 dx``, cancellation-aware.

    See the module-level comment for the rationale. Returns a per-element
    R^2 array that is non-negative by construction:
    ``R2_analytic`` when it is positive (accurate non-cancelling case),
    else ``R2_quadrature`` (cancellation-free fallback). Both branches are
    smooth functions of ``U``, so the resulting estimator is JIT- and
    grad-friendly.
    """
    p_i = int(p)
    dtype = knots.dtype

    if p_i < 2:
        # u_h'' ≡ 0; single non-negative term — no cancellation, no fallback.
        return _residual_R2_quadrature_only(knots, p_i, U, beta, q=q, n_sub=n_sub)

    R2_analytic = _residual_R2_analytic_unclipped(knots, p_i, U, beta)
    R2_quadrature = _residual_R2_quadrature_only(
        knots, p_i, U, beta, q=q, n_sub=n_sub)
    # Pointwise select. The where-condition's only discontinuity is at
    # the cancellation boundary, where the two values agree to within
    # the cancellation FP floor — so the resulting function is
    # numerically continuous and grad-safe.
    return jnp.where(
        R2_analytic > jnp.asarray(0.0, dtype=dtype),
        R2_analytic,
        R2_quadrature,
    )


@functools.partial(jax.jit, static_argnames=("p",))
def residual_element_R2_fully_analytic(
    knots: Array,
    p: int,
    a: float,
    b: float,
    Uloc: Array,
    beta: float,
) -> Array:
    """Single-element ``∫_E (u_h'' + f)^2`` via the analytic monomial
    expansion (kept for diagnostic / back-compat use).

    .. warning::

       This expand-the-square form suffers catastrophic cancellation
       when ``u_h'' ≈ -f`` (the well-trained singular, p = 3, large-N regime).
       Production paths use the cancellation-aware hybrid in
       ``estimator_eta2_terms_analytic_power`` /
       ``loss_terms_from_state`` instead. Callers wanting an
       always-non-negative R^2 should use those.
    """
    dtype = knots.dtype
    beta = _as_dtype(beta, dtype)
    alpha = beta - 2.0
    c0 = beta * (1.0 - beta)

    bcoef = _d2_coeffs_of_uh(knots, int(p), a, b, Uloc)

    if bcoef.size == 0:
        return (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)

    m = jnp.arange(bcoef.size, dtype=dtype)
    M = m[:, None] + m[None, :]

    I_d2 = jnp.sum((bcoef[:, None] * bcoef[None, :]) * _I(a, b, M))
    I_cross = 2.0 * c0 * jnp.sum(bcoef * _I(a, b, m + alpha))
    I_f2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
    return I_d2 + I_cross + I_f2


@functools.partial(jax.jit, static_argnames=("p",))
def oscillation_element_L2sq_analytic_power(
    a: float,
    b: float,
    *,
    p: int,
    beta: float,
) -> Array:
    """Compute ||f - Pi_{p-2} f||_{L2(E)}^2 for the power load."""
    p = int(p)
    dtype = jnp.asarray(a).dtype

    beta = _as_dtype(beta, dtype)
    alpha = beta - 2.0
    c0 = beta * (1.0 - beta)

    if p < 2:
        return (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)

    deg_proj = p - 2
    k = deg_proj + 1
    idx = jnp.arange(k, dtype=dtype)

    M = _I(a, b, idx[:, None] + idx[None, :])
    r = _as_dtype(c0, dtype) * _I(a, b, _as_dtype(alpha, dtype) + idx)
    c = jnp.linalg.solve(M, r)

    I_f2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
    osc2 = I_f2 - jnp.dot(c, r)
    return jnp.maximum(osc2, _as_dtype(0.0, dtype))


@functools.partial(jax.jit, static_argnames=("p", "include_osc"))
def estimator_eta2_terms_analytic_power(
    knots: Array,
    p: int,
    U: Array,
    *,
    beta: float,
    include_osc: bool = False,
) -> tuple[Array, Eta2Terms]:
    """Compute eta^2 for the power problem in 1D.

    Paper-consistent robust scaling:
      eta^2 = sum_E rho_E^2 ||R_E||^2 + (h_B/sigma_B) |N_B|^2 (+ optional osc term),
    with rho_E = min(h_E/sqrt(sigma_E), 1/sqrt(|alpha_E|)).

    For the singular experiment, sigma=1 and alpha=0, therefore rho_E = h_E and h_B/sigma_B = h_B.

    For globally C^1 splines and continuous sigma, interior jumps vanish.
    The training loss in the singular experiment uses ``include_osc=False``. Any oscillation
    contribution is diagnostic-only and must not be added to the optimized loss.
    """
    p = int(p)
    dtype = knots.dtype
    beta = _as_dtype(beta, dtype)

    n_elem = int(knots.shape[0] - 2 * p - 1)

    a = knots[p : p + n_elem]
    b = knots[p + 1 : p + n_elem + 1]
    he = b - a

    cols = jnp.arange(n_elem, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    Uloc = U[cols]

    C = _poly_coeffs_in_x_on_spans(knots, p, a, b)

    alpha = beta - 2.0
    c0 = beta * (1.0 - beta)

    # Per-element ``R^2 = ∫_E (u_h'' + c0 x^alpha)^2 dx`` via the
    # cancellation-free quadrature helper. The old expand-the-square path
    # (R2 = I_d2 + I_cross + I_f2 of three O(1) terms) suffered catastrophic
    # cancellation when u_h'' ≈ -f and yielded ~10 % spurious zeros at
    # p = 3, large N — see ``_residual_R2_per_element_quadrature`` for the
    # fix rationale.
    R2 = _residual_R2_per_element_quadrature(knots, p, U, beta)

    rho_e = he  # sigma=1, alpha=0 -> rho_E = h_E
    eta2_elem = jnp.sum((rho_e * rho_e) * R2)
    eta2_osc = jnp.asarray(0.0, dtype=dtype)

    if include_osc:
        if p < 2:
            osc2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
        else:
            idx = jnp.arange(p - 1, dtype=dtype)
            Mproj = _I(a[:, None, None], b[:, None, None], idx[None, :, None] + idx[None, None, :])
            rproj = _as_dtype(c0, dtype) * _I(a[:, None], b[:, None], _as_dtype(alpha, dtype) + idx[None, :])
            cproj = jnp.linalg.solve(Mproj, rproj[..., None])[..., 0]
            I_f2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
            osc2 = I_f2 - jnp.sum(cproj * rproj, axis=1)
            osc2 = jnp.maximum(osc2, _as_dtype(0.0, dtype))

        eta2_osc = jnp.sum((he * he) * osc2)

    # Neumann boundary residual at x=1: h_last * (u_h'(1)-beta)^2
    h_last = jnp.maximum(he[-1], _as_dtype(1e-15, dtype))
    acoef_last = C[-1] @ Uloc[-1]

    if p >= 1:
        k = jnp.arange(1, p + 1, dtype=dtype)
        duR = jnp.sum(k * acoef_last[1:] * jnp.power(b[-1], k - 1.0))
    else:
        duR = _as_dtype(0.0, dtype)

    flux_err = duR - beta
    rho_B = h_last  # h_B / sigma_B with sigma_B=1
    eta2_bc = rho_B * (flux_err * flux_err)

    # Ulp-level guard only. ``eta2_elem`` (sum over non-negative per-element
    # R²·h_e²), ``eta2_osc`` (non-negative by ``oscillation_element_L2sq``)
    # and ``eta2_bc`` (a square) are individually ≥ 0, so the sum is ≥ 0;
    # ``max(..., 0)`` survives only as a defensive last-ulp clip. Empirically
    # this guard does not fire on any singular sample post catastrophic-cancellation
    # fix (verified across 768 samples covering p=3, N∈{8..256}).
    eta2_total = jnp.maximum(eta2_elem + eta2_osc + eta2_bc, _as_dtype(0.0, dtype))
    terms = Eta2Terms(
        elem=eta2_elem,
        jump=jnp.asarray(0.0, dtype=dtype),
        bc=eta2_bc,
        osc=eta2_osc,
    )
    return eta2_total, terms


@functools.partial(jax.jit, static_argnames=("p", "include_osc"))
def estimator_eta2_analytic_power(
    knots: Array,
    p: int,
    U: Array,
    *,
    beta: float,
    include_osc: bool = False,
) -> Array:
    eta2, _terms = estimator_eta2_terms_analytic_power(knots, p, U, beta=beta, include_osc=include_osc)
    return eta2


@functools.partial(jax.jit, static_argnames=("p", "include_osc"))
def estimator_loss_analytic_power(
    knots: Array,
    p: int,
    U: Array,
    *,
    beta: float,
    include_osc: bool = False,
) -> Array:
    """Training objective: L = 0.5 * eta^2."""
    eta2 = estimator_eta2_analytic_power(knots, p, U, beta=beta, include_osc=include_osc)
    return residual_loss_from_eta2(eta2)


class ResidualLossTerms(NamedTuple):
    total_loss: Array
    volume_loss: Array
    neumann_loss: Array

    @property
    def total(self) -> Array:
        return self.total_loss

    @property
    def volume(self) -> Array:
        return self.volume_loss

    @property
    def neumann(self) -> Array:
        return self.neumann_loss


PowerLossTerms = ResidualLossTerms


def integrate_x_power(a: Array, b: Array, alpha: float) -> Array:
    """Stable evaluation of ``∫_a^b x^alpha dx`` for small cells near the origin."""
    alpha_plus_one = float(alpha + 1.0)

    def scalar(ai: Array, bi: Array) -> Array:
        power = jnp.asarray(alpha_plus_one, dtype=ai.dtype)

        def from_origin(_):
            return jnp.power(bi, power) / power

        def from_positive(_):
            rel = (bi - ai) / ai
            return jnp.power(ai, power) * jnp.expm1(power * jnp.log1p(rel)) / power

        return lax.cond(ai > 0.0, from_positive, from_origin, operand=None)

    return jax.vmap(scalar)(a, b)


def integrate_poly_square_t(coeffs_t: Array, sizes: Array) -> Array:
    idx = jnp.arange(coeffs_t.shape[1], dtype=coeffs_t.dtype)
    gram = jnp.reciprocal(idx[:, None] + idx[None, :] + jnp.asarray(1.0, dtype=coeffs_t.dtype))
    return sizes * jnp.einsum("ei,ij,ej->e", coeffs_t, gram, coeffs_t)


_SERIES_NTERMS = 60
_SERIES_R_THRESHOLD = 0.5


@functools.lru_cache(maxsize=32)
def _gen_binom_coeffs(alpha: float, n: int) -> np.ndarray:
    """Generalized binomial coefficients C(α, j) for j = 0, ..., n-1."""
    c = np.ones(n, dtype=np.float64)
    for j in range(1, n):
        c[j] = c[j - 1] * (alpha - j + 1) / j
    return c


def integrate_poly_times_power_t(coeffs_t: Array, a: Array, b: Array, alpha: float) -> Array:
    r"""Compute  :math:`\int_a^b p(t)\,x^\alpha\,dx`  where
    :math:`p(t)=\sum_k c_k\,t^k` and :math:`t=(x-a)/h`.

    Three-way strategy for full-precision across the mesh:

    * **a = 0** (element at the origin):
      exact formula  :math:`h^{\alpha+1}/(k+\alpha+1)`.
    * **a > 0, h/a < 0.5** (elements far from origin):
      series in :math:`r=h/a` that converges geometrically
      and avoids the catastrophic cancellation of the binomial path.
    * **a > 0, h/a ≥ 0.5** (elements near the origin):
      classical binomial expansion (accurate because
      :math:`a^{k-m}` terms are small).
    """
    h = b - a
    degree = coeffs_t.shape[1] - 1
    dtype = a.dtype
    alpha_f = float(alpha)
    alpha_t = jnp.asarray(alpha_f, dtype=dtype)
    eps_tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)

    # ── 1) Binomial method (original; accurate near origin) ──────────────
    power_integrals = [integrate_x_power(a, b, alpha_f + m) for m in range(degree + 1)]
    binom_result = jnp.zeros_like(a)
    for k in range(degree + 1):
        inner = jnp.zeros_like(a)
        for m in range(k + 1):
            sign = -1.0 if (k - m) % 2 else 1.0
            inner = inner + float(math.comb(k, m) * sign) * jnp.power(a, k - m) * power_integrals[m]
        if k > 0:
            inner = inner / jnp.power(h, k)
        binom_result = binom_result + coeffs_t[:, k] * inner

    # ── 2) Series method (accurate far from origin) ──────────────────────
    #    Change variable x = a(1 + r·t), giving:
    #      ∫_a^b t^k x^α dx = h · a^α · Σ_j C(α,j) r^j / (k+j+1)
    #    where r = h/a and C(α,j) are generalized binomial coefficients.
    #    Converges geometrically in r; no alternating-sign cancellation.
    J = _SERIES_NTERMS
    binom_c = jnp.asarray(_gen_binom_coeffs(alpha_f, J), dtype=dtype)

    a_safe = jnp.maximum(a, eps_tiny)
    r = h / a_safe
    # Clip r for series computation (prevents overflow in r^j);
    # only used where r < _SERIES_R_THRESHOLD, so clipped values are
    # multiplied by zero in the final jnp.where selection.
    r_clip = jnp.minimum(r, jnp.asarray(_SERIES_R_THRESHOLD - 0.01, dtype=dtype))

    j_arr = jnp.arange(J, dtype=dtype)
    k_arr = jnp.arange(degree + 1, dtype=dtype)

    # r^j  via log/exp for numerical safety: shape (n_elem, J)
    log_r_clip = jnp.log(jnp.maximum(r_clip, eps_tiny))
    r_powers = jnp.exp(j_arr[None, :] * log_r_clip[:, None])

    # Weighted series terms: C(α,j) · r^j, shape (n_elem, J)
    weighted = binom_c[None, :] * r_powers

    # 1/(k + j + 1), shape (degree+1, J)
    inv_denom = jnp.reciprocal(k_arr[:, None] + j_arr[None, :] + 1.0)

    # Sum over j for each k: shape (n_elem, degree+1)
    series_k = jnp.einsum("ej,kj->ek", weighted, inv_denom)

    a_alpha = jnp.power(a_safe, alpha_t)
    series_result = jnp.sum(coeffs_t * h[:, None] * a_alpha[:, None] * series_k, axis=1)

    # ── 3) Origin formula: ∫_0^h t^k x^α dx = h^{α+1} / (k+α+1) ───────
    h_pow = jnp.power(h, alpha_t + 1.0)
    origin_k = h_pow[:, None] / (k_arr[None, :] + alpha_t + 1.0)
    origin_result = jnp.sum(coeffs_t * origin_k, axis=1)

    # ── Select per element ───────────────────────────────────────────────
    is_origin = a <= 0.0
    use_series = r < jnp.asarray(_SERIES_R_THRESHOLD, dtype=dtype)

    result = jnp.where(
        is_origin,
        origin_result,
        jnp.where(use_series, series_result, binom_result),
    )
    return result


@functools.partial(jax.jit, static_argnames=("degree",))  # FIX: beta traced (no static) evita recompilar por beta
def loss_terms_from_state(
    u_full: Array,
    cache,
    degree: int,
    beta: float = BETA,
) -> ResidualLossTerms:
    """Residual loss decomposition for the solved state.

    Volume term is the cancellation-free squared-residual integral; see
    ``_residual_R2_per_element_quadrature`` for the rationale. The earlier
    expand-the-square path (``integrate_poly_square_t +
    2·c0·integrate_poly_times_power_t + c0²·integrate_x_power``) suffered
    catastrophic cancellation in the well-trained regime — e.g. p = 3,
    large N — and silently clipped the (slightly negative) result to 0
    via downstream ``max(., 0)`` guards.
    """
    beta_t = jnp.asarray(beta, dtype=u_full.dtype)  # FIX: asarray maneja tracer (beta traced, no recompila)

    u_coeffs_t = element_solution_coeffs_t(u_full, cache, degree)
    du_coeffs_t = differentiate_poly_t_wrt_x(u_coeffs_t, cache.sizes, order=1)

    # Volume term:   0.5 * Σ_e h_e^2 · ∫_E (u_h'' + c0 x^alpha)^2 dx.
    R2 = _residual_R2_per_element_quadrature(
        cache.knots, int(degree), u_full, beta_t)
    volume = 0.5 * jnp.sum((cache.sizes * cache.sizes) * R2)

    # Neumann term at x = 1 (unchanged).
    u_prime_1 = jnp.sum(du_coeffs_t[-1, :])
    neumann = 0.5 * cache.sizes[-1] * (u_prime_1 - beta_t) ** 2
    total = volume + neumann
    return ResidualLossTerms(
        total_loss=total,
        volume_loss=volume,
        neumann_loss=neumann,
    )


residual_loss_terms_from_state = loss_terms_from_state


@functools.partial(jax.jit, static_argnames=("degree", "quadrature_order"))  # FIX: beta traced
def reduced_loss(
    theta: Array,
    degree: int,
    quadrature_order: int,
    h_min: float,
    beta: float = BETA,
) -> Array:
    u_full, cache = solve_state(theta, degree, quadrature_order, h_min, beta)
    return loss_terms_from_state(u_full, cache, degree, beta).total_loss


def compute_validation_loss(
    theta: Array,
    degree: int,
    h_min: float,
    *,
    beta: float = BETA,
    train_loss: float | None = None,
) -> float:
    """Return the exact diagnostic residual loss used for history logging.

    For Problem 1 this quantity is evaluated analytically (monomial integrals),
    so it is independent of any quadrature order.
    """
    if train_loss is not None:
        return float(train_loss)
    return float(jax.device_get(reduced_loss(theta, degree, int(degree) + 1, h_min, beta)))


__all__ = [
    "_I",
    "load_global_analytic_power",
    "residual_element_R2_fully_analytic",
    "oscillation_element_L2sq_analytic_power",
    "estimator_eta2_terms_analytic_power",
    "estimator_eta2_analytic_power",
    "estimator_loss_analytic_power",
    "rule_on_element",
    "stiffness_global_gl",
    "stiffness_global_gl_coo",
    "ResidualLossTerms",
    "PowerLossTerms",
    "integrate_x_power",
    "integrate_poly_square_t",
    "integrate_poly_times_power_t",
    "loss_terms_from_state",
    "residual_loss_terms_from_state",
    "reduced_loss",
    "compute_validation_loss",
]
