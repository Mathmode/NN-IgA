from __future__ import annotations

"""Analytic integrals, assembly quadrature, and residual estimator for Experiment 1.

The manufactured solution is u(x)=x^beta on [0,1], with RHS f(x)=beta(1-beta)x^{beta-2}
(and sigma=1, alpha=0). Many quantities (load, estimator terms, exact errors) are computed
analytically via monomial integrals.

Note: stiffness matrix is assembled with element-wise Gauss-Legendre quadrature (robust
and fast), while the load and estimator pieces use analytic integrals.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from quadrature import (
    as_dtype as _as_dtype,
    monomial_integral as _I,
    rule_gl_on_element,
    rule_gl_on_elements,
)
from optimization import Eta2Terms, residual_loss_from_eta2
from bspline_basis import bspline_basis_local

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


@functools.partial(jax.jit, static_argnames=("p",))
def residual_element_R2_fully_analytic(
    knots: Array,
    p: int,
    a: float,
    b: float,
    Uloc: Array,
    beta: float,
) -> Array:
    """Compute ∫_E (u_h'' + f)^2 for the power RHS (sigma=1, alpha=0)."""
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
    include_osc: bool = True,
) -> tuple[Array, Eta2Terms]:
    """Compute eta^2 for the power problem in 1D.

    eta^2 = sum_E h_E^2 ||R_E||^2 + h_E^2 osc^2 + h_last (u_h'(1)-beta)^2.

    For globally C^1 splines and continuous sigma, interior jumps vanish.
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

    if p < 2:
        R2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
    else:
        m = jnp.arange(p - 1, dtype=dtype)  # 0..p-2
        factor = (m + 1.0) * (m + 2.0)
        B = factor[None, :, None] * C[:, 2:, :]
        bcoef = jnp.einsum("emj,ej->em", B, Uloc)

        M = m[:, None] + m[None, :]
        I_M = _I(a[:, None, None], b[:, None, None], M[None, :, :])
        I_d2 = jnp.einsum("em,en,emn->e", bcoef, bcoef, I_M)

        I_ma = _I(a[:, None], b[:, None], m[None, :] + _as_dtype(alpha, dtype))
        I_cross = 2.0 * c0 * jnp.sum(bcoef * I_ma, axis=1)
        I_f2 = (c0 * c0) * _I(a, b, _as_dtype(2.0, dtype) * alpha)
        R2 = I_d2 + I_cross + I_f2

    eta2_elem = jnp.sum((he * he) * R2)
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
    eta2_bc = h_last * (flux_err * flux_err)

    eta2_total = eta2_elem + eta2_osc + eta2_bc
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
    include_osc: bool = True,
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
    include_osc: bool = True,
) -> Array:
    """Training objective: L = 0.5 * eta^2."""
    eta2 = estimator_eta2_analytic_power(knots, p, U, beta=beta, include_osc=include_osc)
    return residual_loss_from_eta2(eta2)


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
]
