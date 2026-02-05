from __future__ import annotations

"""Analytic error metrics and boundary flux for Experiment 1 (power solution)."""

import functools

import jax
import jax.numpy as jnp

from bspline_basis import bspline_basis_local
from power_quadrature import _I

Array = jnp.ndarray

_TINY = 1e-300


def _piece_intervals(knots: Array, p: int) -> Array:
    p = int(p)
    P = knots[p:-p]
    return jnp.stack([P[:-1], P[1:]], axis=1)


@functools.partial(jax.jit, static_argnames=("p",))
def _poly_coeffs_in_x_on_span(knots: Array, p: int, a: float, b: float) -> Array:
    """Reconstruct polynomial coefficients of the active B-splines on one element.

    Returns C (p+1,p+1) where N_j(x) = sum_k C[k,j] x^k on [a,b].
    """
    p = int(p)
    h = b - a
    r = jnp.arange(p + 1, dtype=knots.dtype)
    t = 0.5 * (1.0 + jnp.cos((2.0 * r + 1.0) * jnp.pi / (2.0 * (p + 1.0))))
    x = a + h * t
    N, _, _, _ = bspline_basis_local(x, knots, p)
    V = jnp.vander(x, N=p + 1, increasing=True)
    C, *_ = jnp.linalg.lstsq(V, N, rcond=None)
    return C


@functools.partial(jax.jit, static_argnames=("p",))
def _uh_poly_coeffs_on_element(knots: Array, p: int, a: float, b: float, Uloc: Array) -> Array:
    C = _poly_coeffs_in_x_on_span(knots, int(p), a, b)
    return C @ Uloc


@functools.partial(jax.jit, static_argnames=("p",))
def _duh_poly_coeffs_on_element(knots: Array, p: int, a: float, b: float, Uloc: Array) -> Array:
    p = int(p)
    acoef = _uh_poly_coeffs_on_element(knots, p, a, b, Uloc)
    if p <= 0:
        return acoef[:0]
    idx = jnp.arange(p, dtype=acoef.dtype)  # 0..p-1
    return (idx + 1.0) * acoef[1:]


@functools.partial(jax.jit, static_argnames=("p", "use_seminorm"))
def _element_error_terms_power(
    knots: Array,
    p: int,
    a: float,
    b: float,
    Uloc: Array,
    beta: float,
    *,
    use_seminorm: bool,
) -> tuple[Array, Array, Array, Array]:
    p = int(p)
    dtype = knots.dtype
    beta = jnp.asarray(beta, dtype=dtype)

    acoef = _uh_poly_coeffs_on_element(knots, p, a, b, Uloc)

    # L2 term
    idx_u = jnp.arange(p + 1, dtype=dtype)
    exp_u = idx_u[:, None] + idx_u[None, :]
    I_poly2_u = jnp.sum((acoef[:, None] * acoef[None, :]) * _I(a, b, exp_u))
    I_cross_u = -2.0 * jnp.sum(acoef * _I(a, b, idx_u + beta))
    I_exact2_u = _I(a, b, 2.0 * beta)
    L2num = jnp.maximum(I_poly2_u + I_cross_u + I_exact2_u, jnp.asarray(0.0, dtype=dtype))
    L2den = jnp.maximum(I_exact2_u, jnp.asarray(0.0, dtype=dtype))

    # H1 seminorm
    if p <= 0:
        I_poly2_du = jnp.asarray(0.0, dtype=dtype)
        I_cross_du = jnp.asarray(0.0, dtype=dtype)
    else:
        dcoef = _duh_poly_coeffs_on_element(knots, p, a, b, Uloc)
        idx_du = jnp.arange(p, dtype=dtype)
        exp_du = idx_du[:, None] + idx_du[None, :]
        I_poly2_du = jnp.sum((dcoef[:, None] * dcoef[None, :]) * _I(a, b, exp_du))
        I_cross_du = -2.0 * beta * jnp.sum(dcoef * _I(a, b, idx_du + beta - 1.0))

    I_exact2_du = (beta * beta) * _I(a, b, 2.0 * beta - 2.0)
    H1num = jnp.maximum(I_poly2_du + I_cross_du + I_exact2_du, jnp.asarray(0.0, dtype=dtype))
    H1den = jnp.maximum(I_exact2_du, jnp.asarray(0.0, dtype=dtype))

    if not use_seminorm:
        H1num = H1num + L2num
        H1den = H1den + L2den

    return H1num, H1den, L2num, L2den


@functools.partial(jax.jit, static_argnames=("p", "relative", "use_seminorm"))
def compute_errors_power_analytic(
    knots: Array,
    p: int,
    U: Array,
    *,
    beta: float,
    relative: bool = True,
    use_seminorm: bool = True,
) -> tuple[Array, Array]:
    """Compute H1 and L2 errors analytically for u(x)=x^beta."""
    p = int(p)
    dtype = knots.dtype
    beta = jnp.asarray(beta, dtype=dtype)

    n_elem = int(knots.shape[0] - 2 * p - 1)
    e = jnp.arange(n_elem, dtype=jnp.int32)

    def elem_terms(ei):
        a = knots[p + ei]
        b = knots[p + ei + 1]
        cols = ei + jnp.arange(p + 1)
        Uloc = U[cols]
        return _element_error_terms_power(knots, p, a, b, Uloc, beta, use_seminorm=use_seminorm)

    H1n, H1d, L2n, L2d = jax.vmap(elem_terms)(e)
    H1n = jnp.sum(H1n)
    H1d = jnp.sum(H1d)
    L2n = jnp.sum(L2n)
    L2d = jnp.sum(L2d)

    H1 = jnp.sqrt(H1n + _TINY)
    L2 = jnp.sqrt(L2n + _TINY)

    if relative:
        H1 = H1 / jnp.sqrt(H1d + _TINY)
        L2 = L2 / jnp.sqrt(L2d + _TINY)

    return H1, L2


@functools.partial(jax.jit, static_argnames=("p",))
def _right_flux(knots: Array, p: int, U: Array) -> Array:
    """Compute u_h'(1) from the polynomial representation on the last element."""
    p = int(p)
    dtype = knots.dtype

    a_last = knots[-p - 2]
    b_last = knots[-p - 1]

    n_elem = int(knots.shape[0] - 2 * p - 1)
    e_last = n_elem - 1

    cols_last = e_last + jnp.arange(p + 1)
    Uloc_last = U[cols_last]

    acoef_last = _uh_poly_coeffs_on_element(knots, p, a_last, b_last, Uloc_last)

    if p <= 0:
        return jnp.asarray(0.0, dtype=dtype)

    k = jnp.arange(1, p + 1, dtype=dtype)
    return jnp.sum(k * acoef_last[1:] * jnp.power(b_last, k - 1.0))


def right_flux(knots: Array, p: int, U: Array) -> float:
    return float(_right_flux(jnp.asarray(knots), int(p), jnp.asarray(U)))


__all__ = [
    "compute_errors_power_analytic",
    "right_flux",
]
