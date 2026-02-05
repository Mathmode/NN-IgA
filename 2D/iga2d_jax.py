from __future__ import annotations

"""JAX IGA utilities for 2D tensor-product patches (square experiments).

This module provides:
  - Gauss–Legendre quadrature rules (cached via NumPy)
  - open-clamped knot vector construction from breakpoints
  - fast p=2/p=3 batch basis evaluation on per-element quadrature grids
  - dense 1D mass/stiffness assembly via scatter-add

The 2D experiments keep ``p in {2,3}`` and rely on tensor-product structure;
we therefore implement only these degrees here (matching the original code).
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from bspline_basis import basis_p2_batch, basis_p3_batch
from quadrature import leggauss

Array = jnp.ndarray


def quad_rule(q_order: int, dtype) -> Tuple[Array, Array]:
    """Gauss–Legendre quadrature on [-1,1]."""
    return leggauss(int(q_order), dtype)


# -----------------------------------------------------------------------------
# 1D knots / matrices / evaluation (open clamped)
# -----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p",))
def make_knot_vector_open(breakpoints: Array, p: int) -> Array:
    bp = jnp.asarray(breakpoints).reshape(-1)
    p = int(p)
    if int(bp.shape[0]) < 2:
        raise ValueError("breakpoints must have length>=2")
    a = bp[0]
    b = bp[-1]
    interior = bp[1:-1]
    left = jnp.full((p + 1,), a, dtype=bp.dtype)
    right = jnp.full((p + 1,), b, dtype=bp.dtype)
    return jnp.concatenate([left, interior, right], axis=0)


@functools.partial(jax.jit, static_argnames=("p",))
def greville_abscissae(knots: Array, p: int) -> Array:
    """Greville points for an open knot vector."""
    p = int(p)
    knots = jnp.asarray(knots).reshape(-1)
    n_basis = int(knots.shape[0] - p - 1)
    if n_basis <= 0:
        return jnp.zeros((0,), dtype=knots.dtype)
    i = jnp.arange(n_basis, dtype=jnp.int32)[:, None]
    j = jnp.arange(1, p + 1, dtype=jnp.int32)[None, :]
    return jnp.sum(knots[i + j], axis=1) / jnp.asarray(float(p), dtype=knots.dtype)


@functools.partial(jax.jit, static_argnames=("p", "q_order", "n_der"))
def eval_1d_on_elements(
    breakpoints: Array,
    p: int,
    *,
    q_order: int = 20,
    n_der: int = 2,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Evaluate basis/derivatives on a per-element Gauss grid.

    Returns xq,wq with shape (n_elem,q_order) and N,dN,d2N with shape (n_elem,q_order,p+1).
    """
    bp = jnp.asarray(breakpoints).reshape(-1)
    p = int(p)
    q_order = int(q_order)
    n_der = int(n_der)
    if n_der < 0 or n_der > 2:
        raise ValueError("n_der must be in {0,1,2}")

    n_elem = int(bp.shape[0] - 1)
    if n_elem <= 0:
        raise ValueError("need at least one element")

    knots = make_knot_vector_open(bp, p)
    xi, wi = quad_rule(q_order, bp.dtype)

    a = bp[:-1]
    b = bp[1:]
    h = b - a
    xq = 0.5 * h[:, None] * xi[None, :] + 0.5 * (a + b)[:, None]
    wq = 0.5 * h[:, None] * wi[None, :]

    start = jnp.arange(n_elem, dtype=jnp.int32)
    g = start[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    idx = start[:, None] + jnp.arange(2 * p + 2, dtype=jnp.int32)[None, :]
    U_local = knots[idx]

    if p == 2:
        N, dN, d2N = basis_p2_batch(xq, U_local)
    elif p == 3:
        N, dN, d2N = basis_p3_batch(xq, U_local)
    else:
        raise ValueError("2D JAX routines currently support p in {2,3}")

    if n_der == 0:
        dN = jnp.zeros_like(N)
        d2N = jnp.zeros_like(N)
    elif n_der == 1:
        d2N = jnp.zeros_like(N)

    return knots, xq, wq, h, g, N, dN, d2N


@functools.partial(jax.jit, static_argnames=("p", "q_order"))
def assemble_1d_mats(breakpoints: Array, p: int, *, q_order: int = 20) -> tuple[Array, Array, Array, Array]:
    """Assemble dense 1D mass M, stiffness K, and load r for f=1."""
    p = int(p)
    knots, xq, wq, h, g, N, dN, _ = eval_1d_on_elements(breakpoints, p, q_order=int(q_order), n_der=1)

    n_elem = int(jnp.asarray(breakpoints).shape[0] - 1)
    n_basis = int(n_elem + p)

    M_e = jnp.einsum("eqi,eq,eqj->eij", N, wq, N)
    K_e = jnp.einsum("eqi,eq,eqj->eij", dN, wq, dN)
    r_e = jnp.einsum("eqi,eq->ei", N, wq)

    M = jnp.zeros((n_basis, n_basis), dtype=jnp.asarray(breakpoints).dtype)
    K = jnp.zeros((n_basis, n_basis), dtype=jnp.asarray(breakpoints).dtype)

    M = M.at[g[:, :, None], g[:, None, :]].add(M_e)
    K = K.at[g[:, :, None], g[:, None, :]].add(K_e)

    r = jnp.zeros((n_basis,), dtype=jnp.asarray(breakpoints).dtype)
    r = r.at[g.reshape(-1)].add(r_e.reshape(-1))

    return knots, M, K, r


__all__ = [
    "quad_rule",
    "make_knot_vector_open",
    "greville_abscissae",
    "eval_1d_on_elements",
    "assemble_1d_mats",
]
