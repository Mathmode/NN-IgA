from __future__ import annotations

"""JAX IGA assembly + solve for the 1D Helmholtz interface experiment."""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from quadrature import leggauss
from bspline_basis import basis_p2_batch, basis_p3_batch

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Quadrature (cached via NumPy, then cast to JAX arrays)
# -----------------------------------------------------------------------------


def quad_rule(q_order: int, dtype) -> Tuple[Array, Array]:
    return leggauss(int(q_order), dtype)


# -----------------------------------------------------------------------------
# Knot vector and span mapping (two-zone, C0 interface)
# -----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_left"))
def make_knot_vector_two_zone(breakpoints: Array, p: int, n_left: int) -> Array:
    """Open knot vector with multiplicity p at the interface breakpoint."""
    p = int(p)
    n_left = int(n_left)

    x0 = breakpoints[0]
    xI = breakpoints[n_left]
    x1 = breakpoints[-1]

    interior_left = breakpoints[1:n_left]
    interior_right = breakpoints[n_left + 1 : -1]

    left = jnp.full((p + 1,), x0, dtype=breakpoints.dtype)
    mid = jnp.full((p,), xI, dtype=breakpoints.dtype)
    right = jnp.full((p + 1,), x1, dtype=breakpoints.dtype)

    return jnp.concatenate([left, interior_left, mid, interior_right, right], axis=0)


@functools.partial(jax.jit, static_argnames=("p", "n_left", "n_elem"))
def spans_two_zone(p: int, n_elem: int, n_left: int) -> Array:
    """Return start indices (span - p) for each physical element (shape (n_elem,))."""
    p = int(p)
    n_left = int(n_left)
    e = jnp.arange(int(n_elem), dtype=jnp.int32)
    shift = jnp.where(e < n_left, jnp.zeros_like(e), jnp.full_like(e, p - 1))
    return e + shift


@functools.partial(jax.jit, static_argnames=("p",))
def elem_dof_map(start: Array, p: int) -> Array:
    p = int(p)
    return start[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]


@functools.partial(jax.jit, static_argnames=("p",))
def knot_windows(knots: Array, start: Array, p: int) -> Array:
    p = int(p)
    idx = start[:, None] + jnp.arange(2 * p + 2, dtype=jnp.int32)[None, :]
    return knots[idx]


# -----------------------------------------------------------------------------
# System assembly + solve
# -----------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=("p", "n_left", "q_order"))
def assemble_system(
    breakpoints: Array,
    p: int,
    *,
    xI: float,
    sigma_L: float,
    sigma_R: float,
    alpha_L: float,
    alpha_R: float,
    n_left: int,
    q_order: int = 20,
) -> tuple[Array, Array, Array]:
    """Assemble dense Galerkin system Au=b for -(sigma u')' + alpha u = f (here f=0)."""
    p = int(p)
    n_left = int(n_left)
    q_order = int(q_order)

    dtype = breakpoints.dtype
    xI_t = jnp.asarray(xI, dtype=dtype)

    n_elem = int(breakpoints.shape[0] - 1)
    knots = make_knot_vector_two_zone(breakpoints, p, n_left)
    n_basis = int(n_elem + 2 * p - 1)

    xi, wi = quad_rule(q_order, dtype)

    a = breakpoints[:-1]
    b = breakpoints[1:]
    h = b - a

    xq = 0.5 * h[:, None] * xi[None, :] + 0.5 * (a + b)[:, None]  # (n_elem,q)
    wq = 0.5 * h[:, None] * wi[None, :]  # (n_elem,q)

    start = spans_two_zone(p, n_elem=n_elem, n_left=n_left)
    g = elem_dof_map(start, p)  # (n_elem,p+1)
    U_local = knot_windows(knots, start, p)  # (n_elem,2p+2)

    if p == 2:
        N, dN, _ = basis_p2_batch(xq, U_local)
    elif p == 3:
        N, dN, _ = basis_p3_batch(xq, U_local)
    else:
        raise ValueError("JAX Helmholtz implementation currently supports p in {2,3}")

    sig_q = jnp.where(xq <= xI_t, jnp.asarray(sigma_L, dtype=dtype), jnp.asarray(sigma_R, dtype=dtype))
    alp_q = jnp.where(xq <= xI_t, jnp.asarray(alpha_L, dtype=dtype), jnp.asarray(alpha_R, dtype=dtype))

    w_sig = wq * sig_q
    w_alp = wq * alp_q

    A_stiff = jnp.einsum("eqi,eq,eqj->eij", dN, w_sig, dN)
    A_mass = jnp.einsum("eqi,eq,eqj->eij", N, w_alp, N)
    A_e = A_stiff + A_mass

    # f == 0 for the paper problem
    b_e = jnp.zeros((n_elem, p + 1), dtype=dtype)

    A = jnp.zeros((n_basis, n_basis), dtype=dtype)
    A = A.at[g[:, :, None], g[:, None, :]].add(A_e)

    rhs = jnp.zeros((n_basis,), dtype=dtype)
    rhs = rhs.at[g.reshape(-1)].add(b_e.reshape(-1))

    return knots, A, rhs


def solve_dirichlet_left(A: Array, b: Array, *, u0: float) -> Array:
    """Solve Au=b with strong Dirichlet u[0]=u0 via reduction (batched)."""
    dtype = A.dtype
    u0_t = jnp.asarray(u0, dtype=dtype)

    A_FF = A[..., 1:, 1:]
    rhs = b[..., 1:] - A[..., 1:, 0] * u0_t

    u_free = jnp.linalg.solve(A_FF, rhs)
    u = jnp.zeros_like(b)
    u = u.at[..., 0].set(u0_t)
    u = u.at[..., 1:].set(u_free)
    return u


def solve_dirichlet_both(A: Array, b: Array, *, u0: float, u1: float) -> Array:
    """Solve Au=b with strong Dirichlet u[0]=u0, u[-1]=u1 via reduction (batched)."""
    dtype = A.dtype
    u0_t = jnp.asarray(u0, dtype=dtype)
    u1_t = jnp.asarray(u1, dtype=dtype)

    A_FF = A[..., 1:-1, 1:-1]
    rhs = b[..., 1:-1] - A[..., 1:-1, 0] * u0_t - A[..., 1:-1, -1] * u1_t

    u_free = jnp.linalg.solve(A_FF, rhs)
    u = jnp.zeros_like(b)
    u = u.at[..., 0].set(u0_t)
    u = u.at[..., -1].set(u1_t)
    u = u.at[..., 1:-1].set(u_free)
    return u


@functools.partial(jax.jit, static_argnames=("p", "n_left", "q_order", "bc_neumann"))
def solve_helmholtz(
    breakpoints: Array,
    p: int,
    *,
    xI: float,
    sigma_L: float,
    sigma_R: float,
    alpha_L: float,
    alpha_R: float,
    n_left: int,
    q_order: int = 20,
    bc_neumann: bool = True,
    u0: float = 0.0,
    u1: float = 0.0,
    gN: float = 0.0,
) -> tuple[Array, Array]:
    """Solve the 2-zone Helmholtz interface problem for the manufactured paper setup."""
    knots, A, rhs = assemble_system(
        breakpoints,
        int(p),
        xI=xI,
        sigma_L=sigma_L,
        sigma_R=sigma_R,
        alpha_L=alpha_L,
        alpha_R=alpha_R,
        n_left=int(n_left),
        q_order=int(q_order),
    )

    if bool(bc_neumann):
        rhs = rhs.at[-1].add(jnp.asarray(gN, dtype=rhs.dtype))
        u = solve_dirichlet_left(A, rhs, u0=u0)
        return u, knots

    u = solve_dirichlet_both(A, rhs, u0=u0, u1=u1)
    return u, knots


__all__ = [
    "quad_rule",
    "make_knot_vector_two_zone",
    "spans_two_zone",
    "elem_dof_map",
    "knot_windows",
    "assemble_system",
    "solve_helmholtz",
    "solve_dirichlet_left",
    "solve_dirichlet_both",
]
