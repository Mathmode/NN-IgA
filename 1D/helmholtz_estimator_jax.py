from __future__ import annotations

"""JAX residual estimator for the 1D Helmholtz interface experiment."""

import functools

import jax
import jax.numpy as jnp

from bspline_basis import basis_p2_batch, basis_p3_batch
from helmholtz_iga_jax import quad_rule, spans_two_zone, elem_dof_map, knot_windows
from optimization import Eta2Terms

Array = jnp.ndarray


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_left", "q_order", "bc_neumann", "include_jump"),
)
def estimator_eta2(
    u_coeffs: Array,
    breakpoints: Array,
    knots: Array,
    p: int,
    *,
    xI: float,
    sigma_L: float,
    sigma_R: float,
    alpha_L: float,
    alpha_R: float,
    gN: float,
    n_left: int,
    q_order: int = 20,
    bc_neumann: bool = True,
    include_jump: bool = True,
    eps_side: float = 1e-6,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Compute eta^2 (and components) for the manufactured 2-zone interface problem."""
    p = int(p)
    n_left = int(n_left)
    q_order = int(q_order)
    dtype = breakpoints.dtype

    xI_t = jnp.asarray(xI, dtype=dtype)
    sigL = jnp.asarray(sigma_L, dtype=dtype)
    sigR = jnp.asarray(sigma_R, dtype=dtype)
    alpL = jnp.asarray(alpha_L, dtype=dtype)
    alpR = jnp.asarray(alpha_R, dtype=dtype)
    gN_t = jnp.asarray(gN, dtype=dtype)

    n_elem = int(breakpoints.shape[0] - 1)
    xi, wi = quad_rule(q_order, dtype)

    a = breakpoints[:-1]
    b = breakpoints[1:]
    h = b - a

    xq = 0.5 * h[:, None] * xi[None, :] + 0.5 * (a + b)[:, None]
    wq = 0.5 * h[:, None] * wi[None, :]

    start = spans_two_zone(p, n_elem=n_elem, n_left=n_left)
    g = elem_dof_map(start, p)
    U_local = knot_windows(knots, start, p)
    u_local = u_coeffs[g]  # (n_elem,p+1)

    if p == 2:
        N, dN, d2N = basis_p2_batch(xq, U_local)
    elif p == 3:
        N, dN, d2N = basis_p3_batch(xq, U_local)
    else:
        raise ValueError("JAX Helmholtz estimator currently supports p in {2,3}")

    uh = jnp.einsum("eqi,ei->eq", N, u_local)
    d2uh = jnp.einsum("eqi,ei->eq", d2N, u_local)

    sig_q = jnp.where(xq <= xI_t, sigL, sigR)
    alp_q = jnp.where(xq <= xI_t, alpL, alpR)

    # f == 0 for the paper problem
    R = -sig_q * d2uh + alp_q * uh
    I_e = jnp.sum(wq * (R * R), axis=1)  # (n_elem,)
    eta2_elem = jnp.sum((h * h) * I_e)

    # One-sided derivative samples at element endpoints (used for jump + Neumann)
    eps_t = jnp.asarray(eps_side, dtype=dtype)
    x_left = a + eps_t * h
    x_right = b - eps_t * h

    uL = x_left[:, None]
    uR = x_right[:, None]

    if p == 2:
        _, dN_L, _ = basis_p2_batch(uL, U_local)
        _, dN_R, _ = basis_p2_batch(uR, U_local)
    else:
        _, dN_L, _ = basis_p3_batch(uL, U_local)
        _, dN_R, _ = basis_p3_batch(uR, U_local)

    dN_L = dN_L[:, 0, :]
    dN_R = dN_R[:, 0, :]
    du_left = jnp.sum(dN_L * u_local, axis=1)
    du_right = jnp.sum(dN_R * u_local, axis=1)

    sig_left = jnp.where(x_left <= xI_t, sigL, sigR)
    sig_right = jnp.where(x_right <= xI_t, sigL, sigR)

    eta2_jump = jnp.asarray(0.0, dtype=dtype)
    if bool(include_jump):
        J = sig_right[:-1] * du_right[:-1] - sig_left[1:] * du_left[1:]
        h_face = jnp.minimum(h[:-1], h[1:])
        eta2_jump = jnp.sum(h_face * (J * J))

    eta2_neu = jnp.asarray(0.0, dtype=dtype)
    if bool(bc_neumann):
        resN = sig_right[-1] * du_right[-1] - gN_t
        eta2_neu = h[-1] * (resN * resN)

    eta2_total = eta2_elem + eta2_jump + eta2_neu
    terms = Eta2Terms(
        elem=eta2_elem,
        jump=eta2_jump,
        bc=eta2_neu,
        osc=jnp.asarray(0.0, dtype=dtype),
    )
    return eta2_total, terms


__all__ = ["estimator_eta2"]
