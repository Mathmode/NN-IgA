from __future__ import annotations

"""JAX error metrics for the 1D Helmholtz interface experiment."""

import functools

import jax
import jax.numpy as jnp

from bspline_basis import basis_p2_batch, basis_p3_batch
from helmholtz_iga_jax import quad_rule, spans_two_zone, elem_dof_map, knot_windows

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("p", "n_left", "q_order"))
def _metrics_kernel(
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
    k_L: float,
    k_R: float,
    A_L: float,
    A_R: float,
    B_R: float,
    gN: float,
    n_left: int,
    q_order: int = 60,
    eps_side: float = 1e-6,
) -> tuple[Array, ...]:
    p = int(p)
    n_left = int(n_left)
    q_order = int(q_order)
    dtype = breakpoints.dtype

    xI_t = jnp.asarray(xI, dtype=dtype)
    sigL = jnp.asarray(sigma_L, dtype=dtype)
    sigR = jnp.asarray(sigma_R, dtype=dtype)
    alpL = jnp.asarray(alpha_L, dtype=dtype)
    alpR = jnp.asarray(alpha_R, dtype=dtype)

    kL = jnp.asarray(k_L, dtype=dtype)
    kR = jnp.asarray(k_R, dtype=dtype)
    AL = jnp.asarray(A_L, dtype=dtype)
    AR = jnp.asarray(A_R, dtype=dtype)
    BR = jnp.asarray(B_R, dtype=dtype)

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
    u_local = u_coeffs[g]

    if p == 2:
        N, dN, d2N = basis_p2_batch(xq, U_local)
    elif p == 3:
        N, dN, d2N = basis_p3_batch(xq, U_local)
    else:
        raise ValueError("JAX Helmholtz metrics currently supports p in {2,3}")

    uh = jnp.einsum("eqi,ei->eq", N, u_local)
    duh = jnp.einsum("eqi,ei->eq", dN, u_local)
    d2uh = jnp.einsum("eqi,ei->eq", d2N, u_local)

    sig = jnp.where(xq <= xI_t, sigL, sigR)
    alp = jnp.where(xq <= xI_t, alpL, alpR)

    u_ex = jnp.where(xq <= xI_t, AL * jnp.sin(kL * xq), AR * jnp.sin(kR * xq) + BR * jnp.cos(kR * xq))
    du_ex = jnp.where(
        xq <= xI_t,
        (AL * kL) * jnp.cos(kL * xq),
        (AR * kR) * jnp.cos(kR * xq) - (BR * kR) * jnp.sin(kR * xq),
    )

    e_u = u_ex - uh
    e_du = du_ex - duh

    I_L2_e = jnp.sum(wq * (e_u * e_u), axis=1)
    I_H1s_e = jnp.sum(wq * (e_du * e_du), axis=1)

    I_L2_u = jnp.sum(wq * (u_ex * u_ex), axis=1)
    I_H1s_u = jnp.sum(wq * (du_ex * du_ex), axis=1)

    elem_idx = jnp.arange(n_elem, dtype=jnp.int32)
    mask_left = elem_idx < n_left

    I_L2_left = jnp.sum(jnp.where(mask_left, I_L2_e, jnp.asarray(0.0, dtype=dtype)))
    I_L2_right = jnp.sum(jnp.where(~mask_left, I_L2_e, jnp.asarray(0.0, dtype=dtype)))
    I_H1s_left = jnp.sum(jnp.where(mask_left, I_H1s_e, jnp.asarray(0.0, dtype=dtype)))
    I_H1s_right = jnp.sum(jnp.where(~mask_left, I_H1s_e, jnp.asarray(0.0, dtype=dtype)))

    I_L2_tot = jnp.sum(I_L2_e)
    I_H1s_tot = jnp.sum(I_H1s_e)
    I_L2_u_tot = jnp.sum(I_L2_u)
    I_H1s_u_tot = jnp.sum(I_H1s_u)

    k2 = jnp.abs(alp)
    I_dG = jnp.sum(wq * (sig * (e_du * e_du) + k2 * (e_u * e_u)))

    R = -sig * d2uh + alp * uh
    I_res = jnp.sum(wq * (R * R))

    L2_abs = jnp.sqrt(I_L2_tot)
    L2_rel = L2_abs / jnp.sqrt(I_L2_u_tot + jnp.asarray(1e-30, dtype=dtype))

    H1_semi_abs = jnp.sqrt(I_H1s_tot)
    H1_semi_rel = H1_semi_abs / jnp.sqrt(I_H1s_u_tot + jnp.asarray(1e-30, dtype=dtype))

    H1_abs = jnp.sqrt(I_L2_tot + I_H1s_tot)
    H1_rel = H1_abs / jnp.sqrt(I_L2_u_tot + I_H1s_u_tot + jnp.asarray(1e-30, dtype=dtype))

    dG_err = jnp.sqrt(I_dG)
    L2_res = jnp.sqrt(I_res)

    L2_left_abs = jnp.sqrt(I_L2_left)
    L2_right_abs = jnp.sqrt(I_L2_right)
    H1_semi_left_abs = jnp.sqrt(I_H1s_left)
    H1_semi_right_abs = jnp.sqrt(I_H1s_right)

    h_min = jnp.min(h)
    h_max = jnp.max(h)
    h_ratio = h_max / (h_min + jnp.asarray(1e-30, dtype=dtype))

    # Flux error at x=1: |sigma(1) u_h'(1) - gN|
    eps_t = jnp.asarray(eps_side, dtype=dtype)
    h_last = h[-1]
    x_end = breakpoints[-1] - eps_t * h_last

    start_last = start[-1:]
    Ue = knot_windows(knots, start_last, p)
    ge = elem_dof_map(start_last, p)
    ce = u_coeffs[ge][0]

    u_ = x_end.reshape(1, 1)
    if p == 2:
        _, dN_end, _ = basis_p2_batch(u_, Ue)
    else:
        _, dN_end, _ = basis_p3_batch(u_, Ue)

    dN_end = dN_end[0, 0, :]
    uh_prime_1 = jnp.dot(dN_end, ce)
    sig_end = jnp.where(x_end <= xI_t, sigL, sigR)
    flux_err = jnp.abs(sig_end * uh_prime_1 - gN_t)

    return (
        L2_abs,
        L2_rel,
        H1_semi_abs,
        H1_semi_rel,
        H1_abs,
        H1_rel,
        L2_left_abs,
        L2_right_abs,
        H1_semi_left_abs,
        H1_semi_right_abs,
        dG_err,
        L2_res,
        flux_err,
        h_min,
        h_max,
        h_ratio,
    )


def compute_error_metrics(
    *,
    u_coeffs: Array,
    breakpoints: Array,
    knots: Array,
    p: int,
    xI: float,
    sigma_L: float,
    sigma_R: float,
    alpha_L: float,
    alpha_R: float,
    k_L: float,
    k_R: float,
    A_L: float,
    A_R: float,
    B_R: float,
    gN: float,
    n_left: int,
    q_order: int = 60,
    eps_side: float = 1e-6,
) -> dict[str, float]:
    out = _metrics_kernel(
        u_coeffs,
        breakpoints,
        knots,
        int(p),
        xI=float(xI),
        sigma_L=float(sigma_L),
        sigma_R=float(sigma_R),
        alpha_L=float(alpha_L),
        alpha_R=float(alpha_R),
        k_L=float(k_L),
        k_R=float(k_R),
        A_L=float(A_L),
        A_R=float(A_R),
        B_R=float(B_R),
        gN=float(gN),
        n_left=int(n_left),
        q_order=int(q_order),
        eps_side=float(eps_side),
    )

    (
        L2_abs,
        L2_rel,
        H1_semi_abs,
        H1_semi_rel,
        H1_abs,
        H1_rel,
        L2_left_abs,
        L2_right_abs,
        H1_semi_left_abs,
        H1_semi_right_abs,
        dG_err,
        L2_res,
        flux_err,
        h_min,
        h_max,
        h_ratio,
    ) = out

    return {
        "L2_abs": float(L2_abs),
        "L2_rel": float(L2_rel),
        "H1_semi_abs": float(H1_semi_abs),
        "H1_semi_rel": float(H1_semi_rel),
        "H1_abs": float(H1_abs),
        "H1_rel": float(H1_rel),
        "L2_left_abs": float(L2_left_abs),
        "L2_right_abs": float(L2_right_abs),
        "H1_semi_left_abs": float(H1_semi_left_abs),
        "H1_semi_right_abs": float(H1_semi_right_abs),
        "dG_err": float(dG_err),
        "L2_res": float(L2_res),
        "flux_err": float(flux_err),
        "h_min": float(h_min),
        "h_max": float(h_max),
        "h_ratio": float(h_ratio),
    }


__all__ = ["compute_error_metrics"]
