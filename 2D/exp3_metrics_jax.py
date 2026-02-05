from __future__ import annotations

"""Error metrics for the 2D square reaction--diffusion problem (JAX)."""

import functools

import jax
import jax.numpy as jnp

from iga2d_jax import eval_1d_on_elements
from exp3_exact_rd_jax import exact_u, exact_grad

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("p", "q_order"))
def _error_metrics_quadrature(
    u_global: Array,
    breakpoints_x: Array,
    breakpoints_y: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    *,
    eps: float,
    sigma: float,
    q_order: int,
) -> tuple[Array, Array, Array, Array]:
    p = int(p)
    dtype = jnp.asarray(u_global).dtype
    eps_t = jnp.asarray(eps, dtype=dtype)
    sig_t = jnp.asarray(sigma, dtype=dtype)

    nBx = int(knots_x.shape[0] - p - 1)
    nBy = int(knots_y.shape[0] - p - 1)
    C = jnp.asarray(u_global, dtype=dtype).reshape(nBy, nBx)

    _, xqx, wx, _, Gx, Nx, Dx, _ = eval_1d_on_elements(breakpoints_x, p, q_order=int(q_order), n_der=1)
    _, yqy, wy, _, Gy, Ny, Dy, _ = eval_1d_on_elements(breakpoints_y, p, q_order=int(q_order), n_der=1)
    C_blocks = C[Gy[:, :, None, None], Gx[None, None, :, :]]  # (Ny,p+1,Nx,p+1)
    C_blocks = jnp.transpose(C_blocks, (0, 2, 1, 3))

    uh = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, Nx)
    ux = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, Dx)
    uy = jnp.einsum("yai,yxij,xbj->yxab", Dy, C_blocks, Nx)

    NyE, qy = int(yqy.shape[0]), int(yqy.shape[1])
    NxE, qx = int(xqx.shape[0]), int(xqx.shape[1])
    X = jnp.broadcast_to(xqx[None, :, None, :], (NyE, NxE, qy, qx))
    Y = jnp.broadcast_to(yqy[:, None, :, None], (NyE, NxE, qy, qx))
    uex = exact_u(X, Y, eps=eps_t, sigma=sig_t)
    uex_x, uex_y = exact_grad(X, Y, eps=eps_t, sigma=sig_t)

    w2d = wy[:, None, :, None] * wx[None, :, None, :]

    e0 = uh - uex
    ex0 = ux - uex_x
    ey0 = uy - uex_y

    L2_abs2 = jnp.sum(w2d * (e0 * e0))
    H1_abs2 = jnp.sum(w2d * (ex0 * ex0 + ey0 * ey0))
    u_L2_2 = jnp.sum(w2d * (uex * uex))
    u_H1_2 = jnp.sum(w2d * (uex_x * uex_x + uex_y * uex_y))

    return L2_abs2, H1_abs2, u_L2_2, u_H1_2


def compute_error_metrics_exp3(
    u_global: Array,
    breakpoints_x: Array,
    breakpoints_y: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    *,
    q_order: int = 20,
    eps: float = 1e-2,
    sigma: float = 1.0,
) -> dict:
    """Compute (L2/H1) abs+rel errors and mesh diagnostics as Python floats."""
    L2_abs2, H1_abs2, u_L2_2, u_H1_2 = _error_metrics_quadrature(
        u_global,
        breakpoints_x,
        breakpoints_y,
        knots_x,
        knots_y,
        int(p),
        eps=float(eps),
        sigma=float(sigma),
        q_order=int(q_order),
    )

    L2_abs2 = float(jax.device_get(L2_abs2))
    H1_abs2 = float(jax.device_get(H1_abs2))
    u_L2_2 = float(jax.device_get(u_L2_2))
    u_H1_2 = float(jax.device_get(u_H1_2))

    L2_abs = float(jnp.sqrt(L2_abs2))
    H1_abs = float(jnp.sqrt(H1_abs2))
    L2_rel = float(jnp.sqrt(L2_abs2 / (u_L2_2 + 1e-30)))
    H1_rel = float(jnp.sqrt(H1_abs2 / (u_H1_2 + 1e-30)))

    bx = jnp.asarray(breakpoints_x)
    by = jnp.asarray(breakpoints_y)
    hx = bx[1:] - bx[:-1]
    hy = by[1:] - by[:-1]
    h_min = float(jnp.minimum(jnp.min(hx), jnp.min(hy)))
    h_max = float(jnp.maximum(jnp.max(hx), jnp.max(hy)))

    return {
        "L2_abs": L2_abs,
        "L2_rel": L2_rel,
        "H1_abs": H1_abs,
        "H1_rel": H1_rel,
        "h_min": h_min,
        "h_max": h_max,
        "h_ratio": float(h_max / max(h_min, 1e-30)),
    }


__all__ = ["compute_error_metrics_exp3"]
