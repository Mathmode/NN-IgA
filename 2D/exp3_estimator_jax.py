from __future__ import annotations

"""Residual estimator for the 2D square reaction--diffusion problem (JAX).

We drive r-adaptivity by minimizing:
    L(theta) = 0.5 * eta(theta)^2

Strong residual on Ω:
    R = -eps * (u_xx + u_yy) + sigma * u

Estimator (volume-only):
    eta^2 = sum_K h_K^2 ||R||^2_{L2(K)}.

Implementation:
  - Element-local tensor-product evaluation (O(N^2) elements)
  - No global (n_pts x n_basis) dense evaluation matrices
  - JIT-friendly with fixed (N,p,q_order) shapes
"""

import functools

import jax
import jax.numpy as jnp

from iga2d_jax import eval_1d_on_elements

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("p", "q_order", "use_h_max"))
def estimator_eta2_exp3(
    u_global: Array,
    breakpoints_x: Array,
    breakpoints_y: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    *,
    eps: float = 1e-2,
    sigma: float = 1.0,
    q_order: int = 12,
    use_h_max: bool = True,
) -> tuple[Array, tuple[Array, Array]]:
    """Return (eta2_total, (eta2_vol, eta2_total)) as JAX scalars."""
    p = int(p)
    dtype = jnp.asarray(u_global).dtype
    eps_t = jnp.asarray(eps, dtype=dtype)
    sig_t = jnp.asarray(sigma, dtype=dtype)

    # Coefficient grid C (row-major: iy, ix)
    nBx = int(knots_x.shape[0] - p - 1)
    nBy = int(knots_y.shape[0] - p - 1)
    C = jnp.asarray(u_global, dtype=dtype).reshape(nBy, nBx)

    _, _, wx, hx, Gx, Nx, _, D2x = eval_1d_on_elements(breakpoints_x, p, q_order=int(q_order), n_der=2)
    _, _, wy, hy, Gy, Ny, _, D2y = eval_1d_on_elements(breakpoints_y, p, q_order=int(q_order), n_der=2)

    # Gather local coefficient blocks per element (ey,ex): (Ny,Nx,p+1,p+1)
    C_blocks = C[Gy[:, :, None, None], Gx[None, None, :, :]]  # (Ny,p+1,Nx,p+1)
    C_blocks = jnp.transpose(C_blocks, (0, 2, 1, 3))

    u = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, Nx)
    uxx = jnp.einsum("yai,yxij,xbj->yxab", Ny, C_blocks, D2x)
    uyy = jnp.einsum("yai,yxij,xbj->yxab", D2y, C_blocks, Nx)

    R = (-eps_t) * (uxx + uyy) + sig_t * u

    # Quadrature weights on each tensor element
    w2d = wy[:, None, :, None] * wx[None, :, None, :]  # (Ny,Nx,qy,qx)

    if bool(use_h_max):
        hK = jnp.maximum(hy[:, None], hx[None, :])
    else:
        hK = jnp.sqrt(hy[:, None] * hy[:, None] + hx[None, :] * hx[None, :])

    eta2_vol = jnp.sum(w2d * (hK[:, :, None, None] ** 2) * (R * R))
    eta2_total = eta2_vol
    return eta2_total, (eta2_vol, eta2_total)


__all__ = ["estimator_eta2_exp3"]
