from __future__ import annotations

"""Single-patch solver for a unit-square reaction--diffusion problem (JAX).

We solve on Ω=(0,1)^2:
    -eps * Δu + sigma * u = 0

with strong Dirichlet data:
    u(0,y) = sin(pi y),  u(1,y) = 0,  u(x,0)=u(x,1)=0.

Implementation notes
--------------------
* No dense 2D matrix is formed (scales to N>32 per axis).
* The interior problem after Dirichlet lifting is solved via a tensor-product
  generalized eigen transform (two 1D eigen problems).
* Fully differentiable in JAX (only uses cholesky/eigh/solve/contract).
"""

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from iga2d_jax import assemble_1d_mats, eval_1d_on_elements
from exp3_exact_rd_jax import g_in

Array = jnp.ndarray


@dataclass(frozen=True)
class SquareSystem:
    breakpoints_x: Array
    breakpoints_y: Array
    knots_x: Array
    knots_y: Array
    p: int
    eps: float
    sigma: float
    nBx: int
    nBy: int


@functools.partial(jax.jit, static_argnames=("p", "q_order_bc"))
def _project_inflow_coeffs(breakpoints_y: Array, p: int, *, q_order_bc: int = 40) -> Array:
    """Compute coefficients on the inflow edge (x=0) so u(0,y)=sin(pi y)."""
    _, My, _, _ = assemble_1d_mats(breakpoints_y, int(p), q_order=int(q_order_bc))
    _, xq, wq, _, g, N, _, _ = eval_1d_on_elements(breakpoints_y, int(p), q_order=int(q_order_bc), n_der=0)

    gy = g_in(xq)
    r_e = jnp.einsum("eqi,eq->ei", N, wq * gy)
    r_g = jnp.zeros((int(My.shape[0]),), dtype=breakpoints_y.dtype)
    r_g = r_g.at[g.reshape(-1)].add(r_e.reshape(-1))

    return jnp.linalg.solve(My, r_g)


@functools.partial(jax.jit, static_argnames=("p", "q_order", "q_order_bc"))
def solve_square_reactdiff_coeffs(
    breakpoints_x: Array,
    breakpoints_y: Array,
    p: int,
    *,
    eps: float = 1e-2,
    sigma: float = 1.0,
    q_order: int = 20,
    q_order_bc: int = 40,
) -> tuple[Array, Array, Array]:
    """Solve and return (u_global, knots_x, knots_y).

    This function is JIT-friendly and returns only arrays. Use
    ``solve_square_reactdiff`` for a convenience wrapper that also returns a
    small Python dataclass with metadata.
    """
    p = int(p)
    dtype = jnp.asarray(breakpoints_x).dtype
    eps_t = jnp.asarray(eps, dtype=dtype)
    sig_t = jnp.asarray(sigma, dtype=dtype)
    alpha = sig_t / eps_t

    knots_x, Mx, Kx, _ = assemble_1d_mats(breakpoints_x, p, q_order=int(q_order))
    knots_y, My, Ky, _ = assemble_1d_mats(breakpoints_y, p, q_order=int(q_order))
    nBx = int(Mx.shape[0])
    nBy = int(My.shape[0])

    # Dirichlet lifting uD (non-zero only at x=0 column)
    c_in = _project_inflow_coeffs(breakpoints_y, p, q_order_bc=int(q_order_bc))
    C_D = jnp.zeros((nBy, nBx), dtype=dtype).at[:, 0].set(c_in)

    # Interior problem for w with homogeneous Dirichlet:
    #   KyI W MxI + MyI W (KxI + alpha MxI) = -[Ky C_D Mx + My C_D (Kx+alpha Mx)]_I
    Ax_full = Kx + alpha * Mx
    F_full = -(Ky @ C_D @ Mx + My @ C_D @ Ax_full)
    F = F_full[1:-1, 1:-1]

    MyI = My[1:-1, 1:-1]
    KyI = Ky[1:-1, 1:-1]
    MxI = Mx[1:-1, 1:-1]
    KxI = Kx[1:-1, 1:-1]
    AxI = KxI + alpha * MxI

    # Generalized eigen in y: KyI v = MyI v lam_y
    Ly = jnp.linalg.cholesky(MyI)
    tmp = solve_triangular(Ly, KyI, lower=True)
    Ky_hat = solve_triangular(Ly, tmp.T, lower=True).T
    Ky_hat = 0.5 * (Ky_hat + Ky_hat.T)
    lam_y, Qy = jnp.linalg.eigh(Ky_hat)
    V = solve_triangular(Ly.T, Qy, lower=False)  # My-orthonormal eigenvectors

    # Generalized eigen in x: (KxI + alpha MxI) w = MxI w lam_x
    Lx = jnp.linalg.cholesky(MxI)
    tmp = solve_triangular(Lx, AxI, lower=True)
    Ax_hat = solve_triangular(Lx, tmp.T, lower=True).T
    Ax_hat = 0.5 * (Ax_hat + Ax_hat.T)
    lam_x, Qx = jnp.linalg.eigh(Ax_hat)
    W = solve_triangular(Lx.T, Qx, lower=False)

    # Modal solve: (lam_y[i] + lam_x[j]) X[i,j] = (V^T F W)[i,j]
    F_tilde = V.T @ F @ W
    denom = lam_y[:, None] + lam_x[None, :]
    X = F_tilde / denom
    W_int = V @ X @ W.T

    C_w = jnp.zeros((nBy, nBx), dtype=dtype).at[1:-1, 1:-1].set(W_int)
    C = C_D + C_w
    u_global = C.reshape(-1)
    return u_global, knots_x, knots_y


def solve_square_reactdiff(
    breakpoints_x: Array,
    breakpoints_y: Array,
    p: int,
    *,
    eps: float = 1e-2,
    sigma: float = 1.0,
    q_order: int = 20,
    q_order_bc: int = 40,
) -> tuple[Array, SquareSystem]:
    """Convenience wrapper returning (u_global, sys) for exporting/postprocessing."""
    u_global, knots_x, knots_y = solve_square_reactdiff_coeffs(
        breakpoints_x,
        breakpoints_y,
        int(p),
        eps=float(eps),
        sigma=float(sigma),
        q_order=int(q_order),
        q_order_bc=int(q_order_bc),
    )

    nBx = int(knots_x.shape[0] - int(p) - 1)
    nBy = int(knots_y.shape[0] - int(p) - 1)
    sys = SquareSystem(
        breakpoints_x=jnp.asarray(breakpoints_x),
        breakpoints_y=jnp.asarray(breakpoints_y),
        knots_x=knots_x,
        knots_y=knots_y,
        p=int(p),
        eps=float(eps),
        sigma=float(sigma),
        nBx=nBx,
        nBy=nBy,
    )
    return u_global, sys


__all__ = [
    "SquareSystem",
    "solve_square_reactdiff_coeffs",
    "solve_square_reactdiff",
]
