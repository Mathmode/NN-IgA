r"""Tensor-product B-spline Galerkin solver for the 2D advection–diffusion
boundary-layer problem (advdiff).

Strong form on Omega = (0, 1)^2 with homogeneous Dirichlet BCs:

    -eps * Laplacian u + b * du/dx = f_nu,    u = 0 on d Omega.

Weak form (test v, with v = 0 on the boundary so the diffusion boundary
term vanishes):

    eps * \int grad u . grad v  +  b * \int (du/dx) v  =  \int f v.

The convective term makes the bilinear form NON-symmetric; the system
matrix is therefore not symmetric and the "Ritz energy" is not a true
energy. We still report ``ritz_energy = 0.5 u^T K u - u^T F`` (as in
``galerkin_solve_arctan``) purely for API parity with the existing
``Solve2DResult`` consumers — it carries no variational meaning here.

Assembly mirrors ``solver_2d._assemble_2d`` exactly for the two
diffusion blocks (K_xx, K_yy) and adds the non-symmetric convective
block:

    K_conv[ex, ey, i, j, k, l] = \int_E (N_i N_j) * (dN_k/dx * N_l) dxdy
                               =  (test (i, j)) * (d/dx trial (k, l)),

so that  K_local = eps * (K_xx + K_yy) + b * K_conv.

All low-level primitives (`_element_data`, `rule_gl_on_elements`,
`basis_batch_for_degree`), the Dirichlet application (`_apply_dirichlet`),
and the result container (`Solve2DResult`) are imported from the existing
modules — nothing is copied. Only the assembly (to add K_conv) and the
all-four-boundary Dirichlet mask are new.
"""
from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from src.nonparametric.solver_2d import (
    Solve2DResult,
    _apply_dirichlet,
    _element_data,
)

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Dirichlet free-mask: u = 0 on ALL four boundaries.
# --------------------------------------------------------------------------


def free_mask_all_boundaries(knots_x: Array, knots_y: Array, p: int) -> Array:
    """Zero on the whole boundary of the unit square.

    For an open-uniform basis the only basis rows/cols touching the
    boundary are indices 0 and n-1 along each axis. Returns shape
    ``(n_x, n_y)`` with 1 = free, 0 = constrained.
    """
    n_x = int(knots_x.shape[0] - p - 1)
    n_y = int(knots_y.shape[0] - p - 1)
    free = jnp.ones((n_x, n_y), dtype=DEFAULT_DTYPE)
    free = free.at[0, :].set(0.0)
    free = free.at[-1, :].set(0.0)
    free = free.at[:, 0].set(0.0)
    free = free.at[:, -1].set(0.0)
    return free


# --------------------------------------------------------------------------
# Assembly: diffusion (symmetric) + convection (non-symmetric) + forcing.
# --------------------------------------------------------------------------


def _assemble_advdiff(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    eps: Array,
    b: Array,
    f_callable,
    q_K: int,
    q_F: int,
):
    """Assemble (K, F) in DENSE form for ``-eps Lap u + b u_x = f``.

    Mirrors ``solver_2d._assemble_2d``: same element data, same GL rules,
    same scatter pattern. Adds the convective block ``K_conv``.
    """
    p_i = int(p)
    nex = int(n_elem_x)
    ney = int(n_elem_y)
    qK = int(q_K)
    qF = int(q_F)
    n_x = nex + p_i
    n_y = ney + p_i
    n_total = n_x * n_y

    ax, bx, U_local_x, idx_local_x = _element_data(knots_x, p_i, nex)
    ay, by, U_local_y, idx_local_y = _element_data(knots_y, p_i, ney)

    # ---- Stiffness assembly with q_K^2 GL per element. ----
    xq_K, wq_x_K = rule_gl_on_elements(ax, bx, qK)        # (nex, qK)
    yq_K, wq_y_K = rule_gl_on_elements(ay, by, qK)        # (ney, qK)
    Nx_K, dNx_K, _ = basis_batch_for_degree(p_i, xq_K, U_local_x)  # (nex, qK, p+1)
    Ny_K, dNy_K, _ = basis_batch_for_degree(p_i, yq_K, U_local_y)

    # Pure GL weights (the diffusion coefficient eps and convection b are
    # applied as scalar multipliers below, so unit weights here).
    w_2d_K = wq_x_K[:, None, :, None] * wq_y_K[None, :, None, :]   # (nex, ney, qK, qK)

    # Indices: e=elem-x, f=elem-y, a=quad-x, b=quad-y; test=(i,j), trial=(k,l).
    # Diffusion (symmetric): grad(test) . grad(trial).
    K_xx = jnp.einsum(
        "efab, eai, fbj, eak, fbl -> efijkl",
        w_2d_K, dNx_K, Ny_K, dNx_K, Ny_K,
    )
    K_yy = jnp.einsum(
        "efab, eai, fbj, eak, fbl -> efijkl",
        w_2d_K, Nx_K, dNy_K, Nx_K, dNy_K,
    )
    # Convection (non-symmetric): test * d/dx(trial).
    #   test  (i, j) -> Nx_K[i] * Ny_K[j]
    #   trial (k, l) -> dNx_K[k] * Ny_K[l]   (x-derivative of the trial)
    K_conv = jnp.einsum(
        "efab, eai, fbj, eak, fbl -> efijkl",
        w_2d_K, Nx_K, Ny_K, dNx_K, Ny_K,
    )

    eps_t = jnp.asarray(eps, dtype=DEFAULT_DTYPE)
    b_t = jnp.asarray(b, dtype=DEFAULT_DTYPE)
    K_local = eps_t * (K_xx + K_yy) + b_t * K_conv   # (nex, ney, p+1, p+1, p+1, p+1)

    # ---- Forcing assembly with q_F^2 GL per element. ----
    xq_F, wq_x_F = rule_gl_on_elements(ax, bx, qF)
    yq_F, wq_y_F = rule_gl_on_elements(ay, by, qF)
    Nx_F, _, _ = basis_batch_for_degree(p_i, xq_F, U_local_x)
    Ny_F, _, _ = basis_batch_for_degree(p_i, yq_F, U_local_y)
    XX_F = jnp.broadcast_to(xq_F[:, None, :, None], (nex, ney, qF, qF))
    YY_F = jnp.broadcast_to(yq_F[None, :, None, :], (nex, ney, qF, qF))
    f_at_F = f_callable(XX_F, YY_F)
    w_2d_F = wq_x_F[:, None, :, None] * wq_y_F[None, :, None, :]
    F_local = jnp.einsum(
        "efab, efab, eai, fbj -> efij", f_at_F, w_2d_F, Nx_F, Ny_F
    )   # (nex, ney, p+1, p+1)

    # ---- Scatter into global K (dense) and F. ----
    g_idx = (
        idx_local_x[:, None, :, None] * n_y + idx_local_y[None, :, None, :]
    )   # (nex, ney, p+1, p+1)
    n_loc = (p_i + 1) ** 2
    K_local_flat = K_local.reshape(nex, ney, n_loc, n_loc)
    g_idx_flat = g_idx.reshape(nex, ney, n_loc)

    K = jnp.zeros((n_total, n_total), dtype=DEFAULT_DTYPE)
    F = jnp.zeros((n_total,), dtype=DEFAULT_DTYPE)
    K = K.at[g_idx_flat[:, :, :, None], g_idx_flat[:, :, None, :]].add(K_local_flat)
    F = F.at[g_idx_flat.reshape(-1)].add(F_local.reshape(-1))
    return K, F


# --------------------------------------------------------------------------
# Top-level solver.
# --------------------------------------------------------------------------


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"),
)
def galerkin_solve_advdiff(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    eps: Array,
    b: Array,
    *,
    q_K: int = 4,
    q_F: int = 40,
) -> Solve2DResult:
    """Solve ``-eps Lap u + b u_x = f_nu`` with homogeneous Dirichlet BCs.

    Returns a ``Solve2DResult``. ``ritz_energy = 0.5 u^T K u - u^T F`` is
    kept for API parity with the symmetric solvers, but the bilinear form
    here is NON-symmetric (convective term), so it is not a true energy —
    do not interpret it variationally.
    """
    from src.nonparametric.advdiff.pde import f_advdiff

    f_callable = lambda x, y: f_advdiff(x, y, eps, b)
    K, F = _assemble_advdiff(
        knots_x, knots_y, p, n_elem_x, n_elem_y, eps, b, f_callable, q_K, q_F,
    )

    free = free_mask_all_boundaries(knots_x, knots_y, p)
    K, F = _apply_dirichlet(K, F, free.reshape(-1))

    u = jnp.linalg.solve(K, F)
    ritz_energy = 0.5 * u @ K @ u - u @ F   # API parity only; not a true energy.

    n_x = n_elem_x + p
    n_y = n_elem_y + p
    return Solve2DResult(
        u_h=u.reshape(n_x, n_y),
        ritz_energy=ritz_energy,
        K=K,
        F=F,
    )


__all__ = [
    "Array",
    "free_mask_all_boundaries",
    "galerkin_solve_advdiff",
]
