"""Tensor-product B-spline Galerkin solver in 2D.

Supports both Aballay 4.3.1 (arctan: arctangent, sigma=1, Neumann on right/top)
and Aballay 4.3.2 (lshape: L-shape, piecewise sigma, all-Dirichlet u=0).

Design
------
The mesh is the tensor product of two open-uniform B-spline knot vectors
``knots_x`` and ``knots_y`` on the unit square. The basis is
``B_i(x) * B_j(y)`` with global DOF index ``gI = i * n_y + j``.

Assembly is element-by-element with Gauss-Legendre quadrature in 2D:

    K[gI, gJ] = sum_{ex, ey} int_{element} sigma(x, y) [
                    (dB_x * B_y)(gI) . (dB_x * B_y)(gJ)
                  + (B_x * dB_y)(gI) . (B_x * dB_y)(gJ)
                ] dx dy

    F[gI]     = sum_{ex, ey} int_{element} f(x, y) B_x(gI) B_y(gI) dx dy
              + (arctan only) Neumann boundary integrals on x=1 and y=1.

Dirichlet BCs are applied via a free/constrained mask of shape
``(n_x, n_y)`` evaluated at the Greville abscissae. Constrained DOFs are
"penalised" via row/column zero-out plus unit diagonal (standard
substitution under the assumption that the Dirichlet value is 0).

JIT path
--------
Every shape is static (compile-time), so the entire assembly+solve is
JIT-compilable. The differentiation path follows the standard adjoint
form (no custom_vjp needed at this stage; ``jnp.linalg.solve`` provides
the gradient automatically).

For the smoke-level work (N <= 10) the dense solve is fine. The cluster
must switch to sparse (BCOO + CG) before scaling to N = 64.
"""
from __future__ import annotations

import functools
from typing import Callable, NamedTuple

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from common.solver_custom_vjp_2d import solve_system_2d

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Knot/element helpers (JAX-traceable; shapes are static)
# --------------------------------------------------------------------------


def _element_data(knots: Array, p: int, n_elem: int):
    """Build (a, b, U_local, idx_local) entirely from JAX ops.

    ``knots`` may be a traced array of static shape ``(n_elem + 2 p + 1,)``.
    """
    a = jax.lax.dynamic_slice(knots, (p,), (n_elem,))
    b = jax.lax.dynamic_slice(knots, (p + 1,), (n_elem,))

    def window_for(e):
        return jax.lax.dynamic_slice(knots, (e,), (2 * p + 2,))

    U_local = jax.vmap(window_for)(jnp.arange(n_elem))
    idx_local = (
        jnp.arange(p + 1, dtype=jnp.int32)[None, :]
        + jnp.arange(n_elem, dtype=jnp.int32)[:, None]
    )
    return a, b, U_local, idx_local


def greville_abscissae(knots: Array, p: int) -> Array:
    """Greville abscissae of an open-uniform B-spline basis.

    ``greville[i] = mean(knots[i + 1 : i + p + 1])`` for i in [0, n_basis - 1].
    Returns a vector of length ``n_basis = len(knots) - p - 1``.

    Implementation: direct per-window mean (no cumulative sum).

    An earlier implementation computed the same quantity as the cumsum-
    difference ``(cumsum[i + p + 1] - cumsum[i + 1]) / p``. The two are
    equal in exact arithmetic but NOT in float64: ``jnp.cumsum`` accumulates
    round-off left-to-right, so the difference at different starting
    indices carries asymmetric error. This broke reflection symmetry of
    the Greville vector for symmetric knot vectors — at N=30, p=2 the
    Greville at the multiplicity-p knot 0.5 drifted to
    ``0.4999999999999998`` (2 ULPs below 0.5), which then asymmetrically
    flipped the Dirichlet free-mask classification of ~30 DOFs and broke
    the assembled (K, F) by ~40% for the L-shape. See
    audit_p3_diagnostic.md and test_eta_jump_p3_symmetry.py for the trace.

    The per-window mean below has round-off that depends ONLY on the
    window contents, not on its position in the knot vector, so it is
    bit-symmetric under reversal of a symmetric knot vector.
    """
    knots = jnp.asarray(knots)
    n_basis = int(knots.shape[0] - p - 1)

    def window(i):
        return jax.lax.dynamic_slice(knots, (i + 1,), (p,))

    win = jax.vmap(window)(jnp.arange(n_basis))   # (n_basis, p)
    return win.mean(axis=1)


# --------------------------------------------------------------------------
# Assembly: element-by-element with 2D GL quadrature.
# --------------------------------------------------------------------------


def _assemble_2d(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma_callable: Callable,
    f_callable: Callable,
    q_K: int,
    q_F: int,
):
    """Assemble (K, F) in DENSE form.

    Shape contracts:
        sigma_callable(x, y) -> array of x.shape (broadcasted).
        f_callable(x, y)     -> array of x.shape (broadcasted).

    Returns:
        K of shape (n_x * n_y, n_x * n_y)
        F of shape (n_x * n_y,)
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
    xq_K, wq_x_K = rule_gl_on_elements(ax, bx, qK)   # (nex, qK)
    yq_K, wq_y_K = rule_gl_on_elements(ay, by, qK)   # (ney, qK)
    Nx_K, dNx_K, _ = basis_batch_for_degree(p_i, xq_K, U_local_x)  # (nex, qK, p+1)
    Ny_K, dNy_K, _ = basis_batch_for_degree(p_i, yq_K, U_local_y)

    XX_K = jnp.broadcast_to(xq_K[:, None, :, None], (nex, ney, qK, qK))
    YY_K = jnp.broadcast_to(yq_K[None, :, None, :], (nex, ney, qK, qK))
    sigma_at_K = sigma_callable(XX_K, YY_K)              # (nex, ney, qK, qK)
    w_2d_K = wq_x_K[:, None, :, None] * wq_y_K[None, :, None, :]  # (nex, ney, qK, qK)
    sigma_w_K = sigma_at_K * w_2d_K

    # K_local[ex, ey, i, j, k, l] : 4 basis indices (test (i, j), trial (k, l)).
    # Two contributions: ∂x.∂x and ∂y.∂y.
    K_xx = jnp.einsum(
        "efab, eai, fbj, eak, fbl -> efijkl",
        sigma_w_K, dNx_K, Ny_K, dNx_K, Ny_K,
    )
    K_yy = jnp.einsum(
        "efab, eai, fbj, eak, fbl -> efijkl",
        sigma_w_K, Nx_K, dNy_K, Nx_K, dNy_K,
    )
    K_local = K_xx + K_yy   # (nex, ney, p+1, p+1, p+1, p+1)

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


def _assemble_neumann_1d(
    g_callable: Callable,
    knots: Array,
    p: int,
    n_elem: int,
    q: int,
) -> Array:
    """Compute G[j] = int_0^1 g(t) B_j(t) dt over the 1D mesh.

    Used by arctan to assemble the Neumann boundary contributions on x=1 and y=1.
    """
    a, b, U_local, idx_local = _element_data(knots, p, n_elem)
    xq, wq = rule_gl_on_elements(a, b, q)
    N, _, _ = basis_batch_for_degree(p, xq, U_local)
    g_q = g_callable(xq)            # (n_elem, q)
    F_local = jnp.einsum("eq, eq, eqi -> ei", g_q, wq, N)   # (n_elem, p+1)
    n_basis = int(n_elem + p)
    F = jnp.zeros((n_basis,), dtype=DEFAULT_DTYPE)
    F = F.at[idx_local.reshape(-1)].add(F_local.reshape(-1))
    return F


# --------------------------------------------------------------------------
# Boundary conditions (Dirichlet mask, applied multiplicatively).
# --------------------------------------------------------------------------


def _apply_dirichlet(
    K: Array,
    F: Array,
    free_flat: Array,
) -> tuple[Array, Array]:
    """Apply zero-Dirichlet BCs via mask multiplication.

    ``free_flat`` is a (n_total,) array of 0/1 entries (1 = free, 0 = constrained).
    The Dirichlet value is assumed to be zero; non-zero values would require
    a substitution offset.

    Replaces constrained rows/columns of K with the identity, and zeros the
    corresponding entries of F.
    """
    free = free_flat
    # K' = (free_i)(free_j) K_ij + (1 - free_i) delta_ij
    # The diagonal of the second term needs to ONLY be added when i is constrained.
    n = K.shape[0]
    constrained = jnp.asarray(1.0, dtype=K.dtype) - free
    K_free = K * free[:, None] * free[None, :]
    K_diag = jnp.diag(constrained)
    K_new = K_free + K_diag
    F_new = F * free
    return K_new, F_new


# --------------------------------------------------------------------------
# Problem-specific Dirichlet free-masks at the Greville abscissae.
# --------------------------------------------------------------------------


def free_mask_arctan(knots_x: Array, knots_y: Array, p: int) -> Array:
    """arctan Dirichlet: u = 0 on {x = 0} ∪ {y = 0}.

    For an open-uniform basis, only the i = 0 row of basis functions has
    nonzero value at x = 0 (idem for j = 0 at y = 0). Returns shape (n_x, n_y)
    with 1 = free.
    """
    n_x = int(knots_x.shape[0] - p - 1)
    n_y = int(knots_y.shape[0] - p - 1)
    free = jnp.ones((n_x, n_y), dtype=DEFAULT_DTYPE)
    free = free.at[0, :].set(0.0)
    free = free.at[:, 0].set(0.0)
    return free


def free_mask_lshape(knots_x: Array, knots_y: Array, p: int) -> Array:
    """lshape Dirichlet: u = 0 on the external boundary AND on the removed
    quadrant (greville at x>=0.5 AND y<=0.5).

    Returns shape (n_x, n_y) with 1 = free, 0 = constrained.
    """
    from common.dirichlet_masking import lshape_removed_quadrant_mask

    gx = greville_abscissae(knots_x, p)
    gy = greville_abscissae(knots_y, p)
    # Collapse the ~1-ULP XLA-fusion drift of the x=0.5 / y=0.5 interface
    # Greville (the p-fold mean of 0.5 drifts <1e-15 below 0.5 under fusion for
    # p>=3, flipping step_ge(0.5,0.5) and un-constraining the reentrant row).
    # A literal-0.5 snap is fusion-stable; jnp.round is NOT (the /1e12 returns
    # the drift). Window 1e-9 sits between the ~1e-16 drift and the nearest
    # legitimate interface Greville ~3e-8 (even at h_min=1e-7). p=2 unaffected.
    gx = jnp.where(jnp.abs(gx - 0.5) < 1e-9, jnp.asarray(0.5, gx.dtype), gx)
    gy = jnp.where(jnp.abs(gy - 0.5) < 1e-9, jnp.asarray(0.5, gy.dtype), gy)
    XX, YY = jnp.meshgrid(gx, gy, indexing="ij")
    removed = lshape_removed_quadrant_mask(XX, YY)

    n_x = int(knots_x.shape[0] - p - 1)
    n_y = int(knots_y.shape[0] - p - 1)
    free = jnp.ones((n_x, n_y), dtype=DEFAULT_DTYPE)
    # External boundary: rows/cols 0 and n-1.
    free = free.at[0, :].set(0.0)
    free = free.at[-1, :].set(0.0)
    free = free.at[:, 0].set(0.0)
    free = free.at[:, -1].set(0.0)
    # Removed-quadrant Greville: union with external.
    free = free * (jnp.asarray(1.0, dtype=DEFAULT_DTYPE) - removed)
    return free


# --------------------------------------------------------------------------
# Top-level solvers (problem-specific).
# --------------------------------------------------------------------------


class Solve2DResult(NamedTuple):
    """Result of a 2D Galerkin solve.

    Attributes
    ----------
    u_h : Array of shape (n_x, n_y)
        Coefficient matrix on the tensor-product B-spline basis.
    ritz_energy : float
        J = 0.5 u^T K u - u^T F (negative for the minimiser of a coercive
        bilinear form, by convention).
    K : Array
        Global stiffness matrix (post-Dirichlet).
    F : Array
        Global RHS vector (post-Dirichlet).
    """
    u_h: Array
    ritz_energy: Array
    K: Array
    F: Array


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"),
)
def galerkin_solve_arctan(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    alpha: Array,
    s1: Array,
    s2: Array,
    *,
    q_K: int = 4,
    q_F: int = 50,
) -> Solve2DResult:
    """Solve the arctan (arctangent) problem on a tensor-product B-spline mesh.

    sigma = 1, f = -Laplacian u^sigma analytic, Neumann data on x=1 and y=1,
    Dirichlet u = 0 on x=0 and y=0.
    """
    from src.nonparametric.arctan.pde import (
        f_sigma,
        g_sigma_neumann_right,
        g_sigma_neumann_top,
    )

    sigma_callable = lambda x, y: jnp.ones_like(x)
    f_callable = lambda x, y: f_sigma(x, y, alpha, s1, s2)
    K, F = _assemble_2d(
        knots_x, knots_y, p, n_elem_x, n_elem_y,
        sigma_callable, f_callable, q_K, q_F,
    )

    # Neumann contributions on x=1 (last x-basis row) and y=1 (last y-basis col).
    F_right = _assemble_neumann_1d(
        lambda y: g_sigma_neumann_right(y, alpha, s1, s2),
        knots_y, p, n_elem_y, q_F,
    )
    F_top = _assemble_neumann_1d(
        lambda x: g_sigma_neumann_top(x, alpha, s1, s2),
        knots_x, p, n_elem_x, q_F,
    )

    n_x = n_elem_x + p
    n_y = n_elem_y + p
    F_grid = F.reshape(n_x, n_y)
    F_grid = F_grid.at[n_x - 1, :].add(F_right)
    F_grid = F_grid.at[:, n_y - 1].add(F_top)
    # Note: the corner (n_x-1, n_y-1) gets contributions from BOTH edges, which
    # is the correct treatment for a corner with both Neumann edges meeting.
    F = F_grid.reshape(-1)

    # Apply Dirichlet on x=0 and y=0.
    free = free_mask_arctan(knots_x, knots_y, p)
    K, F = _apply_dirichlet(K, F, free.reshape(-1))

    u = solve_system_2d(K, F)
    ritz_energy = 0.5 * u @ K @ u - u @ F
    return Solve2DResult(
        u_h=u.reshape(n_x, n_y),
        ritz_energy=ritz_energy,
        K=K,
        F=F,
    )


@functools.partial(
    jax.jit,
    static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"),
)
def galerkin_solve_lshape(
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    sigma1: Array,
    sigma2: Array,
    *,
    q_K: int = 4,
    q_F: int = 2,
) -> Solve2DResult:
    """Solve the lshape (L-shape) problem on a tensor-product B-spline mesh.

    sigma piecewise constant, f = 1, Dirichlet u = 0 on external boundary
    AND inside the removed quadrant.
    """
    from src.nonparametric.lshape.pde import f_lshape, sigma_field

    sigma_callable = lambda x, y: sigma_field(x, y, sigma1, sigma2)
    f_callable = lambda x, y: f_lshape(x, y)
    K, F = _assemble_2d(
        knots_x, knots_y, p, n_elem_x, n_elem_y,
        sigma_callable, f_callable, q_K, q_F,
    )

    free = free_mask_lshape(knots_x, knots_y, p)
    K, F = _apply_dirichlet(K, F, free.reshape(-1))

    u = solve_system_2d(K, F)
    ritz_energy = 0.5 * u @ K @ u - u @ F

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
    "Solve2DResult",
    "greville_abscissae",
    "free_mask_arctan",
    "free_mask_lshape",
    "galerkin_solve_arctan",
    "galerkin_solve_lshape",
]
