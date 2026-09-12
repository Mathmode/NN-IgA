"""Fast-Diagonalization (Kronecker) exact O(N³) solver for `arctan` (Opt B).

`arctan` is the Poisson problem −Δu = f with **σ = 1** on a tensor-product
B-spline mesh, with **separable** zero-Dirichlet BCs (u = 0 on {x=0} ∪ {y=0}).
On such a mesh the stiffness matrix is EXACTLY a sum of Kronecker products

    K = K_x ⊗ M_y + M_x ⊗ K_y

with K_* the 1D stiffness (∫ N_i' N_j') and M_* the 1D mass (∫ N_i N_j). The
separable Dirichlet (drop index i=0 in x and j=0 in y) keeps the Kronecker
structure on the free block. The **Fast Diagonalization Method (FDM)** then
solves K u = F by two small 1D generalized eigenproblems + a spectral divide,
in **O(N³)** — EXACT (direct, not iterative) and well-conditioned by
construction (the generalized eigenvalues of SPD (K_free, M_free) are positive).

This is a NEW, parallel path. The validated dense solver
(`solver_2d.galerkin_solve_arctan`, which uses `solve_system_2d`) is untouched
and remains the default/oracle; the FDM is verified bit-for-bit against it
(forward AND gradient) in `tests/test_fdm_arctan_2d.py`.

Adjoint: a `custom_vjp` keeps the discrete adjoint (dF = λ, the factor
sensitivities below) and **reuses the forward diagonalization** for the adjoint
solve — it does NOT differentiate the eigendecomposition. The factor cotangents
(consistent with the dense dK = −λuᵀ, verified end-to-end vs. the dense path):

    dK_x = −λ M_y Uᵀ      dM_y = −λᵀ K_x U
    dM_x = −λ K_y Uᵀ      dK_y = −λᵀ M_x U      dF = λ

where U is the (free) solution matrix and λ solves K λ = ḡ (same FDM).

SCOPE: arctan only. lshape (non-rectangular L domain → no global Kronecker) and
advdiff (non-symmetric, variable σ) are out of scope, as is 1D.
"""
from __future__ import annotations

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import functools

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from src.nonparametric.solver_2d import (
    Solve2DResult,
    _assemble_2d,
    _assemble_neumann_1d,
    _element_data,
)

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# 1D stiffness + mass, assembled with the SAME primitives as `_assemble_2d`
# (`_element_data` / `rule_gl_on_elements` / `basis_batch_for_degree`) so the
# Kronecker product reproduces the dense 2D K bit-for-bit. Reimplemented here
# (not imported from the 1D tree) to keep a clean 2D-only dependency and that
# bit-consistency guarantee.
# ---------------------------------------------------------------------------
def stiffness_mass_1d(knots: Array, p: int, n_elem: int, nq: int) -> tuple[Array, Array]:
    """Return (K_1d, M_1d), the dense 1D stiffness ∫N_i'N_j' and mass ∫N_iN_j."""
    a, b, U_local, idx_local = _element_data(knots, int(p), int(n_elem))
    xq, wq = rule_gl_on_elements(a, b, int(nq))          # (n_elem, nq)
    N, dN, _ = basis_batch_for_degree(int(p), xq, U_local)   # (n_elem, nq, p+1)
    Kloc = jnp.einsum("eai,ea,eaj->eij", dN, wq, dN)
    Mloc = jnp.einsum("eai,ea,eaj->eij", N, wq, N)
    n_ctrl = int(n_elem + p)
    K1d = jnp.zeros((n_ctrl, n_ctrl), dtype=DEFAULT_DTYPE)
    M1d = jnp.zeros((n_ctrl, n_ctrl), dtype=DEFAULT_DTYPE)
    K1d = K1d.at[idx_local[:, :, None], idx_local[:, None, :]].add(Kloc)
    M1d = M1d.at[idx_local[:, :, None], idx_local[:, None, :]].add(Mloc)
    return K1d, M1d


# ---------------------------------------------------------------------------
# Generalized symmetric eigenproblem  K φ = λ M φ  (M SPD), M-orthonormal
# eigenvectors, via the Cholesky route (jax 0.5.2 has no 2-arg eigh):
#   M = L Lᵀ ;  C = L⁻¹ K L⁻ᵀ (symmetric) ;  eigh(C) = (Λ, V) ;  U = L⁻ᵀ V
# Then  Uᵀ M U = I  and  Uᵀ K U = diag(Λ).
# ---------------------------------------------------------------------------
def generalized_eigh_spd(K: Array, M: Array) -> tuple[Array, Array]:
    """(Λ, U) with K U = M U diag(Λ), Uᵀ M U = I. K, M symmetric, M SPD."""
    L = jnp.linalg.cholesky(M)
    Linv_K = jax.scipy.linalg.solve_triangular(L, K, lower=True)            # L⁻¹ K
    C = jax.scipy.linalg.solve_triangular(L, Linv_K.T, lower=True).T        # L⁻¹ K L⁻ᵀ
    C = 0.5 * (C + C.T)
    Lam, V = jnp.linalg.eigh(C)
    U = jax.scipy.linalg.solve_triangular(L, V, lower=True, trans="T")      # L⁻ᵀ V
    return Lam, U


def _spectral_solve(Ux, Lam_x, Uy, Lam_y, Phi):
    """Solve (K_x U M_y + M_x U K_y) = Phi given the two M-orthonormal
    diagonalizations: U = U_x [ (U_xᵀ Phi U_y) / (Λ_x ⊕ Λ_y) ] U_yᵀ."""
    Ftil = Ux.T @ Phi @ Uy
    denom = Lam_x[:, None] + Lam_y[None, :]
    return Ux @ (Ftil / denom) @ Uy.T


# ---------------------------------------------------------------------------
# FDM linear solve on the FREE block, with the explicit discrete-adjoint VJP.
# Inputs are the free-restricted 1D factors and the free RHS matrix.
# ---------------------------------------------------------------------------
@jax.custom_vjp
def solve_fdm_factors(Kx: Array, Mx: Array, Ky: Array, My: Array, Phi: Array) -> Array:
    """U solving  K_x U M_y + M_x U K_y = Phi  (all matrices free-restricted)."""
    Lam_x, Ux = generalized_eigh_spd(Kx, Mx)
    Lam_y, Uy = generalized_eigh_spd(Ky, My)
    return _spectral_solve(Ux, Lam_x, Uy, Lam_y, Phi)


def _solve_fdm_fwd(Kx, Mx, Ky, My, Phi):
    Lam_x, Ux = generalized_eigh_spd(Kx, Mx)
    Lam_y, Uy = generalized_eigh_spd(Ky, My)
    U = _spectral_solve(Ux, Lam_x, Uy, Lam_y, Phi)
    # Save the diagonalization (reused for the adjoint solve) + factors + U.
    return U, (Ux, Lam_x, Uy, Lam_y, U, Kx, Mx, Ky, My)


def _solve_fdm_bwd(res, g):
    Ux, Lam_x, Uy, Lam_y, U, Kx, Mx, Ky, My = res
    # Adjoint solve REUSES the same diagonalization (K is self-adjoint).
    lam = _spectral_solve(Ux, Lam_x, Uy, Lam_y, g)
    dKx = -lam @ My @ U.T
    dMy = -lam.T @ (Kx @ U)
    dMx = -lam @ Ky @ U.T
    dKy = -lam.T @ (Mx @ U)
    dPhi = lam
    return dKx, dMx, dKy, dMy, dPhi


solve_fdm_factors.defvjp(_solve_fdm_fwd, _solve_fdm_bwd)


# ---------------------------------------------------------------------------
# Top-level arctan solver via FDM — SAME signature/return as
# `galerkin_solve_arctan`. F (incl. Neumann) is assembled EXACTLY as the dense
# path (reused, not re-derived); only the stiffness solve changes.
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_K", "q_F"))
def galerkin_solve_arctan_fdm(
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
    """FDM/Kronecker exact O(N³) arctan solver. Bit-identical to
    `galerkin_solve_arctan` (forward + gradient); see module docstring."""
    from src.nonparametric.arctan.pde import (
        f_sigma,
        g_sigma_neumann_right,
        g_sigma_neumann_top,
    )

    n_x = n_elem_x + p
    n_y = n_elem_y + p

    # --- RHS F: assembled exactly as in galerkin_solve_arctan (reused). ---
    sigma_callable = lambda x, y: jnp.ones_like(x)
    f_callable = lambda x, y: f_sigma(x, y, alpha, s1, s2)
    _K_unused, F = _assemble_2d(
        knots_x, knots_y, p, n_elem_x, n_elem_y, sigma_callable, f_callable, q_K, q_F,
    )
    F_right = _assemble_neumann_1d(
        lambda y: g_sigma_neumann_right(y, alpha, s1, s2), knots_y, p, n_elem_y, q_F,
    )
    F_top = _assemble_neumann_1d(
        lambda x: g_sigma_neumann_top(x, alpha, s1, s2), knots_x, p, n_elem_x, q_F,
    )
    F_grid = F.reshape(n_x, n_y)
    F_grid = F_grid.at[n_x - 1, :].add(F_right)
    F_grid = F_grid.at[:, n_y - 1].add(F_top)

    # --- 1D factors and separable free restriction. arctan Dirichlet is
    # constrained = {i=0} ∪ {j=0} (verified in Phase 0 against free_mask_arctan),
    # so the free block is the STATIC slice [1:, 1:] (jit-safe; no data-dependent
    # indexing). The test suite asserts free_mask_arctan == this pattern. ---
    Kx, Mx = stiffness_mass_1d(knots_x, p, n_elem_x, q_K)
    Ky, My = stiffness_mass_1d(knots_y, p, n_elem_y, q_K)
    Kxf, Mxf = Kx[1:, 1:], Mx[1:, 1:]
    Kyf, Myf = Ky[1:, 1:], My[1:, 1:]
    Phi_f = F_grid[1:, 1:]

    # --- FDM solve on the free block, then reinsert Dirichlet zeros. ---
    U_f = solve_fdm_factors(Kxf, Mxf, Kyf, Myf, Phi_f)
    u_grid = jnp.zeros((n_x, n_y), dtype=DEFAULT_DTYPE)
    u_grid = u_grid.at[1:, 1:].set(U_f)
    u = u_grid.reshape(-1)

    # ritz_energy reported with the dense post-Dirichlet (K, F) for parity with
    # the dense result; the FDM does not form K, so report u·F-based energy
    # consistently (0.5 uᵀKu − uᵀF == −0.5 uᵀF at the solution, K u = F).
    F_full = F_grid.reshape(-1)
    ritz_energy = -0.5 * (u @ F_full)
    return Solve2DResult(u_h=u.reshape(n_x, n_y), ritz_energy=ritz_energy, K=None, F=F_full)


__all__ = [
    "stiffness_mass_1d",
    "generalized_eigh_spd",
    "solve_fdm_factors",
    "galerkin_solve_arctan_fdm",
]
