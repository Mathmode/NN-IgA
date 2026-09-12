"""Explicit discrete-adjoint linear solve for SPD systems (2D analog of
``1D/src/nonparametric/solver.py``).

The 2D Galerkin solvers (``solver_2d.galerkin_solve_arctan`` / ``_lshape``)
call ``solve_system_2d`` below. They previously used ``jnp.linalg.solve``
and relied on JAX's implicit adjoint (differentiation through the linear
solve via the implicit function theorem); this module
provides a mathematically equivalent solve whose backward pass is the
EXPLICIT discrete adjoint of Proposition X.Y: it reuses the forward
Cholesky factor ``L`` in a single triangular adjoint solve, rather than
retro-propagating through the factorization.

It lives in ``common/`` because the machinery is problem-agnostic — it
works for any SPD ``K`` regardless of which 2D PDE produced it (arctan
arctangent, lshape L-shape; both are SPD Poisson). The advection–diffusion
advdiff matrix is non-symmetric and would need an LU variant — out of scope.

Two entry points (parallel to the 1D module):
  * ``solve_system_2d``          — Cholesky solve with explicit custom_vjp.
  * ``solve_system_2d_autodiff`` — same Cholesky solve, NO custom_vjp
    (JAX differentiates the factorization itself); reference for the
    consistency test.

Both are ``jax.jit``- and ``jax.vmap``-compatible: ``K`` and ``F`` carry
leading batch dimensions handled naturally by ``...`` einsum/broadcast
patterns.
"""
from __future__ import annotations

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


def _solve_from_cholesky_2d(L: Array, rhs: Array) -> Array:
    """Two triangular solves: ``L y = rhs`` then ``L^T x = y``.

    Mirrors ``1D/src/nonparametric/solver.py:_solve_from_cholesky`` exactly,
    including the vector/matrix squeeze handling so a vector ``rhs`` of
    shape ``(..., n)`` and a matrix ``rhs`` of shape ``(..., n, m)`` both
    work, with arbitrary leading batch dims (``L`` is ``(..., n, n)``).
    """
    squeeze = rhs.ndim == L.ndim - 1
    rhs_mat = rhs[..., None] if squeeze else rhs
    y = lax.linalg.triangular_solve(L, rhs_mat, left_side=True, lower=True)
    x = lax.linalg.triangular_solve(
        L, y, left_side=True, lower=True, transpose_a=True
    )
    return x[..., 0] if squeeze else x


@jax.custom_vjp
def solve_system_2d(K: Array, F: Array) -> Array:
    """Solve ``K @ u = F`` for SPD ``K`` via Cholesky, with an explicit
    discrete-adjoint VJP that reuses the forward factorization.

    Mathematically equivalent to ``jnp.linalg.solve(K, F)`` for SPD ``K``,
    but the backward pass does NOT differentiate through the Cholesky
    steps: it reuses the forward ``L`` and performs a single triangular
    adjoint solve, giving ``dK = -lam u^T`` and ``dF = lam`` with
    ``lam = K^{-1} g`` (Proposition X.Y).

    vmap/JIT compatible: ``K`` is ``(..., n, n)``, ``F`` is ``(..., n)``;
    leading batch dims pass through the ``...`` einsum patterns.
    """
    L = jnp.linalg.cholesky(K)
    return _solve_from_cholesky_2d(L, F)


def _solve_system_2d_fwd(K: Array, F: Array):
    L = jnp.linalg.cholesky(K)
    U = _solve_from_cholesky_2d(L, F)
    return U, (L, U)   # save L (forward factor) + U for the backward pass


def _solve_system_2d_bwd(residuals, g: Array):
    L, U = residuals
    # Adjoint solve REUSING the forward Cholesky factor L (no re-factorize).
    lam = _solve_from_cholesky_2d(L, g)
    lam_mat = lam[..., None] if lam.ndim == L.ndim - 1 else lam
    U_mat = U[..., None] if U.ndim == L.ndim - 1 else U
    # dK = -lam U^T (outer product, batched); symmetric K so this is the
    # correct discrete-adjoint sensitivity.
    dK = -jnp.einsum("...ik,...jk->...ij", lam_mat, U_mat)
    dF = lam
    return dK, dF


solve_system_2d.defvjp(_solve_system_2d_fwd, _solve_system_2d_bwd)
solve_system_2d = jax.jit(solve_system_2d)


@jax.jit
def solve_system_2d_autodiff(K: Array, F: Array) -> Array:
    """Same Cholesky solve as ``solve_system_2d`` but WITHOUT ``custom_vjp``.

    JAX differentiates the Cholesky factorization and triangular solves
    itself (its own implicit/AD adjoint). This is the numerical reference
    against which the explicit ``solve_system_2d`` VJP is checked.
    """
    L = jnp.linalg.cholesky(K)
    return _solve_from_cholesky_2d(L, F)


__all__ = [
    "Array",
    "solve_system_2d",
    "solve_system_2d_autodiff",
    "_solve_from_cholesky_2d",
]
