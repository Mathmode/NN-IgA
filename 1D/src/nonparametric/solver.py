from __future__ import annotations

import functools

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
from jax import lax
from jax.experimental import sparse as jsparse

Array = jnp.ndarray


def _solve_from_cholesky(L: Array, rhs: Array) -> Array:
    squeeze = rhs.ndim == L.ndim - 1
    rhs_mat = rhs[..., None] if squeeze else rhs
    y = lax.linalg.triangular_solve(L, rhs_mat, left_side=True, lower=True)
    x = lax.linalg.triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)
    return x[..., 0] if squeeze else x


@jax.custom_vjp
def solve_system(K: Array, F: Array) -> Array:
    L = jnp.linalg.cholesky(K)
    return _solve_from_cholesky(L, F)


def _solve_system_fwd(K: Array, F: Array):
    L = jnp.linalg.cholesky(K)
    U = _solve_from_cholesky(L, F)
    return U, (L, U)


def _solve_system_bwd(residuals, g: Array):
    L, U = residuals
    lam = _solve_from_cholesky(L, g)
    lam_mat = lam[..., None] if lam.ndim == L.ndim - 1 else lam
    U_mat = U[..., None] if U.ndim == L.ndim - 1 else U
    dK = -jnp.einsum("...ik,...jk->...ij", lam_mat, U_mat)
    dF = lam
    return dK, dF


solve_system.defvjp(_solve_system_fwd, _solve_system_bwd)
solve_system = jax.jit(solve_system)


@jax.jit
def solve_system_autodiff(K: Array, F: Array) -> Array:
    L = jnp.linalg.cholesky(K)
    return _solve_from_cholesky(L, F)


@functools.partial(jax.jit, static_argnames=("tol", "maxiter", "precond"))
def solve_system_sparse_cg(
    rows: Array,
    cols: Array,
    vals: Array,
    F: Array,
    *,
    tol: float = 1e-8,
    maxiter: int = 2000,
    precond: str = "jacobi",
) -> Array:
    n = int(F.shape[0])
    idx = jnp.stack([rows, cols], axis=1)
    A = jsparse.BCOO((vals, idx), shape=(n, n))

    precond_l = str(precond).lower().strip()
    M = None
    if precond_l == "jacobi":
        diag = jnp.zeros((n,), dtype=vals.dtype)
        diag = diag.at[rows].add(jnp.where(rows == cols, vals, jnp.asarray(0.0, dtype=vals.dtype)))
        eps = jnp.asarray(1e-14, dtype=vals.dtype)
        inv_diag = jnp.where(jnp.abs(diag) > eps, 1.0 / diag, 1.0)
        M = lambda x: inv_diag * x
    elif precond_l != "none":
        raise ValueError("precond must be one of: none, jacobi")

    x, _ = jsla.cg(lambda v: A @ v, F, M=M, tol=float(tol), atol=0.0, maxiter=int(maxiter))
    return x


__all__ = ["solve_system", "solve_system_autodiff", "solve_system_sparse_cg"]
