from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray


@jax.jit
def solve_system(K: Array, F: Array) -> Array:
    """Solve K x = F via Cholesky.

    Assumes K is SPD (after imposing Dirichlet values).
    """
    L = jnp.linalg.cholesky(K)
    squeeze = (F.ndim == K.ndim - 1)
    rhs = F[..., None] if squeeze else F
    y = lax.linalg.triangular_solve(L, rhs, left_side=True, lower=True)
    x = lax.linalg.triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)
    return x[..., 0] if squeeze else x


__all__ = ["solve_system"]
