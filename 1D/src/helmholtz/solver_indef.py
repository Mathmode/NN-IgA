from __future__ import annotations

"""Indefinite symmetric linear solve B c = l with an explicit discrete-adjoint VJP.

The Helmholtz Galerkin operator B(omega) = K(sigma) - omega^2 M(rho) is symmetric
but INDEFINITE (not SPD): Cholesky (the singular/arctan/lshape path) would fail. We use a direct
LU solve (jnp.linalg.solve) and supply the same explicit discrete adjoint used by
``1D/src/nonparametric/solver.py`` but valid for any symmetric B:

    forward :  c   = B^{-1} l
    backward:  lam = B^{-1} g          (B symmetric => B^{-T} = B^{-1})
               dB  = - lam c^T
               dl  =   lam

This is mathematically identical to JAX's implicit derivative of ``jnp.linalg.solve``;
we make it explicit for parity with the rest of the project. NEVER use CG here (the
operator is indefinite and resonance-near-singular).
"""

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@jax.custom_vjp
def solve_indef(B: Array, l: Array) -> Array:
    """Solve B c = l for symmetric indefinite B via a direct (LU) solve."""
    return jnp.linalg.solve(B, l)


def _fwd(B: Array, l: Array):
    c = jnp.linalg.solve(B, l)
    return c, (B, c)


def _bwd(res, g: Array):
    B, c = res
    lam = jnp.linalg.solve(B, g)            # B symmetric => no transpose needed
    dB = -jnp.outer(lam, c)
    dl = lam
    return dB, dl


solve_indef.defvjp(_fwd, _bwd)
solve_indef = jax.jit(solve_indef)


@jax.jit
def solve_indef_autodiff(B: Array, l: Array) -> Array:
    """Same solve WITHOUT custom_vjp (JAX's implicit adjoint); consistency ref."""
    return jnp.linalg.solve(B, l)


__all__ = ["solve_indef", "solve_indef_autodiff"]
