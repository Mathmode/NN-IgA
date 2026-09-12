from __future__ import annotations

"""Shared quadrature utilities for the repo (1D/2D).

This module centralizes:
  - Gauss–Legendre rules (cached via NumPy, cast to JAX arrays)
  - Vectorized mapping of rules to physical elements
  - A small analytic helper: monomial integrals ∫ x^nu dx (used in Exp. 1)

Notes
-----
Gauss–Legendre nodes/weights are generated with NumPy and then converted to JAX
arrays. When called from inside `jax.jit` with a static order, these become XLA
constants (no runtime NumPy usage).
"""

import functools
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

Array = jnp.ndarray


def as_dtype(x, dtype) -> Array:
    return jnp.asarray(x, dtype=dtype)


# -----------------------------------------------------------------------------
# Gauss–Legendre rules (cached via NumPy)
# -----------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def leggauss_np(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype("float64"), w.astype("float64")


def leggauss(n: int, dtype) -> Tuple[Array, Array]:
    """Return (xi, wi) for Gauss–Legendre on [-1,1] as JAX arrays."""
    x, w = leggauss_np(int(n))
    return as_dtype(x, dtype), as_dtype(w, dtype)


@functools.partial(jax.jit, static_argnames=("n",))
def rule_gl_on_elements(a: Array, b: Array, n: int) -> Tuple[Array, Array]:
    """Map an n-point GL rule from [-1,1] to many elements [a,b] (vectorized)."""
    a = jnp.asarray(a)
    b = jnp.asarray(b, dtype=a.dtype)
    xi, wi = leggauss(int(n), a.dtype)

    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid[..., None] + half[..., None] * xi[None, ...]
    w = half[..., None] * wi[None, ...]
    return x, w


@functools.partial(jax.jit, static_argnames=("n",))
def rule_gl_on_element(a: float, b: float, n: int) -> Tuple[Array, Array]:
    """Map an n-point GL rule from [-1,1] to one element [a,b]."""
    x, w = rule_gl_on_elements(jnp.asarray(a), jnp.asarray(b), int(n))
    return x.reshape(-1), w.reshape(-1)


# -----------------------------------------------------------------------------
# Analytic helper (Exp. 1): monomial integrals
# -----------------------------------------------------------------------------


def monomial_integral(a: float | Array, b: float | Array, nu: float | Array, *, eps: float = 1e-12) -> Array:
    """Compute ∫_a^b x^nu dx with a stable log-form near nu == -1.

    Returns +inf for the divergent case a=0 and nu<=-1 (broadcasted).
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b, dtype=a.dtype)
    nu = jnp.asarray(nu, dtype=a.dtype)
    mu = nu + 1.0

    eps_t = as_dtype(eps, a.dtype)
    log_term = jnp.log(jnp.clip(b, 1e-300)) - jnp.log(jnp.clip(a, 1e-300))
    gen_term = (jnp.power(b, mu) - jnp.power(a, mu)) / mu
    out = jnp.where(jnp.abs(mu) < eps_t, log_term, gen_term)

    diverge = jnp.logical_and(a == 0.0, nu <= -1.0)
    return jnp.where(diverge, as_dtype(jnp.inf, out.dtype), out)


__all__ = [
    "Array",
    "as_dtype",
    "leggauss_np",
    "leggauss",
    "rule_gl_on_elements",
    "rule_gl_on_element",
    "monomial_integral",
]
