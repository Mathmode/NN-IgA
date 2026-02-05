from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

Array = jnp.ndarray

_TINY = 1e-15


@functools.partial(jax.jit, static_argnames=("p",))
def mesh_diagnostics(knots: Array, p: int) -> tuple[Array, Array, Array]:
    """Basic mesh diagnostics for a 1D open-clamped knot vector."""
    p = int(p)
    P = knots[p:-p]
    h = P[1:] - P[:-1]
    h_min = jnp.min(h)
    h_max = jnp.max(h)
    h_safe = jnp.maximum(h_min, jnp.asarray(_TINY, dtype=knots.dtype))
    h_ratio = h_max / h_safe
    return h_min, h_max, h_ratio


__all__ = ["mesh_diagnostics"]
