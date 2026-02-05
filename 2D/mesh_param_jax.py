from __future__ import annotations

"""JAX mesh parametrization for the 2D experiments.

We use the same softmax element-size parametrization as the PyTorch code, but in
JAX so it can be JIT-compiled and differentiated:

    h_i = h_min + (L - n*h_min) * softmax(theta)_i,    sum_i h_i = L.

This enforces strict positivity and prevents mesh collapse during training.
"""

import functools

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=())
def build_breakpoints_softmax(theta: Array, a: float, b: float, *, h_min: float = 1e-6) -> Array:
    """Map unconstrained ``theta`` to breakpoints in [a,b] with a strict h_min."""
    theta = jnp.asarray(theta).reshape(-1)
    n = int(theta.shape[0])
    if n == 0:
        dtype = theta.dtype
        return jnp.asarray([a, b], dtype=dtype)

    dtype = theta.dtype
    a_t = jnp.asarray(a, dtype=dtype)
    b_t = jnp.asarray(b, dtype=dtype)
    L = b_t - a_t

    h_min_t = jnp.asarray(h_min, dtype=dtype)
    n_t = jnp.asarray(n, dtype=dtype)
    h_min_eff = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L / n_t)

    w = jax.nn.softmax(theta, axis=0)
    h = h_min_eff + (L - n_t * h_min_eff) * w

    z0 = jnp.zeros((1,), dtype=dtype)
    bp = a_t + jnp.concatenate([z0, jnp.cumsum(h)], axis=0)
    bp = bp.at[0].set(a_t).at[-1].set(b_t)
    return bp


def init_theta_softmax(n_elem: int, *, seed: int = 0, scale: float = 0.0, dtype=jnp.float64) -> Array:
    """Initializer for the softmax parametrization (theta=0 yields a uniform mesh)."""
    n_elem = int(n_elem)
    if n_elem <= 0:
        return jnp.zeros((0,), dtype=dtype)

    theta = jnp.zeros((n_elem,), dtype=dtype)
    if float(scale) > 0.0:
        key = jax.random.PRNGKey(int(seed))
        theta = theta + jnp.asarray(float(scale), dtype=dtype) * jax.random.normal(key, theta.shape, dtype=dtype)
    return theta


__all__ = [
    "build_breakpoints_softmax",
    "init_theta_softmax",
]

