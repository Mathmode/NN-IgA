from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("degree", "return_h"))
def theta_to_knots(
    theta: Array,
    a: float,
    b: float,
    degree: int,
    *,
    h_min: float = 1e-6,
    return_h: bool = False,
) -> Array | tuple[Array, Array]:
    """Map unconstrained parameters ``theta`` to an open-clamped knot vector.

    The interior element sizes are given by a softmax distribution.
    """
    theta = jnp.atleast_1d(theta)
    dtype = theta.dtype
    N = int(theta.shape[0])
    degree = int(degree)

    w = jax.nn.softmax(theta)

    # Strict minimum element size: h_i >= h_min and sum h_i = (b-a)
    L = jnp.asarray(b - a, dtype=dtype)
    h_min_t = jnp.asarray(h_min, dtype=dtype)
    # Safety clamp: ensures N*h_min < L even if the user passes a too-large value.
    h_min_t = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L / jnp.asarray(N, dtype=dtype))
    h = h_min_t + (L - jnp.asarray(N, dtype=dtype) * h_min_t) * w

    a_t = jnp.asarray(a, dtype=dtype)
    P_int = a_t + jnp.cumsum(h[:-1])
    left = jnp.full((degree + 1,), a, dtype=dtype)
    right = jnp.full((degree + 1,), b, dtype=dtype)
    knots = jnp.concatenate([left, P_int, right])

    if return_h:
        P_all = jnp.concatenate([jnp.array([a], dtype=dtype), P_int, jnp.array([b], dtype=dtype)])
        h_out = P_all[1:] - P_all[:-1]
        return knots, h_out
    return knots


def uniform_knots(
    N: int,
    degree: int,
    a: float = 0.0,
    b: float = 1.0,
    dtype=jnp.float64,
) -> Array:
    """Uniform open-clamped knots with ``N`` elements on [a,b]."""
    P = jnp.linspace(a, b, int(N) + 1, dtype=dtype)
    left = jnp.full((degree + 1,), a, dtype=dtype)
    right = jnp.full((degree + 1,), b, dtype=dtype)
    return jnp.concatenate([left, P[1:-1], right])


def breakpoints_from_knots(knots: Array, degree: int) -> Array:
    """Return physical breakpoints P = knots[p:-p]."""
    degree = int(degree)
    return jnp.asarray(knots[degree:-degree])


def refine_knots_with_midpoints(knots: Array, degree: int) -> Array:
    """Refine a knot vector by inserting midpoints of each element."""
    degree = int(degree)
    internal = knots[degree:-degree]
    midpoints = (internal[:-1] + internal[1:]) / 2.0
    refined_internal = jnp.sort(jnp.concatenate([internal, midpoints]))
    left = jnp.full((degree + 1,), knots[0], dtype=knots.dtype)
    right = jnp.full((degree + 1,), knots[-1], dtype=knots.dtype)
    return jnp.concatenate([left, refined_internal, right])


def progressive_refinement(
    *,
    degree: int = 3,
    a: float = 0.0,
    b: float = 1.0,
    max_elements: int = 16,
    seed: int = 0,
) -> list[Array]:
    """Generate a sequence of refined knot vectors starting from 2 elements."""
    import jax.random as jr

    key = jr.PRNGKey(int(seed))
    theta = jr.normal(key, (2,))
    knots = theta_to_knots(theta, a, b, degree)

    refinement_levels: list[Array] = [knots]
    current_elements = 2

    while current_elements < max_elements:
        knots = refine_knots_with_midpoints(knots, degree)
        refinement_levels.append(knots)
        current_elements *= 2

    return refinement_levels


__all__ = [
    "theta_to_knots",
    "uniform_knots",
    "breakpoints_from_knots",
    "refine_knots_with_midpoints",
    "progressive_refinement",
]
