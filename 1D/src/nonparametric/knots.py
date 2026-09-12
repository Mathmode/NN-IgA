from __future__ import annotations

"""Mesh parametrization utilities for the singular experiment."""

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray

H_MIN_DEFAULT = 1.0e-8


@jax.jit
def stable_softmax(theta: Array) -> Array:
    shifted = theta - jnp.max(theta)
    exp_theta = jnp.exp(shifted)
    return exp_theta / jnp.sum(exp_theta)


@jax.jit
def theta_to_sizes(theta: Array, h_min: float = H_MIN_DEFAULT) -> Array:
    theta = jnp.asarray(theta, dtype=DEFAULT_DTYPE).reshape(-1)
    n_elem = theta.shape[0]
    h_min_t = jnp.asarray(h_min, dtype=theta.dtype)
    scale = jnp.asarray(1.0, dtype=theta.dtype) - jnp.asarray(n_elem, dtype=theta.dtype) * h_min_t
    return h_min_t + scale * stable_softmax(theta)


@functools.partial(jax.jit, static_argnames=("degree", "return_h"))
def theta_to_knots(
    theta: Array,
    a: float,
    b: float,
    degree: int,
    *,
    h_min: float = H_MIN_DEFAULT,
    return_h: bool = False,
) -> Array | tuple[Array, Array]:
    theta = jnp.asarray(theta, dtype=DEFAULT_DTYPE).reshape(-1)
    sizes = theta_to_sizes(theta, h_min=h_min)
    dtype = sizes.dtype
    degree = int(degree)

    a_t = jnp.asarray(a, dtype=dtype)
    b_t = jnp.asarray(b, dtype=dtype)
    interior = a_t + jnp.cumsum(sizes[:-1])
    left = jnp.full((degree + 1,), a_t, dtype=dtype)
    right = jnp.full((degree + 1,), b_t, dtype=dtype)
    knots = jnp.concatenate([left, interior, right], axis=0)

    if return_h:
        return knots, sizes
    return knots


def uniform_knots(
    n_elem: int,
    degree: int,
    a: float = 0.0,
    b: float = 1.0,
    dtype=DEFAULT_DTYPE,
) -> Array:
    points = jnp.linspace(a, b, int(n_elem) + 1, dtype=dtype)
    left = jnp.full((degree + 1,), a, dtype=dtype)
    right = jnp.full((degree + 1,), b, dtype=dtype)
    return jnp.concatenate([left, points[1:-1], right])


def breakpoints_from_knots(knots: Array, degree: int) -> Array:
    degree = int(degree)
    return jnp.asarray(knots[degree:-degree])


def inverse_softmax_from_sizes(sizes: np.ndarray, h_min: float = H_MIN_DEFAULT) -> Array:
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
    weights = np.clip(sizes - float(h_min), 1.0e-300, None)
    theta = np.log(weights)
    theta -= theta.mean()
    return jnp.asarray(theta, dtype=DEFAULT_DTYPE)


def bisect_sizes(sizes: np.ndarray) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=np.float64).reshape(-1)
    return np.repeat(0.5 * sizes, 2)


def check_admissible_h_min(n_elem: int, h_min: float) -> None:
    if (1.0 - float(n_elem) * float(h_min)) <= 0.0:
        raise ValueError(
            f"Infeasible h_min={float(h_min):.3e} for N={int(n_elem)}: require N*h_min < 1."
        )


def refine_knots_with_midpoints(knots: Array, degree: int) -> Array:
    degree = int(degree)
    internal = jnp.asarray(knots[degree:-degree])
    midpoints = 0.5 * (internal[:-1] + internal[1:])
    refined_internal = jnp.sort(jnp.concatenate([internal, midpoints]))
    left = jnp.full((degree + 1,), knots[0], dtype=knots.dtype)
    right = jnp.full((degree + 1,), knots[-1], dtype=knots.dtype)
    return jnp.concatenate([left, refined_internal, right])


__all__ = [
    "Array",
    "H_MIN_DEFAULT",
    "stable_softmax",
    "theta_to_sizes",
    "theta_to_knots",
    "uniform_knots",
    "breakpoints_from_knots",
    "inverse_softmax_from_sizes",
    "bisect_sizes",
    "check_admissible_h_min",
    "refine_knots_with_midpoints",
]
