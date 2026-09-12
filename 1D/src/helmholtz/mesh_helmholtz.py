from __future__ import annotations

"""Two-patch (C^0 interface) mesh policy for the helmholtz experiment.

The transmission problem has a material interface at x_I = 1/2. The mesh keeps a
FIXED knot of multiplicity p at x_I (full C^0 continuity reduction) at every level
and grades the two halves [0, 1/2] and [1/2, 1] INDEPENDENTLY (the left layer
carries the shorter wavelength k1 > k2). This ports the per-half logic of the 2D
lshape mesh (``positional_density_network_2d.knots_p3_from_network_axis``) to 1D.

A level N (even) has n_half = N/2 elements per half; the network is collocated at
the n_half cell midpoints of each half, so the same weights serve any N.
"""

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import functools

import jax
import jax.numpy as jnp

from src.nonparametric.knots import stable_softmax

Array = jnp.ndarray
X_I = 0.5
H_MIN_DEFAULT = 1.0e-8


@jax.jit
def sizes_in_budget(theta: Array, budget: float, h_min: float) -> Array:
    """Softmax sizes over one half summing to `budget`, each >= h_min."""
    theta = jnp.asarray(theta, dtype=DEFAULT_DTYPE).reshape(-1)
    n = theta.shape[0]
    h = jnp.asarray(h_min, dtype=theta.dtype)
    bud = jnp.asarray(budget, dtype=theta.dtype)
    scale = bud - jnp.asarray(n, dtype=theta.dtype) * h
    return h + scale * stable_softmax(theta)


@functools.partial(jax.jit, static_argnames=("degree",))
def knots_two_patch(theta_left: Array, theta_right: Array, degree: int,
                    *, h_min: float = H_MIN_DEFAULT) -> Array:
    """Open knot vector on [0,1] with a C^0 interface (multiplicity `degree`) at
    x_I, built from per-half logit vectors.

    Each half gets budget 0.5; interior knots are the cumulative sizes; the
    interface node 0.5 is inserted with multiplicity `degree`.
    """
    p = int(degree)
    sizes_l = sizes_in_budget(theta_left, 0.5, h_min)
    sizes_r = sizes_in_budget(theta_right, 0.5, h_min)
    dtype = sizes_l.dtype
    interior_l = jnp.cumsum(sizes_l)[:-1]                       # n_half-1 pts in (0, 0.5)
    interior_r = jnp.asarray(X_I, dtype=dtype) + jnp.cumsum(sizes_r)[:-1]
    interface = jnp.full((p,), X_I, dtype=dtype)               # C^0 (multiplicity p)
    left = jnp.zeros((p + 1,), dtype=dtype)
    right = jnp.ones((p + 1,), dtype=dtype)
    return jnp.concatenate([left, interior_l, interface, interior_r, right])


def uniform_two_patch_knots(n_elem: int, degree: int) -> Array:
    """Uniform two-patch knots: n_elem/2 equal cells per half, C^0 interface."""
    p = int(degree)
    n = int(n_elem)
    if n % 2 != 0:
        raise ValueError(f"helmholtz needs even N (got {n}); n_half = N/2 per patch.")
    half = n // 2
    zero = jnp.zeros((), dtype=DEFAULT_DTYPE)
    interior_l = jnp.linspace(0.0, X_I, half + 1, dtype=DEFAULT_DTYPE)[1:-1]
    interior_r = jnp.linspace(X_I, 1.0, half + 1, dtype=DEFAULT_DTYPE)[1:-1]
    interface = jnp.full((p,), X_I, dtype=DEFAULT_DTYPE)
    left = jnp.zeros((p + 1,), dtype=DEFAULT_DTYPE)
    right = jnp.ones((p + 1,), dtype=DEFAULT_DTYPE)
    return jnp.concatenate([left, interior_l, interface, interior_r, right])


def n_elem_from_knots(knots: Array, degree: int) -> int:
    return int(knots.shape[0] - 2 * int(degree) - 1)


__all__ = [
    "X_I", "H_MIN_DEFAULT", "sizes_in_budget",
    "knots_two_patch", "uniform_two_patch_knots", "n_elem_from_knots",
]
