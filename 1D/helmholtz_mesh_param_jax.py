from __future__ import annotations

"""JAX mesh parametrization for the 1D Helmholtz interface experiment.

Key requirement for JIT:
    The interface index ``n_left`` must be fixed so array shapes stay static.

We therefore use a *fixed split* softmax parametrization:
    - first ``n_left`` elements live on [a, xI]
    - remaining ``n_right`` elements live on [xI, b]

Each side gets its own softmax distribution and enforces a strict minimum
element size ``h_min``.
"""

import functools
import numpy as np

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def allocate_by_phase(
    n_elem_total: int,
    xI: float,
    k_left: float,
    k_right: float,
    *,
    p: int = 1,
    a: float = 0.0,
    b: float = 1.0,
) -> tuple[int, int]:
    """Allocate (n_left, n_right) proportional to the integrated phase |k|/p."""
    n_elem_total = int(n_elem_total)
    if n_elem_total < 2:
        raise ValueError("n_elem_total must be >= 2")
    if not (a < xI < b):
        raise ValueError("xI must lie strictly inside (a,b)")

    p = int(max(1, p))
    L1 = float(xI - a)
    L2 = float(b - xI)
    S1 = abs(float(k_left)) * L1 / float(p)
    S2 = abs(float(k_right)) * L2 / float(p)

    if (S1 + S2) == 0.0:
        n_left = n_elem_total // 2
    else:
        n_left = int(round(n_elem_total * S1 / (S1 + S2)))

    n_left = max(1, min(n_elem_total - 1, n_left))
    n_right = int(n_elem_total - n_left)
    return int(n_left), int(n_right)


def build_piecewise_uniform_breakpoints(
    n_elem: int,
    xI: float,
    n_left: int,
    *,
    a: float = 0.0,
    b: float = 1.0,
) -> np.ndarray:
    """Piecewise-uniform breakpoints that include the interface xI."""
    n_elem = int(n_elem)
    n_left = int(n_left)
    if n_elem < 2:
        raise ValueError("n_elem must be >= 2")
    if not (a < xI < b):
        raise ValueError("xI must lie strictly inside (a,b)")
    if not (1 <= n_left <= n_elem - 1):
        raise ValueError("n_left must be in 1..n_elem-1")

    n_right = n_elem - n_left
    bpL = np.linspace(a, xI, n_left + 1)
    bpR = np.linspace(xI, b, n_right + 1)[1:]
    return np.concatenate([bpL, bpR])


@jax.jit
def _softmax(theta: Array) -> Array:
    return jax.nn.softmax(theta.reshape(-1), axis=0)


@functools.partial(jax.jit, static_argnames=("n_left",))
def build_breakpoints_fixed_split(
    theta: Array,
    *,
    xI: float,
    n_left: int,
    a: float = 0.0,
    b: float = 1.0,
    h_min: float = 1e-6,
) -> Array:
    """Map unconstrained ``theta`` to breakpoints with a fixed interface index.

    Returns bp of shape (n_elem+1,) such that bp[n_left] == xI exactly.
    """
    theta = jnp.asarray(theta).reshape(-1)
    n_elem = int(theta.shape[0])
    n_left = int(n_left)
    n_right = int(n_elem - n_left)

    dtype = theta.dtype
    a_t = jnp.asarray(a, dtype=dtype)
    b_t = jnp.asarray(b, dtype=dtype)
    xI_t = jnp.asarray(xI, dtype=dtype)

    L1 = xI_t - a_t
    L2 = b_t - xI_t

    h_min_t = jnp.asarray(h_min, dtype=dtype)
    h_min_L = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L1 / jnp.asarray(n_left, dtype=dtype))
    h_min_R = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L2 / jnp.asarray(n_right, dtype=dtype))

    thL = theta[:n_left]
    thR = theta[n_left:]
    wL = _softmax(thL)
    wR = _softmax(thR)

    hL = h_min_L + (L1 - jnp.asarray(n_left, dtype=dtype) * h_min_L) * wL
    hR = h_min_R + (L2 - jnp.asarray(n_right, dtype=dtype) * h_min_R) * wR

    z0 = jnp.zeros((1,), dtype=dtype)
    bpL = a_t + jnp.concatenate([z0, jnp.cumsum(hL)], axis=0)  # (n_left+1,)
    bpR = xI_t + jnp.concatenate([z0, jnp.cumsum(hR)], axis=0)  # (n_right+1,)

    bp = jnp.concatenate([bpL, bpR[1:]], axis=0)
    bp = bp.at[n_left].set(xI_t).at[0].set(a_t).at[-1].set(b_t)
    return bp


@functools.partial(jax.jit, static_argnames=("n_left",))
def theta_from_breakpoints_fixed_split(
    breakpoints: Array,
    *,
    xI: float,
    n_left: int,
    a: float = 0.0,
    b: float = 1.0,
    h_min: float = 1e-6,
) -> Array:
    """Invert the fixed-split parametrization approximately.

    Produces ``theta`` such that `build_breakpoints_fixed_split(theta, ...)`
    reproduces the element-size distribution (up to numerical tolerances).
    """
    bp = jnp.asarray(breakpoints).reshape(-1)
    dtype = bp.dtype

    n_elem = int(bp.shape[0] - 1)
    n_left = int(n_left)
    n_right = int(n_elem - n_left)

    a_t = jnp.asarray(a, dtype=dtype)
    b_t = jnp.asarray(b, dtype=dtype)
    xI_t = jnp.asarray(xI, dtype=dtype)

    L1 = xI_t - a_t
    L2 = b_t - xI_t

    h = bp[1:] - bp[:-1]
    hL = h[:n_left]
    hR = h[n_left:]

    h_min_t = jnp.asarray(h_min, dtype=dtype)
    h_min_L = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L1 / jnp.asarray(n_left, dtype=dtype))
    h_min_R = jnp.minimum(h_min_t, jnp.asarray(0.999, dtype=dtype) * L2 / jnp.asarray(n_right, dtype=dtype))

    denomL = jnp.maximum(L1 - jnp.asarray(n_left, dtype=dtype) * h_min_L, jnp.asarray(1e-30, dtype=dtype))
    denomR = jnp.maximum(L2 - jnp.asarray(n_right, dtype=dtype) * h_min_R, jnp.asarray(1e-30, dtype=dtype))

    wL = (hL - h_min_L) / denomL
    wR = (hR - h_min_R) / denomR

    eps = jnp.asarray(1e-32, dtype=dtype)
    wL = jnp.clip(wL, eps, None)
    wR = jnp.clip(wR, eps, None)
    wL = wL / jnp.sum(wL)
    wR = wR / jnp.sum(wR)

    thL = jnp.log(wL)
    thR = jnp.log(wR)
    thL = thL - jnp.mean(thL)
    thR = thR - jnp.mean(thR)

    return jnp.concatenate([thL, thR], axis=0)


__all__ = [
    "allocate_by_phase",
    "build_piecewise_uniform_breakpoints",
    "build_breakpoints_fixed_split",
    "theta_from_breakpoints_fixed_split",
]
