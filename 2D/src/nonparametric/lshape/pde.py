"""lshape (Aballay 4.3.2) L-shape problem with piecewise-constant diffusion.

Domain
------
Omega = (0, 1)^2 \\ closure((0.5, 1] x [0, 0.5)).

We implement Omega on a tensor-product mesh over the full unit square and
zero out the removed quadrant via Dirichlet masking (see
``src.shared.dirichlet_masking``).

PDE (Aballay eq. 18b)
---------------------
    -div(sigma(x) grad u) = 1  in Omega,
    u = 0  on  boundary(Omega).

Piecewise-constant diffusion coefficient (no analytic solution; reference
Ritz energy must be computed numerically):

    sigma(x, y) =
        1            if  (x, y) in (0, 0.5) x (0.5, 1),  (top-left)
        sigma1       if  (x, y) in (0, 0.5)^2,             (bottom-left)
        sigma2       if  (x, y) in (0.5, 1)^2,             (top-right)
        --- removed quadrant: u = 0, sigma is irrelevant ---

This module exposes a JIT-compatible ``sigma_field(x, y, sigma1, sigma2)``
that returns sigma at any point (x, y) and is differentiable w.r.t.
``sigma1`` and ``sigma2``. The discontinuity in ``x``, ``y`` is handled by
masked indicators (built on ``jnp.heaviside``); the gradient w.r.t. the
parameters is well-defined and smooth across the parameter space.
"""
from __future__ import annotations

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp

from common.dirichlet_masking import step_ge, step_le

Array = jnp.ndarray


def sigma_field(
    x: Array, y: Array, sigma1: float, sigma2: float
) -> Array:
    """Piecewise-constant diffusion coefficient sigma(x, y).

    Regions (all inclusive at interfaces; the value at exactly x=0.5 or
    y=0.5 sits on a measure-zero boundary so the choice is harmless):

        top-left     (x < 0.5 and y > 0.5):  value 1
        bottom-left  (x < 0.5 and y < 0.5):  value sigma1
        top-right    (x > 0.5 and y > 0.5):  value sigma2
        bottom-right (removed quadrant):     value 1 (does not enter assembly)
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)
    s1 = jnp.asarray(sigma1, dtype=x.dtype)
    s2 = jnp.asarray(sigma2, dtype=x.dtype)
    one = jnp.asarray(1.0, dtype=x.dtype)

    # Strict half-plane indicators via heaviside, opened on one side.
    # We use a tiny eps to break ties cleanly at x=0.5 / y=0.5 (which we
    # avoid by node placement: the fixed nodes at 0.5 form interfaces,
    # and the cell interior away from those interfaces is what matters).
    eps = jnp.asarray(1e-12, dtype=x.dtype)
    x_left = step_le(x, 0.5 - eps)
    x_right = step_ge(x, 0.5 + eps)
    y_low = step_le(y, 0.5 - eps)
    y_high = step_ge(y, 0.5 + eps)

    in_top_left = x_left * y_high
    in_bottom_left = x_left * y_low
    in_top_right = x_right * y_high
    in_bottom_right = x_right * y_low  # removed, but we still must return a value

    # Note: the removed quadrant gets `one` (filler) since its DOFs are
    # zeroed out post-assembly. Choosing 1 there avoids issues if the
    # quadrature picks up an unmasked cell in the corner.
    return (one * in_top_left
            + s1 * in_bottom_left
            + s2 * in_top_right
            + one * in_bottom_right)


def f_lshape(x: Array, y: Array) -> Array:
    """Forcing f = 1 on the whole domain."""
    x = jnp.asarray(x)
    return jnp.ones_like(x)


__all__ = ["Array", "sigma_field", "f_lshape"]
