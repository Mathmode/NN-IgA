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
piecewise-constant masks; the gradient w.r.t. the
parameters is well-defined and smooth across the parameter space.
"""
from __future__ import annotations

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp


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
    return jnp.where((x < 0.5) & (y < 0.5), s1,
                     jnp.where((x > 0.5) & (y > 0.5), s2, jnp.ones_like(x)))



def f_lshape(x: Array, y: Array) -> Array:
    """Forcing f = 1 on the whole domain."""
    x = jnp.asarray(x)
    return jnp.ones_like(x)


__all__ = ["Array", "sigma_field", "f_lshape"]
