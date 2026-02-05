from __future__ import annotations

r"""JAX exact solution for the classical L-shaped Laplace benchmark.

Domain:
    Omega = (-1,1)^2 \ ([0,1] x [-1,0])

Exact solution (re-entrant corner singularity at (0,0)):
    u(r,θ) = r^{2/3} sin( 2θ/3 ),   θ ∈ [0, 3π/2].

We provide (u, ux, uy) in Cartesian coordinates and a helper for Dirichlet data.
"""

import jax.numpy as jnp

Array = jnp.ndarray


def _wrap_theta(theta: Array) -> Array:
    """Map atan2 output to [0, 2π)."""
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=theta.dtype)
    return jnp.where(theta < 0.0, theta + two_pi, theta)


def u_exact(x: Array, y: Array) -> Array:
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)
    r = jnp.sqrt(x * x + y * y)
    theta = _wrap_theta(jnp.arctan2(y, x))
    return jnp.power(r, jnp.asarray(2.0 / 3.0, dtype=x.dtype)) * jnp.sin(jnp.asarray(2.0 / 3.0, dtype=x.dtype) * theta)


def grad_u_exact(x: Array, y: Array) -> tuple[Array, Array]:
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)
    r = jnp.sqrt(x * x + y * y)
    theta = _wrap_theta(jnp.arctan2(y, x))

    r_safe = jnp.where(r > 0.0, r, jnp.ones_like(r))
    fac = jnp.asarray(2.0 / 3.0, dtype=x.dtype) * jnp.power(r_safe, jnp.asarray(-1.0 / 3.0, dtype=x.dtype))

    ux = -fac * jnp.sin(theta / 3.0)
    uy = fac * jnp.cos(theta / 3.0)

    ux = jnp.where(r > 0.0, ux, jnp.zeros_like(ux))
    uy = jnp.where(r > 0.0, uy, jnp.zeros_like(uy))
    return ux, uy


def eval_exact(x: Array, y: Array) -> tuple[Array, Array, Array]:
    """Return (u, ux, uy) for convenience."""
    u = u_exact(x, y)
    ux, uy = grad_u_exact(x, y)
    return u, ux, uy


__all__ = [
    "u_exact",
    "grad_u_exact",
    "eval_exact",
]

