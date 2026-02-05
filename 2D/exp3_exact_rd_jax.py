from __future__ import annotations

"""Exact solution for a 2D reaction--diffusion boundary layer on (0,1)^2.

We solve:
    -eps * Δu + sigma * u = 0

with Dirichlet data:
    u(0,y)=sin(pi y),  u(1,y)=0,  u(x,0)=u(x,1)=0.

Separation-of-variables yields:
    u(x,y) = A(x) * sin(pi y),
where A solves:
    -eps A'' + (eps*pi^2 + sigma) A = 0,   A(0)=1, A(1)=0,
so:
    A(x) = sinh(lam*(1-x)) / sinh(lam),
    lam = sqrt((eps*pi^2 + sigma)/eps).
"""

import jax.numpy as jnp

Array = jnp.ndarray


def g_in(y: Array) -> Array:
    """Inflow Dirichlet data at x=0."""
    return jnp.sin(jnp.pi * y)


def _lambda(eps: float | Array, sigma: float | Array, dtype) -> Array:
    eps_t = jnp.asarray(eps, dtype=dtype)
    sig_t = jnp.asarray(sigma, dtype=dtype)
    return jnp.sqrt((eps_t * (jnp.pi**2) + sig_t) / eps_t)


def exact_u(x: Array, y: Array, *, eps: float = 1e-2, sigma: float = 1.0) -> Array:
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)
    lam = _lambda(eps, sigma, x.dtype)
    A = jnp.sinh(lam * (1.0 - x)) / jnp.sinh(lam)
    return A * jnp.sin(jnp.pi * y)


def exact_grad(x: Array, y: Array, *, eps: float = 1e-2, sigma: float = 1.0) -> tuple[Array, Array]:
    x = jnp.asarray(x)
    y = jnp.asarray(y, dtype=x.dtype)
    lam = _lambda(eps, sigma, x.dtype)
    denom = jnp.sinh(lam)
    A = jnp.sinh(lam * (1.0 - x)) / denom
    Ax = -(lam * jnp.cosh(lam * (1.0 - x))) / denom
    ux = Ax * jnp.sin(jnp.pi * y)
    uy = A * (jnp.pi * jnp.cos(jnp.pi * y))
    return ux, uy


__all__ = [
    "g_in",
    "exact_u",
    "exact_grad",
]

