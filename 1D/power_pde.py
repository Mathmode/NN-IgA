from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

Array = jnp.ndarray


@dataclass(frozen=True)
class PdeData1D:
    """Container for 1D PDE coefficients and boundary data."""

    sigma: Callable[[Array], Array]
    alpha: Callable[[Array], Array]
    f: Callable[[Array], Array]
    gN: float


BETA = 1.6


def pde_singular_power(beta: float = BETA) -> PdeData1D:
    """Problem 1 (power solution) PDE data.

    PDE: -(sigma u')' + alpha u = f, with sigma=1, alpha=0,
         f(x) = beta(1-beta) x^{beta-2}.
    BC: u(0)=0, sigma u'(1)=beta.
    """

    sigma = lambda x: jnp.ones_like(x)
    alpha = lambda x: jnp.zeros_like(x)
    f = lambda x: beta * (1.0 - beta) * jnp.power(x, beta - 2.0)
    return PdeData1D(sigma=sigma, alpha=alpha, f=f, gN=float(beta))


def u_exact_power(x: Array, beta: float) -> Array:
    return jnp.power(x, beta)


def du_exact_power(x: Array, beta: float) -> Array:
    return beta * jnp.power(x, beta - 1.0)


def u_exact(x: Array) -> Array:
    return u_exact_power(x, BETA)


def du_exact(x: Array) -> Array:
    return du_exact_power(x, BETA)


__all__ = [
    "PdeData1D",
    "BETA",
    "pde_singular_power",
    "u_exact_power",
    "du_exact_power",
    "u_exact",
    "du_exact",
]
