from __future__ import annotations

"""Exact PDE data and manufactured solution for the singular experiment."""

from dataclasses import dataclass
import math
from typing import Callable

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

Array = jnp.ndarray

BETA = 1.6
EXACT_D2_COEFF = BETA * (BETA - 1.0)
FORCING_COEFF = -EXACT_D2_COEFF


@dataclass(frozen=True)
class PdeData1D:
    """Container for 1D PDE coefficients and boundary data."""

    sigma: Callable[[Array], Array]
    alpha: Callable[[Array], Array]
    f: Callable[[Array], Array]
    gN: float


@jax.jit
def forcing_power(x: Array, beta: float = BETA) -> Array:
    beta_t = jnp.asarray(beta, dtype=x.dtype)
    return -(beta_t * (beta_t - 1.0)) * jnp.power(x, beta_t - 2.0)


@jax.jit
def u_exact_power(x: Array, beta: float = BETA) -> Array:
    beta_t = jnp.asarray(beta, dtype=x.dtype)
    return jnp.power(x, beta_t)


@jax.jit
def du_exact_power(x: Array, beta: float = BETA) -> Array:
    beta_t = jnp.asarray(beta, dtype=x.dtype)
    return beta_t * jnp.power(x, beta_t - 1.0)


@jax.jit
def d2u_exact_power(x: Array, beta: float = BETA) -> Array:
    beta_t = jnp.asarray(beta, dtype=x.dtype)
    return beta_t * (beta_t - 1.0) * jnp.power(x, beta_t - 2.0)


def exact_l2_norm_power(beta: float = BETA) -> float:
    return math.sqrt(1.0 / (2.0 * float(beta) + 1.0))


def exact_energy_norm_power(beta: float = BETA) -> float:
    beta_f = float(beta)
    return math.sqrt((beta_f * beta_f) / (2.0 * beta_f - 1.0))


def exact_h1_norm_power(beta: float = BETA) -> float:
    beta_f = float(beta)
    l2_sq = 1.0 / (2.0 * beta_f + 1.0)
    h1_semi_sq = (beta_f * beta_f) / (2.0 * beta_f - 1.0)
    return math.sqrt(l2_sq + h1_semi_sq)


def pde_singular_power(beta: float = BETA) -> PdeData1D:
    sigma = lambda x: jnp.ones_like(x)
    alpha = lambda x: jnp.zeros_like(x)
    f = lambda x: forcing_power(x, beta)
    return PdeData1D(sigma=sigma, alpha=alpha, f=f, gN=float(beta))


def u_exact(x: Array) -> Array:
    return u_exact_power(x, BETA)


def du_exact(x: Array) -> Array:
    return du_exact_power(x, BETA)


def d2u_exact(x: Array) -> Array:
    return d2u_exact_power(x, BETA)


__all__ = [
    "Array",
    "BETA",
    "EXACT_D2_COEFF",
    "FORCING_COEFF",
    "PdeData1D",
    "forcing_power",
    "u_exact_power",
    "du_exact_power",
    "d2u_exact_power",
    "exact_l2_norm_power",
    "exact_energy_norm_power",
    "exact_h1_norm_power",
    "pde_singular_power",
    "u_exact",
    "du_exact",
    "d2u_exact",
]
