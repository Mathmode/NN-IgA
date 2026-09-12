"""arctan (Aballay 4.3.1) manufactured arctangent problem.

PDE on Omega = (0, 1)^2:

    -Laplacian u^sigma = f^sigma  in Omega,
    u^sigma = 0  on  {x = 0} cup {y = 0},
    grad u^sigma . n = g^sigma  on  {x = 1} cup {y = 1}.

Manufactured solution
---------------------
u^sigma(x, y) = u1(x) u2(y),  with
    u_j(t) = arctan(alpha (t - s_j)) + arctan(alpha s_j),  j = 1, 2.

The "+ arctan(alpha s_j)" offset enforces u_j(0) = 0, hence u^sigma = 0
on the {x = 0} and {y = 0} sides of the unit square.

Analytic derivatives (used by the FE assembly and by the H1 metric):
    u_j'(t)  = alpha / (1 + alpha^2 (t - s_j)^2)
    u_j''(t) = -2 alpha^3 (t - s_j) / (1 + alpha^2 (t - s_j)^2)^2

so the forcing is

    f^sigma = -[u_1''(x) u_2(y) + u_1(x) u_2''(y)].

And Neumann data on {x = 1} or {y = 1} (outward normal +x or +y):
    g^sigma(1, y) = u_1'(1) u_2(y),
    g^sigma(x, 1) = u_1(x) u_2'(1).
"""
from __future__ import annotations

from typing import Tuple

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp

Array = jnp.ndarray


# --------------------------------------------------------------------------
# 1D primitives (u_j and its derivatives) — broadcastable over arrays.
# --------------------------------------------------------------------------


def u_axis(t: Array, alpha, s) -> Array:
    t = jnp.asarray(t)
    a = jnp.asarray(alpha, dtype=t.dtype)
    s_ = jnp.asarray(s, dtype=t.dtype)
    return jnp.arctan(a * (t - s_)) + jnp.arctan(a * s_)


def du_axis(t: Array, alpha, s) -> Array:
    t = jnp.asarray(t)
    a = jnp.asarray(alpha, dtype=t.dtype)
    s_ = jnp.asarray(s, dtype=t.dtype)
    return a / (jnp.asarray(1.0, dtype=t.dtype) + (a * (t - s_)) ** 2)


def d2u_axis(t: Array, alpha, s) -> Array:
    t = jnp.asarray(t)
    a = jnp.asarray(alpha, dtype=t.dtype)
    s_ = jnp.asarray(s, dtype=t.dtype)
    denom = jnp.asarray(1.0, dtype=t.dtype) + (a * (t - s_)) ** 2
    return -2.0 * (a ** 3) * (t - s_) / (denom ** 2)


# --------------------------------------------------------------------------
# 2D solution / forcing / Neumann data
# --------------------------------------------------------------------------


def u_sigma(x: Array, y: Array, alpha: float, s1: float, s2: float) -> Array:
    """Manufactured u^sigma(x, y)."""
    return u_axis(x, alpha, s1) * u_axis(y, alpha, s2)


def grad_u_sigma(
    x: Array, y: Array, alpha: float, s1: float, s2: float
) -> Tuple[Array, Array]:
    """Analytic gradient (partial_x u, partial_y u)."""
    u1 = u_axis(x, alpha, s1)
    u2 = u_axis(y, alpha, s2)
    du1 = du_axis(x, alpha, s1)
    du2 = du_axis(y, alpha, s2)
    return (du1 * u2, u1 * du2)


def f_sigma(x: Array, y: Array, alpha: float, s1: float, s2: float) -> Array:
    """Forcing f^sigma = -Laplacian u^sigma, by Aballay eq. (23)."""
    u1 = u_axis(x, alpha, s1)
    u2 = u_axis(y, alpha, s2)
    d2u1 = d2u_axis(x, alpha, s1)
    d2u2 = d2u_axis(y, alpha, s2)
    return -(d2u1 * u2 + u1 * d2u2)


def g_sigma_neumann_right(y: Array, alpha: float, s1: float, s2: float) -> Array:
    """Neumann data on x = 1 with outward normal (+1, 0): grad u . n = du1/dx (1)."""
    du1_at_1 = du_axis(jnp.asarray(1.0, dtype=y.dtype), alpha, s1)
    return du1_at_1 * u_axis(y, alpha, s2)


def g_sigma_neumann_top(x: Array, alpha: float, s1: float, s2: float) -> Array:
    """Neumann data on y = 1 with outward normal (0, +1): grad u . n = du2/dy (1)."""
    du2_at_1 = du_axis(jnp.asarray(1.0, dtype=x.dtype), alpha, s2)
    return u_axis(x, alpha, s1) * du2_at_1


def neumann_at_point(
    x: Array,
    y: Array,
    nx: Array,
    ny: Array,
    alpha: float,
    s1: float,
    s2: float,
) -> Array:
    """grad u(x, y) . (nx, ny) for general unit normal (nx, ny)."""
    gx, gy = grad_u_sigma(x, y, alpha, s1, s2)
    return gx * nx + gy * ny


__all__ = [
    "Array",
    "u_axis",
    "du_axis",
    "d2u_axis",
    "u_sigma",
    "grad_u_sigma",
    "f_sigma",
    "g_sigma_neumann_right",
    "g_sigma_neumann_top",
    "neumann_at_point",
]
