"""advdiff: manufactured advection–diffusion boundary-layer problem.

PDE on Omega = (0, 1)^2:

    -eps * Laplacian u + b * du/dx = f_nu   in Omega,
                                 u = 0       on  the whole boundary  d Omega,

with nu = (eps, b). The convective term ``b du/dx`` makes the operator
non-symmetric; for small ``eps`` the solution develops an anisotropic
boundary layer of width ~ delta = eps / b at the outflow boundary x = 1.

Manufactured solution
---------------------
Chosen so that u = 0 on all four sides of the unit square:

    u*(x, y; nu) = x * (1 - exp(z)) * sin(pi y),
        with  z = (x - 1) / delta,   delta = eps / b.

  * At x = 0: z = -1/delta -> exp(z) ~ 0, and the leading factor x = 0,
    so u* = 0.
  * At x = 1: z = 0 -> (1 - exp(0)) = 0, so u* = 0.
  * sin(pi y) = 0 at y = 0 and y = 1, so u* = 0 on the bottom/top edges.

The (1 - exp(z)) factor produces the exp((x-1)/delta) boundary layer at
x = 1; for eps = 1e-2, b = 1 the layer width is delta = 1e-2.

Analytic gradient (used by the FE assembly and the H1-seminorm metric):
    du*/dx = sin(pi y) * A'(x)
    du*/dy = pi * A(x) * cos(pi y)
  where
    A(x)  = x * (1 - exp(z))
    A'(x) = (1 - exp(z)) - (x / delta) * exp(z).

Manufactured source (derived analytically by the author; substituted
literally here — NOT re-derived symbolically):

    f_nu(x, y) = sin(pi y) * [ b * (1 + exp(z)) + eps * pi^2 * x * (1 - exp(z)) ],

with the same z = (x - 1)/delta, delta = eps/b.

Author's boundary sanity checks (recorded for reference, not re-run):
  * x = 0, (eps=1e-2, b=1): z = -100, exp(z) ~ 0 -> f(0,y) ~ b sin(pi y),
    u*(0,y) = 0.
  * x = 1: z = 0, exp(z) = 1 -> f(1,y) = 2 b sin(pi y), u*(1,y) = 0.
  * b -> 0 limit: f -> eps pi^2 x sin(pi y), u* -> x sin(pi y), and
    -eps Laplacian u* = eps pi^2 x sin(pi y) = f. Consistent.

Conventions match ``arctan/pde.py``: pure JAX, broadcasting over
``(x, y)`` of arbitrary shape, ``eps`` and ``b`` as scalars, ``DEFAULT_DTYPE``
precision, ``ensure_double_precision()`` at module load.
"""
from __future__ import annotations

from typing import Tuple

from common._precision import ensure_double_precision

ensure_double_precision()

import jax.numpy as jnp

Array = jnp.ndarray


# --------------------------------------------------------------------------
# Internal: boundary-layer variable z = (x - 1) / delta, delta = eps / b.
# --------------------------------------------------------------------------


def _z_and_delta(x: Array, eps, b) -> Tuple[Array, Array]:
    x = jnp.asarray(x)
    eps_t = jnp.asarray(eps, dtype=x.dtype)
    b_t = jnp.asarray(b, dtype=x.dtype)
    delta = eps_t / b_t
    z = (x - jnp.asarray(1.0, dtype=x.dtype)) / delta
    return z, delta


# --------------------------------------------------------------------------
# Manufactured solution / gradient / source.
# --------------------------------------------------------------------------


def u_exact_advdiff(x: Array, y: Array, eps, b) -> Array:
    """Manufactured solution u*(x, y; nu) = x (1 - exp(z)) sin(pi y)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z, _delta = _z_and_delta(x, eps, b)
    pi = jnp.asarray(jnp.pi, dtype=x.dtype)
    return x * (jnp.asarray(1.0, dtype=x.dtype) - jnp.exp(z)) * jnp.sin(pi * y)


def grad_u_exact_advdiff(x: Array, y: Array, eps, b) -> Tuple[Array, Array]:
    """Analytic gradient (partial_x u*, partial_y u*).

        du/dx = sin(pi y) * A'(x),     du/dy = pi * A(x) * cos(pi y),
        A(x)  = x (1 - exp(z)),
        A'(x) = (1 - exp(z)) - (x / delta) exp(z).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z, delta = _z_and_delta(x, eps, b)
    one = jnp.asarray(1.0, dtype=x.dtype)
    pi = jnp.asarray(jnp.pi, dtype=x.dtype)
    ez = jnp.exp(z)
    A = x * (one - ez)
    dA = (one - ez) - (x / delta) * ez
    du_dx = jnp.sin(pi * y) * dA
    du_dy = pi * A * jnp.cos(pi * y)
    return du_dx, du_dy


def f_advdiff(x: Array, y: Array, eps, b) -> Array:
    """Manufactured source (verbatim from the spec; not re-derived):

        f = sin(pi y) [ b (1 + exp(z)) + eps pi^2 x (1 - exp(z)) ].
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z, _delta = _z_and_delta(x, eps, b)
    one = jnp.asarray(1.0, dtype=x.dtype)
    pi = jnp.asarray(jnp.pi, dtype=x.dtype)
    eps_t = jnp.asarray(eps, dtype=x.dtype)
    b_t = jnp.asarray(b, dtype=x.dtype)
    ez = jnp.exp(z)
    bracket = b_t * (one + ez) + eps_t * (pi * pi) * x * (one - ez)
    return jnp.sin(pi * y) * bracket


__all__ = [
    "Array",
    "u_exact_advdiff",
    "grad_u_exact_advdiff",
    "f_advdiff",
]
