from __future__ import annotations

"""Residual a-posteriori estimator for the helmholtz experiment (1D Helmholtz transmission).

eta^2 = sum_E h_E^2 int_E r_E^2 dx                      (element reaction residual)
        + h_I * [sigma u_h']_{x_I}^2                     (material-interface flux jump)
        + h_B * (g_N - sigma(1) u_h'(1))^2               (Neumann residual at x=1)

with the element strong residual  r_E = (sigma u_h')' - alpha u_h
                                      = sigma u_h'' + omega^2 rho u_h
(sigma, rho constant per element; f = 0). All pieces are polynomial-in-u_h, so the
volume term integrates exactly by Gauss quadrature.

NOTE (indefiniteness / pollution). Because B(omega) = K - omega^2 M is indefinite,
the residual estimator's reliability constant is FREQUENCY-DEPENDENT (the standard
pollution effect for Helmholtz): eta still drives mesh adaptation but is NOT a
guaranteed error bound uniform in omega. This mirrors the advdiff caveat (loss of
norm-equivalence) and is acceptable as a training signal.
"""

import functools

from common._precision import ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import bspline_basis_local
from common.quadrature import rule_gl_on_elements
from src.helmholtz.discretization_helmholtz import uh_deriv_at
from src.helmholtz.mesh_helmholtz import X_I, n_elem_from_knots
from src.helmholtz.pde_helmholtz import SIGMA1, SIGMA2, RHO1, RHO2, G_N

Array = jnp.ndarray
_DELTA = 1.0e-9


@functools.partial(jax.jit, static_argnames=("degree", "nq"))
def eta_squared(knots: Array, degree: int, u_full: Array, omega: Array, nq: int) -> Array:
    """Total residual estimator eta^2 (element + interface jump + Neumann)."""
    p = int(degree)
    dtype = knots.dtype
    omega_t = jnp.asarray(omega, dtype=dtype)
    n_eff = n_elem_from_knots(knots, p)
    a = knots[p : p + n_eff]
    b = knots[p + 1 : p + n_eff + 1]
    he = b - a

    xq, wq = rule_gl_on_elements(a, b, int(nq))
    N_flat, _dN, d2N_flat, _spans = bspline_basis_local(xq.reshape(-1), knots, p)
    Nb = N_flat.reshape(n_eff, int(nq), p + 1)
    d2Nb = d2N_flat.reshape(n_eff, int(nq), p + 1)

    cols = jnp.arange(n_eff, dtype=jnp.int32)[:, None] + jnp.arange(p + 1, dtype=jnp.int32)[None, :]
    u_loc = u_full[cols]                                   # (n_eff, p+1)

    uh = jnp.einsum("eqi,ei->eq", Nb, u_loc)
    uh_xx = jnp.einsum("eqi,ei->eq", d2Nb, u_loc)

    xm = 0.5 * (a + b)
    sig = jnp.where(xm < jnp.asarray(X_I, dtype), jnp.asarray(SIGMA1, dtype), jnp.asarray(SIGMA2, dtype))
    rho = jnp.where(xm < jnp.asarray(X_I, dtype), jnp.asarray(RHO1, dtype), jnp.asarray(RHO2, dtype))

    r = sig[:, None] * uh_xx + (omega_t * omega_t) * rho[:, None] * uh   # (n_eff, nq)
    int_r2 = jnp.sum(wq * r * r, axis=1)                                  # (n_eff,)
    eta2_elem = jnp.sum((he * he) * int_r2)

    # Interface flux jump [sigma u_h'] at x_I (u_h' discontinuous across the C^0 knot).
    duL = uh_deriv_at(knots, p, u_full, jnp.asarray([X_I - _DELTA], dtype=dtype))[0]
    duR = uh_deriv_at(knots, p, u_full, jnp.asarray([X_I + _DELTA], dtype=dtype))[0]
    jump = jnp.asarray(SIGMA2, dtype) * duR - jnp.asarray(SIGMA1, dtype) * duL
    h_I = jnp.minimum(he[0], he[-1])                       # local size near the interface
    eta2_jump = h_I * jump * jump

    # Neumann residual g_N - sigma(1) u_h'(1).
    du1 = uh_deriv_at(knots, p, u_full, jnp.asarray([1.0 - _DELTA], dtype=dtype))[0]
    flux_err = jnp.asarray(G_N, dtype) - jnp.asarray(SIGMA2, dtype) * du1
    eta2_bc = he[-1] * flux_err * flux_err

    return eta2_elem + eta2_jump + eta2_bc


@functools.partial(jax.jit, static_argnames=("degree", "nq"))
def eta(knots: Array, degree: int, u_full: Array, omega: Array, nq: int) -> Array:
    return jnp.sqrt(jnp.maximum(eta_squared(knots, degree, u_full, omega, nq),
                                jnp.asarray(0.0, dtype=knots.dtype)))


__all__ = ["eta_squared", "eta"]
