"""Residual a-posteriori estimator η² for the 2D advection–diffusion
problem (advdiff), VOLUME TERM ONLY.

This is the volume-only residual estimator (no inter-element jump terms:
the IGA basis is C^{p-1}, so internal flux jumps vanish). The element weight
follows the manuscript's coefficient-dependent scaling ρ_E of eq. (12): with
σ ≡ ε and μ_E = α − ½∇·β = 0 here, ρ_E² = h_E²/σ_E = h_E²/ε (audit D6 / T3
fix — the previous code used h_E² without the σ division, i.e. dropped the
diffusion weight).

Element residual of the strong form ``-eps Lap u + b u_x - f``:

    R_E(x, y) = -eps (uhxx + uhyy) + b uhx - f_nu(x, y).

Per-element contribution and global estimator (eq. (13)):

    η²_E = (h_E²/ε) ∫_E R_E² dxdy,   h_E² = h_x² + h_y²,
    η²   = Σ_E η²_E.

Because ε = 10^{ℓε} is spatially constant per parameter ν, the 1/ε factor is a
global multiplier: it scales η by ε^{-1/2} but cancels in the normalized
training loss (28) (numerator and uniform-mesh denominator share it), so the
trained meshes are unchanged and no retraining is required. It does change the
reported effectivity index I_eff = η/|e|_{H1} by ε^{-1/2}, aligning it with the
manuscript's ε^{1/2}-scaling argument in §5.5.

Mirrors ``_eta_volume_arctan``: same einsum patterns, same ``q_est``
quadrature parameter (default 50), same ``jnp.maximum(., 0.0)`` ulp-guard,
same static_argnames. Low-level helpers are imported from the existing
modules — nothing is copied.
"""
from __future__ import annotations

import functools

from common._precision import DEFAULT_DTYPE, ensure_double_precision

ensure_double_precision()

import jax
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from src.nonparametric.eta_estimator_2d import _gather_local_coeffs
from src.nonparametric.solver_2d import _element_data

Array = jnp.ndarray


@functools.partial(jax.jit, static_argnames=("p", "n_elem_x", "n_elem_y", "q_est"))
def eta_squared_advdiff(
    u_coeffs: Array,
    knots_x: Array,
    knots_y: Array,
    p: int,
    n_elem_x: int,
    n_elem_y: int,
    eps: Array,
    b: Array,
    *,
    q_est: int = 50,
) -> Array:
    """Residual a-posteriori estimator η² for the adv-diff problem,
    volume term only. Non-negative scalar, JIT-compatible.

    R_E = -eps (uhxx + uhyy) + b uhx - f;  η² = Σ_E (h_E²/ε) ‖R_E‖²_{L²(E)}.
    """
    from src.nonparametric.advdiff.pde import f_advdiff

    a_x, b_x, U_local_x, idx_local_x = _element_data(knots_x, p, n_elem_x)
    a_y, b_y, U_local_y, idx_local_y = _element_data(knots_y, p, n_elem_y)

    xq, wq_x = rule_gl_on_elements(a_x, b_x, q_est)
    yq, wq_y = rule_gl_on_elements(a_y, b_y, q_est)
    Nx, dNx, d2Nx = basis_batch_for_degree(p, xq, U_local_x)
    Ny, dNy, d2Ny = basis_batch_for_degree(p, yq, U_local_y)

    C_loc = _gather_local_coeffs(u_coeffs, idx_local_x, idx_local_y)
    # einsum indices (same as _eta_volume_arctan):
    #   x = element-x, y = element-y, i = basis-x, j = basis-y,
    #   q = quad-x,    r = quad-y.
    uhx = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, dNx, Ny)    # ∂u_h/∂x
    uhxx = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, d2Nx, Ny)  # ∂²u_h/∂x²
    uhyy = jnp.einsum("xyij,xqi,yrj->xyqr", C_loc, Nx, d2Ny)  # ∂²u_h/∂y²

    XX = jnp.broadcast_to(xq[:, None, :, None], uhx.shape)
    YY = jnp.broadcast_to(yq[None, :, None, :], uhx.shape)
    f_vals = f_advdiff(XX, YY, eps, b)

    eps_t = jnp.asarray(eps, dtype=DEFAULT_DTYPE)
    b_t = jnp.asarray(b, dtype=DEFAULT_DTYPE)
    R = -eps_t * (uhxx + uhyy) + b_t * uhx - f_vals

    w_2d = wq_x[:, None, :, None] * wq_y[None, :, None, :]
    R2 = jnp.sum(R * R * w_2d, axis=(2, 3))   # (n_elem_x, n_elem_y)
    hx = b_x - a_x
    hy = b_y - a_y
    hE_sq = hx[:, None] ** 2 + hy[None, :] ** 2
    # rho_E^2 = h_E^2 / sigma_E = h_E^2 / eps  (eq. 12; sigma = eps here). Audit
    # D6/T3: the previous code omitted the 1/eps diffusion weight.
    return jnp.maximum(jnp.sum((hE_sq / eps_t) * R2), 0.0)


def eta_uniform_squared_advdiff(
    nu: Array,
    N: int,
    p: int,
    *,
    q_est: int = 50,
    q_K: int = None,
    q_F: int = 50,
) -> Array:
    """Compute η²(θ_uniform; ν) for advdiff (advection–diffusion) — the eq-26
    residual-loss denominator on the uniform ``N×N`` mesh at the current
    level. ``nu = (logeps, b)``; mirrors ``eta_uniform_squared_arctan``."""
    from common.h1_seminorm_2d import open_uniform_knots
    from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff

    if q_K is None:
        q_K = p + 1
    eps = jnp.power(jnp.asarray(10.0, dtype=DEFAULT_DTYPE), nu[0])
    b = nu[1]
    knots = jnp.asarray(open_uniform_knots(int(N), int(p)), dtype=DEFAULT_DTYPE)
    res = galerkin_solve_advdiff(
        knots, knots, int(p), int(N), int(N),
        eps, b, q_K=int(q_K), q_F=int(q_F),
    )
    return eta_squared_advdiff(
        res.u_h, knots, knots, int(p), int(N), int(N),
        eps, b, q_est=int(q_est),
    )


__all__ = [
    "Array",
    "eta_squared_advdiff",
    "eta_uniform_squared_advdiff",
]
