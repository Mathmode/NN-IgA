"""Regression test for audit defect D6-Exp5 (remediation T3).

The advection-diffusion residual estimator must weight each element by
rho_E^2 = h_E^2 / sigma_E = h_E^2 / eps  (eq. 12; sigma = eps, mu_E = 0), not by
the bare h_E^2 that the shipped code used.  Because eps is spatially constant per
parameter, this is a global 1/eps factor: eta scales by eps^{-1/2}, but it cancels
in the normalized training loss (28), so no retraining is required.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from common.bspline_basis import basis_batch_for_degree
from common.quadrature import rule_gl_on_elements
from common.h1_seminorm_2d import open_uniform_knots
from src.nonparametric.eta_estimator_2d import _gather_local_coeffs
from src.nonparametric.solver_2d import _element_data
from src.nonparametric.advdiff.solver_advdiff import galerkin_solve_advdiff
from src.nonparametric.advdiff.eta_estimator_advdiff import eta_squared_advdiff
from src.nonparametric.advdiff.pde import f_advdiff


def _eta_sq_reference(u_h, kx, ky, p, N, eps, b, q_est, *, with_eps_weight):
    """Independent recomputation of eta^2 with either the h_E^2 (old) or
    h_E^2/eps (eq. 12) weight."""
    a_x, b_x, U_x, idx_x = _element_data(kx, p, N)
    a_y, b_y, U_y, idx_y = _element_data(ky, p, N)
    xq, wx = rule_gl_on_elements(a_x, b_x, q_est)
    yq, wy = rule_gl_on_elements(a_y, b_y, q_est)
    Nx, dNx, d2Nx = basis_batch_for_degree(p, xq, U_x)
    Ny, dNy, d2Ny = basis_batch_for_degree(p, yq, U_y)
    C = _gather_local_coeffs(u_h, idx_x, idx_y)
    uhx = jnp.einsum("xyij,xqi,yrj->xyqr", C, dNx, Ny)
    uhxx = jnp.einsum("xyij,xqi,yrj->xyqr", C, d2Nx, Ny)
    uhyy = jnp.einsum("xyij,xqi,yrj->xyqr", C, Nx, d2Ny)
    XX = jnp.broadcast_to(xq[:, None, :, None], uhx.shape)
    YY = jnp.broadcast_to(yq[None, :, None, :], uhx.shape)
    f = f_advdiff(XX, YY, eps, b)
    R = -eps * (uhxx + uhyy) + b * uhx - f
    w = wx[:, None, :, None] * wy[None, :, None, :]
    R2 = jnp.sum(R * R * w, axis=(2, 3))
    hx = b_x - a_x
    hy = b_y - a_y
    hE_sq = hx[:, None] ** 2 + hy[None, :] ** 2
    weight = hE_sq / eps if with_eps_weight else hE_sq
    return float(jnp.maximum(jnp.sum(weight * R2), 0.0))


def test_advdiff_estimator_carries_inverse_eps_weight():
    p, N = 2, 12
    for logeps, b in [(-2.0, 2.0), (-1.5, 0.5)]:
        eps = 10.0 ** logeps
        kn = jnp.asarray(open_uniform_knots(N, p))
        res = galerkin_solve_advdiff(kn, kn, p, N, N, jnp.asarray(eps), jnp.asarray(b),
                                     q_K=4, q_F=50)
        eta_sq = float(eta_squared_advdiff(res.u_h, kn, kn, p, N, N,
                                           jnp.asarray(eps), jnp.asarray(b), q_est=50))
        ref_weighted = _eta_sq_reference(res.u_h, kn, kn, p, N, jnp.asarray(eps),
                                         jnp.asarray(b), 50, with_eps_weight=True)
        ref_bare = _eta_sq_reference(res.u_h, kn, kn, p, N, jnp.asarray(eps),
                                     jnp.asarray(b), 50, with_eps_weight=False)

        # (1) The estimator matches the eq. (12) h_E^2/eps weight, not bare h_E^2.
        assert eta_sq == pytest.approx(ref_weighted, rel=1e-10)
        # (2) eta^2 = (bare h_E^2 value) / eps  exactly (global 1/eps factor).
        assert eta_sq == pytest.approx(ref_bare / eps, rel=1e-10)
        # (3) The two conventions genuinely differ (eps << 1).
        assert not np.isclose(eta_sq, ref_bare, rtol=1e-3)
