"""Regression test for audit defect R1 (remediation T2).

The Helmholtz estimator's interface flux-jump term must be weighted by the
smaller of the two INTERFACE-ADJACENT element sizes, h_I = min(h_{E+}, h_{E-}),
not by min(he[0], he[-1]) (the domain-boundary elements at x=0 and x=1).

We build an asymmetric graded mesh where the interface-adjacent elements are far
smaller than the boundary elements, so the two conventions give different
weights, then verify that the jump contribution backed out of ``eta2`` equals
h_I * jump^2 (interface-adjacent) and NOT min(he[0], he[-1]) * jump^2.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from common.quadrature import rule_gl_on_elements
from common.bspline_basis import bspline_basis_local
from src.helmholtz.mesh_helmholtz_global import (
    global_knot_map, solve_u, eta2, uh_deriv, quad_order, _spans,
    K2, SIGMA1, SIGMA2, RHO2, OMEGA, G_N, X_I,
)


def _elem_neumann_only(knots, p, u, rho1, nq):
    """Recompute the element-residual + Neumann terms of eta^2 (everything except
    the interface jump), so the jump contribution can be isolated as a residual."""
    dtype = knots.dtype
    om = jnp.asarray(OMEGA, dtype)
    n, a, b = _spans(knots, p)
    he = b - a
    xq, wq = rule_gl_on_elements(a, b, int(nq))
    Nf, _d, d2f, _s = bspline_basis_local(xq.reshape(-1), knots, p)
    Nb = Nf.reshape(n, int(nq), p + 1)
    d2b = d2f.reshape(n, int(nq), p + 1)
    cols = jnp.arange(n)[:, None] + jnp.arange(p + 1)[None, :]
    ul = u[cols]
    uh = jnp.einsum("eqi,ei->eq", Nb, ul)
    uxx = jnp.einsum("eqi,ei->eq", d2b, ul)
    xm = 0.5 * (a + b)
    sig = jnp.where(xm < X_I, SIGMA1, SIGMA2)
    rho = jnp.where(xm < X_I, rho1, RHO2)
    r = sig[:, None] * uxx + (om * om) * rho[:, None] * uh
    e_elem = float(jnp.sum((he * he) * jnp.sum(wq * r * r, axis=1)))
    du1 = float(uh_deriv(knots, p, u, jnp.asarray([1.0 - 1e-9]))[0])
    fe = G_N - SIGMA2 * du1
    e_neu = float(np.asarray(he)[-1] * fe * fe)
    return e_elem, e_neu


def test_jump_weight_uses_interface_adjacent_elements():
    p, N, c = 2, 16, 3.0
    h_min = 1e-7
    rho1 = (c * K2) ** 2 * SIGMA1
    nq = quad_order(p)

    # Grade elements toward x_I = 0.5: softmax maps large logits to large sizes,
    # so give the boundary segments the largest logits (a parabola with its
    # minimum at xi=0.5). The interface-adjacent elements then become the smallest,
    # while the domain-boundary elements (he[0], he[-1]) stay large -- making
    # min(h_left,h_right) far smaller than min(he[0],he[-1]).
    n_seg = N - 1
    xi = (np.arange(n_seg) + 0.5) / n_seg
    theta = jnp.asarray(5.0 * (2.0 * (xi - 0.5)) ** 2, dtype=jnp.float64)
    knots = global_knot_map(theta, N, p, c, h_min, use_hmax=False)

    n, a, b = _spans(knots, p)
    he = np.asarray(b - a)
    xm = 0.5 * (np.asarray(a) + np.asarray(b))
    real = he > 1e-13
    # interface-adjacent real element sizes
    left = he[(xm < X_I) & real]
    right = he[(xm > X_I) & real]
    h_left, h_right = left[-1], right[0]     # nearest to x_I on each side
    h_I_expected = min(h_left, h_right)
    boundary_min = min(he[real][0], he[real][-1])

    # The test is only meaningful if the two conventions differ substantially.
    assert h_I_expected < 0.3 * boundary_min, (
        f"mesh not graded enough: h_I={h_I_expected:.3e} vs boundary_min={boundary_min:.3e}"
    )

    u = solve_u(knots, p, jnp.asarray(rho1), nq)
    eta_sq = float(eta2(knots, p, u, jnp.asarray(rho1), nq))
    e_elem, e_neu = _elem_neumann_only(knots, p, u, rho1, nq)
    e_jump = eta_sq - e_elem - e_neu

    duL = float(uh_deriv(knots, p, u, jnp.asarray([X_I - 1e-9]))[0])
    duR = float(uh_deriv(knots, p, u, jnp.asarray([X_I + 1e-9]))[0])
    jump = SIGMA2 * duR - SIGMA1 * duL
    assert abs(jump) > 0.0

    weight_backed_out = e_jump / (jump * jump)
    # Matches the interface-adjacent weight, NOT the domain-boundary weight.
    assert weight_backed_out == pytest.approx(h_I_expected, rel=1e-6)
    assert not np.isclose(weight_backed_out, boundary_min, rtol=1e-3)
