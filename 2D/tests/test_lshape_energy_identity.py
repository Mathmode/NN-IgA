from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from src.nonparametric.lshape.conformity import (
    design_matrix,
    geometric_lshape_knots,
    support_rule_free_mask,
)
from src.nonparametric.lshape.energy_error import (
    discrete_energy,
    endpoint_order,
    fitted_order,
    local_orders,
    relative_energy_error,
)
from src.nonparametric.solver_2d import galerkin_solve_lshape

SIGMAS = [(1.0, 1.0), (0.1, 10.0), (3.0, 0.2)]


def _solve(kn, p, s1, s2):
    n = kn.shape[0] - 2 * p - 1
    return galerkin_solve_lshape(jnp.asarray(kn), jnp.asarray(kn), p, n, n,
                                 jnp.asarray(s1), jnp.asarray(s2), q_K=p + 1, q_F=p + 1)


def _prolong(u_c, kn_c, kn_f, p):
    xs = np.linspace(0.0, 1.0, 8 * kn_f.shape[0])
    Bc, Bf = design_matrix(kn_c, p, xs), design_matrix(kn_f, p, xs)
    P = np.linalg.lstsq(Bf, Bc, rcond=None)[0]
    return P @ np.asarray(u_c) @ P.T


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("sigma", SIGMAS)
def test_energy_identity_between_nested_spaces(p, sigma):
    kn_c = geometric_lshape_knots(p, 2, 1.0)
    kn_f = geometric_lshape_knots(p, 4, 1.0)
    coarse, fine = _solve(kn_c, p, *sigma), _solve(kn_f, p, *sigma)
    u_cf = _prolong(coarse.u_h, kn_c, kn_f, p)
    assert np.max(np.abs(u_cf * ~support_rule_free_mask(kn_f, kn_f, p))) < 1e-12
    e = (np.asarray(fine.u_h) - u_cf).reshape(-1)
    a_ee = float(e @ np.asarray(fine.K) @ e)
    F_c, F_f = discrete_energy(coarse.ritz_energy), discrete_energy(fine.ritz_energy)
    assert F_c <= F_f
    assert a_ee == pytest.approx(F_f - F_c, rel=1e-9, abs=1e-15)
    assert relative_energy_error(F_c, F_f) == pytest.approx(np.sqrt(a_ee / F_f), rel=1e-8)


def test_orders_of_exact_power_law():
    n = np.array([4.0, 8.0, 16.0, 32.0])
    e = 3.0 * n ** -1.5
    assert endpoint_order(e, n) == pytest.approx(1.5)
    assert fitted_order(e, n) == pytest.approx(1.5)
    assert np.allclose(local_orders(e, n), 1.5)


def test_ngsolve_reference_bounds_iga_energy():
    pytest.importorskip("ngsolve")
    from src.nonparametric.lshape.ngsolve_energy import (
        INTERFACE, OUTER, boundary_names, build_lshape_mesh, solve_energy,
    )
    mesh = build_lshape_mesh(levels=10)
    assert set(boundary_names(mesh)) == {OUTER, INTERFACE}
    for s1, s2 in SIGMAS:
        F_ng, _ = solve_energy(mesh, s1, s2, 8)
        F_h = discrete_energy(_solve(geometric_lshape_knots(3, 8, 1.0), 3, s1, s2).ritz_energy)
        assert F_h <= F_ng
        assert 0.0 < relative_energy_error(F_h, F_ng) < 0.2
