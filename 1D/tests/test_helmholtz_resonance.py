from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import src.helmholtz.mesh_helmholtz_global as G
import src.helmholtz.resonance as R


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("c", [2.0, 3.1, 4.4])
def test_independent_solver_matches_repository_solver(p, c):
    kn = np.asarray(G.uniform_two_patch_knots(32, p))
    u = G.solve_u(jnp.asarray(kn), p, jnp.asarray(G.rho1_of_c(c)), G.quad_order(p))
    err_repo = float(G.h1_rel(kn, p, u, G.ExactContrast(c)))
    err_indep = R.h1_rel(kn, p, R.solve(kn, p, c), c)
    assert err_indep == pytest.approx(err_repo, rel=1e-8)


def test_exact_solution_satisfies_interface_and_boundary_conditions():
    c, eps = 3.7, 1e-9
    u, du = R.exact_solution(c), R.exact_derivative(c)
    assert abs(u(np.array([0.0]))[0]) < 1e-12
    assert u(np.array([R.X_I - eps]))[0] == pytest.approx(u(np.array([R.X_I + eps]))[0], abs=1e-6)
    flux_l = R.SIGMA1 * du(np.array([R.X_I - eps]))[0]
    flux_r = R.SIGMA2 * du(np.array([R.X_I + eps]))[0]
    assert flux_l == pytest.approx(flux_r, rel=1e-6)
    assert R.SIGMA2 * du(np.array([1.0]))[0] == pytest.approx(R.G_N, rel=1e-12)


def test_discrete_resonances_converge_to_continuous_ones():
    shifts = []
    for N in (32, 64, 128):
        c_h = R.discrete_resonances(R.uniform_knots(N, 3), 3)
        shifts.append(R.associated_by_index(c_h, 4.8) - 4.8)
    assert shifts[0] > shifts[1] > shifts[2] > 0.0
    assert shifts[2] < 1e-4


def test_system_is_singular_at_a_discrete_resonance():
    kn = R.uniform_knots(32, 2)
    blocks = R.assemble_blocks(kn, 2)
    c_h = R.associated_by_index(R.discrete_resonances(kn, 2, blocks), 4.8)
    smin = lambda c: np.linalg.svd(R.system(kn, 2, c, blocks)[1:, 1:], compute_uv=False)[-1]
    assert smin(c_h) < 1e-8 * smin(c_h + 0.05)


def test_continuous_resonances_are_two_fifths():
    assert np.allclose(R.continuous_resonances(5.5)[-3:], [4.4, 4.8, 5.2])
    assert R.mode_index(4.8) == 11
