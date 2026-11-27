from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import T_ARCTAN, T_LSHAPE, T_LSHAPE_BY_P
from src.nonparametric import alg1_ladder_2d as A

REPO = Path(__file__).resolve().parents[2]
SIGMA = (0.6951927961775606, 7.847599703514611)
LEVELS = (4, 8, 16, 32)


def _exact_energy(sigma):
    E = pd.read_csv(REPO / "references_v2" / "lshape_energy_ngsolve.csv")
    d = np.abs(E.sigma1 - sigma[0]) + np.abs(E.sigma2 - sigma[1])
    assert d.min() < 1e-9
    return -0.5 * float(E.F_order14.iloc[int(np.argmin(d))])


@pytest.mark.parametrize("p", [2, 3])
def test_bound_comes_from_config(p):
    assert A.lshape_problem(SIGMA).bound(p) == float(T_LSHAPE_BY_P.get(p, T_LSHAPE))
    assert A.arctan_problem((10.0, 0.5, 0.5)).bound(p) == float(T_ARCTAN)
    assert A.oblique_problem(0.0).bound(p) == float(T_ARCTAN)


def test_objective_is_normalized_by_the_uniform_mesh():
    prob = A.lshape_problem(SIGMA)
    zeros = np.zeros(prob.n_blocks * prob.block_size(8))
    assert float(A.objective(prob, "ritz", 2, 8)(zeros)[0]) == pytest.approx(-1.0, abs=1e-12)
    assert float(A.objective(prob, "eta", 2, 8)(zeros)[0]) == pytest.approx(1.0, abs=1e-12)


def test_lshape_ritz_bounded_ladder_stays_admissible():
    p = 2
    prob = A.lshape_problem(SIGMA)
    T = prob.bound(p)
    J_exact = _exact_energy(SIGMA)
    out = A.run_ladder(prob, "ritz", p, LEVELS, 200)
    errors = []
    for N in LEVELS:
        theta = np.asarray(out[N]["theta"])
        assert np.all(np.isfinite(theta))
        assert np.max(np.abs(theta)) <= T + 1e-12
        r, kx, ky, n, eta2 = A.solution(prob, out[N]["theta"], p, N)
        J_h = float(r.ritz_energy)
        assert math.isfinite(J_h) and math.isfinite(eta2)
        assert J_h >= J_exact
        h = np.diff(np.unique(np.asarray(kx)))
        assert h.max() / h.min() <= math.exp(2 * T) * (1 + 1e-9)
        errors.append(math.sqrt(-2.0 * (J_h - J_exact) / (2.0 * J_exact)))
    assert all(b < a for a, b in zip(errors, errors[1:]))
    r0, *_ = A.solution(prob, np.zeros_like(theta), p, LEVELS[-1])
    e0 = math.sqrt(-2.0 * (float(r0.ritz_energy) - J_exact) / (2.0 * J_exact))
    assert errors[-1] <= 0.3 * e0

def test_lbfgs_does_not_stop_on_the_relative_reduction():
    prob = A.lshape_problem((0.4281332398719394, 2.3357214690901213))
    out = A.run_ladder(prob, "ritz", 3, LEVELS, 200)
    assert out[LEVELS[-1]]["nit"] > 5
