from __future__ import annotations

import numpy as np
import pytest

from src.config import T_SCHEDULE
from src.nonparametric.alg1_ladder import bound_for, normalized_objective, run_ladder

BETA = 1.5543313068309408
LEVELS = (2, 4, 8, 16, 32)


@pytest.mark.parametrize("p", [2, 3])
def test_bound_comes_from_config(p):
    for N in LEVELS:
        assert bound_for(p, N) == float(T_SCHEDULE[p][N])


@pytest.mark.parametrize("kind,sign", [("eta", 1.0), ("ritz", -1.0)])
def test_objective_is_normalized_by_the_uniform_mesh(kind, sign):
    f = normalized_objective(kind, 2, 8, BETA)
    assert float(f(np.zeros(8))) == pytest.approx(sign, abs=1e-12)


@pytest.mark.parametrize("kind", ["eta", "ritz"])
def test_ladder_stays_in_the_box_and_improves_on_uniform(kind):
    p = 2
    out = run_ladder(kind, p, BETA, LEVELS, 300)
    for N in LEVELS:
        theta = np.asarray(out[N][0])
        assert np.all(np.isfinite(theta))
        assert np.max(np.abs(theta)) <= bound_for(p, N) + 1e-12
    f = normalized_objective(kind, p, LEVELS[-1], BETA)
    assert float(f(out[LEVELS[-1]][0])) < float(f(np.zeros(LEVELS[-1])))
