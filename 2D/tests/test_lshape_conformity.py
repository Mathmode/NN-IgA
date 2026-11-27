from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.nonparametric.lshape.conformity import (
    active_basis_sum_on_quadrant,
    dirichlet_violation,
    geometric_lshape_knots,
    random_graded_lshape_knots,
    support_rule_free_mask,
)
from src.nonparametric.solver_2d import free_mask_lshape

REPO = Path(__file__).resolve().parents[2]
REFERENCE_V1 = REPO / "references" / "reference_lshape.pkl"
REFERENCE_V2 = REPO / "references_v2" / "reference_lshape.pkl"


def _knot_cases():
    rng = np.random.default_rng(0)
    cases = []
    for p in (2, 3, 5):
        for n_half in (4, 8, 16):
            cases.append((p, random_graded_lshape_knots(rng, p, n_half), random_graded_lshape_knots(rng, p, n_half)))
    cases.append((5, geometric_lshape_knots(5, 64, 1.15), geometric_lshape_knots(5, 64, 1.15)))
    return cases


KNOT_CASES = _knot_cases()


@pytest.mark.parametrize("case", range(len(KNOT_CASES)))
def test_active_basis_vanishes_on_closed_quadrant(case):
    p, kx, ky = KNOT_CASES[case]
    free = support_rule_free_mask(kx, ky, p)
    assert float(np.max(active_basis_sum_on_quadrant(kx, ky, p, free))) == 0.0


@pytest.mark.parametrize("case", range(len(KNOT_CASES)))
@pytest.mark.parametrize("jit", [False, True])
def test_solver_mask_matches_support_rule(case, jit):
    p, kx, ky = KNOT_CASES[case]
    fn = jax.jit(free_mask_lshape, static_argnums=2) if jit else free_mask_lshape
    mask = np.asarray(fn(jnp.asarray(kx), jnp.asarray(ky), p)).astype(bool)
    assert np.array_equal(mask, support_rule_free_mask(kx, ky, p))


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _worst_violation(cache):
    worst = {"quadrant": 0.0, "boundary": 0.0, "coefficients": 0.0}
    for entry in cache.values():
        v = dirichlet_violation(entry["u_coeffs"], entry["knots_x"], entry["knots_y"], int(entry["p"]))
        worst = {k: max(worst[k], v[k]) for k in worst}
    return worst


@pytest.mark.skipif(not REFERENCE_V2.exists(), reason="references_v2 cache not generated yet (Hyperion job)")
def test_reference_v2_vanishes_on_quadrant_and_boundary():
    worst = _worst_violation(_load(REFERENCE_V2))
    assert worst["coefficients"] == 0.0
    assert worst["quadrant"] < 1e-14
    assert worst["boundary"] < 1e-14


@pytest.mark.xfail(strict=True, reason="v1 cache leaves the re-entrant row y=1/2, x>1/2 unconstrained")
@pytest.mark.skipif(not REFERENCE_V1.exists(), reason="v1 reference cache not present")
def test_reference_v1_vanishes_on_quadrant_and_boundary():
    worst = _worst_violation(_load(REFERENCE_V1))
    assert worst["coefficients"] == 0.0
    assert worst["quadrant"] < 1e-14
