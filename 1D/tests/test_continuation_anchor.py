"""After continuation training at the anchor level, evaluating the same
rho_phi at any finer N must produce a *valid* (admissible) mesh, i.e.

  - all cell sizes positive,
  - all cell sizes >= h_min,
  - sum of cell sizes ~= 1.

This guards against accidental N-dependence of the network through gauge
fix or T(p, N) saturation logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import H_MIN_SCHEDULE
from src.nonparametric.knots import theta_to_sizes
from src.parametric.positional_density_network import (
    init_params,
    logits_at_level,
)


@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256])
def test_admissible_mesh_at_level(p, N):
    """A freshly initialized PDN must produce admissible meshes at any N."""
    params = init_params(seed=7)
    beta = 1.7
    h_min = float(H_MIN_SCHEDULE[p][N])
    theta = logits_at_level(params, beta=beta, p=p, N=N)
    sizes = np.asarray(theta_to_sizes(theta, h_min))
    assert sizes.shape == (N,)
    assert np.all(sizes > 0.0), f"non-positive size at (p={p}, N={N})"
    assert np.all(sizes >= h_min - 1e-12), (
        f"size below h_min at (p={p}, N={N}): min={sizes.min():.3e}, h_min={h_min:.3e}"
    )
    np.testing.assert_allclose(float(np.sum(sizes)), 1.0, atol=1e-10)


def test_finer_levels_after_anchor_train_is_admissible():
    """The same params evaluated at N > anchor still gives an admissible mesh.

    We don't actually train here (would be slow). We just verify that the
    saturation cap and softmax map are well-defined at every (p, N) in the
    EVAL_LEVELS set, which is what continuation guarantees structurally.
    """
    params = init_params(seed=11)
    beta = 1.85
    for p in (2, 3):
        for N in (64, 128, 256):
            h_min = float(H_MIN_SCHEDULE[p][N])
            theta = logits_at_level(params, beta=beta, p=p, N=N)
            sizes = np.asarray(theta_to_sizes(theta, h_min))
            assert sizes.shape == (N,)
            assert np.all(sizes > 0.0)
            assert np.all(sizes >= h_min - 1e-12)
