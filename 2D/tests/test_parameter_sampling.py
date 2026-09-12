"""Tests for src.shared.parameter_sampling."""
from __future__ import annotations

import numpy as np
import pytest

from common.parameter_sampling import (
    build_arctan_grid,
    build_lshape_grid,
    cartesian_grid,
    sample_alpha_log_inv_base2,
    sample_uniform_linear,
    sample_uniform_log_base10,
    split_with_forced_corners,
)


# --------------------------------------------------------------------------
# Marginals
# --------------------------------------------------------------------------


def test_alpha_endpoints_match_aballay_range():
    """First / last alpha must hit alpha_min=1 and alpha_max=20."""
    alpha = sample_alpha_log_inv_base2(20, 1.0, 20.0)
    assert alpha.shape == (20,)
    # ascending order (we sort inside the helper)
    assert np.all(np.diff(alpha) >= 0)
    # endpoints
    assert np.isclose(alpha[0], 1.0, atol=1e-12)
    assert np.isclose(alpha[-1], 20.0, atol=1e-12)


def test_alpha_biases_toward_max():
    """Most points clustered near alpha_max under the reversed-base-2 rule."""
    alpha = sample_alpha_log_inv_base2(20, 1.0, 20.0)
    median = float(np.median(alpha))
    # Reverse-base-2 -> median above the linear midpoint (10.5).
    assert median > 10.5


def test_sigma_log10_extremes():
    sigma = sample_uniform_log_base10(20, -1.0, 1.0)
    assert np.isclose(sigma[0], 1e-1, atol=1e-12)
    assert np.isclose(sigma[-1], 1e1, atol=1e-12)
    # uniform in log-space
    log_sigma = np.log10(sigma)
    diffs = np.diff(log_sigma)
    assert np.allclose(diffs, diffs[0], atol=1e-12)


def test_uniform_linear():
    s = sample_uniform_linear(10, 0.1, 0.9)
    assert np.isclose(s[0], 0.1)
    assert np.isclose(s[-1], 0.9)
    assert np.allclose(np.diff(s), (0.9 - 0.1) / 9.0)


# --------------------------------------------------------------------------
# Cartesian grid + split
# --------------------------------------------------------------------------


def test_cartesian_grid_shape_and_order():
    a = np.array([1.0, 2.0])
    b = np.array([10.0, 20.0, 30.0])
    grid = cartesian_grid(a, b)
    assert grid.shape == (6, 2)
    # First axis varies slowest (ij order):
    assert grid[0].tolist() == [1.0, 10.0]
    assert grid[1].tolist() == [1.0, 20.0]
    assert grid[2].tolist() == [1.0, 30.0]
    assert grid[3].tolist() == [2.0, 10.0]


def test_split_with_forced_corners_partitions():
    a = np.linspace(0, 1, 4)
    b = np.linspace(0, 1, 5)
    axes = (a, b)
    grid = cartesian_grid(*axes)
    splits = split_with_forced_corners(grid, axes, train_frac=0.7, seed=0)
    train = splits["train"]
    test = splits["test"]
    # Sizes
    assert len(train) + len(test) == grid.shape[0]
    # No overlap (by index)
    intersect = np.intersect1d(splits["train_idx"], splits["test_idx"])
    assert intersect.size == 0


def test_split_forces_all_corners_into_train():
    a = np.linspace(0, 1, 4)
    b = np.linspace(0, 1, 5)
    axes = (a, b)
    grid = cartesian_grid(*axes)
    splits = split_with_forced_corners(grid, axes, train_frac=0.7, seed=0)
    expected_corners = np.array([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]
    ])
    train = splits["train"]
    for c in expected_corners:
        # match within tolerance
        matched = np.any(np.all(np.abs(train - c[None, :]) < 1e-12, axis=1))
        assert matched, f"corner {c.tolist()} not found in training set"


def test_split_is_seed_deterministic():
    a = np.linspace(0, 1, 6)
    b = np.linspace(0, 1, 7)
    axes = (a, b)
    grid = cartesian_grid(*axes)
    s1 = split_with_forced_corners(grid, axes, 0.7, seed=42)
    s2 = split_with_forced_corners(grid, axes, 0.7, seed=42)
    s3 = split_with_forced_corners(grid, axes, 0.7, seed=43)
    assert np.array_equal(s1["train_idx"], s2["train_idx"])
    # Different seed -> different ordering of non-corner indices in general.
    assert not np.array_equal(s1["train_idx"], s3["train_idx"])


# --------------------------------------------------------------------------
# Convenience builders match Aballay sizes
# --------------------------------------------------------------------------


def test_p2_grid_sizes():
    # Aballay 4.3.1 grid: 20 (alpha) x 10 (s1) x 10 (s2) = 2000 tuples,
    # split 70/30 -> 1400 train, 600 test.
    s = build_arctan_grid(seed=0)
    assert s["grid"].shape == (2000, 3)
    assert s["train"].shape == (1400, 3)
    assert s["test"].shape == (600, 3)
    # Corners present
    corner_alpha_values = (1.0, 20.0)
    corner_s_values = (0.1, 0.9)
    for ca in corner_alpha_values:
        for cs1 in corner_s_values:
            for cs2 in corner_s_values:
                tgt = np.array([ca, cs1, cs2])
                hit = np.any(np.all(np.abs(s["train"] - tgt) < 1e-10, axis=1))
                assert hit, f"arctan corner {tgt.tolist()} missing from training"


def test_p3_grid_sizes():
    s = build_lshape_grid(seed=0)
    assert s["grid"].shape == (400, 2)
    assert s["train"].shape == (280, 2)
    assert s["test"].shape == (120, 2)
    # Corners present
    for c1 in (0.1, 10.0):
        for c2 in (0.1, 10.0):
            tgt = np.array([c1, c2])
            hit = np.any(np.all(np.abs(s["train"] - tgt) < 1e-10, axis=1))
            assert hit, f"lshape corner {tgt.tolist()} missing from training"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
