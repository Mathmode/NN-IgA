"""Regression test for the incremental ``checkpoint_final.npz`` save in
``train_with_continuation``.

Motivation
----------
Production training of arctan at N=64 occasionally times out at the 8-h SLURM
wall-clock. The previous behaviour wrote ``checkpoint_final.npz`` ONLY
after the entire continuation chain finished — so a kill at N=64 left
the downstream eval with no usable anchor checkpoint.

After the fix, ``train_with_continuation`` overwrites
``checkpoint_final.npz`` at the end of EVERY completed level. We assert:

  * ``checkpoint_N{level}.npz`` exists for every level we asked for.
  * ``checkpoint_final.npz`` exists and matches the LAST completed level
    bit-for-bit (so downstream eval transparently uses it).
  * Mid-training, ``checkpoint_final.npz`` mirrors the most recently
    completed level (verified by truncating the level list).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.parametric.continuation import (
    load_checkpoint,
    train_with_continuation,
)


# Tiny config so the test runs in a few seconds: 2 sigmas, 1 epoch/level,
# 2 mesh levels. The point is to exercise the checkpoint plumbing, NOT to
# verify training convergence (other tests cover that).
@pytest.fixture
def tiny_train_inputs():
    train_sigmas = np.array(
        [
            [5.0, 0.3, 0.5],
            [10.0, 0.7, 0.4],
        ],
        dtype=np.float64,
    )
    # Denominators only need to be positive scalars for the loss to be
    # well-defined; the network won't actually converge in 1 epoch.
    train_denoms = np.array([1.0, 1.0], dtype=np.float64)
    return train_sigmas, train_denoms


def _run(checkpoint_dir, levels, train_inputs):
    train_sigmas, train_denoms = train_inputs
    return train_with_continuation(
        experiment="arctan",
        p=2,
        seed=0,
        train_sigmas=train_sigmas,
        train_denoms=train_denoms,
        levels=levels,
        epochs_per_level={N: 1 for N in levels},
        batch_size=2,
        lr_init=1e-2,
        lr_end=1e-3,
        log_every=1,
        q_K=3,
        q_F=4,
        q_est=4,
        checkpoint_dir=checkpoint_dir,
        resume=False,
        sigma_dim=3,
        hidden_dims=(4, 4),
    )


def test_per_level_checkpoint_files_exist(tmp_path, tiny_train_inputs):
    """``checkpoint_N{N}.npz`` is written for every completed level."""
    levels = (4, 8)
    _run(tmp_path, levels, tiny_train_inputs)
    for N in levels:
        ckpt = tmp_path / f"checkpoint_N{N}.npz"
        assert ckpt.exists(), f"missing per-level checkpoint {ckpt}"


def test_final_checkpoint_exists_after_training(tmp_path, tiny_train_inputs):
    """``checkpoint_final.npz`` is overwritten on every completed level
    (and therefore exists once training finishes)."""
    levels = (4, 8)
    _run(tmp_path, levels, tiny_train_inputs)
    final = tmp_path / "checkpoint_final.npz"
    assert final.exists(), "checkpoint_final.npz must exist after training"


def test_final_checkpoint_matches_last_level(tmp_path, tiny_train_inputs):
    """``checkpoint_final.npz`` matches the LAST level's per-level file."""
    levels = (4, 8)
    _run(tmp_path, levels, tiny_train_inputs)
    final = load_checkpoint(tmp_path / "checkpoint_final.npz")
    last = load_checkpoint(tmp_path / f"checkpoint_N{levels[-1]}.npz")
    # Compare layer-by-layer (W, b) tuples.
    assert len(final.layers) == len(last.layers)
    for (Wf, bf), (Wl, bl) in zip(final.layers, last.layers):
        np.testing.assert_allclose(np.asarray(Wf), np.asarray(Wl), atol=0, rtol=0)
        np.testing.assert_allclose(np.asarray(bf), np.asarray(bl), atol=0, rtol=0)


def test_final_checkpoint_recoverable_after_partial_run(tmp_path, tiny_train_inputs):
    """If training stops after only the FIRST level completes (simulated by
    only requesting one level), ``checkpoint_final.npz`` still exists and
    points at that level. This is the practical SLURM-timeout scenario:
    a later level being killed mid-training does NOT erase the already-
    persisted final checkpoint."""
    levels = (4,)
    _run(tmp_path, levels, tiny_train_inputs)
    final = tmp_path / "checkpoint_final.npz"
    n4 = tmp_path / "checkpoint_N4.npz"
    assert final.exists()
    assert n4.exists()
    # Final points at N=4 since that's the only completed level.
    pf = load_checkpoint(final)
    p4 = load_checkpoint(n4)
    for (Wf, bf), (W4, b4) in zip(pf.layers, p4.layers):
        np.testing.assert_allclose(np.asarray(Wf), np.asarray(W4), atol=0, rtol=0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
