"""Verify that the evaluation phase keeps memory under reasonable bounds.

The previous failure mode on Hyperion was unbounded growth of the JAX
compilation cache during evaluation (~17 GB peak after ~22 minutes).
With ``clear_caches_between_levels=True`` the cache is dropped after every
level, keeping peak RSS bounded.
"""
from __future__ import annotations

import gc
import resource
import sys


from src import config
from src.parametric.continuation import train_with_continuation
from src.parametric.evaluation import evaluate_test_split
from src.lhs_sampling import generate_splits


def _peak_rss_mb() -> float:
    """Peak RSS in MB. Linux: ru_maxrss is KB; macOS: bytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS reports bytes regardless of magnitude
        return rss / (1024 ** 2)
    return rss / 1024  # KB → MB (Linux)


def test_evaluate_test_split_memory_bounded(tmp_path):
    """Eval peak memory should stay well under 4 GB on the smoke config."""
    # Smoke-size overrides (in-place mutation of cfg.TRAIN dict propagates)
    config.TRAIN["max_epochs"] = 10
    config.TRAIN["patience"] = 5
    config.TRAIN["train_size"] = 16
    config.TRAIN["val_size"] = 8
    config.TRAIN["test_size"] = 8

    train_betas, val_betas, test_betas = generate_splits(0)

    cont = train_with_continuation(
        p=3, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4, 8),
    )

    gc.collect()

    rows = evaluate_test_split(
        cont.params_final,
        p=3,
        seed=0,
        test_betas=test_betas,
        eval_levels=(2, 4, 8, 16),
        incremental_csv_path=tmp_path / "summary.csv",
        clear_caches_between_levels=True,
        verbose=False,
    )

    peak = _peak_rss_mb()
    # 4 GB ceiling — generous, but well below the 17 GB observed without
    # cache clearing.
    assert peak < 4 * 1024, (
        f"Peak memory {peak:.0f} MB exceeds 4 GB; cache-clear is not effective"
    )
    # Sanity on row count: 4 levels * 8 betas * 3 methods = 96.
    assert len(rows) == 4 * 8 * 3, f"Wrong row count: {len(rows)}"
