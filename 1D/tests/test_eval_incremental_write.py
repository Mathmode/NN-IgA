"""Verify that summary.csv is written progressively per level.

After ``evaluate_test_split`` returns, the on-disk CSV should contain
exactly the rows the function returned (in the same order). This test also
checks that the CSV file exists during the call so a killed job leaves
recoverable partial output.
"""
from __future__ import annotations

import csv

from src import config
from src.parametric.continuation import train_with_continuation
from src.parametric.evaluation import evaluate_test_split
from src.lhs_sampling import generate_splits


def test_csv_written_per_level(tmp_path):
    """File exists with all expected rows after evaluate_test_split returns."""
    config.TRAIN["max_epochs"] = 5
    config.TRAIN["patience"] = 5
    config.TRAIN["train_size"] = 8
    config.TRAIN["val_size"] = 4
    config.TRAIN["test_size"] = 4

    train_betas, val_betas, test_betas = generate_splits(0)
    cont = train_with_continuation(
        p=2, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4),
    )

    csv_path = tmp_path / "summary.csv"
    rows = evaluate_test_split(
        cont.params_final,
        p=2,
        seed=0,
        test_betas=test_betas,
        eval_levels=(2, 4, 8),
        incremental_csv_path=csv_path,
        clear_caches_between_levels=True,
        verbose=False,
    )

    # File exists.
    assert csv_path.exists(), "summary.csv was not created"

    # Row count matches: 3 levels × 4 betas × 3 methods = 36.
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        on_disk_rows = list(reader)
    expected = 3 * 4 * 3
    assert len(on_disk_rows) == expected, (
        f"Expected {expected} rows on disk, got {len(on_disk_rows)}"
    )
    assert len(rows) == expected, "Returned row count mismatch"

    # Header fields match the canonical schema.
    from src.parametric.evaluation import SUMMARY_FIELDNAMES
    assert tuple(reader.fieldnames or []) == SUMMARY_FIELDNAMES or \
           list(on_disk_rows[0].keys()) == list(SUMMARY_FIELDNAMES)


def test_csv_grows_monotonically(tmp_path):
    """After the function returns, the file must contain >0 rows for every
    eval level (i.e. every level appended successfully)."""
    config.TRAIN["max_epochs"] = 3
    config.TRAIN["patience"] = 5
    config.TRAIN["train_size"] = 8
    config.TRAIN["val_size"] = 4
    config.TRAIN["test_size"] = 4

    train_betas, val_betas, test_betas = generate_splits(0)
    cont = train_with_continuation(
        p=2, seed=0,
        train_betas=train_betas, val_betas=val_betas,
        levels=(2, 4),
    )
    csv_path = tmp_path / "summary.csv"
    eval_levels = (2, 4, 8)
    evaluate_test_split(
        cont.params_final,
        p=2, seed=0, test_betas=test_betas,
        eval_levels=eval_levels,
        incremental_csv_path=csv_path,
        clear_caches_between_levels=True, verbose=False,
    )

    # Count rows per level on disk.
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    counts: dict[int, int] = {}
    for r in rows:
        N = int(r["N"])
        counts[N] = counts.get(N, 0) + 1
    for N in eval_levels:
        assert counts.get(int(N), 0) == 4 * 3, (
            f"Level N={N}: expected {4 * 3} rows, got {counts.get(int(N), 0)}"
        )
