"""Regression test for the per-level CSV flush in ``evaluate_arctan``.

Motivation
----------
Production eval previously accumulated all rows in memory and wrote the
CSV only at the very end (or on graceful exit). A SLURM timeout in the
middle therefore lost ALL eval rows, even from earlier levels that had
already finished.

After the fix, ``evaluate_arctan`` (and ``evaluate_lshape``) appends each level's
rows to the CSV with ``fsync=True`` immediately when that level
completes. Practically:

  * The CSV header is created before the level loop starts.
  * After level N=4 finishes, those rows are on disk.
  * After level N=8 finishes, those rows are appended.
  * If a kill arrives between levels, the partial CSV is still valid
    and contains the levels that DID complete.

We verify by:

  1. running a tiny end-to-end ``evaluate_arctan`` and checking the CSV has
     header + rows for every (level, sigma, method) combination;
  2. intercepting ``evaluate_arctan`` via a monkey-patched
     ``append_csv_dicts`` to confirm the call happens with ``fsync=True``
     once per level — proving the flush is per-level, not deferred to end.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.parametric.evaluation_v2 import CSV_COLUMNS, evaluate_arctan
from src.parametric.positional_density_network_2d import init_params


@pytest.fixture
def trained_params():
    # Untrained params are fine — we just need a valid forward pass.
    return init_params(seed=0, sigma_dim=3, hidden_dims=(4, 4))


@pytest.fixture
def tiny_test_sigmas():
    return np.array(
        [
            [5.0, 0.3, 0.5],
            [10.0, 0.6, 0.4],
        ],
        dtype=np.float64,
    )


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_evaluate_p2_writes_header_immediately(tmp_path, trained_params, tiny_test_sigmas):
    """Header is created up-front, before any level finishes."""
    levels = (4, 8)
    out_csv = evaluate_arctan(
        trained_params, tiny_test_sigmas,
        p=2, levels=levels, seed=0, output_dir=tmp_path,
        q_K=3, q_F=8, q_metric=8, corrector_max_iter=2,
    )
    assert out_csv.exists()
    with out_csv.open() as f:
        header = next(csv.reader(f))
    assert header == list(CSV_COLUMNS)


def test_evaluate_p2_row_count_matches_expected(tmp_path, trained_params, tiny_test_sigmas):
    """Total rows = n_levels × n_sigmas × 3 methods.

    With 2 levels, 2 sigmas, and 3 methods (uniform / positional /
    positional_corrected) we expect 12 rows."""
    levels = (4, 8)
    out_csv = evaluate_arctan(
        trained_params, tiny_test_sigmas,
        p=2, levels=levels, seed=0, output_dir=tmp_path,
        q_K=3, q_F=8, q_metric=8, corrector_max_iter=2,
    )
    rows = _read_csv_rows(out_csv)
    expected = len(levels) * len(tiny_test_sigmas) * 3
    assert len(rows) == expected, f"want {expected} rows; got {len(rows)}"


def test_evaluate_p2_appends_once_per_level_with_fsync(
    tmp_path, trained_params, tiny_test_sigmas, monkeypatch
):
    """``evaluate_arctan`` must call ``append_csv_dicts(..., fsync=True)``
    exactly once per level (not once-at-end or once-per-sigma)."""
    from src.parametric import evaluation_v2 as mod

    calls = []
    real_append = mod.append_csv_dicts

    def spy(path, rows, fieldnames, *, fsync=False):
        calls.append({"n_rows": len(list(rows)) if not isinstance(rows, list) else len(rows),
                       "fsync": bool(fsync)})
        return real_append(path, rows, fieldnames, fsync=fsync)

    monkeypatch.setattr(mod, "append_csv_dicts", spy)

    levels = (4, 8)
    evaluate_arctan(
        trained_params, tiny_test_sigmas,
        p=2, levels=levels, seed=0, output_dir=tmp_path,
        q_K=3, q_F=8, q_metric=8, corrector_max_iter=2,
    )
    # Exactly one append call per level.
    assert len(calls) == len(levels), f"want {len(levels)} appends; got {len(calls)}"
    # And each call has fsync=True (the whole point of the per-level flush).
    for c in calls:
        assert c["fsync"] is True, f"per-level append must use fsync=True; got {c}"
    # And each level should have written n_sigmas * 3 rows.
    n_expected = len(tiny_test_sigmas) * 3
    for c in calls:
        assert c["n_rows"] == n_expected, (
            f"per-level batch should have {n_expected} rows; got {c['n_rows']}"
        )


def test_evaluate_p2_partial_csv_recoverable_on_simulated_crash(
    tmp_path, trained_params, tiny_test_sigmas, monkeypatch
):
    """Simulate a crash AFTER the first level's flush: the CSV must still
    contain the first level's rows (header + n_sigmas*3 data rows)."""
    from src.parametric import evaluation_v2 as mod

    seen_levels: list[int] = []
    real_append = mod.append_csv_dicts

    def crashing_append(path, rows, fieldnames, *, fsync=False):
        # First call writes legit rows; second call simulates a kill BEFORE
        # the write happens (e.g. SLURM SIGTERM mid-loop).
        if len(seen_levels) == 0:
            seen_levels.append(1)
            return real_append(path, rows, fieldnames, fsync=fsync)
        raise KeyboardInterrupt("simulated kill mid-eval")

    monkeypatch.setattr(mod, "append_csv_dicts", crashing_append)

    levels = (4, 8)
    with pytest.raises(KeyboardInterrupt):
        evaluate_arctan(
            trained_params, tiny_test_sigmas,
            p=2, levels=levels, seed=0, output_dir=tmp_path,
            q_K=3, q_F=8, q_metric=8, corrector_max_iter=2,
        )

    out_csv = tmp_path / "summary_p2_seed0.csv"
    assert out_csv.exists(), "CSV must exist on disk after partial run"
    rows = _read_csv_rows(out_csv)
    # First level (N=4) should be fully there: 2 sigmas * 3 methods = 6 rows.
    assert len(rows) == len(tiny_test_sigmas) * 3, (
        f"partial CSV should hold the first level's rows; got {len(rows)}"
    )
    # All N values should equal the first level.
    Ns = {int(r["N"]) for r in rows}
    assert Ns == {levels[0]}, f"only N=4 should be persisted; got Ns={Ns}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
