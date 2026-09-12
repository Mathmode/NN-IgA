"""Tests for the superseded-directory guard (remediation T11, audit D4).

Run with:  python -m pytest scripts/test_data_guards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_guards import assert_not_superseded, is_superseded, mark_superseded


def test_name_token_flags_old_dir(tmp_path):
    d = tmp_path / "advdiff_OLD_jun15_mac" / "p2" / "eval"
    d.mkdir(parents=True)
    assert is_superseded(d)
    with pytest.raises(RuntimeError):
        assert_not_superseded(d)


def test_marker_flags_parent(tmp_path):
    root = tmp_path / "data_results" / "advdiff_stale"
    (root / "p2" / "eval").mkdir(parents=True)
    mark_superseded(root, replaced_by="data_results/advdiff/")
    # A subdirectory is flagged via the parent marker (stopping at data_results).
    assert is_superseded(root / "p2" / "eval")
    with pytest.raises(RuntimeError):
        assert_not_superseded(root / "p2" / "eval")


def test_current_dir_passes(tmp_path):
    d = tmp_path / "data_results" / "advdiff" / "p2" / "eval"
    d.mkdir(parents=True)
    assert not is_superseded(d)
    assert_not_superseded(d)   # must not raise
