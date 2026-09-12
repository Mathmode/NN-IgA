"""Tests for the NGSolve lshape reference module.

The module is import-guarded: it must import cleanly even when NGSolve is
not installed. The actual reference solve is only exercised when NGSolve
is available — otherwise the relevant tests are skipped.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from src.nonparametric.lshape import ngsolve_reference as ng_ref


def test_module_imports_without_ngsolve():
    """Importing the module is always safe; functions raise only on call."""
    # NGSOLVE_AVAILABLE is True or False; both are valid module states.
    assert hasattr(ng_ref, "NGSOLVE_AVAILABLE")
    assert hasattr(ng_ref, "compute_p3_ngsolve_reference")
    assert hasattr(ng_ref, "build_p3_reference_cache")
    assert hasattr(ng_ref, "load_p3_reference_cache")


def test_clean_error_message_when_ngsolve_missing():
    """If NGSolve is not installed, calling compute_p3_ngsolve_reference must
    raise a clear ImportError with the pip-install hint."""
    if ng_ref.NGSOLVE_AVAILABLE:
        pytest.skip("NGSolve is installed; this test exercises the missing path.")
    with pytest.raises(ImportError, match="NGSolve is required"):
        ng_ref.compute_p3_ngsolve_reference(1.0, 1.0)
    with pytest.raises(ImportError, match="NGSolve is required"):
        ng_ref.build_p3_reference_cache(
            np.array([[1.0, 1.0]]), cache_path="/tmp/_unused.pkl"
        )


def test_load_cache_roundtrip(tmp_path):
    """``load_p3_reference_cache`` should pickle-roundtrip without NGSolve."""
    cache = {
        (1.0, 1.0): {
            "sigma1": 1.0, "sigma2": 1.0, "energy": -0.012345,
            "h1_seminorm_sq": 0.04321, "sigma_h1_seminorm_sq": 0.04321,
            "int_u": 0.005, "dof_count": 12345,
            "max_h": 0.01, "order": 3, "refine_corner": 3,
        },
    }
    cache_path = tmp_path / "ngsolve_cache.pkl"
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)

    loaded = ng_ref.load_p3_reference_cache(cache_path)
    assert loaded == cache


@pytest.mark.skipif(not ng_ref.NGSOLVE_AVAILABLE, reason="NGSolve not installed")
def test_compute_reference_smoke_if_available():
    """If NGSolve IS installed locally, run one coarse smoke at (1, 1) and
    verify the returned dictionary has the expected keys and positive H1²."""
    entry = ng_ref.compute_p3_ngsolve_reference(
        1.0, 1.0, max_h=0.05, order=2, refine_corner=1,
    )
    for k in ("energy", "h1_seminorm_sq", "sigma_h1_seminorm_sq",
              "int_u", "dof_count", "max_h", "order"):
        assert k in entry
    assert entry["dof_count"] > 0
    assert entry["sigma_h1_seminorm_sq"] > 0
    assert entry["h1_seminorm_sq"] > 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
