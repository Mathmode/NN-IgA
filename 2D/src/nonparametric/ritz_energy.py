"""DIAGNOSTIC helper for the Aballay-style relative energy error.

NOTE: this module exists ONLY to compute the CSV column ``ritz_rel_error``
for comparability with Aballay et al. 2025. The training loss in
``src/parametric/training.py`` is the residual a-posteriori estimator and
does NOT depend on this file.
"""
from __future__ import annotations

import math


def ritz_relative_error(j_h: float, j_ref: float) -> float:
    """Aballay's relative energy error: ``sqrt[(j_h - j_ref) / j_ref]``.

    With both J values negative and ``j_h > j_ref`` (Galerkin is sub-optimal in
    V_h), the numerator ``j_h - j_ref > 0`` and denominator ``|j_ref| > 0``.
    Returns a real number; the absolute value inside the sqrt keeps it real
    if signs invert (which can happen with approximate masking — see the
    audit memo).

    Convention: returns ``-sqrt(|...|)`` when ``j_h < j_ref`` to flag the
    inversion.
    """
    j_h = float(j_h)
    j_ref = float(j_ref)
    if j_ref == 0.0:
        return float("nan")
    diff = j_h - j_ref
    return math.copysign(math.sqrt(abs(diff / j_ref)), diff)


__all__ = ["ritz_relative_error"]
