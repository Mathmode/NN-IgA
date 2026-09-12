"""Guards against reading superseded result directories (remediation T11).

Audit defect D4: Table 5 had been generated from ``data_results/advdiff_OLD_jun15_mac/``
(a superseded run) instead of the current ``data_results/advdiff/``. To make that
class of mistake impossible to repeat, any table/figure generator that consumes an
eval directory must call :func:`assert_not_superseded` on it first.

A directory is considered superseded if EITHER:
  * its path contains a superseded token (``_OLD``, ``_old_``, ``superseded``,
    ``deprecated``, ``backup``), OR
  * it (or a parent up to ``data_results``) contains a ``SUPERSEDED.txt`` marker.

Create the marker with :func:`mark_superseded`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SUPERSEDED_MARKER = "SUPERSEDED.txt"
_SUPERSEDED_TOKENS = ("_old", "superseded", "deprecated", "backup", "_bak")


def _find_marker(path: Path) -> Optional[Path]:
    """Return the SUPERSEDED.txt marker at ``path`` or any parent up to (and
    including) a ``data_results`` directory, else None."""
    p = Path(path).resolve()
    for cand in (p, *p.parents):
        m = cand / SUPERSEDED_MARKER
        if m.exists():
            return m
        if cand.name == "data_results":
            break
    return None


def is_superseded(path) -> bool:
    """True if ``path`` is a superseded results directory (by name token or marker)."""
    p = Path(path)
    name_lc = str(p).lower()
    if any(tok in name_lc for tok in _SUPERSEDED_TOKENS):
        return True
    return _find_marker(p) is not None


def assert_not_superseded(path) -> None:
    """Raise if ``path`` is a superseded results directory. Call this before
    reading any eval directory that feeds a manuscript table or figure."""
    p = Path(path)
    marker = _find_marker(p)
    name_lc = str(p).lower()
    hit = next((tok for tok in _SUPERSEDED_TOKENS if tok in name_lc), None)
    if hit is not None or marker is not None:
        reason = (f"path token '{hit}'" if hit is not None
                  else f"marker {marker}")
        raise RuntimeError(
            f"Refusing to read SUPERSEDED results directory:\n  {p}\n"
            f"  (flagged by {reason}).\n"
            f"This directory was replaced by a newer run; using it would "
            f"reproduce audit defect D4 (a stale table). Point the generator at "
            f"the current results directory instead."
        )


def mark_superseded(path, *, replaced_by: str, note: str = "") -> Path:
    """Write a SUPERSEDED.txt marker into ``path`` describing its replacement.
    Returns the marker path."""
    p = Path(path)
    marker = p / SUPERSEDED_MARKER
    lines = [
        "This results directory is SUPERSEDED and must not feed manuscript "
        "tables or figures.",
        f"Replaced by: {replaced_by}",
    ]
    if note:
        lines.append(f"Note: {note}")
    lines.append(
        "Enforced by scripts/_data_guards.assert_not_superseded (remediation T11, "
        "audit D4)."
    )
    marker.write_text("\n".join(lines) + "\n")
    return marker


__all__ = [
    "SUPERSEDED_MARKER",
    "is_superseded",
    "assert_not_superseded",
    "mark_superseded",
]
