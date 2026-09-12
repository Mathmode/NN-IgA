from __future__ import annotations

"""CSV I/O helpers shared across experiments."""

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def summary_is_complete(path: str | Path, required_levels: Iterable[int]) -> bool:
    """True if a per-seed evaluation summary CSV already covers every requested
    mesh level.

    Used by the eval entry points to SKIP a seed whose summary already exists
    and is complete (unless ``--force`` is passed), so re-running an evaluation
    array does not recompute finished seeds. "Complete" means the file exists,
    is readable, has an ``N`` column, and the set of ``N`` values present is a
    superset of ``required_levels`` (a partial/crashed run with a missing level
    is treated as incomplete and will be redone). Any read error returns False
    (re-evaluate rather than wrongly skip).
    """
    try:
        path = Path(path)
        if not path.exists():
            return False
        required = {int(N) for N in required_levels}
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not required:
            # Nothing specifically required: treat a non-empty summary as done.
            return len(rows) > 0
        present = {int(float(r["N"])) for r in rows if r.get("N") not in (None, "")}
        return required.issubset(present)
    except Exception:
        return False


def write_csv_rows(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write a CSV file from a header and row sequences."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def write_csv_dicts(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write a CSV file from a list of dict rows."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def append_csv_dicts(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    *,
    fsync: bool = False,
) -> None:
    """Append dict rows to a CSV file, creating the header on first write.

    Parameters
    ----------
    fsync
        If True, call ``f.flush()`` and ``os.fsync(f.fileno())`` before
        closing the file so the OS commits the bytes to disk immediately.
        Useful for long cluster jobs where each per-level batch should
        be recoverable even if a later batch never completes (timeout /
        kill). Default False keeps the legacy fast path.
    """
    import os
    path = Path(path)
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        if fsync:
            f.flush()
            os.fsync(f.fileno())
