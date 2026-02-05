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
