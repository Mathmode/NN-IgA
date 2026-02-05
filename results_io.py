from __future__ import annotations

"""Standard results layout used by all experiments.

This module exists to keep I/O paths consistent across 1D and 2D scripts.

Layout
------
<root>/
  p{p}/N{N}/
    summary.csv
    uniform/...
    r_adapt/...

Per-degree series:
  <root>/{tag}_p{p}.csv
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from csv_utils import ensure_dir, write_csv_dicts


@dataclass(frozen=True)
class StandardResults:
    """Filesystem layout + CSV helpers for the repo."""

    root: Path = Path("results")
    tag: str = "experiment"

    def __init__(self, root: str | Path = "results", tag: str = "experiment"):
        object.__setattr__(self, "root", Path(root))
        object.__setattr__(self, "tag", str(tag))

    def case_root(self, p: int, N: int) -> Path:
        d = self.root / f"p{int(p)}" / f"N{int(N)}"
        ensure_dir(d)
        return d

    def case_dir(self, p: int, N: int, method: str | None = None) -> Path:
        """Return the per-case directory (optionally including a method subdir)."""
        d = self.case_root(p, N)
        if method is None:
            return d
        d = d / str(method)
        ensure_dir(d)
        return d

    def method_dir(self, p: int, N: int, method: str) -> Path:
        return self.case_dir(p, N, method)

    def summary_path(self, p: int, N: int) -> Path:
        return self.case_root(p, N) / "summary.csv"

    def series_path(self, p: int) -> Path:
        ensure_dir(self.root)
        return self.root / f"{self.tag}_p{int(p)}.csv"

    def write_series(self, p: int, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
        write_csv_dicts(self.series_path(p), list(rows), list(fieldnames))


__all__ = ["StandardResults"]

