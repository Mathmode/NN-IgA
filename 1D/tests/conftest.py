"""Shared test setup for 1D experiments.

Adds:
  - ``<repo>/`` (the project root, parent of ``1D/``) so
    ``import common.X`` works.
  - ``<repo>/1D/`` so ``import src.X`` works.

Also forces JAX into float64 on CPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ONE_D_ROOT = Path(__file__).resolve().parents[1]    # .../<repo>/1D
PROJECT_ROOT = ONE_D_ROOT.parent                     # .../<repo>

for p in (str(PROJECT_ROOT), str(ONE_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
