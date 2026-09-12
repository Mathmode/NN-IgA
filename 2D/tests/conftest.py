"""Shared test setup for 2D experiments.

Adds:
  - ``<repo>/`` (project root, parent of ``2D/``) so
    ``import common.X`` works.
  - ``<repo>/2D/`` so ``import src.X`` works.

Also forces JAX into float64 on CPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

TWO_D_ROOT = Path(__file__).resolve().parents[1]    # .../<repo>/2D
PROJECT_ROOT = TWO_D_ROOT.parent                     # .../<repo>

for p in (str(PROJECT_ROOT), str(TWO_D_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
