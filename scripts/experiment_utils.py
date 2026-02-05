from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]


def _safe_version(dist_name: str) -> str | None:
    try:
        return metadata.version(dist_name)
    except Exception:
        return None


def _git_info(repo_root: Path) -> dict[str, Any]:
    if not (repo_root / ".git").exists():
        return {}

    def _run(args: Sequence[str]) -> str | None:
        try:
            out = subprocess.check_output(list(args), cwd=str(repo_root), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    is_dirty = None
    try:
        subprocess.check_call(["git", "diff", "--quiet"], cwd=str(repo_root))
        subprocess.check_call(["git", "diff", "--cached", "--quiet"], cwd=str(repo_root))
        is_dirty = False
    except subprocess.CalledProcessError:
        is_dirty = True
    except Exception:
        is_dirty = None

    return {"commit": commit, "dirty": is_dirty}


def write_run_metadata(
    out_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    argv: Sequence[str] | None = None,
    args: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = "run_metadata.json",
) -> Path:
    """Write a JSON metadata file to support reproducibility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if argv is None:
        argv = list(sys.argv)

    def _to_jsonable(x):
        if is_dataclass(x):
            return asdict(x)
        if isinstance(x, Path):
            return str(x)
        return x

    payload: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "argv": list(argv),
        "python": sys.version,
        "platform": platform.platform(),
        "env": {
            "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        },
        "versions": {
            "numpy": _safe_version("numpy"),
            "jax": _safe_version("jax"),
            "jaxlib": _safe_version("jaxlib"),
            "optax": _safe_version("optax"),
        },
    }

    if args is not None:
        payload["args"] = {k: _to_jsonable(v) for k, v in dict(args).items()}
    if extra is not None:
        payload["extra"] = {k: _to_jsonable(v) for k, v in dict(extra).items()}

    if repo_root is not None:
        payload["git"] = _git_info(Path(repo_root))

    path = out_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


__all__ = [
    "parse_int_list",
    "write_run_metadata",
]

