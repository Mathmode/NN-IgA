"""Lightweight profiling instrumentation for the IGA r-adaptivity codebase.

Usage:
    from common.profiling import Profiler

    prof = Profiler()

    # Time a block:
    with prof.region("assembly"):
        ...

    # Time a JAX computation (includes block_until_ready):
    with prof.jax_region("solve"):
        result = solve(...)
        # automatically calls jax.block_until_ready on exit

    # Record a scalar metric:
    prof.record("cg_iters", 42)

    # Print summary:
    prof.summary()

    # Export to CSV:
    prof.to_csv("profiling_results.csv")
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class _RegionStats:
    """Accumulated wall-time statistics for a named region."""
    total_sec: float = 0.0
    count: int = 0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def record(self, elapsed: float) -> None:
        self.total_sec += elapsed
        self.count += 1
        self.min_sec = min(self.min_sec, elapsed)
        self.max_sec = max(self.max_sec, elapsed)

    @property
    def mean_sec(self) -> float:
        return self.total_sec / self.count if self.count else 0.0


class Profiler:
    """Minimal profiler for JAX-based IGA code.

    Designed to measure:
      - Wall time per region (assembly, solve, adjoint, matvec, validation, etc.)
      - Scalar metrics (CG iterations, peak memory, etc.)
      - Per-step timing breakdown

    All timing uses time.perf_counter + jax.block_until_ready for accurate
    device-inclusive measurements.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._regions: dict[str, _RegionStats] = defaultdict(_RegionStats)
        self._scalars: dict[str, list[float]] = defaultdict(list)
        self._step_times: list[dict[str, float]] = []
        self._current_step: dict[str, float] | None = None
        self._mem_snapshots: list[dict[str, Any]] = []

    # ---- Context managers for timing ----

    @contextmanager
    def region(self, name: str):
        """Time a CPU/Python region (no device sync)."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self._regions[name].record(elapsed)
        if self._current_step is not None:
            self._current_step[name] = self._current_step.get(name, 0.0) + elapsed

    @contextmanager
    def jax_region(self, name: str):
        """Time a JAX region with device synchronization via block_until_ready.

        Calls jax.block_until_ready on the default backend after the block,
        ensuring all device computation is included in the measurement.
        """
        if not self.enabled:
            yield
            return
        # Sync before to get a clean start
        _sync_device()
        t0 = time.perf_counter()
        yield
        _sync_device()
        elapsed = time.perf_counter() - t0
        self._regions[name].record(elapsed)
        if self._current_step is not None:
            self._current_step[name] = self._current_step.get(name, 0.0) + elapsed

    # ---- Per-step tracking ----

    def begin_step(self) -> None:
        """Mark the beginning of an optimization step."""
        if not self.enabled:
            return
        self._current_step = {"_t0": time.perf_counter()}

    def end_step(self) -> None:
        """Mark the end of an optimization step and record the breakdown."""
        if not self.enabled or self._current_step is None:
            return
        self._current_step["_total"] = time.perf_counter() - self._current_step.pop("_t0", time.perf_counter())
        self._step_times.append(self._current_step)
        self._current_step = None

    # ---- Scalar metrics ----

    def record(self, name: str, value: float) -> None:
        """Record a scalar metric (e.g., CG iteration count, memory)."""
        if self.enabled:
            self._scalars[name].append(float(value))

    # ---- Memory snapshots ----

    def snapshot_memory(self, label: str = "") -> None:
        """Capture JAX device memory statistics (if available)."""
        if not self.enabled:
            return
        try:
            devices = jax.local_devices()
            if devices:
                stats = devices[0].memory_stats()
                if stats:
                    self._mem_snapshots.append({
                        "label": label,
                        "peak_bytes": stats.get("peak_bytes_in_use", -1),
                        "current_bytes": stats.get("bytes_in_use", -1),
                        "num_allocs": stats.get("num_allocs", -1),
                    })
        except Exception:
            pass  # memory_stats not available on all backends

    # ---- Reporting ----

    def summary(self) -> str:
        """Return a formatted profiling summary."""
        lines = []
        lines.append("=" * 72)
        lines.append("PROFILING SUMMARY")
        lines.append("=" * 72)

        if self._regions:
            lines.append("")
            lines.append("Region Timings:")
            lines.append(f"  {'Region':<30s} {'Calls':>6s} {'Total(s)':>10s} {'Mean(ms)':>10s} {'Min(ms)':>10s} {'Max(ms)':>10s} {'%':>6s}")
            lines.append("  " + "-" * 78)
            total_all = sum(r.total_sec for r in self._regions.values())
            sorted_regions = sorted(self._regions.items(), key=lambda kv: -kv[1].total_sec)
            for name, stats in sorted_regions:
                pct = 100.0 * stats.total_sec / total_all if total_all > 0 else 0.0
                lines.append(
                    f"  {name:<30s} {stats.count:>6d} {stats.total_sec:>10.3f} "
                    f"{stats.mean_sec * 1e3:>10.2f} {stats.min_sec * 1e3:>10.2f} "
                    f"{stats.max_sec * 1e3:>10.2f} {pct:>5.1f}%"
                )

        if self._scalars:
            lines.append("")
            lines.append("Scalar Metrics:")
            for name, vals in sorted(self._scalars.items()):
                import statistics
                n = len(vals)
                mean = statistics.mean(vals)
                mn = min(vals)
                mx = max(vals)
                lines.append(f"  {name:<30s}  n={n:>5d}  mean={mean:>10.2f}  min={mn:>10.2f}  max={mx:>10.2f}")

        if self._step_times:
            lines.append("")
            lines.append("Per-Step Breakdown (last 5 steps):")
            for i, step in enumerate(self._step_times[-5:]):
                total = step.get("_total", 0.0)
                parts = {k: v for k, v in step.items() if not k.startswith("_")}
                parts_str = "  ".join(f"{k}={v * 1e3:.1f}ms" for k, v in sorted(parts.items(), key=lambda x: -x[1]))
                lines.append(f"  Step {len(self._step_times) - 5 + i}: total={total * 1e3:.1f}ms  {parts_str}")

        if self._mem_snapshots:
            lines.append("")
            lines.append("Memory Snapshots:")
            for snap in self._mem_snapshots:
                peak_mb = snap["peak_bytes"] / (1024 ** 2) if snap["peak_bytes"] >= 0 else -1
                curr_mb = snap["current_bytes"] / (1024 ** 2) if snap["current_bytes"] >= 0 else -1
                lines.append(f"  [{snap['label']}] peak={peak_mb:.1f}MB  current={curr_mb:.1f}MB  allocs={snap['num_allocs']}")

        lines.append("")
        lines.append("=" * 72)
        result = "\n".join(lines)
        print(result)
        return result

    def to_csv(self, path: str) -> None:
        """Export region timings to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["region", "calls", "total_sec", "mean_ms", "min_ms", "max_ms"])
            for name, stats in sorted(self._regions.items(), key=lambda kv: -kv[1].total_sec):
                writer.writerow([
                    name,
                    stats.count,
                    f"{stats.total_sec:.6f}",
                    f"{stats.mean_sec * 1e3:.4f}",
                    f"{stats.min_sec * 1e3:.4f}",
                    f"{stats.max_sec * 1e3:.4f}",
                ])

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._regions.clear()
        self._scalars.clear()
        self._step_times.clear()
        self._current_step = None
        self._mem_snapshots.clear()


def _sync_device():
    """Block until all JAX computations on default device have completed."""
    try:
        jax.effects_barrier()
    except AttributeError:
        # Fallback for older JAX versions: create a small computation and wait
        jnp.zeros(1).block_until_ready()


# ---- Global profiler instance (opt-in) ----

_GLOBAL_PROFILER: Profiler | None = None


def get_profiler() -> Profiler:
    """Get or create the global profiler instance."""
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is None:
        _GLOBAL_PROFILER = Profiler(enabled=True)
    return _GLOBAL_PROFILER


def set_profiler(prof: Profiler) -> None:
    """Set the global profiler instance."""
    global _GLOBAL_PROFILER
    _GLOBAL_PROFILER = prof


def disable_profiling() -> None:
    """Disable the global profiler."""
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is not None:
        _GLOBAL_PROFILER.enabled = False
