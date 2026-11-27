from __future__ import annotations

import os
import platform
import sys


def _peak_rss_mb_windows():
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return float("nan")
    return counters.PeakWorkingSetSize / 1024 ** 2


def peak_rss_mb():
    if sys.platform == "win32":
        return _peak_rss_mb_windows()
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 ** 2 if sys.platform == "darwin" else rss / 1024


def hardware_info():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"), "host": platform.node()}
    try:
        with open("/proc/cpuinfo") as fh:
            info["cpu_model"] = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu_model"] = platform.processor()
    return info


__all__ = ["peak_rss_mb", "hardware_info"]
