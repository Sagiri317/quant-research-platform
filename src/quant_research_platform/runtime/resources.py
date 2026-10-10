"""Batch sizing and actual local CPU capacity."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

DEFAULT_BATCH_ROWS = 65536


def _cgroup_v2_directories() -> Iterator[Path]:
    if sys.platform != "linux":
        return
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").exists():
        return
    membership = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    unified = [line[3:] for line in membership if line.startswith("0::")]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise ValueError("invalid_cgroup_membership")
    relative = Path(unified[0].lstrip("/"))
    if ".." in relative.parts:
        raise ValueError("invalid_cgroup_membership")
    directory = root / relative
    if not directory.is_dir():
        raise RuntimeError("cgroup_membership_not_visible")
    while True:
        yield directory
        if directory == root:
            break
        directory = directory.parent


def model_cpu_capacity() -> tuple[int, int]:
    """Discover physical cores and the usable affinity/bandwidth capacity."""
    import psutil

    physical = psutil.cpu_count(logical=False)
    if type(physical) is not int or physical <= 0:
        raise RuntimeError("physical_core_count_unavailable")
    available = physical
    try:
        if sys.platform == "linux":
            available = min(available, len(os.sched_getaffinity(0)))
        elif sys.platform == "win32":
            available = min(available, len(psutil.Process().cpu_affinity()))
        for directory in _cgroup_v2_directories():
            maximum = directory / "cpu.max"
            if not maximum.exists():
                continue
            quota, period = maximum.read_text(encoding="ascii").split()
            period_us = _exact_positive_int(
                int(period), error="invalid_cgroup_cpu_period"
            )
            if quota != "max":
                quota_us = _exact_positive_int(
                    int(quota), error="invalid_cgroup_cpu_quota"
                )
                available = min(available, max(1, quota_us // period_us))
        available = _exact_positive_int(available, error="invalid_runtime_cpu_capacity")
    except (OSError, ValueError, RuntimeError):
        raise RuntimeError("runtime_cpu_capacity_discovery_failed") from None
    return physical, available


def _exact_positive_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(error)
    return value
