"""Batch sizing and actual local CPU/disk capacity."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

DEFAULT_BATCH_ROWS = 65536


def require_free_disk_headroom(path: Path, *, reserve_bytes: int) -> int:
    """Require owner-specified free-space headroom on the target volume."""

    reserve = _exact_nonnegative_int(
        reserve_bytes,
        error="invalid_runtime_resource_disk_reserve",
    )
    from quant_research_platform.storage.identity import (
        filesystem_exists,
        native_filesystem_path,
    )

    candidate = Path(path)
    while not filesystem_exists(candidate) and candidate != candidate.parent:
        candidate = candidate.parent
    free = shutil.disk_usage(native_filesystem_path(candidate)).free
    if free < reserve:
        raise RuntimeError("insufficient_free_disk")
    return free


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


def _exact_nonnegative_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(error)
    return value
