"""Process identity for recoverable same-volume storage scratch paths."""

from __future__ import annotations

import os
from pathlib import Path

import psutil


def owned_scratch_name(prefix: str, unique: str) -> str:
    """Return a compact scratch name bound to this process incarnation."""

    if not prefix or not unique or "-" in unique:
        raise ValueError("invalid_storage_scratch_name")
    return f"{owned_scratch_prefix(prefix)}{unique}"


def owned_scratch_prefix(prefix: str) -> str:
    """Return the owner-qualified prefix used by random scratch-name creators."""

    if not prefix:
        raise ValueError("invalid_storage_scratch_name")
    return f"{prefix}q{_base36(os.getpid())}-"


def scratch_owner_is_stale(path: Path, *, prefix: str) -> bool:
    """Return whether a recognized scratch owner is no longer this incarnation."""

    name = Path(path).name
    normalized_name = os.path.normcase(name)
    normalized_prefix = os.path.normcase(prefix)
    if not normalized_name.startswith(normalized_prefix):
        return False
    owner_parts = normalized_name[len(normalized_prefix) :].split("-", 1)
    if len(owner_parts) != 2 or not owner_parts[0].startswith("q"):
        return False
    try:
        pid = int(owner_parts[0][1:], 36)
    except ValueError:
        return False
    if pid <= 0 or not owner_parts[1]:
        return False
    try:
        process = psutil.Process(pid)
        return not (
            process.is_running()
            and Path(path).stat().st_ctime >= float(process.create_time())
        )
    except (FileNotFoundError, psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except psutil.AccessDenied:
        return False


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    remaining = int(value)
    if remaining <= 0:
        raise ValueError("invalid_storage_scratch_owner_identity")
    digits: list[str] = []
    while remaining:
        remaining, digit = divmod(remaining, 36)
        digits.append(alphabet[digit])
    return "".join(reversed(digits))
