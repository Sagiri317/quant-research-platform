"""Process identity for recoverable same-volume storage scratch paths."""

from __future__ import annotations

import os
from pathlib import Path

import psutil

_CREATE_TIME_SCALE = 1_000
_OWNER_MARKER = ".owner.start"


def owned_scratch_name(prefix: str, unique: str) -> str:
    """Return a compact scratch name bound to this process incarnation."""

    if not prefix or not unique or "-" in unique:
        raise ValueError("invalid_storage_scratch_name")
    return f"{owned_scratch_prefix(prefix)}{unique}"


def owned_scratch_prefix(prefix: str) -> str:
    """Return the owner-qualified prefix used by random scratch-name creators."""

    if not prefix:
        raise ValueError("invalid_storage_scratch_name")
    pid, _ = _current_process_identity()
    return f"{prefix}{_base36(pid)}-"


def write_scratch_owner_identity(path: Path) -> None:
    """Persist the process-incarnation token before scratch receives private data."""

    _, create_time_token = _current_process_identity()
    (Path(path) / _OWNER_MARKER).write_text(
        _base36(create_time_token),
        encoding="ascii",
    )


def remove_scratch_owner_identity(path: Path) -> None:
    """Remove the owner token before a staged directory is published."""

    (Path(path) / _OWNER_MARKER).unlink()


def scratch_owner_is_alive(path: Path, *, prefix: str) -> bool:
    """Return whether a scratch name still belongs to the live process incarnation."""

    name = Path(path).name
    if not name.startswith(prefix):
        return False
    owner_parts = name[len(prefix) :].split("-", 1)
    if len(owner_parts) != 2:
        return False
    try:
        pid = int(owner_parts[0], 36)
    except ValueError:
        return False
    if pid <= 0 or not owner_parts[1]:
        return False
    try:
        process = psutil.Process(pid)
        observed_create_time = _create_time_token(process.create_time())
        try:
            expected_create_time = int(
                (Path(path) / _OWNER_MARKER).read_text(encoding="ascii"),
                36,
            )
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return process.is_running()
        return (
            expected_create_time > 0
            and process.is_running()
            and observed_create_time == expected_create_time
        )
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True


def _current_process_identity() -> tuple[int, int]:
    pid = os.getpid()
    return pid, _create_time_token(psutil.Process(pid).create_time())


def _create_time_token(create_time: float) -> int:
    return max(1, int(round(float(create_time) * _CREATE_TIME_SCALE)))


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
