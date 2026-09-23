"""Owner-qualified same-parent scratch and no-replace primitives."""

from __future__ import annotations

import errno
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryFile, gettempdir
from typing import Any

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.identity import native_filesystem_path


@contextmanager
def handle_preserving_primary(handle: Any) -> Iterator[Any]:
    """Close a handle without replacing an active primary exception."""

    try:
        yield handle
    except BaseException as primary:
        try:
            handle.close()
        except BaseException as close_error:
            primary.add_note(
                f"scratch handle close also failed: {type(close_error).__name__}"
            )
        raise
    else:
        handle.close()


def move_path_no_replace(source: Path, target: Path) -> None:
    """Move one path without replacing an existing destination."""

    source = Path(source)
    target = Path(target)
    if os.name == "nt":
        os.rename(native_filesystem_path(source), native_filesystem_path(target))
        return
    if not sys.platform.startswith("linux"):
        raise OSError(errno.ENOTSUP, "no-replace move is unsupported", target)
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "renameat2 is required", target)
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), target)


def remove_owned_scratch(
    path: Path,
    *,
    onexc: Callable[[Callable[[str], object], str, BaseException], object]
    | None = None,
) -> None:
    path = Path(path)
    target = native_filesystem_path(path)
    try:
        current = os.lstat(target)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode):
        shutil.rmtree(target, onexc=onexc)
    else:
        os.unlink(target)


def open_owned_scratch_file(
    path: Path,
    *,
    encoding: str | None = None,
) -> Any:
    """Create and claim one exclusive ordinary scratch file."""

    path = Path(path)
    descriptor = os.open(
        native_filesystem_path(path), os.O_RDWR | os.O_CREAT | os.O_EXCL
    )
    try:
        if encoding is None:
            handle = os.fdopen(descriptor, "w+b")
        else:
            handle = os.fdopen(descriptor, "w+", encoding=encoding, newline="")
        return handle
    except BaseException as primary:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            primary.add_note(
                f"scratch descriptor close also failed: {type(cleanup_error).__name__}"
            )
        try:
            remove_owned_scratch(path)
        except BaseException as cleanup_error:
            primary.add_note(
                f"scratch cleanup also failed: {type(cleanup_error).__name__}"
            )
        raise


def owned_scratch_name(prefix: str, unique: str) -> str:
    """Return a compact per-operation scratch name."""

    if not prefix or not unique or any(part in unique for part in ("/", "\\")):
        raise ValueError("invalid_storage_scratch_name")
    return f"{prefix}{unique}"


class ScratchReservation:
    """Admit owned scratch growth, charging RAM only on memory filesystems.

    Close files/remove directories before releasing this reservation. A resize
    is a pre-write upper bound; shrink only after successful truncation.
    """

    def __init__(
        self, parent: Path, *, resource_policy: RuntimeResourcePolicy | None = None
    ):
        self.parent = Path(parent)
        self._bytes = 0
        self._resources = ExitStack()
        try:
            self._policy = self._resources.enter_context(
                runtime_resource_policy_scope(resource_policy)
            )
            self._lease = (
                self._policy.lease("storage.scratch.tmpfs", 0)
                if _memory_filesystem(self.parent)
                else None
            )
        except BaseException:
            self._resources.__exit__(*sys.exc_info())
            raise

    @property
    def reserved_bytes(self) -> int:
        return self._bytes

    @property
    def memory_capacity_bytes(self) -> int | None:
        return (
            None
            if self._lease is None
            else self._bytes + self._policy.working_set_headroom_bytes
        )

    def resize(self, total_bytes: int) -> None:
        if type(total_bytes) is not int or total_bytes < 0:
            raise ValueError("invalid_storage_scratch_bytes")
        growth = total_bytes - self._bytes
        if growth > 0:
            require_free_disk_headroom(self.parent, reserve_bytes=growth)
            if self._lease is not None:
                self._policy.require_working_set(
                    growth, error="storage_scratch_exceeds_memory_capacity"
                )
                self._lease.grow(growth)
        elif self._lease is not None:
            self._lease.shrink_to(total_bytes)
        self._bytes = total_bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                f"scratch reservation close also failed: {type(cleanup_error).__name__}"
            )
        return False

    def close(self) -> None:
        if self._bytes:
            error = RuntimeError("storage_scratch_not_released")
            self._resources.__exit__(type(error), error, None)
            raise error
        if self._lease is not None:
            self._lease.close()
        self._resources.close()


def _memory_filesystem(path: Path) -> bool:
    if not sys.platform.startswith("linux"):
        return False
    resolved = path.resolve()
    selected = None
    with Path("/proc/self/mountinfo").open(encoding="utf-8") as mounts:
        for line in mounts:
            fields, filesystem = line.rstrip().split(" - ", 1)
            mount = Path(
                re.sub(
                    r"\\([0-7]{3})",
                    lambda match: chr(int(match[1], 8)),
                    fields.split()[4],
                )
            )
            if resolved.is_relative_to(mount) and (
                selected is None or len(mount.parts) >= selected[0]
            ):
                selected = (
                    len(mount.parts),
                    filesystem.split()[0] in {"tmpfs", "ramfs"},
                )
    if selected is None:
        raise RuntimeError("storage_scratch_mount_unknown")
    return selected[1]


@contextmanager
def reserved_temporary_file(*, resource_policy: RuntimeResourcePolicy | None = None):
    """Own one binary spool and its capacity until the actual handle closes."""
    reservation = ScratchReservation(
        Path(gettempdir()), resource_policy=resource_policy
    )
    with handle_preserving_primary(reservation):
        handle = TemporaryFile(mode="w+b")
        try:
            with handle_preserving_primary(handle):
                yield handle, reservation
        finally:
            if handle.closed:
                reservation.resize(0)
