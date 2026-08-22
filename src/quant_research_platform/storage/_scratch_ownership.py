"""Owner-qualified same-parent scratch and no-replace primitives."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def handle_preserving_primary(handle: Any) -> Iterator[Any]:
    """Close a handle without replacing an active primary exception."""

    try:
        yield handle
    finally:
        if sys.exception() is None:
            handle.close()
        else:
            try:
                handle.close()
            except BaseException:
                pass


def move_path_no_replace(source: Path, target: Path) -> None:
    """Move one path without replacing an existing destination."""

    source = Path(source)
    target = Path(target)
    if os.name == "nt":
        os.rename(source, target)
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


def scratch_identity_matches(path: Path, identity: os.stat_result) -> bool:
    try:
        current = Path(path).lstat()
    except FileNotFoundError:
        return False
    return os.path.samestat(identity, current)


def require_owned_scratch(path: Path, identity: os.stat_result) -> None:
    if not scratch_identity_matches(path, identity):
        raise ValueError("storage_scratch_identity_changed")


def _windows_extended_path(path: Path) -> str:
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return f"\\\\?\\UNC\\{absolute[2:]}"
    return f"\\\\?\\{absolute}"


def remove_owned_scratch(
    path: Path,
    identity: os.stat_result,
    *,
    onexc: Callable[[Callable[[str], object], str, BaseException], object]
    | None = None,
) -> None:
    path = Path(path)
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if not os.path.samestat(identity, current):
        raise ValueError("storage_scratch_identity_changed")
    if stat.S_ISDIR(current.st_mode):
        target: str | Path = _windows_extended_path(path) if os.name == "nt" else path
        shutil.rmtree(target, onexc=onexc)
    else:
        path.unlink()


def create_owned_scratch(path: Path, *, mode: int = 0o777) -> os.stat_result:
    path = Path(path)
    os.mkdir(path, mode)
    identity = path.lstat()
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("storage_scratch_identity_changed")
    return identity


def open_owned_scratch_file(
    path: Path,
    *,
    encoding: str | None = None,
) -> tuple[Any, os.stat_result]:
    """Create and claim one exclusive ordinary scratch file."""

    path = Path(path)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
    identity = os.fstat(descriptor)
    try:
        if encoding is None:
            handle = os.fdopen(descriptor, "w+b")
        else:
            handle = os.fdopen(descriptor, "w+", encoding=encoding, newline="")
        return handle, identity
    except BaseException as primary:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            primary.add_note(
                "scratch descriptor close also failed: "
                f"{type(cleanup_error).__name__}"
            )
        try:
            remove_owned_scratch(path, identity)
        except BaseException as cleanup_error:
            primary.add_note(
                "scratch cleanup also failed: " f"{type(cleanup_error).__name__}"
            )
        raise


def owned_scratch_name(prefix: str, unique: str) -> str:
    """Return a compact per-operation scratch name."""

    if not prefix or not unique or any(part in unique for part in ("/", "\\")):
        raise ValueError("invalid_storage_scratch_name")
    return f"{prefix}{unique}"
