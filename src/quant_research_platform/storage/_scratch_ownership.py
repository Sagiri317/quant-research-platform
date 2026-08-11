"""Process identity for recoverable same-volume storage scratch paths."""

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

import psutil


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


def require_owned_scratch(path: Path, identity: os.stat_result) -> None:
    if not scratch_identity_matches(path, identity):
        raise ValueError("storage_scratch_identity_changed")


def scratch_identity_matches(path: Path, identity: os.stat_result) -> bool:
    try:
        current = Path(path).lstat()
    except FileNotFoundError:
        return False
    return os.path.samestat(identity, current)


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
    parent_identity = path.parent.lstat()
    os.mkdir(path, mode)
    identity = path.lstat()
    reparse_flag = getattr(identity, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
    )
    if not stat.S_ISDIR(identity.st_mode) or reparse_flag:
        raise ValueError("storage_scratch_identity_changed")
    if not os.path.samestat(parent_identity, path.parent.lstat()):
        raise ValueError("storage_scratch_parent_identity_changed")
    return identity


def open_owned_scratch_file(
    path: Path,
    *,
    encoding: str | None = None,
) -> tuple[Any, os.stat_result]:
    """Create and claim one exclusive ordinary scratch file."""

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
    identity: os.stat_result | None = None
    handle = None
    try:
        identity = os.fstat(descriptor)
        if encoding is None:
            handle = os.fdopen(descriptor, "w+b")
        else:
            handle = os.fdopen(
                descriptor,
                "w+",
                encoding=encoding,
                newline="",
            )
        return handle, identity
    except BaseException:
        if handle is not None:
            try:
                handle.close()
            except BaseException:
                pass
        elif identity is None:
            try:
                identity = os.fstat(descriptor)
            except BaseException:
                pass
            try:
                os.close(descriptor)
            except BaseException:
                pass
        else:
            try:
                current = os.fstat(descriptor)
                if os.path.samestat(identity, current):
                    os.close(descriptor)
            except BaseException:
                pass
        if identity is not None:
            try:
                remove_owned_scratch(path, identity)
            except BaseException:
                pass
        raise


def cleanup_stale_owned_scratch(
    parent: Path,
    *,
    parent_identity: os.stat_result,
    prefix: str,
    directory: bool,
    onexc: Callable[[Callable[[str], object], str, BaseException], object]
    | None = None,
) -> None:
    """Remove stale exact owner-qualified children while attempting every candidate."""

    parent = Path(parent)
    first_failure: BaseException | None = None

    def require_parent() -> None:
        try:
            require_owned_scratch(parent, parent_identity)
        except BaseException as error:
            if first_failure is not None:
                raise first_failure from None
            raise error

    require_parent()
    try:
        candidates = iter(parent.glob(f"{prefix}*"))
    except BaseException as error:
        first_failure = error
        candidates = iter(())
    while True:
        try:
            candidate = next(candidates)
        except StopIteration:
            break
        except BaseException as error:
            if first_failure is None:
                first_failure = error
            continue
        try:
            identity = candidate.lstat()
        except FileNotFoundError:
            continue
        except BaseException as error:
            if first_failure is None:
                first_failure = error
            continue
        reparse_flag = getattr(identity, "st_file_attributes", 0) & getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0,
        )
        expected_kind = (
            stat.S_ISDIR(identity.st_mode)
            if directory
            else stat.S_ISREG(identity.st_mode) and int(identity.st_nlink) == 1
        )
        if reparse_flag or not expected_kind:
            continue
        try:
            stale = scratch_owner_is_stale(candidate, prefix=prefix)
        except BaseException as error:
            if first_failure is None:
                first_failure = error
            continue
        if not stale:
            continue
        require_parent()
        try:
            remove_owned_scratch(candidate, identity, onexc=onexc)
        except BaseException as error:
            if first_failure is None:
                first_failure = error
            require_parent()
    require_parent()
    if first_failure is not None:
        raise first_failure


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
