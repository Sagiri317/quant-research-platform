"""Fail-if-exists atomic write helpers."""

from __future__ import annotations

import errno
import os
import re
import stat
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_open_descriptor,
    filesystem_remove_tree,
    filesystem_rename,
    filesystem_stat,
    filesystem_unlink,
)


def is_atomic_directory_staging_name(name: str) -> bool:
    """Identify unpublished directory transactions created by this owner."""
    return re.fullmatch(r"\.a-[0-9a-f]{12}", name) is not None


def reject_if_exists(path: Path) -> None:
    candidate = Path(path)
    if filesystem_exists(candidate):
        raise OutputAlreadyExistsError()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_chunks(path, (text,), encoding=encoding)


def atomic_write_chunks(
    path: Path,
    chunks: Iterable[str] | Iterable[bytes],
    *,
    encoding: str | None = "utf-8",
) -> Path:
    final_path, temp_path = _prepare_file_target(path)
    claimed_temp = False
    try:
        handle = open_owned_scratch_file(temp_path, encoding=encoding)
        claimed_temp = True
        with closing_iterator(handle):
            with closing_iterator(iter(chunks)) as source:
                for chunk in source:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        publish_no_replace(temp_path, final_path)
    except BaseException as primary:
        _cleanup_claimed(temp_path, claimed_temp, primary)
        raise
    return final_path


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Publish an already-rendered binary file without replacing an earlier result."""
    return atomic_write_chunks(path, (payload,), encoding=None)


def atomic_write_directory[T](root: Path, writer: Callable[[Path], T]) -> T:
    final_root = Path(root)
    reject_if_exists(final_root)
    filesystem_mkdir(final_root.parent, parents=True, exist_ok=True)
    temp_root = final_root.with_name(f".a-{uuid4().hex[:12]}")
    claimed_temp = False
    try:
        filesystem_mkdir(temp_root)
        claimed_temp = True
        result = writer(temp_root)
        publish_no_replace(temp_root, final_root)
    except BaseException as primary:
        _cleanup_claimed(temp_root, claimed_temp, primary)
        raise
    return result


def _prepare_file_target(path: Path) -> tuple[Path, Path]:
    final_path = Path(path)
    reject_if_exists(final_path)
    filesystem_mkdir(final_path.parent, parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".f-{uuid4().hex[:12]}")
    return (final_path, temp_path)


def _cleanup_claimed(path: Path, claimed: bool, primary: BaseException) -> None:
    if not claimed:
        return
    try:
        remove_owned_scratch(path)
    except BaseException as cleanup_error:
        primary.add_note(
            f"atomic scratch cleanup also failed: {type(cleanup_error).__name__}"
        )


def publish_no_replace(source: Path, target: Path) -> None:
    source = Path(source)
    target = Path(target)
    try:
        move_path_no_replace(source, target)
    except FileExistsError as exc:
        if filesystem_exists(target):
            raise OutputAlreadyExistsError() from exc
        raise


def move_path_no_replace(source: Path, target: Path) -> None:
    """Move one path without replacing an existing destination."""
    source = Path(source)
    target = Path(target)
    if os.name == "nt":
        filesystem_rename(source, target)
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
    try:
        current = filesystem_stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode):
        filesystem_remove_tree(path, onexc=onexc)
    else:
        filesystem_unlink(path)


def open_owned_scratch_file(path: Path, *, encoding: str | None = None) -> Any:
    """Create and claim one exclusive ordinary scratch file."""
    path = Path(path)
    descriptor = filesystem_open_descriptor(path, os.O_RDWR | os.O_CREAT | os.O_EXCL)
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
