"""Fail-if-exists atomic write helpers."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_name,
    scratch_owner_is_stale,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError

_MAX_ATOMIC_TARGET_MARKER_BYTES = 32 * 1024


def reject_if_exists(path: Path) -> None:
    if Path(path).exists():
        raise OutputAlreadyExistsError()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_sibling(final_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    claimed_temp = False
    try:
        with _open_file_preserving_primary(
            temp_path,
            "x",
            encoding=encoding,
        ) as handle:
            claimed_temp = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        reject_if_exists(final_path)
        publish_no_replace(temp_path, final_path)
    except BaseException:
        if claimed_temp:
            try:
                _remove_path(temp_path)
            except BaseException:
                pass
        raise
    return final_path


def atomic_write_json(path: Path, payload: object) -> Path:
    return atomic_write_text(
        path,
        render_qrp_json_object(payload) + "\n",
        encoding="utf-8",
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_sibling(final_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    claimed_temp = False
    try:
        temp_path.touch(exist_ok=False)
        claimed_temp = True
        frame.to_csv(temp_path, **kwargs)
        reject_if_exists(final_path)
        publish_no_replace(temp_path, final_path)
    except BaseException:
        if claimed_temp:
            try:
                _remove_path(temp_path)
            except BaseException:
                pass
        raise
    return final_path


def atomic_write_directory[T](root: Path, writer: Callable[[Path], T]) -> T:
    final_root = Path(root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    target_identity = _atomic_target_identity(final_root)
    scratch_prefix = _atomic_directory_scratch_prefix(final_root)
    _cleanup_stale_atomic_directories(
        final_root,
        prefix=scratch_prefix,
        target_identity=target_identity,
    )
    reject_if_exists(final_root)
    temp_root = final_root.with_name(
        owned_scratch_name(scratch_prefix, uuid4().hex[:12])
    )
    target_marker = _atomic_target_marker(temp_root)
    claimed_temp = False
    claimed_marker = False
    try:
        temp_root.mkdir()
        claimed_temp = True
        with _open_file_preserving_primary(
            target_marker,
            "x",
            encoding="utf-8",
        ) as handle:
            claimed_marker = True
            handle.write(target_identity)
            handle.flush()
            os.fsync(handle.fileno())
        result = writer(temp_root)
        reject_if_exists(final_root)
        try:
            publish_no_replace(temp_root, final_root)
        except FileExistsError as exc:
            if final_root.exists():
                raise OutputAlreadyExistsError() from exc
            raise
        _remove_atomic_target_marker(target_marker)
    except BaseException:
        if claimed_temp:
            try:
                _remove_path(temp_root)
            except BaseException:
                pass
        if claimed_marker:
            try:
                _remove_atomic_target_marker(target_marker)
            except BaseException:
                pass
        raise
    return result


def _cleanup_stale_atomic_directories(
    final_root: Path,
    *,
    prefix: str,
    target_identity: str,
) -> None:
    first_failure: BaseException | None = None
    try:
        stage_candidates = iter(final_root.parent.glob(f"{prefix}*"))
    except BaseException as exc:
        first_failure = exc
        stage_candidates = iter(())
    while True:
        try:
            temp_root = next(stage_candidates)
        except StopIteration:
            break
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        try:
            is_directory = temp_root.is_dir()
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        if not is_directory:
            continue
        try:
            is_stale = scratch_owner_is_stale(temp_root, prefix=prefix)
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        if not is_stale:
            continue
        try:
            target_marker = _atomic_target_marker(temp_root)
            stage_is_empty = not any(temp_root.iterdir())
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        try:
            marker_is_file = target_marker.is_file()
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        if marker_is_file:
            try:
                observed_identity = _read_atomic_target_marker(target_marker)
            except OSError as exc:
                if exc.args != ("invalid_atomic_target_marker",):
                    if first_failure is None:
                        first_failure = exc
                    continue
                if not stage_is_empty:
                    continue
            except UnicodeError:
                if not stage_is_empty:
                    continue
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc
                continue
            else:
                if observed_identity != target_identity and not stage_is_empty:
                    continue
        elif not stage_is_empty:
            continue
        try:
            _remove_path(temp_root)
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        try:
            _remove_atomic_target_marker(target_marker)
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
    try:
        marker_candidates = iter(final_root.parent.glob(f"{prefix}*.target"))
    except BaseException as exc:
        if first_failure is None:
            first_failure = exc
        marker_candidates = iter(())
    while True:
        try:
            target_marker = next(marker_candidates)
        except StopIteration:
            break
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        try:
            temp_root = Path(str(target_marker)[: -len(".target")])
            stage_exists = temp_root.exists()
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        if stage_exists:
            continue
        try:
            is_stale = scratch_owner_is_stale(temp_root, prefix=prefix)
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
            continue
        if not is_stale:
            continue
        try:
            _remove_atomic_target_marker(target_marker)
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
    if first_failure is not None:
        raise first_failure


def _atomic_directory_scratch_prefix(final_root: Path) -> str:
    target_digest = hashlib.sha256(
        os.path.normcase(final_root.name).encode("utf-8")
    ).hexdigest()[:8]
    return f".atomic-{target_digest}-"


def _atomic_target_identity(final_root: Path) -> str:
    return os.path.normcase(str(final_root.resolve(strict=False)))


def _atomic_target_marker(temp_root: Path) -> Path:
    return Path(f"{temp_root}.target")


def _read_atomic_target_marker(path: Path) -> str:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or int(metadata.st_nlink) != 1
            or metadata.st_size > _MAX_ATOMIC_TARGET_MARKER_BYTES
        ):
            raise OSError("invalid_atomic_target_marker")
        with _open_file_preserving_primary(candidate, "rb") as handle:
            raw = handle.read(_MAX_ATOMIC_TARGET_MARKER_BYTES + 1)
    except OSError:
        raise
    if len(raw) > _MAX_ATOMIC_TARGET_MARKER_BYTES:
        raise OSError("invalid_atomic_target_marker")
    return raw.decode("utf-8", errors="strict")


def _remove_atomic_target_marker(target_marker: Path) -> None:
    target_marker.unlink(missing_ok=True)


@contextmanager
def _open_file_preserving_primary(
    path: Path,
    *args: Any,
    **kwargs: Any,
) -> Iterator[Any]:
    handle = path.open(*args, **kwargs)
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


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f".tmp-{uuid4().hex[:12]}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_no_replace(
    source: Path,
    target: Path,
    *,
    attempts: int = 3,
    delay_seconds: float = 0.10,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            _rename_no_replace(source, target)
            return
        except FileExistsError as exc:
            if Path(target).exists():
                raise OutputAlreadyExistsError() from exc
            raise
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


def _rename_no_replace(source: Path, target: Path) -> None:
    if os.name == "nt":
        source.rename(target)
        return
    if not sys.platform.startswith("linux"):
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unsupported",
            target,
        )

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOTSUP,
            "renameat2 is required for atomic no-replace directory publication",
            target,
        )
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


__all__ = [
    "atomic_write_csv",
    "atomic_write_directory",
    "atomic_write_json",
    "atomic_write_text",
    "reject_if_exists",
]
