"""Fail-if-exists atomic write helpers."""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pandas as pd

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._scratch_ownership import (
    create_owned_scratch,
    move_path_no_replace,
    owned_scratch_name,
    remove_owned_scratch,
    require_owned_scratch,
    scratch_identity_matches,
    scratch_owner_is_stale,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError

_MAX_ATOMIC_TARGET_MARKER_BYTES = 32 * 1024


def reject_if_exists(path: Path) -> None:
    candidate = require_unaliased_path(Path(path), error="invalid_storage_write_path")
    if candidate.exists():
        raise OutputAlreadyExistsError()


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_storage_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = final_path.parent.lstat()
    require_unaliased_path(final_path, error="invalid_storage_write_path")
    scratch_owner.cleanup_stale_owned_scratch(
        final_path.parent,
        parent_identity=parent_identity,
        prefix=".f-",
        directory=False,
    )
    temp_path = _temp_sibling(final_path)
    require_unaliased_path(temp_path, error="invalid_storage_write_path")
    claimed_temp: os.stat_result | None = None
    try:
        handle, claimed_temp = scratch_owner.open_owned_scratch_file(
            temp_path,
            encoding=encoding,
        )
        with scratch_owner.handle_preserving_primary(handle):
            require_owned_scratch(final_path.parent, parent_identity)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        reject_if_exists(final_path)
        require_unaliased_path(temp_path, error="invalid_storage_write_path")
        require_owned_scratch(temp_path, claimed_temp)
        publish_no_replace(
            temp_path,
            final_path,
            source_identity=claimed_temp,
        )
    except BaseException:
        if claimed_temp is not None:
            try:
                _remove_claimed_path(temp_path, claimed_temp)
            except BaseException:
                pass
        raise
    return final_path


def atomic_write_json(
    path: Path,
    payload: object,
) -> Path:
    return atomic_write_text(
        path,
        render_qrp_json_object(payload) + "\n",
        encoding="utf-8",
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_storage_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = final_path.parent.lstat()
    require_unaliased_path(final_path, error="invalid_storage_write_path")
    scratch_owner.cleanup_stale_owned_scratch(
        final_path.parent,
        parent_identity=parent_identity,
        prefix=".f-",
        directory=False,
    )
    temp_path = _temp_sibling(final_path)
    require_unaliased_path(temp_path, error="invalid_storage_write_path")
    claimed_temp: os.stat_result | None = None
    try:
        handle, claimed_temp = scratch_owner.open_owned_scratch_file(
            temp_path,
            encoding=str(kwargs.get("encoding", "utf-8")),
        )
        with scratch_owner.handle_preserving_primary(handle):
            require_owned_scratch(final_path.parent, parent_identity)
            frame.to_csv(handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        reject_if_exists(final_path)
        require_unaliased_path(temp_path, error="invalid_storage_write_path")
        require_owned_scratch(temp_path, claimed_temp)
        publish_no_replace(temp_path, final_path, source_identity=claimed_temp)
    except BaseException:
        if claimed_temp is not None:
            try:
                _remove_claimed_path(temp_path, claimed_temp)
            except BaseException:
                pass
        raise
    return final_path


def atomic_write_directory[T](
    root: Path,
    writer: Callable[[Path], T],
    *,
    finalize_memory: bool = False,
) -> T:
    final_root = Path(root)
    require_unaliased_path(final_root.parent, error="invalid_storage_write_path")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = final_root.parent.lstat()
    require_unaliased_path(final_root, error="invalid_storage_write_path")
    target_identity = _atomic_target_identity(final_root)
    scratch_prefix = _atomic_directory_scratch_prefix(final_root)
    _cleanup_stale_atomic_directories(
        final_root,
        parent_identity=parent_identity,
        prefix=scratch_prefix,
        target_identity=target_identity,
    )
    reject_if_exists(final_root)
    temp_root = final_root.with_name(
        owned_scratch_name(scratch_prefix, uuid4().hex[:8])
    )
    target_marker = _atomic_target_marker(temp_root)
    claimed_temp: os.stat_result | None = None
    claimed_marker: os.stat_result | None = None
    try:
        claimed_temp = create_owned_scratch(temp_root)
        require_owned_scratch(final_root.parent, parent_identity)
        require_unaliased_path(temp_root, error="invalid_storage_write_path")
        require_owned_scratch(temp_root, claimed_temp)
        handle, claimed_marker = scratch_owner.open_owned_scratch_file(
            target_marker,
            encoding="utf-8",
        )
        require_owned_scratch(final_root.parent, parent_identity)
        require_owned_scratch(temp_root, claimed_temp)
        with scratch_owner.handle_preserving_primary(handle):
            handle.write(target_identity)
            handle.flush()
            os.fsync(handle.fileno())
        require_owned_scratch(target_marker, claimed_marker)
        require_owned_scratch(temp_root, claimed_temp)
        require_owned_scratch(final_root.parent, parent_identity)
        result = writer(temp_root)
        reject_if_exists(final_root)
        require_unaliased_path(temp_root, error="invalid_storage_write_path")
        require_owned_scratch(temp_root, claimed_temp)
        require_owned_scratch(target_marker, claimed_marker)
        _remove_claimed_path(target_marker, claimed_marker)
        claimed_marker = None
        policy = current_runtime_resource_policy() if finalize_memory else None
        if finalize_memory and policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        try:
            publish_no_replace(temp_root, final_root, source_identity=claimed_temp)
        except FileExistsError as exc:
            try:
                target_exists = final_root.exists()
            except BaseException as classification_error:
                exc.add_note(
                    "atomic target classification also failed: "
                    f"{type(classification_error).__name__}"
                )
                raise exc from classification_error
            if target_exists:
                raise OutputAlreadyExistsError() from exc
            raise
        if policy is not None:
            policy.finish_memory_measurement(
                on_reject=lambda: _remove_claimed_path(final_root, claimed_temp)
            )
    except BaseException:
        if claimed_temp is not None:
            try:
                _remove_claimed_path(temp_root, claimed_temp)
            except BaseException:
                pass
        if claimed_marker is not None:
            try:
                _remove_claimed_path(target_marker, claimed_marker)
            except BaseException:
                pass
        raise
    return result


def _cleanup_stale_atomic_directories(
    final_root: Path,
    *,
    parent_identity: os.stat_result,
    prefix: str,
    target_identity: str,
) -> None:
    parent = require_unaliased_path(
        final_root.parent,
        error="invalid_storage_write_path",
    )
    first_failure: BaseException | None = None

    def remember(error: BaseException) -> None:
        nonlocal first_failure
        if first_failure is None:
            first_failure = error

    def require_parent() -> None:
        try:
            require_owned_scratch(parent, parent_identity)
        except BaseException as error:
            if first_failure is not None:
                raise first_failure from None
            raise error

    require_parent()
    try:
        for temp_root in parent.glob(f"{prefix}*"):
            try:
                temp_root = require_unaliased_path(
                    temp_root,
                    error="invalid_storage_write_path",
                )
                temp_identity = temp_root.lstat()
                if not stat.S_ISDIR(temp_identity.st_mode):
                    continue
                if not scratch_owner_is_stale(temp_root, prefix=prefix):
                    continue
                target_marker = require_unaliased_path(
                    _atomic_target_marker(temp_root),
                    error="invalid_storage_write_path",
                )
                stage_is_empty = next(temp_root.iterdir(), None) is None
                marker_identity: os.stat_result | None = None
                try:
                    candidate_identity = target_marker.lstat()
                except FileNotFoundError:
                    candidate_identity = None
                if candidate_identity is not None and stat.S_ISREG(
                    candidate_identity.st_mode
                ):
                    marker_identity = candidate_identity
                    try:
                        observed_identity = _read_atomic_target_marker(target_marker)
                    except OSError as error:
                        if error.args != ("invalid_atomic_target_marker",):
                            raise
                        if not stage_is_empty:
                            continue
                    except UnicodeError:
                        if not stage_is_empty:
                            continue
                    else:
                        if observed_identity != target_identity and not stage_is_empty:
                            continue
                    require_owned_scratch(target_marker, marker_identity)
                elif not stage_is_empty:
                    continue
                require_parent()
                remove_owned_scratch(temp_root, temp_identity)
                if marker_identity is not None:
                    require_parent()
                    remove_owned_scratch(target_marker, marker_identity)
            except FileNotFoundError:
                continue
            except BaseException as error:
                remember(error)
                require_parent()
    except BaseException as error:
        if error is first_failure:
            raise
        remember(error)
        require_parent()

    require_parent()
    try:
        for target_marker in parent.glob(f"{prefix}*.target"):
            try:
                target_marker = require_unaliased_path(
                    target_marker,
                    error="invalid_storage_write_path",
                )
                marker_identity = target_marker.lstat()
                temp_root = require_unaliased_path(
                    Path(str(target_marker)[: -len(".target")]),
                    error="invalid_storage_write_path",
                )
                if temp_root.exists() or not scratch_owner_is_stale(
                    temp_root,
                    prefix=prefix,
                ):
                    continue
                try:
                    _read_atomic_target_marker(target_marker)
                except (OSError, UnicodeError):
                    pass
                require_owned_scratch(target_marker, marker_identity)
                require_parent()
                remove_owned_scratch(target_marker, marker_identity)
            except FileNotFoundError:
                continue
            except BaseException as error:
                remember(error)
                require_parent()
    except BaseException as error:
        if error is first_failure:
            raise
        remember(error)
        require_parent()
    require_parent()
    if first_failure is not None:
        raise first_failure


def _atomic_directory_scratch_prefix(final_root: Path) -> str:
    target_digest = hashlib.sha256(
        os.path.normcase(final_root.name).encode("utf-8")
    ).hexdigest()[:8]
    return f".a-{target_digest}-"


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
        handle = candidate.open("rb")
        with scratch_owner.handle_preserving_primary(handle):
            raw = handle.read(_MAX_ATOMIC_TARGET_MARKER_BYTES + 1)
    except OSError:
        raise
    if len(raw) > _MAX_ATOMIC_TARGET_MARKER_BYTES:
        raise OSError("invalid_atomic_target_marker")
    return raw.decode("utf-8", errors="strict")


def _temp_sibling(path: Path) -> Path:
    return path.with_name(owned_scratch_name(".f-", uuid4().hex[:8]))


def _remove_claimed_path(
    path: Path,
    identity: os.stat_result | None,
) -> None:
    if identity is not None:
        remove_owned_scratch(path, identity)


def publish_no_replace(
    source: Path,
    target: Path,
    *,
    source_identity: os.stat_result,
) -> None:
    source = require_unaliased_path(Path(source), error="invalid_storage_write_path")
    target = require_unaliased_path(Path(target), error="invalid_storage_write_path")
    policy = current_runtime_resource_policy()
    if policy is not None:
        policy.assert_memory_within_budget()
    for attempt in range(1, 4):
        require_unaliased_path(source, error="invalid_storage_write_path")
        require_unaliased_path(target, error="invalid_storage_write_path")
        require_owned_scratch(source, source_identity)
        primary: BaseException | None = None
        try:
            move_path_no_replace(source, target)
        except BaseException as error:
            primary = error

        target_exact = False
        target_probe_error: BaseException | None = None
        try:
            target_exact = scratch_identity_matches(target, source_identity)
        except BaseException as error:
            target_probe_error = error

        source_exact = False
        source_probe_error: BaseException | None = None
        try:
            source_exact = scratch_identity_matches(source, source_identity)
        except BaseException as error:
            source_probe_error = error

        if target_exact and source_probe_error is not None:
            active_primary = primary or source_probe_error
            try:
                remove_owned_scratch(target, source_identity)
            except BaseException:
                raise active_primary from None
            raise active_primary
        if target_probe_error is not None or source_probe_error is not None:
            raise primary or target_probe_error or source_probe_error  # type: ignore[misc]
        if target_exact and not source_exact:
            if primary is None or isinstance(primary, OSError):
                return
            raise primary
        if primary is not None:
            if isinstance(primary, FileExistsError):
                try:
                    target_exists = target.exists()
                except BaseException as classification_error:
                    primary.add_note(
                        "atomic target classification also failed: "
                        f"{type(classification_error).__name__}"
                    )
                    raise primary from None
                if target_exists:
                    raise OutputAlreadyExistsError() from primary
                raise primary
            if isinstance(primary, PermissionError) and source_exact:
                try:
                    target_missing = not target.exists()
                except BaseException:
                    raise primary from None
                if target_missing and attempt < 3:
                    try:
                        time.sleep(0.10)
                    except BaseException:
                        raise primary from None
                    continue
            raise primary
        raise ValueError("storage_scratch_identity_changed")


__all__ = [
    "atomic_write_csv",
    "atomic_write_directory",
    "atomic_write_json",
    "atomic_write_text",
    "reject_if_exists",
]
