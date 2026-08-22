"""Fail-if-exists atomic write helpers."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._scratch_ownership import (
    move_path_no_replace,
    open_owned_scratch_file,
    owned_scratch_name,
    remove_owned_scratch,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError


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
    final_path, temp_path = _prepare_file_target(path)
    claimed_temp = False
    try:
        handle = open_owned_scratch_file(temp_path, encoding=encoding)
        claimed_temp = True
        with scratch_owner.handle_preserving_primary(handle):
            require_unaliased_path(
                final_path.parent,
                error="invalid_storage_write_path",
            )
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        require_unaliased_path(final_path.parent, error="invalid_storage_write_path")
        publish_no_replace(temp_path, final_path)
    except BaseException as primary:
        _cleanup_claimed(temp_path, claimed_temp, primary)
        raise
    return final_path


def atomic_write_json(path: Path, payload: object) -> Path:
    return atomic_write_text(
        path,
        render_qrp_json_object(payload) + "\n",
        encoding="utf-8",
    )


def atomic_write_directory[T](
    root: Path,
    writer: Callable[[Path], T],
) -> T:
    final_root = require_unaliased_path(Path(root), error="invalid_storage_write_path")
    require_unaliased_path(final_root.parent, error="invalid_storage_write_path")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    reject_if_exists(final_root)
    temp_root = final_root.with_name(
        owned_scratch_name(".a-", uuid4().hex[:12])
    )
    claimed_temp = False
    try:
        os.mkdir(temp_root)
        claimed_temp = True
        require_unaliased_path(
            final_root.parent,
            error="invalid_storage_write_path",
        )
        result = writer(temp_root)
        require_unaliased_path(
            final_root.parent,
            error="invalid_storage_write_path",
        )
        publish_no_replace(temp_root, final_root)
    except BaseException as primary:
        _cleanup_claimed(temp_root, claimed_temp, primary)
        raise
    return result


def _prepare_file_target(path: Path) -> tuple[Path, Path]:
    final_path = require_unaliased_path(Path(path), error="invalid_storage_write_path")
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_storage_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    require_unaliased_path(final_path, error="invalid_storage_write_path")
    temp_path = final_path.with_name(
        owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_storage_write_path")
    return final_path, temp_path


def _cleanup_claimed(
    path: Path,
    claimed: bool,
    primary: BaseException,
) -> None:
    if not claimed:
        return
    try:
        remove_owned_scratch(path)
    except BaseException as cleanup_error:
        primary.add_note(
            "atomic scratch cleanup also failed: "
            f"{type(cleanup_error).__name__}"
        )


def publish_no_replace(
    source: Path,
    target: Path,
) -> None:
    source = require_unaliased_path(Path(source), error="invalid_storage_write_path")
    target = require_unaliased_path(Path(target), error="invalid_storage_write_path")
    try:
        move_path_no_replace(source, target)
    except FileExistsError as exc:
        if target.exists():
            raise OutputAlreadyExistsError() from exc
        raise


__all__ = [
    "atomic_write_directory",
    "atomic_write_json",
    "atomic_write_text",
    "publish_no_replace",
    "reject_if_exists",
]
