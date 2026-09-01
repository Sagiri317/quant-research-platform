"""Filesystem and persisted-file identity primitives."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_INSTANCE_IDENTITY_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DatasetFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


def validate_instance_identity(value: object) -> str:
    if type(value) is not str or _INSTANCE_IDENTITY_RE.fullmatch(value) is None:
        raise ValueError("invalid_instance_identity")
    return value


def dataset_file_identity_from_stat(value: os.stat_result) -> DatasetFileIdentity:
    return DatasetFileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        size=int(value.st_size),
        modified_ns=int(value.st_mtime_ns),
    )


def _file_identity_bytes(identity: DatasetFileIdentity) -> bytes:
    return b"\0".join(
        str(value).encode("ascii")
        for value in (
            identity.device,
            identity.inode,
            identity.size,
            identity.modified_ns,
        )
    )


def dataset_file_identity(path: Path) -> DatasetFileIdentity:
    candidate = require_unaliased_path(
        Path(path),
        error="invalid_dataset_part_files",
    )
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_ino) <= 0:
        raise ValueError("invalid_dataset_part_files")
    return dataset_file_identity_from_stat(before)


def content_hash_for_files(
    files: Iterable[tuple[str, Path]],
    *,
    error: str,
) -> str:
    ordered = sorted(tuple(files), key=lambda item: item[0])
    if not ordered or len({logical_id for logical_id, _ in ordered}) != len(ordered):
        raise ValueError(error)
    digest = hashlib.sha256()
    for logical_id, path in ordered:
        if type(logical_id) is not str or not logical_id:
            raise ValueError(error)
        logical_bytes = logical_id.encode("utf-8")
        try:
            identity = dataset_file_identity(path)
            digest.update(len(logical_bytes).to_bytes(8, "big"))
            digest.update(logical_bytes)
            digest.update(identity.size.to_bytes(8, "big"))
            with Path(path).open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                after = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
        except (OSError, ValueError) as exc:
            raise ValueError(error) from exc
        if after != identity:
            raise ValueError(error)
    return digest.hexdigest()


def require_artifact_root(root: Path) -> Path:
    candidate = Path(root)
    if candidate.name == "data":
        raise ValueError("dataset_root_must_be_artifact_root")
    return require_unaliased_path(candidate, error="invalid_dataset_part_files")


def require_unaliased_path(path: Path, *, error: str) -> Path:
    candidate = Path(path)
    current = Path(candidate.anchor) if candidate.is_absolute() else Path.cwd()
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    try:
        for component in (*reversed(current.parents), current):
            if path_is_filesystem_alias(component):
                raise ValueError(error)
        for part in parts:
            if part == ".":
                continue
            if part == "..":
                current = current.parent
                continue
            current /= part
            if path_is_filesystem_alias(current):
                raise ValueError(error)
    except OSError as exc:
        raise ValueError(error) from exc
    return candidate


def path_is_filesystem_alias(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_symlink() or (
        hasattr(candidate, "is_junction") and candidate.is_junction()
    )
