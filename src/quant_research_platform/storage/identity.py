"""Filesystem and persisted-file identity primitives."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


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
