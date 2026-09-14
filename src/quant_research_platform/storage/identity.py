"""Filesystem and persisted-file identity primitives."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator

_INSTANCE_IDENTITY_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class DatasetFileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int


class _DigestingReader:
    def __init__(self, handle) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self._digest.update(chunk)
        return chunk

    def drain(self) -> None:
        while self.read(64 * 1024):
            pass

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


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
        before = os.stat(native_filesystem_path(candidate), follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_ino) <= 0:
        raise ValueError("invalid_dataset_part_files")
    return dataset_file_identity_from_stat(before)


def content_hash_for_files(
    files: Iterable[tuple[str, Path] | tuple[str, Path, DatasetFileIdentity]],
    *,
    error: str,
    resource_policy: object | None = None,
) -> str:
    """Hash logical member ids and bytes without using paths as identity."""

    entries: list[tuple[str, Path, DatasetFileIdentity | None]] = []
    for item in files:
        if isinstance(item, tuple):
            if len(item) == 2:
                logical_id, path = item
                expected = None
            elif len(item) == 3:
                logical_id, path, expected = item
            else:
                raise ValueError(error)
            if (
                type(logical_id) is not str
                or not logical_id
                or not isinstance(path, Path)
                or expected is not None
                and not isinstance(expected, DatasetFileIdentity)
            ):
                raise ValueError(error)
        else:
            raise ValueError(error)
        entries.append((logical_id, path, expected))
    if not entries:
        raise ValueError(error)
    member_hashes: list[tuple[str, str]] = []
    try:
        for index, (logical_id, path, expected) in enumerate(entries):
            before = dataset_file_identity(path)
            if expected is not None and before != expected:
                raise ValueError(error)
            entries[index] = (logical_id, path, before)
            if resource_policy is None:
                member_hash = _hash_file_content(path, before, error)
            else:
                headroom = resource_policy.operation_headroom_bytes
                if headroom <= 0:
                    raise ValueError(error)
                with resource_policy.lease(
                    "storage.content.identity",
                    min(1024 * 1024, headroom),
                ) as lease:
                    member_hash = _hash_file_content(
                        path,
                        before,
                        error,
                        chunk_size=lease.reserved_bytes,
                    )
            member_hashes.append((logical_id, member_hash))
        digest = hashlib.sha256()
        for logical_id, member_hash in sorted(member_hashes):
            for value in (logical_id, member_hash):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        for _, path, expected in entries:
            if dataset_file_identity(path) != expected:
                raise ValueError(error)
        return digest.hexdigest()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise ValueError(error) from exc


def _hash_file_content(
    path: Path,
    expected: DatasetFileIdentity,
    error: str,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    if chunk_size <= 0:
        raise ValueError(error)
    digest = hashlib.sha256()
    with closing_iterator(
        open(native_filesystem_path(path), "rb", buffering=0)
    ) as handle:
        handle_identity = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
        if handle_identity != expected:
            raise ValueError(error)
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            del chunk
        final_handle_identity = dataset_file_identity_from_stat(
            os.fstat(handle.fileno())
        )
    if final_handle_identity != expected or dataset_file_identity(path) != expected:
        raise ValueError(error)
    return digest.hexdigest()


def native_filesystem_path(path: Path) -> str | Path:
    """Return a Windows extended path when native APIs need long-path support."""

    if os.name != "nt":
        return Path(path)
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return f"\\\\?\\UNC\\{absolute[2:]}"
    return f"\\\\?\\{absolute}"


def open_filesystem_file(path: Path, mode: str = "rb"):
    """Open a local file while retaining normal Path seams below MAX_PATH."""

    candidate = Path(path)
    if os.name == "nt" and len(os.path.abspath(candidate)) >= 240:
        return open(native_filesystem_path(candidate), mode)
    return candidate.open(mode)


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
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError(error) from exc
    return candidate


def path_is_filesystem_alias(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_symlink() or candidate.is_junction()
