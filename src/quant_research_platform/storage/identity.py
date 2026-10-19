"""Filesystem and persisted-file identity primitives."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator


class _DigestingReader:
    def __init__(self, handle) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self._digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def read1(self, size: int = -1) -> bytes:
        read = getattr(self._handle, "read1", self._handle.read)
        chunk = read(size)
        self._digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    def readinto(self, buffer) -> int:
        chunk = self.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def readinto1(self, buffer) -> int:
        chunk = self.read1(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def drain(self) -> None:
        while self.read(64 * 1024):
            pass

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _hash_file_content(path: Path) -> str:
    digest = hashlib.sha256()
    with closing_iterator(open_filesystem_file(path, "rb")) as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
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


def filesystem_exists(path: Path) -> bool:
    return os.path.exists(native_filesystem_path(path))


def filesystem_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
    return os.stat(native_filesystem_path(path), follow_symlinks=follow_symlinks)


def filesystem_is_file(path: Path) -> bool:
    return os.path.isfile(native_filesystem_path(path))


def filesystem_is_dir(path: Path) -> bool:
    return os.path.isdir(native_filesystem_path(path))


def filesystem_iterdir(path: Path):
    logical_root = Path(path)
    for child in Path(native_filesystem_path(logical_root)).iterdir():
        yield (logical_root / child.name)


def filesystem_glob(path: Path, pattern: str):
    logical_root = Path(path)
    native_root = Path(native_filesystem_path(logical_root))
    for child in native_root.glob(pattern):
        yield (logical_root / child.relative_to(native_root))


def filesystem_scandir(path: Path):
    return os.scandir(native_filesystem_path(path))


def filesystem_walk(path: Path, *, follow_symlinks: bool = False, on_error=None):
    logical_root = Path(path)
    native_root = Path(native_filesystem_path(logical_root))
    for parent, directories, files in os.walk(
        native_root, followlinks=follow_symlinks, onerror=on_error
    ):
        yield (logical_root / Path(parent).relative_to(native_root), directories, files)


def filesystem_mkdir(
    path: Path, *, mode: int = 511, parents: bool = False, exist_ok: bool = False
) -> None:
    Path(native_filesystem_path(path)).mkdir(
        mode=mode, parents=parents, exist_ok=exist_ok
    )


def open_filesystem_file(
    path: Path,
    mode: str = "rb",
    *,
    buffering: int = -1,
    encoding: str | None = None,
    newline: str | None = None,
):
    return open(
        native_filesystem_path(path),
        mode,
        buffering=buffering,
        encoding=encoding,
        newline=newline,
    )


def filesystem_write_bytes(path: Path, data: bytes) -> int:
    return Path(native_filesystem_path(path)).write_bytes(data)


def filesystem_write_text(path: Path, data: str, *, encoding: str) -> int:
    return Path(native_filesystem_path(path)).write_text(data, encoding=encoding)


def filesystem_rename(source: Path, target: Path) -> None:
    os.rename(native_filesystem_path(source), native_filesystem_path(target))


def filesystem_unlink(path: Path, *, missing_ok: bool = False) -> None:
    try:
        os.unlink(native_filesystem_path(path))
    except FileNotFoundError:
        if not missing_ok:
            raise


def filesystem_open_descriptor(path: Path, flags: int, mode: int = 511) -> int:
    return os.open(native_filesystem_path(path), flags, mode)


def filesystem_remove_tree(path: Path, *, onexc=None) -> None:
    shutil.rmtree(native_filesystem_path(path), onexc=onexc)


def filesystem_gzip_open(path: Path, mode: str, *, encoding: str):
    return gzip.open(native_filesystem_path(path), mode, encoding=encoding)


def require_artifact_root(root: Path) -> Path:
    candidate = Path(root)
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
    native = Path(native_filesystem_path(candidate))
    return native.is_symlink() or native.is_junction()
