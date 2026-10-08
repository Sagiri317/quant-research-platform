"""Filesystem and persisted-file identity primitives."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import shutil
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from quant_research_platform.policy.lifecycle import closing_iterator

_INSTANCE_IDENTITY_RE = re.compile("^[0-9a-f]{64}$")


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
    return b"\x00".join(
        str(value).encode("ascii")
        for value in (
            identity.device,
            identity.inode,
            identity.size,
            identity.modified_ns,
        )
    )


def dataset_file_identity(path: Path) -> DatasetFileIdentity:
    candidate = require_unaliased_path(Path(path), error="invalid_dataset_part_files")
    try:
        before = filesystem_stat(candidate, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_ino) <= 0:
        raise ValueError("invalid_dataset_part_files")
    return dataset_file_identity_from_stat(before)


def content_hash_for_files(
    files: Iterable[tuple[str, Path] | tuple[str, Path, DatasetFileIdentity]],
    *,
    error: str,
) -> str:
    """Hash logical member ids and bytes with exact before/after identities."""
    entries = []
    hashes = []
    with closing_iterator(iter(files)) as members:
        try:
            for item in members:
                if not isinstance(item, tuple) or len(item) not in (2, 3):
                    raise ValueError(error)
                logical_id, path = item[:2]
                expected = item[2] if len(item) == 3 else None
                if (
                    type(logical_id) is not str
                    or not logical_id
                    or (not isinstance(path, Path))
                    or (
                        expected is not None
                        and (not isinstance(expected, DatasetFileIdentity))
                    )
                ):
                    raise ValueError(error)
                identity = dataset_file_identity(path)
                if expected is not None and identity != expected:
                    raise ValueError(error)
                entries.append((path, identity))
                hashes.append((logical_id, _hash_file_content(path, identity, error)))
            if not entries:
                raise ValueError(error)
            digest = hashlib.sha256()
            for logical_id, member_hash in sorted(hashes):
                for value in (logical_id, member_hash):
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
            for path, identity in entries:
                if dataset_file_identity(path) != identity:
                    raise ValueError(error)
            return digest.hexdigest()
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise ValueError(error) from exc


@contextmanager
def verified_file_handle(
    path: Path,
    handle: BinaryIO,
    *,
    expected_identity: DatasetFileIdentity,
    error: str = "dataset_file_identity_changed",
) -> Iterator[BinaryIO]:
    """Borrow one handle, binding its bytes to the captured published identity."""

    def verify() -> None:
        if (
            dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            != expected_identity
            or dataset_file_identity(path) != expected_identity
        ):
            raise ValueError(error)

    verify()
    primary = None
    try:
        yield handle
    except BaseException as body_error:
        primary = body_error
        raise
    finally:
        try:
            verify()
        except BaseException as verification_error:
            if primary is None:
                raise
            primary.add_note(
                "file identity verification also failed: "
                f"{type(verification_error).__name__}"
            )


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
    with closing_iterator(open_filesystem_file(path, "rb", buffering=0)) as handle:
        with verified_file_handle(
            path, handle, expected_identity=expected, error=error
        ):
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
                del chunk
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
    for parent, directories, files in native_root.walk(
        follow_symlinks=follow_symlinks, on_error=on_error
    ):
        yield (logical_root / parent.relative_to(native_root), directories, files)


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


def filesystem_read_bytes(path: Path) -> bytes:
    return Path(native_filesystem_path(path)).read_bytes()


def filesystem_write_bytes(path: Path, data: bytes) -> int:
    return Path(native_filesystem_path(path)).write_bytes(data)


def filesystem_read_text(path: Path, *, encoding: str) -> str:
    return Path(native_filesystem_path(path)).read_text(encoding=encoding)


def filesystem_write_text(path: Path, data: str, *, encoding: str) -> int:
    return Path(native_filesystem_path(path)).write_text(data, encoding=encoding)


def filesystem_rename(source: Path, target: Path) -> None:
    os.rename(native_filesystem_path(source), native_filesystem_path(target))


def filesystem_replace(source: Path, target: Path) -> None:
    os.replace(native_filesystem_path(source), native_filesystem_path(target))


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
    native = Path(native_filesystem_path(candidate))
    return native.is_symlink() or native.is_junction()
