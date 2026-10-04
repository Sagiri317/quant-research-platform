"""One package-level production software identity."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    filesystem_scandir,
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
    verified_file_handle,
)

NUMERICAL_MODEL_DISTRIBUTIONS = (
    "arch",
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "xgboost",
)
_LOWER_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True, init=False)
class ProductionSoftwareIdentity:
    """Canonical package payload and its identity hash."""

    _payload: Mapping[str, object]
    identity_hash: str

    def __init__(
        self,
        payload: Mapping[str, object],
        identity_hash: str,
    ) -> None:
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "identity_hash", identity_hash)
        self.__post_init__()

    def __post_init__(self) -> None:
        payload = _plain_value(_freeze_value(self._payload))
        _validate_payload(payload)
        payload["dependency_versions"] = dict(
            sorted(payload["dependency_versions"].items())
        )
        if type(self.identity_hash) is not str or not _is_hash(self.identity_hash):
            raise ValueError("software_identity_hash_invalid")
        if self.identity_hash != canonical_json_hash(payload):
            raise ValueError("software_identity_hash_mismatch")
        object.__setattr__(self, "_payload", _freeze_value(payload))

    @property
    def payload(self) -> dict[str, object]:
        return _plain_value(self._payload)


def production_software_identity(
    *,
    package_root: Path | None = None,
    dependency_versions: Mapping[str, str] | None = None,
) -> ProductionSoftwareIdentity:
    """Hash production package content plus resolved backend versions."""

    root = (
        Path(package_root)
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    payload = {
        "package_content_hash": _package_content_hash(root),
        "dependency_versions": _dependency_versions(dependency_versions),
    }
    return ProductionSoftwareIdentity(
        payload=payload,
        identity_hash=canonical_json_hash(payload),
    )


def _package_content_hash(root: Path) -> str:
    root = require_unaliased_path(
        Path(root),
        error="production_package_root_invalid",
    )
    files, directories = _package_inventory(root)

    if not files:
        raise ValueError("production_package_empty")
    digest = hashlib.sha256()
    for relative, path, expected_identity in files:
        relative_bytes = relative.encode("utf-8")
        try:
            digest.update(len(relative_bytes).to_bytes(8, "big"))
            digest.update(relative_bytes)
            digest.update(expected_identity[2].to_bytes(8, "big"))
            with (
                handle_preserving_primary(open_filesystem_file(path)) as handle,
                verified_file_handle(
                    path,
                    handle,
                    expected_identity=DatasetFileIdentity(*expected_identity),
                    error="production_package_identity_changed",
                ),
            ):
                while content := handle.read(1024 * 1024):
                    digest.update(content)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("production_package_file_unreadable") from exc
        if _file_identity(path) != expected_identity:
            raise ValueError("production_package_identity_changed")
    if _package_inventory(root) != (files, directories):
        raise ValueError("production_package_identity_changed")
    return digest.hexdigest()


def _package_inventory(root: Path):
    root_before = _directory_identity(root)
    files: list[tuple[str, Path, tuple[int, int, int, int]]] = []
    directories: dict[Path, tuple[int, int, int, int]] = {root: root_before}
    pending = [(root, "")]
    while pending:
        directory, relative_parent = pending.pop()
        try:
            with handle_preserving_primary(filesystem_scandir(directory)) as entries:
                for entry in entries:
                    entry_path = require_unaliased_path(
                        directory / entry.name, error="production_package_alias"
                    )
                    if entry.is_symlink():
                        raise ValueError("production_package_alias")
                    relative = (
                        f"{relative_parent}/{entry.name}"
                        if relative_parent
                        else entry.name
                    )
                    if "__pycache__" in Path(relative).parts:
                        continue
                    if entry.name.casefold().endswith(".pyc"):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        identity = _directory_identity(entry_path)
                        directories[entry_path] = identity
                        pending.append((entry_path, relative))
                    elif entry.is_file(follow_symlinks=False):
                        files.append(
                            (
                                relative.replace("\\", "/"),
                                entry_path,
                                _file_identity(entry_path),
                            )
                        )
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("production_package_enumeration_failed") from exc
        if _directory_identity(directory) != directories[directory]:
            raise ValueError("production_package_identity_changed")

    for directory, expected_identity in directories.items():
        if _directory_identity(directory) != expected_identity:
            raise ValueError("production_package_identity_changed")
    return tuple(sorted(files, key=lambda item: item[0])), directories


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        value = filesystem_stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("production_package_file_unreadable") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError("production_package_alias")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _directory_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        value = filesystem_stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("production_package_root_invalid") from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError("production_package_root_invalid")
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)


def _dependency_versions(
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("software_dependency_versions_invalid")
        result = dict(supplied)
        if set(result) != set(NUMERICAL_MODEL_DISTRIBUTIONS) or any(
            type(name) is not str or not name or type(value) is not str or not value
            for name, value in result.items()
        ):
            raise ValueError("software_dependency_versions_invalid")
        return dict(sorted(result.items()))
    result: dict[str, str] = {}
    for distribution in NUMERICAL_MODEL_DISTRIBUTIONS:
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError as exc:
            raise ValueError("software_dependency_version_unavailable") from exc
    return result


def _validate_payload(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "package_content_hash",
        "dependency_versions",
    }:
        raise ValueError("software_identity_payload_invalid")
    package_hash = value["package_content_hash"]
    if type(package_hash) is not str or not _is_hash(package_hash):
        raise ValueError("software_package_content_hash_invalid")
    versions = value["dependency_versions"]
    if type(versions) is not dict or set(versions) != set(
        NUMERICAL_MODEL_DISTRIBUTIONS
    ):
        raise ValueError("software_dependency_versions_invalid")
    if any(
        type(name) is not str
        or not name
        or type(version_value) is not str
        or not version_value
        for name, version_value in versions.items()
    ):
        raise ValueError("software_dependency_versions_invalid")


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(character in _LOWER_HEX for character in value)


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(child) for key, child in value.items()}
        )
    if type(value) in {list, tuple}:
        return tuple(_freeze_value(child) for child in value)
    return value


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(child) for key, child in value.items()}
    if type(value) in {list, tuple}:
        return [_plain_value(child) for child in value]
    return value
