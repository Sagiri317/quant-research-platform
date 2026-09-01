"""One package-level production software identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType

from quant_research_platform.contracts.json_codec import canonical_json_hash

NUMERICAL_MODEL_DISTRIBUTIONS = (
    "arch",
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "xgboost",
)


@dataclass(frozen=True, slots=True)
class ProductionSoftwareIdentity:
    """Canonical package payload and its identity hash."""

    _payload: Mapping[str, object]
    identity_hash: str

    @property
    def payload(self) -> dict[str, object]:
        return _plain_value(self._payload)


def production_software_identity(
    *,
    package_root: Path | None = None,
    dependency_versions: Mapping[str, str] | None = None,
) -> ProductionSoftwareIdentity:
    """Hash production package bytes plus resolved numerical/model versions."""

    root = Path(package_root) if package_root is not None else _package_root()
    files = _package_files(root)
    versions = _dependency_versions(dependency_versions)
    payload = {
        "package_files": files,
        "dependency_versions": versions,
    }
    return ProductionSoftwareIdentity(
        _payload=_freeze_value(payload),
        identity_hash=canonical_json_hash(payload),
    )


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _package_files(root: Path) -> list[dict[str, str]]:
    try:
        if not root.is_dir() or root.is_symlink():
            raise ValueError("production_package_root_invalid")
        paths = tuple(
            path
            for path in root.rglob("*")
            if "__pycache__" not in path.parts and path.suffix.lower() != ".pyc"
        )
    except OSError as exc:
        raise ValueError("production_package_root_invalid") from exc
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in sorted(
        paths,
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if path.is_symlink():
            raise ValueError("production_package_alias")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in seen:
            raise ValueError("production_package_path_collision")
        seen.add(folded)
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("production_package_file_unreadable") from exc
        files.append({"path": relative, "content_hash": content_hash})
    return files


def _dependency_versions(
    supplied: Mapping[str, str] | None,
) -> dict[str, str]:
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("software_dependency_versions_invalid")
        result = dict(supplied)
        if any(
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


def _freeze_value(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_value(child) for key, child in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_value(child) for child in value)
    return value


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(child) for key, child in value.items()}
    if type(value) in {list, tuple}:
        return [_plain_value(child) for child in value]
    return value
