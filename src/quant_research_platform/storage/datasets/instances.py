"""Read-only discovery of immutable dataset instances."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from quant_research_platform.contracts.business_manifest import (
    validate_business_manifest_envelope,
)
from quant_research_platform.storage.datasets.schema import (
    COMMON_MANIFEST_METADATA_KEY,
)
from quant_research_platform.storage.identity import (
    require_unaliased_path,
    validate_instance_identity,
)
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

MAX_DISCOVERED_INSTANCES = 1024


def discover_dataset_instances(
    logical_root: Path,
    *,
    max_instances: int = MAX_DISCOVERED_INSTANCES,
) -> tuple[Path, ...]:
    """Return valid immutable instance directories in canonical name order."""

    if type(max_instances) is not int or max_instances <= 0:
        raise ValueError("dataset_instance_limit_invalid")
    root = require_unaliased_path(
        Path(logical_root),
        error="dataset_instance_root_invalid",
    )
    if not root.exists():
        return ()
    root_identity = _directory_identity(root, "dataset_instance_root_invalid")
    instances_root = root / "instances"
    require_unaliased_path(instances_root, error="dataset_instance_root_invalid")
    if not instances_root.exists():
        return ()
    instances_identity = _directory_identity(
        instances_root,
        "dataset_instance_root_invalid",
    )
    names = _instance_names(instances_root, max_instances, instances_identity)

    result: list[Path] = []
    for name in names:
        entry = instances_root / name
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ValueError("dataset_instance_child_invalid")
        try:
            validate_instance_identity(name)
        except ValueError as exc:
            raise ValueError("dataset_instance_child_invalid") from exc
        entry_identity = _directory_identity(
            entry,
            "dataset_instance_child_invalid",
        )
        try:
            manifest = read_dataset_manifest_evidence(entry)
            envelope = validate_business_manifest_envelope(
                manifest[COMMON_MANIFEST_METADATA_KEY]
            )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError("dataset_instance_invalid") from exc
        if envelope["instance_identity"] != name:
            raise ValueError("dataset_instance_identity_mismatch")
        if (
            envelope["logical_artifact_id"] != manifest["data_id"]
            or manifest["data_id"] != root.name
        ):
            raise ValueError("dataset_instance_logical_id_mismatch")
        if (
            _directory_identity(entry, "dataset_instance_child_invalid")
            != entry_identity
        ):
            raise ValueError("dataset_instance_identity_changed")
        result.append(entry)
    if (
        _directory_identity(instances_root, "dataset_instance_root_invalid")
        != instances_identity
    ):
        raise ValueError("dataset_instance_identity_changed")
    if _directory_identity(root, "dataset_instance_root_invalid") != root_identity:
        raise ValueError("dataset_instance_identity_changed")
    return tuple(result)


def _instance_names(
    instances_root: Path,
    max_instances: int,
    expected_identity: tuple[int, int, int, int],
) -> tuple[str, ...]:
    names: list[str] = []
    try:
        with os.scandir(instances_root) as entries:
            for entry in entries:
                if len(names) >= max_instances:
                    raise ValueError("dataset_instance_limit_exceeded")
                names.append(entry.name)
    except OSError as exc:
        raise ValueError("dataset_instance_root_invalid") from exc
    if (
        _directory_identity(instances_root, "dataset_instance_root_invalid")
        != expected_identity
    ):
        raise ValueError("dataset_instance_identity_changed")
    return tuple(sorted(names))


def _directory_identity(path: Path, error: str) -> tuple[int, int, int, int]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ValueError(error) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ValueError(error)
    return (value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns)
