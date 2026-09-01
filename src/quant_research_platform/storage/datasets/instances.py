"""Read-only discovery of immutable dataset instances."""

from __future__ import annotations

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
    if not root.is_dir():
        raise ValueError("dataset_instance_root_invalid")
    instances_root = root / "instances"
    require_unaliased_path(instances_root, error="dataset_instance_root_invalid")
    if not instances_root.exists():
        return ()
    if not instances_root.is_dir():
        raise ValueError("dataset_instance_root_invalid")

    entries: list[Path] = []
    try:
        for entry in instances_root.iterdir():
            if len(entries) >= max_instances:
                raise ValueError("dataset_instance_limit_exceeded")
            entries.append(entry)
    except OSError as exc:
        raise ValueError("dataset_instance_root_invalid") from exc

    result: list[Path] = []
    for entry in sorted(entries, key=lambda candidate: candidate.name):
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            validate_instance_identity(entry.name)
        except ValueError:
            continue
        try:
            manifest = read_dataset_manifest_evidence(entry)
            envelope = manifest[COMMON_MANIFEST_METADATA_KEY]
            validate_business_manifest_envelope(envelope)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError("dataset_instance_invalid") from exc
        if envelope["instance_identity"] != entry.name:
            raise ValueError("dataset_instance_identity_mismatch")
        result.append(entry)
    return tuple(result)
