"""Strict physical validation for persisted output bundles."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from quant_research_platform.artifacts._output_manifest import (
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts._json import parse_qrp_json_object


def validate_output_bundle(
    root: Path,
    *,
    expected_run_id: str | None = None,
    expected_output_id: str | None = None,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    """Validate one complete output bundle and return its declared files in order."""

    bundle_root = Path(root)
    resolved_root = _validated_bundle_root(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    manifest_identity = _validated_regular_member(
        manifest_path,
        resolved_root=resolved_root,
        error="output_manifest_physical_identity_invalid",
    )
    try:
        manifest = parse_qrp_json_object(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("output_manifest_unreadable") from exc
    validate_output_manifest_business_envelope(manifest)

    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("output_manifest_run_id_mismatch")
    if expected_output_id is not None and manifest["output_id"] != expected_output_id:
        raise ValueError("output_manifest_output_id_mismatch")

    declared_names = tuple(manifest["files"])
    expected_names = ("manifest.json", *declared_names)
    observed_entries = tuple(bundle_root.iterdir())
    observed_names = tuple(entry.name for entry in observed_entries)
    if len({name.casefold() for name in observed_names}) != len(observed_names):
        raise ValueError("output_member_set_mismatch")
    if set(observed_names) != set(expected_names) or len(observed_names) != len(
        expected_names
    ):
        raise ValueError("output_member_set_mismatch")

    physical_identities = {manifest_identity}
    declared_paths: list[Path] = []
    for name in declared_names:
        path = bundle_root / name
        identity = _validated_regular_member(
            path,
            resolved_root=resolved_root,
            error="output_member_physical_identity_invalid",
        )
        if identity in physical_identities:
            raise ValueError("output_member_physical_identity_invalid")
        physical_identities.add(identity)
        declared_paths.append(path)
    return manifest, tuple(declared_paths)


def _validated_bundle_root(root: Path) -> Path:
    try:
        if root.is_symlink() or root.is_junction() or not root.is_dir():
            raise ValueError("output_bundle_root_invalid")
        return root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("output_bundle_root_invalid") from exc


def _validated_regular_member(
    path: Path,
    *,
    resolved_root: Path,
    error: str,
) -> tuple[int, int]:
    try:
        if path.is_symlink() or path.is_junction():
            raise ValueError(error)
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(error)
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(error) from exc
    if resolved_path.parent != resolved_root:
        raise ValueError(error)
    return metadata.st_dev, metadata.st_ino


__all__ = ["validate_output_bundle"]
