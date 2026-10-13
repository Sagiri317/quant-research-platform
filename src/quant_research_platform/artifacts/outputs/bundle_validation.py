"""Strict physical validation for persisted output bundles."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.outputs.manifest import (
    validate_output_manifest,
)
from quant_research_platform.storage.identity import (
    filesystem_scandir,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file


def validate_output_bundle(
    root: Path, *, expected_definition_id: str | None = None, observer=None
) -> tuple[dict[str, object], tuple[Path, ...]]:
    """Read the manifest once and check its declared member paths."""
    root = require_unaliased_path(Path(root), error="output_bundle_root_invalid")
    if not root.is_dir():
        raise ValueError("output_bundle_root_invalid")
    manifest = read_qrp_json_file(root / "manifest.json")[1]
    validate_output_manifest(manifest)
    if (
        expected_definition_id is not None
        and manifest["output_definition_id"] != expected_definition_id
    ):
        raise ValueError("output_manifest_definition_id_mismatch")
    _require_exact_member_set(root, ("manifest.json", *manifest["files"]))
    members = tuple(root / name for name in manifest["files"])
    for path in members:
        require_unaliased_path(path, error="output_member_path_invalid")
        if not path.is_file():
            raise ValueError("output_member_path_invalid")
    if observer is not None:
        observer(manifest, members)
    return manifest, members


def _require_exact_member_set(root: Path, expected_names: tuple[str, ...]) -> None:
    try:
        with filesystem_scandir(root) as entries:
            observed_names = tuple(entry.name for entry in entries)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("output_member_set_mismatch") from exc
    if (
        len({name.casefold() for name in observed_names}) != len(observed_names)
        or len(observed_names) != len(expected_names)
        or set(observed_names) != set(expected_names)
    ):
        raise ValueError("output_member_set_mismatch")
