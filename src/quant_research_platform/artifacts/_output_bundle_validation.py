"""Strict physical validation for persisted output bundles."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path

from quant_research_platform.artifacts._output_manifest import (
    validate_output_manifest_business_envelope,
)
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    dataset_file_identity,
)
from quant_research_platform.storage.manifests import read_bounded_qrp_json


def validate_output_bundle(
    root: Path,
    *,
    expected_run_id: str | None = None,
    expected_output_id: str | None = None,
    observer: Callable[
        [
            dict[str, object],
            tuple[tuple[Path, DatasetFileIdentity], ...],
        ],
        None,
    ]
    | None = None,
) -> tuple[
    dict[str, object],
    tuple[tuple[Path, DatasetFileIdentity], ...],
]:
    """Validate one complete output bundle and retain every admitted identity.

    The optional observer runs synchronously and read-only after identity capture and
    before the operation-final recheck.  The supported caller domain is a fully
    published bundle kept quiescent for this validation window, with no concurrent
    writer.
    """

    bundle_root = Path(root)
    resolved_root, root_identity = _validated_bundle_root(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    manifest_identity = _validated_regular_member(
        manifest_path,
        resolved_root=resolved_root,
        error="output_manifest_physical_identity_invalid",
    )
    try:
        _manifest_bytes, manifest = read_bounded_qrp_json(
            manifest_path,
            expected_identity=manifest_identity,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("output_manifest_unreadable") from exc
    validate_output_manifest_business_envelope(manifest)
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("output_manifest_run_id_mismatch")
    if expected_output_id is not None and manifest["output_id"] != expected_output_id:
        raise ValueError("output_manifest_output_id_mismatch")

    declared_names = tuple(manifest["files"])
    expected_names = ("manifest.json", *declared_names)
    _require_exact_member_set(bundle_root, expected_names)

    physical_identities = {manifest_identity}
    captured_identities = [(manifest_path, manifest_identity)]
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
        captured_identities.append((path, identity))
    admitted_identities = tuple(captured_identities)
    try:
        if observer is not None:
            observer(manifest, admitted_identities)
    finally:
        primary = sys.exception()
        try:
            final_resolved_root, final_root_identity = _validated_bundle_root(
                bundle_root
            )
            if final_resolved_root != resolved_root or not os.path.samestat(
                root_identity, final_root_identity
            ):
                raise ValueError("output_bundle_identity_changed")
            _require_exact_member_set(bundle_root, expected_names)
            try:
                identities_changed = any(
                    dataset_file_identity(path) != identity
                    for path, identity in admitted_identities
                )
            except (OSError, ValueError) as exc:
                raise ValueError("output_bundle_identity_changed") from exc
            if identities_changed:
                raise ValueError("output_bundle_identity_changed")
        except BaseException:
            if primary is None:
                raise
    return manifest, admitted_identities


def _require_exact_member_set(root: Path, expected_names: tuple[str, ...]) -> None:
    observed_names = tuple(entry.name for entry in root.iterdir())
    if (
        len({name.casefold() for name in observed_names}) != len(observed_names)
        or len(observed_names) != len(expected_names)
        or set(observed_names) != set(expected_names)
    ):
        raise ValueError("output_member_set_mismatch")


def _validated_bundle_root(root: Path) -> tuple[Path, os.stat_result]:
    try:
        identity = root.lstat()
        require_unaliased_path(root, error="output_bundle_root_invalid")
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError("output_bundle_root_invalid")
        resolved = root.resolve(strict=True)
        if not os.path.samestat(identity, root.lstat()):
            raise ValueError("output_bundle_root_invalid")
        return resolved, identity
    except OSError as exc:
        raise ValueError("output_bundle_root_invalid") from exc


def _validated_regular_member(
    path: Path,
    *,
    resolved_root: Path,
    error: str,
) -> DatasetFileIdentity:
    try:
        identity = dataset_file_identity(path)
        resolved_path = path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError(error) from exc
    if resolved_path.parent != resolved_root:
        raise ValueError(error)
    return identity


__all__ = ["validate_output_bundle"]
