from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_research_platform.artifacts._output_bundle_validation import (
    validate_output_bundle,
)


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    files = ("stage_profile.csv", "stage_profile.json", "stage_profile.md")
    payload = {
        "schema_id": "qrp-output-manifest",
        "output_id": "diagnostic_output",
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": "test_run",
        "metadata": {
            "market_id": "a_share",
            "run_type": "diagnostics",
            "scope": "LOCAL_ONLY",
        },
        "files": list(files),
    }
    (root / "manifest.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    for name in files:
        (root / name).write_bytes(name.encode("utf-8"))


def test_validate_output_bundle_returns_declared_member_order(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)

    manifest, members = validate_output_bundle(
        root,
        expected_run_id="test_run",
        expected_output_id="diagnostic_output",
    )

    assert manifest["files"] == [
        "stage_profile.csv",
        "stage_profile.json",
        "stage_profile.md",
    ]
    assert members == (
        root / "stage_profile.csv",
        root / "stage_profile.json",
        root / "stage_profile.md",
    )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_validate_output_bundle_requires_exact_member_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    if mutation == "missing":
        (root / "stage_profile.csv").unlink()
    else:
        (root / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_declared_hardlink_alias(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    (root / "stage_profile.json").unlink()
    try:
        os.link(root / "stage_profile.csv", root / "stage_profile.json")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_member_physical_identity_invalid"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_external_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    try:
        os.link(root / "stage_profile.csv", tmp_path / "external.bin")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_member_physical_identity_invalid"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_symlinked_manifest(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    external = tmp_path / "external-manifest.json"
    external.write_bytes((root / "manifest.json").read_bytes())
    (root / "manifest.json").unlink()
    try:
        (root / "manifest.json").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_manifest_physical_identity_invalid"):
        validate_output_bundle(root)
