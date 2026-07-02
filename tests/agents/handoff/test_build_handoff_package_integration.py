from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import (
        _build_test_package,
        _git,
        _init_handoff_repo,
        _load_handoff_module,
        _write_receipts,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _build_test_package,
        _git,
        _init_handoff_repo,
        _load_handoff_module,
        _write_receipts,
    )

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_handoff_package_minimal_layout_and_sidecar(
    tmp_path: Path,
) -> None:
    repo, package, sidecar, _base, final_head = _build_test_package(tmp_path)
    handoff_contract = _load_handoff_module("handoff_contract")

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
        task_metadata = json.loads(archive.read("task_metadata.json").decode("utf-8"))
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))

    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert _git(repo, "rev-parse", "HEAD").strip() == final_head
    assert names == set(handoff_contract.REQUIRED_HANDOFF_ENTRIES)
    assert set(sidecar_payload) == set(handoff_contract.ALLOWED_SIDECAR_KEYS)
    assert set(sidecar_payload["status_summary"]) == set(
        handoff_contract.ALLOWED_STATUS_SUMMARY_KEYS
    )
    assert "handoff_package_path" not in task_metadata
    assert "handoff_sidecar_path" not in task_metadata
    assert validation_summary["schema_id"] == "qrp-handoff-validation-summary"
    assert {row["name"] for row in validation_summary["checks"]} == set(
        handoff_contract.MANDATORY_VALIDATION_ROWS
    )
    assert {entry["path"] for entry in manifest["entries"]} == names - {
        "artifact_manifest.json"
    }
    for old_entry in handoff_contract.FORBIDDEN_HANDOFF_ENTRIES:
        assert old_entry not in names


def test_builder_rejects_manual_evidence_task_fields(
    tmp_path: Path,
) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    receipts_dir = tmp_path / "receipts"
    _write_receipts(receipts_dir, repo=repo, final_head=final_head)

    task = {
        "schema_id": "qrp-handoff-task-metadata",
        "mode": "evidence",
        "slug": "manual-field",
        "branch": "feature/synthetic-handoff-review",
        "base": base,
        "final_head": final_head,
        "handoff_package_path": "handoffs/old.zip",
    }

    with pytest.raises(SystemExit, match="task_metadata_forbidden_field"):
        builder.build_handoff_package(
            repo_root=repo,
            out_dir=tmp_path / "out",
            receipts_dir=receipts_dir,
            task=task,
            verify=False,
        )


def test_builder_rejects_old_receipt_filename(tmp_path: Path) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    receipts_dir = tmp_path / "receipts"
    _write_receipts(receipts_dir, repo=repo, final_head=final_head)
    (receipts_dir / "private-smoke-status.json").write_text(
        '{"schema_id":"qrp-old","status":"PASS"}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="forbidden_receipt_file"):
        builder.build_handoff_package(
            repo_root=repo,
            out_dir=tmp_path / "out",
            receipts_dir=receipts_dir,
            task={
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": "old-name",
                "branch": "feature/synthetic-handoff-review",
                "base": base,
                "final_head": final_head,
            },
            verify=False,
        )
