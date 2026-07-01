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
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _build_test_package,
        _git,
        _init_handoff_repo,
        _load_handoff_module,
    )

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_handoff_package_receipts_dir_positive_path(tmp_path: Path) -> None:
    repo, package, sidecar, base, final_head = _build_test_package(tmp_path)

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
        generation = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
        package_smoke = json.loads(
            archive.read("handoff_evidence/package-smoke-receipt.json").decode(
                "utf-8"
            )
        )

    assert _git(repo, "rev-parse", "HEAD").strip() == final_head
    assert sidecar.exists()
    assert "final-branch.bundle" in names
    assert "artifact_manifest.json" in names
    assert "task_metadata.json" in names
    assert "git_status.txt" in names
    assert "validation_evidence.json" not in names
    assert "handoff_evidence/source-tree-scan.json" not in names
    assert "handoff_evidence/verification-matrix.json" not in names
    assert "handoff_evidence/package-smoke.json" not in names
    assert "handoff_evidence/hard-scan-classification.json" not in names
    assert "handoff_evidence/test-checks.json" not in names
    assert "handoff_evidence/smoke-compare-status.json" not in names
    assert "handoff_evidence/package-smoke-receipt.json" in names
    assert "handoff_evidence/hard-scan-receipt.json" in names
    assert "handoff_evidence/check-receipts.json" in names
    assert "handoff_evidence/private-smoke-status.json" in names
    assert "handoff_evidence/validation-summary.json" in names
    assert "handoff_evidence/generation.json" in names
    assert "handoff_evidence/bundle-reconstruction.json" in names
    assert generation["base"] == base
    assert generation["final_head"] == final_head
    assert generation["hard_scan_receipt_path"] == (
        "handoff_evidence/hard-scan-receipt.json"
    )
    assert "source_tree_scan_path" not in generation
    assert validation_summary["schema_id"] == "qrp-handoff-validation-summary"
    assert {
        "wheel",
        "sdist",
        "install_target",
    } == set(package_smoke["content_scan"]["scanned_artifact_kinds"])


@pytest.mark.parametrize(
    "manual_key",
    ("handoff_evidence", "handoff_markdown_evidence", "validation_evidence"),
)
def test_builder_rejects_manual_evidence_task_fields(
    tmp_path: Path,
    manual_key: str,
) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    receipts_dir = tmp_path / "receipts"
    _write_minimal_receipts(receipts_dir, repo=repo, final_head=final_head)

    task = {
        "schema_id": "qrp-handoff-task-metadata",
        "mode": "evidence",
        "slug": "manual-field",
        "branch": "feature/synthetic-handoff-review",
        "base": base,
        "final_head": final_head,
        manual_key: {},
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
    _write_minimal_receipts(receipts_dir, repo=repo, final_head=final_head)
    (receipts_dir / "package-smoke.json").write_text(
        '{"schema_id":"qrp-old","status":"PASS"}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="old_handoff_evidence_name_forbidden"):
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


def _write_minimal_receipts(
    receipts_dir: Path,
    *,
    repo: Path,
    final_head: str,
) -> None:
    try:
        from tests.agents.handoff.conftest import _write_receipts
    except ModuleNotFoundError:  # pragma: no cover - addopts='' import fallback
        from conftest import _write_receipts  # type: ignore[no-redef]

    _write_receipts(receipts_dir, repo=repo, final_head=final_head)
