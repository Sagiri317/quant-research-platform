from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tests.agents.handoff.conftest import (
    init_handoff_test_repo,
    run_git,
)
from tests.agents.handoff.receipt_contract_fixtures import (
    make_check_receipts_payload,
    make_hard_scan_receipt_payload,
    make_package_smoke_receipt_payload,
)
from tests.helpers.governance_git_repos import run_governance_script
from tests.helpers.maintenance_script_loader import load_handoff_script_module

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_handoff_package_single_scripted_entrypoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _base, final_head = init_handoff_test_repo(repo)
    builder_path = (
        repo
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
        / "build_handoff_package.py"
    )

    result = run_governance_script(
        repo,
        str(builder_path),
        "--repo-root",
        ".",
        "--out-dir",
        ".run/handoffs",
        "--task-name",
        "handoff-test",
        "--base-ref",
        "origin/main",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    package = repo / ".run" / "handoffs" / "handoff-test.zip"
    sidecar = repo / ".run" / "handoffs" / "handoff-test.json"
    assert package.is_file()
    assert sidecar.is_file()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        task_metadata = json.loads(archive.read("task_metadata.json").decode("utf-8"))
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
        check_receipts = json.loads(
            archive.read("handoff_evidence/check-receipts.json").decode("utf-8")
        )
        package_smoke = json.loads(
            archive.read("handoff_evidence/package-smoke-receipt.json").decode("utf-8")
        )
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    handoff_contract = load_handoff_script_module("handoff_contract")

    assert run_git(repo, "rev-parse", "HEAD").strip() == final_head
    assert names == set(handoff_contract.REQUIRED_HANDOFF_ENTRIES)
    assert set(sidecar_payload) == {"schema_id", "zip_filename", "zip_sha256"}
    assert set(task_metadata) == {
        "schema_id",
        "task_name",
        "branch",
        "base",
        "final_head",
    }
    assert task_metadata["task_name"] == "handoff-test"
    assert task_metadata["final_head"] == final_head
    assert set(validation_summary) == {"schema_id", "status", "rows"}
    assert validation_summary["status"] == "PASS"
    assert {row["name"] for row in validation_summary["rows"]} == set(
        handoff_contract.MANDATORY_VALIDATION_ROWS
    )
    assert {row["status"] for row in validation_summary["rows"]} <= {
        "PASS",
        "FAIL",
        "NOT_RUN",
        "BLOCKED",
    }
    assert {row["command_id"] for row in check_receipts["checks"]} == {
        "profile-smoke"
    }
    assert set(package_smoke) == {
        "schema_id",
        "status",
        "build",
        "wheel_extract",
        "import",
        "content_scan",
    }
    profile_smoke = next(
        row
        for row in check_receipts["checks"]
        if row["command_id"] == "profile-smoke"
    )
    assert profile_smoke["command_argv"] == [
        "python",
        "tests/tools/run_profile.py",
        "--profile",
        "profile-smoke",
        "--run-dir",
        ".run/handoff-replay/profile-smoke",
    ]
    assert isinstance(profile_smoke["duration_seconds"], float)
    assert {entry["path"] for entry in manifest["entries"]} == names - {
        "artifact_manifest.json"
    }


def test_builder_generates_machine_receipts_from_bundle_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    init_handoff_test_repo(repo)
    builder = load_handoff_script_module("build_handoff_package")
    receipt_roots: list[Path] = []

    class Verification:
        ok = True
        zip_sha256 = "0" * 64
        handoff_package_verify_status = "PASS"
        share_safety_status = "PASS"
        failures: tuple[str, ...] = ()

    def fake_build_machine_receipts(
        receipt_root: Path,
        final_head: str,
        **_kwargs: object,
    ) -> dict[str, dict[str, object]]:
        receipt_roots.append(receipt_root)
        assert receipt_root != repo
        assert (receipt_root / ".git").is_dir()
        assert run_git(receipt_root, "rev-parse", "HEAD").strip() == final_head
        return {
            builder.CHECK_RECEIPTS_ENTRY: make_check_receipts_payload(),
            builder.PACKAGE_SMOKE_RECEIPT_ENTRY: (
                make_package_smoke_receipt_payload()
            ),
            builder.HARD_SCAN_RECEIPT_ENTRY: make_hard_scan_receipt_payload(
                scanned_head=final_head
            ),
        }

    monkeypatch.setattr(builder, "build_machine_receipts", fake_build_machine_receipts)
    monkeypatch.setattr(
        builder,
        "validate_handoff_package_payload",
        lambda *_args, **_kwargs: Verification(),
    )

    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=tmp_path / "out",
        task_name="handoff-test",
        base_ref="origin/main",
    )

    assert result["ok"] is True
    assert len(receipt_roots) == 1


def test_builder_fails_when_replayed_validation_row_is_not_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    init_handoff_test_repo(repo)
    builder = load_handoff_script_module("build_handoff_package")
    final_head = run_git(repo, "rev-parse", "HEAD").strip()
    good_receipts = {
        builder.CHECK_RECEIPTS_ENTRY: make_check_receipts_payload(),
        builder.PACKAGE_SMOKE_RECEIPT_ENTRY: make_package_smoke_receipt_payload(),
        builder.HARD_SCAN_RECEIPT_ENTRY: make_hard_scan_receipt_payload(
            scanned_head=final_head
        ),
    }
    check_receipts = good_receipts[builder.CHECK_RECEIPTS_ENTRY]
    profile_smoke = next(
        row
        for row in check_receipts["checks"]
        if row["command_id"] == "profile-smoke"
    )
    profile_smoke["status"] = "FAIL"
    profile_smoke["exit_code"] = 1
    profile_smoke["failure_class"] = "pytest_failed"
    check_receipts["status"] = "FAIL"

    class Verification:
        ok = True
        zip_sha256 = "0" * 64
        handoff_package_verify_status = "PASS"
        share_safety_status = "PASS"
        failures: tuple[str, ...] = ()

    monkeypatch.setattr(builder, "build_machine_receipts", lambda *_: good_receipts)
    monkeypatch.setattr(
        builder,
        "validate_handoff_package_payload",
        lambda *_args, **_kwargs: Verification(),
    )

    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=tmp_path / "out",
        task_name="handoff-test",
        base_ref="origin/main",
    )

    assert result["ok"] is False
    assert (
        "validation_summary_row_not_pass:scripted_checks:FAIL"
        in result["verify_failures"]
    )
    assert (
        "validation_summary_row_not_pass:profile_smoke:FAIL"
        in result["verify_failures"]
    )


def test_builder_fails_when_package_smoke_timeout_blocks_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    init_handoff_test_repo(repo)
    builder = load_handoff_script_module("build_handoff_package")
    final_head = run_git(repo, "rev-parse", "HEAD").strip()
    receipts = {
        builder.CHECK_RECEIPTS_ENTRY: make_check_receipts_payload(),
        builder.PACKAGE_SMOKE_RECEIPT_ENTRY: make_package_smoke_receipt_payload(),
        builder.HARD_SCAN_RECEIPT_ENTRY: make_hard_scan_receipt_payload(
            scanned_head=final_head
        ),
    }
    package_smoke = receipts[builder.PACKAGE_SMOKE_RECEIPT_ENTRY]
    package_smoke["status"] = "BLOCKED"
    package_smoke["build"] = {"status": "BLOCKED", "artifact_kinds": []}
    package_smoke["wheel_extract"] = {
        "status": "NOT_RUN",
        "method": "zipfile_extract",
    }
    package_smoke["import"] = {
        "status": "NOT_RUN",
        "module": "quant_research_platform",
    }
    package_smoke["content_scan"] = {
        "status": "NOT_RUN",
        "forbidden_member_count": 0,
        "scanned_artifact_kinds": [],
    }

    class Verification:
        ok = True
        zip_sha256 = "0" * 64
        handoff_package_verify_status = "PASS"
        share_safety_status = "PASS"
        failures: tuple[str, ...] = ()

    monkeypatch.setattr(builder, "build_machine_receipts", lambda *_: receipts)
    monkeypatch.setattr(
        builder,
        "validate_handoff_package_payload",
        lambda *_args, **_kwargs: Verification(),
    )

    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=tmp_path / "out",
        task_name="handoff-test",
        base_ref="origin/main",
    )

    assert result["ok"] is False
    assert (
        "validation_summary_row_not_pass:package_smoke:BLOCKED"
        in result["verify_failures"]
    )
