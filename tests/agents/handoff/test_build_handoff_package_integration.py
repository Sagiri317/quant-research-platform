from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import (
        _git,
        _init_handoff_repo,
        _load_handoff_module,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _git,
        _init_handoff_repo,
        _load_handoff_module,
    )

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_build_handoff_package_single_scripted_entrypoint(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _base, final_head = _init_handoff_repo(repo)
    builder_path = (
        repo
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
        / "build_handoff_package.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(builder_path),
            "--repo-root",
            ".",
            "--out-dir",
            ".run/handoffs",
            "--task-name",
            "handoff-test",
            "--base-ref",
            "origin/main",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
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
    handoff_contract = _load_handoff_module("handoff_contract")

    assert _git(repo, "rev-parse", "HEAD").strip() == final_head
    assert names == set(handoff_contract.REQUIRED_HANDOFF_ENTRIES)
    assert set(sidecar_payload) == {"schema_id", "zip_filename", "zip_sha256"}
    assert set(task_metadata) == {
        "schema_id",
        "task_name",
        "branch",
        "base",
        "final_head",
    }
    assert "mode" not in task_metadata
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
        "profile-smoke",
        "code-map",
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


def test_builder_cli_has_no_external_task_or_receipts_args() -> None:
    builder = _load_handoff_module("build_handoff_package")
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert '"--task"' not in source
    assert '"--receipts-dir"' not in source
    assert '"--mode"' not in source
    assert '"--verify"' not in source


def test_builder_fails_when_replayed_validation_row_is_not_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_handoff_repo(repo)
    builder = _load_handoff_module("build_handoff_package")
    good_receipts = builder.build_machine_receipts(
        repo, _git(repo, "rev-parse", "HEAD").strip()
    )
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
        builder, "verify_handoff_package_result", lambda *_: Verification()
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
