from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import _git, _init_handoff_repo, _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _git, _init_handoff_repo, _load_handoff_module  # type: ignore[no-redef]

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
    assert set(validation_summary) == {"schema_id", "rows"}
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
        "install",
        "import",
        "content_scan",
    }
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
