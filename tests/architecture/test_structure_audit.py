from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from tests.helpers.maintenance_script_loader import load_maintenance_script_module

structure_audit = load_maintenance_script_module("structure_audit")


def test_structure_audit_reports_all_checks_passing(capsys) -> None:
    exit_code = structure_audit.main(["--repo-root", ".", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "PASS"
    assert payload["checks"]
    assert {row["status"] for row in payload["checks"]} == {"PASS"}


def test_dependency_sync_detects_missing_environment_dependency(tmp_path: Path) -> None:
    shutil.copyfile("pyproject.toml", tmp_path / "pyproject.toml")
    environment = Path("environment.yml").read_text(encoding="utf-8")
    (tmp_path / "environment.yml").write_text(
        "\n".join(
            line for line in environment.splitlines() if "ruff>=" not in line
        )
        + "\n",
        encoding="utf-8",
    )

    check = structure_audit._tooling_environment_dependency_sync(tmp_path)

    assert check.status == "FAIL"
    assert check.findings == ("environment.yml:dependency_missing:ruff->ruff",)


def test_script_boundary_detects_tests_import(tmp_path: Path) -> None:
    script = (
        tmp_path
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "bad.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("from tests.helpers import bad\n", encoding="utf-8")

    check = structure_audit._agents_scripts_self_contained_boundary(tmp_path)

    assert check.status == "FAIL"
    assert check.findings == (
        ".agents/skills/quant-research-platform-maintenance/scripts/bad.py:"
        "tests.helpers",
    )


def test_machine_id_check_detects_duplicate_validation_row(tmp_path: Path) -> None:
    previous_path = tuple(sys.path)
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("handoff_contract", "receipt_contract")
    }
    handoff_root = tmp_path / structure_audit.HANDOFF_ROOT
    handoff_root.mkdir(parents=True)
    (handoff_root / "handoff_contract.py").write_text(
        "\n".join(
            (
                "from types import SimpleNamespace",
                "MANDATORY_VALIDATION_ROWS = ('governance', 'governance')",
                "VALIDATION_ROW_SOURCE_CONTRACT = {'governance': object()}",
                "REPLAY_COMMAND_CONTRACT = {",
                "    'profile-smoke': SimpleNamespace(command_id='profile-smoke')",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (handoff_root / "receipt_contract.py").write_text(
        "REQUIRED_CHECK_IDS = ('profile-smoke',)\n",
        encoding="utf-8",
    )

    check = structure_audit._no_duplicate_validation_or_command_ids(tmp_path)

    assert check.status == "FAIL"
    assert check.findings == ("validation_row:governance",)
    assert tuple(sys.path) == previous_path
    assert {
        name: sys.modules.get(name)
        for name in ("handoff_contract", "receipt_contract")
    } == previous_modules
