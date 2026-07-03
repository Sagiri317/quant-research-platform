from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

STRUCTURE_AUDIT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py"
)


def test_structure_audit_script_exists_and_is_internal_only() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert STRUCTURE_AUDIT.is_file()
    assert "structure_audit" not in pyproject


def test_structure_audit_json_gate_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(STRUCTURE_AUDIT),
            "--repo-root",
            ".",
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_id"] == "qrp-structure-audit"
    assert payload["status"] == "PASS"
    assert payload["checks"]
    check_names = {check["name"] for check in payload["checks"]}
    assert {
        "mandatory_validation_all_pass_gate",
        "synthetic_fixture_profile_runner",
        "source_tree_scan_behavior_tests",
        "profile_smoke_lightweight",
        "validation_fail_packages_rejected",
        "no_duplicate_source_tree_scanner_regex",
        "package_build_no_isolation",
        "package_build_timeout",
        "package_import_timeout",
        "conda_build_backend_contract",
        "package_content_scan_truthfulness",
    } <= check_names


def test_structure_audit_detects_package_build_isolation_mutation(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    _write_handoff_source(
        tmp_path,
        replay_evidence=(
            "subprocess.run([sys.executable, '-m', 'build', '--sdist', "
            "'--wheel', '--outdir', str(dist)], "
            "timeout=PACKAGE_BUILD_TIMEOUT_SECONDS)\n"
        ),
    )

    check = audit._package_build_no_isolation(tmp_path)

    assert check.status == "FAIL"
    assert "replay_evidence:package_build_missing_no_isolation" in check.findings


def test_structure_audit_detects_forbidden_package_build_options(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    _write_handoff_source(
        tmp_path,
        replay_evidence=(
            "subprocess.run([sys.executable, '-m', 'build', '--no-isolation', "
            "'--installer', 'uv', '--skip-dependency-check', '--sdist', "
            "'--wheel', '--outdir', str(dist)], "
            "timeout=PACKAGE_BUILD_TIMEOUT_SECONDS)\n"
        ),
    )

    check = audit._package_build_no_isolation(tmp_path)

    assert check.status == "FAIL"
    assert "replay_evidence:package_build_forbidden_installer" in check.findings
    assert "replay_evidence:package_build_forbidden_skip_dependency_check" in (
        check.findings
    )


def test_structure_audit_detects_missing_package_build_timeout(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    _write_handoff_source(
        tmp_path,
        replay_evidence=(
            "from handoff_contract import PACKAGE_IMPORT_TIMEOUT_SECONDS\n"
            "subprocess.run([sys.executable, '-m', 'build', '--no-isolation'], "
            "timeout=120)\n"
        ),
    )

    check = audit._package_build_timeout(tmp_path)

    assert check.status == "FAIL"
    assert "replay_evidence:package_build_timeout_constant_not_used" in check.findings


def test_structure_audit_detects_missing_package_import_timeout(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    _write_handoff_source(
        tmp_path,
        replay_evidence=(
            "from handoff_contract import PACKAGE_BUILD_TIMEOUT_SECONDS\n"
            "subprocess.run([sys.executable, '-c', 'import q'], timeout=120)\n"
        ),
    )

    check = audit._package_import_timeout(tmp_path)

    assert check.status == "FAIL"
    assert "replay_evidence:package_import_timeout_constant_not_used" in check.findings


def test_structure_audit_detects_missing_conda_build_backend_dependency(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=80,<81", "wheel>=0.45,<1"]',
                'build-backend = "setuptools.build_meta"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "environment.yml").write_text(
        "dependencies:\n  - python-build>=1.5,<2\n  - setuptools>=80,<81\n",
        encoding="utf-8",
    )

    check = audit._conda_build_backend_contract(tmp_path)

    assert check.status == "FAIL"
    assert "environment.yml:missing_build_backend_dependency:wheel" in check.findings


def test_structure_audit_detects_content_scan_fixed_pass_mutation(
    tmp_path: Path,
) -> None:
    audit = _load_structure_audit()
    _write_handoff_source(
        tmp_path,
        replay_evidence=(
            "content_status = 'PASS' if not forbidden else 'FAIL'\n"
            "'scanned_artifact_kinds': list(PACKAGE_SCANNED_ARTIFACT_KINDS)\n"
        ),
    )

    check = audit._package_content_scan_truthfulness(tmp_path)

    assert check.status == "FAIL"
    assert "replay_evidence:content_scan_not_run_guard_missing" in check.findings
    assert "replay_evidence:content_scan_fixed_scanned_kinds" in check.findings


def test_negative_handoff_evidence_fields_are_semantic_specs() -> None:
    contract = _load_handoff_contract()
    specs = contract.ACTIVE_DENIED_HANDOFF_EVIDENCE_FIELDS

    assert specs
    assert all(
        {
            "field_name",
            "owner",
            "risk",
            "lifecycle",
            "acceptance_gate",
        }
        <= set(spec)
        for spec in specs
    )


def _load_handoff_contract():
    path = Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_contract.py"
    )
    spec = importlib.util.spec_from_file_location("handoff_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_structure_audit():
    spec = importlib.util.spec_from_file_location("structure_audit", STRUCTURE_AUDIT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_handoff_source(tmp_path: Path, *, replay_evidence: str = "") -> None:
    handoff_root = (
        tmp_path
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
    )
    handoff_root.mkdir(parents=True)
    (handoff_root / "replay_evidence.py").write_text(
        replay_evidence,
        encoding="utf-8",
    )
    (handoff_root / "handoff_contract.py").write_text(
        "PACKAGE_BUILD_TIMEOUT_SECONDS = 300\n"
        "PACKAGE_IMPORT_TIMEOUT_SECONDS = 120\n",
        encoding="utf-8",
    )
