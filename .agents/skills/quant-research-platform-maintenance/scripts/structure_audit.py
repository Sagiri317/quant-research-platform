"""Internal structural audit gate for QRP maintenance work."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import governance_contract  # noqa: E402

HANDOFF_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)
STRUCTURE_AUDIT_PATH = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py"
)
REQUIRED_SOURCE_TREE_TEXT_ROOTS = {
    "src",
    "tests",
    ".agents",
    "docs",
    "AGENTS.md",
    "README.md",
    "data/README.md",
    "pyproject.toml",
    "environment.yml",
}
OLD_HELPERS = {
    "_" + "_".join((("in" + "stall"), "package", "smoke", "target")),
    "_".join(("BUNDLE", "CONTENT", "SCAN", "EXEMPT", "ROOTS")),
    "_".join(("BUNDLE", "CONTENT", "SCAN", "EXEMPT", "FILES")),
    "_".join(("is", "bundle", "content", "scan", "exempt")),
    "_" + "_".join(("load", "handoff", "builder", "integration", "tests")),
}
PIP_INSTALL_PATTERN = re.compile(
    r"(?<![\w-])(?:python\s+-m\s+pip|pip)\s+" + ("in" + r"stall\b")
)
COMMIT_HASH_PATTERN = re.compile(r"\b[0-9a-f]{40}\b")
LOCAL_PATH_PATTERN = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?!path[\\/])")


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    findings: tuple[str, ...] = ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    checks = run_audit(repo_root)
    status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
    payload = {
        "schema_id": "qrp-structure-audit",
        "status": status,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "findings": list(check.findings),
            }
            for check in checks
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for check in checks:
            print(f"{check.status} {check.name}")
            for finding in check.findings:
                print(f"  {finding}")
    return 0 if status == "PASS" else 1


def run_audit(repo_root: Path) -> tuple[AuditCheck, ...]:
    return (
        _profile_scopes_exist(repo_root),
        _profile_smoke_uses_official_runner(repo_root),
        _profile_smoke_is_lightweight(repo_root),
        _no_executable_pip_package_command(repo_root),
        _package_smoke_schema_closed(repo_root),
        _package_build_no_isolation(repo_root),
        _package_build_timeout(repo_root),
        _package_import_timeout(repo_root),
        _conda_build_backend_contract(repo_root),
        _package_content_scan_truthfulness(repo_root),
        _verifier_has_no_root_wide_exemptions(repo_root),
        _mandatory_validation_all_pass_gate(repo_root),
        _synthetic_fixture_profile_runner(repo_root),
        _source_tree_scan_roots_configured(),
        _source_tree_scan_behavior_tests(repo_root),
        _test_owner_placement(repo_root),
        _validation_fail_packages_rejected(repo_root),
        _handoff_replay_environment_preflight_closure(repo_root),
        _price_volume_reference_owner_closure(repo_root),
        _helper_owner_inversion_closure(repo_root),
        _test_helper_boundary_closure(repo_root),
        _replay_environment_requirements_single_source(repo_root),
        _environment_dependency_sync(repo_root),
        _no_duplicate_source_tree_scanner_regex(repo_root),
        _negative_field_specs(repo_root),
        _active_open_issues_current_only(repo_root),
        _old_helper_references_absent(repo_root),
        _no_duplicate_validation_or_command_ids(repo_root),
        _no_repo_state_hardcoding(repo_root),
    )


def _profile_scopes_exist(repo_root: Path) -> AuditCheck:
    runner = _load_module(repo_root / "tests" / "tools" / "run_profile.py")
    findings = tuple(
        f"{name}:{scope}"
        for name, profile in sorted(runner.PROFILES.items())
        for scope in profile.scopes
        if not (repo_root / scope).exists()
    )
    return _check("profile_scopes_exist", findings)


def _profile_smoke_uses_official_runner(repo_root: Path) -> AuditCheck:
    handoff_contract = _load_module(repo_root / HANDOFF_ROOT / "handoff_contract.py")
    spec = handoff_contract.replay_command_spec("profile-smoke")
    expected = (
        "python",
        "tests/tools/run_profile.py",
        "--profile",
        "profile-smoke",
        "--run-dir",
        ".run/handoff-replay/profile-smoke",
    )
    findings = ()
    if tuple(spec.command_argv) != expected:
        findings = (f"profile-smoke:{spec.command}",)
    return _check("profile_smoke_official_runner", findings)


def _profile_smoke_is_lightweight(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    for path in (repo_root / "tests" / "agents" / "handoff").glob("test_*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        if "valid_handoff_package" not in source:
            continue
        if _module_has_excluded_marker(source):
            continue
        findings.extend(
            f"{path.as_posix()}:{name}"
            for name in _unmarked_valid_handoff_package_tests(source)
        )
    return _check("profile_smoke_lightweight", tuple(findings))


def _no_executable_pip_package_command(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    for path in _tracked_text_files(repo_root):
        text = (repo_root / path).read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not PIP_INSTALL_PATTERN.search(line):
                continue
            lowered = line.lower()
            if (
                "forbidden" in lowered
                or "禁止" in lowered
                or "do not" in lowered
                or "无可执行" in line
                or "no executable" in lowered
            ):
                continue
            findings.append(f"{path.as_posix()}:{line_number}:{line.strip()}")
    return _check("no_executable_pip_package_command", tuple(findings))


def _package_smoke_schema_closed(repo_root: Path) -> AuditCheck:
    combined = _handoff_source(repo_root)
    retired_tokens = (
        "_".join(("no", "deps")),
        "_".join(("target", "kind")),
        '"' + ("in" + "stall") + '"',
        "_".join(("temporary", "no", "deps", ("in" + "stall"))),
    )
    findings = tuple(token for token in retired_tokens if token in combined)
    return _check("package_smoke_wheel_extract_only", findings)


def _package_build_no_isolation(repo_root: Path) -> AuditCheck:
    replay = _handoff_file_text(repo_root, "replay_evidence.py")
    findings: list[str] = []
    if "--no-isolation" not in replay:
        findings.append("replay_evidence:package_build_missing_no_isolation")
    if "--installer" in replay:
        findings.append("replay_evidence:package_build_forbidden_installer")
    if "--skip-dependency-check" in replay:
        findings.append(
            "replay_evidence:package_build_forbidden_skip_dependency_check"
        )
    return _check("package_build_no_isolation", tuple(findings))


def _package_build_timeout(repo_root: Path) -> AuditCheck:
    replay = _handoff_file_text(repo_root, "replay_evidence.py")
    contract = _handoff_file_text(repo_root, "handoff_contract.py")
    findings: list[str] = []
    if "PACKAGE_BUILD_TIMEOUT_SECONDS" not in contract:
        findings.append("handoff_contract:package_build_timeout_constant_missing")
    if "timeout=PACKAGE_BUILD_TIMEOUT_SECONDS" not in replay:
        findings.append("replay_evidence:package_build_timeout_constant_not_used")
    return _check("package_build_timeout", tuple(findings))


def _package_import_timeout(repo_root: Path) -> AuditCheck:
    replay = _handoff_file_text(repo_root, "replay_evidence.py")
    contract = _handoff_file_text(repo_root, "handoff_contract.py")
    findings: list[str] = []
    if "PACKAGE_IMPORT_TIMEOUT_SECONDS" not in contract:
        findings.append("handoff_contract:package_import_timeout_constant_missing")
    if "timeout=PACKAGE_IMPORT_TIMEOUT_SECONDS" not in replay:
        findings.append("replay_evidence:package_import_timeout_constant_not_used")
    return _check("package_import_timeout", tuple(findings))


def _conda_build_backend_contract(repo_root: Path) -> AuditCheck:
    pyproject_path = repo_root / "pyproject.toml"
    environment_path = repo_root / "environment.yml"
    findings: list[str] = []
    if not pyproject_path.exists():
        return _check("conda_build_backend_contract", ("pyproject.toml:missing",))
    if not environment_path.exists():
        return _check("conda_build_backend_contract", ("environment.yml:missing",))
    with pyproject_path.open("rb") as handle:
        pyproject = tomllib.load(handle)
    build_requires = _dependency_specs(
        pyproject.get("build-system", {}).get("requires", [])
    )
    dev = _dependency_specs(
        pyproject.get("project", {})
        .get("optional-dependencies", {})
        .get("dev", [])
    )
    environment_specs = _environment_dependency_specs(
        environment_path.read_text(encoding="utf-8")
    )
    if "build" not in dev:
        findings.append("pyproject.toml:missing_dev_dependency:build")
    if "python-build" not in environment_specs:
        findings.append("environment.yml:missing_build_frontend:python-build")
    for package_name in ("setuptools", "wheel"):
        if package_name not in build_requires:
            findings.append(
                f"pyproject.toml:missing_build_backend_dependency:{package_name}"
            )
            continue
        if package_name not in environment_specs:
            findings.append(
                f"environment.yml:missing_build_backend_dependency:{package_name}"
            )
            continue
        if not _conda_spec_is_same_or_stricter(
            environment_specs[package_name], build_requires[package_name]
        ):
            findings.append(
                f"environment.yml:loose_build_backend_dependency:{package_name}"
            )
    return _check("conda_build_backend_contract", tuple(findings))


def _package_content_scan_truthfulness(repo_root: Path) -> AuditCheck:
    replay = _handoff_file_text(repo_root, "replay_evidence.py")
    findings: list[str] = []
    if "content_status = \"NOT_RUN\"" not in replay:
        findings.append("replay_evidence:content_scan_not_run_guard_missing")
    if "if build_status == \"PASS\"" not in replay:
        findings.append("replay_evidence:content_scan_build_pass_guard_missing")
    if "\"scanned_artifact_kinds\": scanned_artifact_kinds" not in replay:
        findings.append("replay_evidence:content_scan_actual_scanned_kinds_missing")
    if "list(PACKAGE_SCANNED_ARTIFACT_KINDS)" in replay:
        findings.append("replay_evidence:content_scan_fixed_scanned_kinds")
    return _check("package_content_scan_truthfulness", tuple(findings))


def _verifier_has_no_root_wide_exemptions(repo_root: Path) -> AuditCheck:
    verifier = (repo_root / HANDOFF_ROOT / "verify_handoff_package.py").read_text(
        encoding="utf-8"
    )
    findings = tuple(token for token in OLD_HELPERS if token in verifier)
    return _check("verifier_no_root_wide_exemptions", findings)


def _mandatory_validation_all_pass_gate(repo_root: Path) -> AuditCheck:
    contract = (repo_root / HANDOFF_ROOT / "handoff_contract.py").read_text(
        encoding="utf-8"
    )
    builder = (repo_root / HANDOFF_ROOT / "build_handoff_package.py").read_text(
        encoding="utf-8"
    )
    verifier = (repo_root / HANDOFF_ROOT / "verify_handoff_package.py").read_text(
        encoding="utf-8"
    )
    findings = []
    if "def mandatory_validation_failures(" not in contract:
        findings.append("handoff_contract:mandatory_validation_failures_missing")
    if "mandatory_validation_failures(validation_summary)" not in builder:
        findings.append("builder:mandatory_validation_failures_not_called")
    if (
        "mandatory_validation_failures(package_payload.validation_summary)"
        not in verifier
    ):
        findings.append("verifier:mandatory_validation_failures_not_called")
    return _check("mandatory_validation_all_pass_gate", tuple(findings))


def _synthetic_fixture_profile_runner(repo_root: Path) -> AuditCheck:
    conftest = (
        repo_root / "tests" / "agents" / "handoff" / "conftest.py"
    ).read_text(encoding="utf-8")
    findings = []
    if '"tests" / "tools" / "run_profile.py"' not in conftest:
        findings.append("synthetic_repo_missing_tests_tools_run_profile")
    if "'profile-smoke'" not in conftest:
        findings.append("synthetic_runner_missing_profile_smoke")
    if "validation_summary[\"status\"] == \"PASS\"" not in conftest:
        findings.append("synthetic_fixture_missing_validation_pass_assertion")
    return _check("synthetic_fixture_profile_runner", tuple(findings))


def _source_tree_scan_roots_configured() -> AuditCheck:
    configured = {
        Path(root).as_posix() for root in governance_contract.SOURCE_TREE_TEXT_ROOTS
    }
    return _check(
        "source_tree_scan_roots",
        tuple(sorted(REQUIRED_SOURCE_TREE_TEXT_ROOTS - configured)),
    )


def _source_tree_scan_behavior_tests(repo_root: Path) -> AuditCheck:
    test_sources = (
        (
            repo_root / "tests" / "agents" / "test_verify_governance_active_docs.py"
        ).read_text(encoding="utf-8"),
        (
            repo_root
            / "tests"
            / "architecture"
            / "test_governance_source_tree_scan_contract.py"
        ).read_text(encoding="utf-8"),
    )
    test_source = "\n".join(test_sources)
    required = (
        "src/quant_research_platform/private_payload.py",
        "docs/architecture/private-payload.md",
        "README.md",
        "build_source_tree_scan_input(repo)",
        "source_tree_sensitive_negative_literal_findings(scan_input)",
    )
    findings = tuple(token for token in required if token not in test_source)
    return _check("source_tree_scan_behavior_tests", findings)


def _test_owner_placement(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    for path in (repo_root / "tests" / "architecture").glob("test_*.py"):
        source = path.read_text(encoding="utf-8", errors="ignore")
        if _imports_handoff_integration_helper(source):
            findings.append(f"{path.as_posix()}:dynamic_handoff_integration_import")
    for path in (repo_root / "tests" / "catalog").glob("test_*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        forbidden = ("ast.parse", "import_graph", 'rglob("*.py")', "rglob('*.py')")
        if any(token in text for token in forbidden):
            findings.append(path.as_posix())
    return _check("test_owner_placement", tuple(findings))


def _validation_fail_packages_rejected(repo_root: Path) -> AuditCheck:
    verifier_tests = (
        repo_root / "tests" / "agents" / "handoff" / "test_handoff_package_verifier.py"
    ).read_text(encoding="utf-8")
    builder_tests = (
        repo_root
        / "tests"
        / "agents"
        / "handoff"
        / "test_build_handoff_package_integration.py"
    ).read_text(encoding="utf-8")
    findings = []
    if (
        "test_verifier_rejects_self_consistent_failed_validation_summary"
        not in verifier_tests
    ):
        findings.append("verifier_self_consistent_fail_test_missing")
    if (
        "test_builder_fails_when_replayed_validation_row_is_not_pass"
        not in builder_tests
    ):
        findings.append("builder_validation_fail_test_missing")
    if "validation_summary_row_not_pass:profile_smoke:FAIL" not in (
        verifier_tests + builder_tests
    ):
        findings.append("profile_smoke_not_pass_assertion_missing")
    return _check("validation_fail_packages_rejected", tuple(findings))


def _handoff_replay_environment_preflight_closure(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    replay_path = repo_root / HANDOFF_ROOT / "replay_evidence.py"
    builder_path = repo_root / HANDOFF_ROOT / "build_handoff_package.py"
    verifier_path = repo_root / HANDOFF_ROOT / "verify_handoff_package.py"
    handoff_contract_path = repo_root / HANDOFF_ROOT / "handoff_contract.py"
    receipt_contract_path = repo_root / HANDOFF_ROOT / "receipt_contract.py"
    replay_tree = _parse_python_file(replay_path)
    builder_tree = _parse_python_file(builder_path)
    verifier_tree = _parse_python_file(verifier_path)
    receipt_tree = _parse_python_file(receipt_contract_path)

    replay_defs = _top_level_definition_names(replay_tree)
    for name in (
        "ReplayEnvironmentBlocked",
        "collect_replay_environment_failures",
        "assert_replay_environment_ready",
    ):
        if name not in replay_defs:
            findings.append(f"replay_evidence:{name}_missing")
    if not _function_calls_name(
        replay_tree,
        "build_machine_receipts",
        "assert_replay_environment_ready",
    ):
        findings.append("replay_evidence:build_machine_receipts_missing_preflight")
    if "ReplayEnvironmentBlocked" not in _imported_or_defined_names(builder_tree):
        findings.append("builder:ReplayEnvironmentBlocked_not_used")
    if "ReplayEnvironmentBlocked" not in _imported_or_defined_names(verifier_tree):
        findings.append("verifier:ReplayEnvironmentBlocked_not_used")
    builder_text = builder_path.read_text(encoding="utf-8", errors="ignore")
    verifier_text = verifier_path.read_text(encoding="utf-8", errors="ignore")
    if "replay_environment_blocked:" not in builder_text:
        findings.append("builder:replay_environment_blocked_failure_missing")
    if "replay_environment_blocked:" not in verifier_text:
        findings.append("verifier:replay_environment_blocked_failure_missing")
    if _required_handoff_entries_contain(
        repo_root,
        ("dependency", "preflight", "environment"),
    ):
        findings.append("handoff_contract:preflight_entry_added")
    forbidden_schema_terms = (
        "preflight",
        "environment_dump",
        "pip_freeze",
        "raw_stdout",
        "raw_stderr",
        "raw_pytest_log",
    )
    for field in _receipt_schema_field_names(receipt_tree):
        if any(term in field for term in forbidden_schema_terms):
            findings.append(f"receipt_contract:forbidden_schema_field:{field}")
    if "dependency_preflight_receipt" in _required_handoff_entry_text(
        handoff_contract_path
    ):
        findings.append("handoff_contract:dependency_preflight_receipt_entry_added")
    return _check("handoff_replay_environment_preflight_closure", tuple(findings))


def _price_volume_reference_owner_closure(repo_root: Path) -> AuditCheck:
    helper_path = repo_root / "tests" / "helpers" / "price_volume_reference.py"
    tree = _parse_python_file(helper_path)
    denied_exact = {
        "assert_forced_tiny_processed_chunks_match_default",
        "write_minimal_interim_inputs",
        "write_correctness_processed_inputs",
        "research_rows_for_dates",
        "correctness_processed_catalog",
        "correctness_research_dataset_catalog",
    }
    names = _top_level_definition_names(tree) | _top_level_assignment_names(tree)
    findings = sorted(
        f"price_volume_reference:{name}"
        for name in names
        if name in denied_exact
        or name.startswith("correctness_")
        or (name.startswith("minimal_") and name.endswith("_frame"))
    )
    return _check("price_volume_reference_owner_closure", tuple(findings))


def _helper_owner_inversion_closure(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    keyed_diff = repo_root / "tests" / "helpers" / "keyed_frame_diff.py"
    if "tests.helpers.price_volume_reference" in _imported_modules(keyed_diff):
        findings.append(
            "tests/helpers/keyed_frame_diff.py:tests.helpers.price_volume_reference"
        )
    return _check("helper_owner_inversion_closure", tuple(findings))


def _test_helper_boundary_closure(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    retired_helper = repo_root / "tests" / "correctness" / "_helpers.py"
    if retired_helper.exists():
        findings.append("tests/correctness/_helpers.py:present")
    for path in (repo_root / "tests").rglob("*.py"):
        owner = _test_owner(path, repo_root)
        for imported in _imported_modules(path):
            if imported == "tests.correctness._helpers":
                findings.append(
                    f"{path.relative_to(repo_root).as_posix()}:"
                    "tests.correctness._helpers"
                )
            parts = imported.split(".")
            if len(parts) >= 3 and parts[0] == "tests" and parts[2] == "_helpers":
                imported_owner = parts[1]
                if imported_owner != owner:
                    findings.append(
                        f"{path.relative_to(repo_root).as_posix()}:{imported}"
                    )
    retired_terms = repo_root / "tests" / "architecture" / "test_retired_terms.py"
    if retired_terms.exists() and "tests/correctness/_helpers.py" in (
        retired_terms.read_text(encoding="utf-8", errors="ignore")
    ):
        findings.append("test_retired_terms:retired_correctness_helper_root")
    unclear_names = {"common.py", "utils.py", "new_helpers.py", "correctness_data.py"}
    helpers_root = repo_root / "tests" / "helpers"
    if helpers_root.exists():
        findings.extend(
            f"tests/helpers/{path.name}:unclear_owner"
            for path in sorted(helpers_root.glob("*.py"))
            if path.name in unclear_names
        )
    return _check("test_helper_boundary_closure", tuple(findings))


def _replay_environment_requirements_single_source(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    replay_path = repo_root / HANDOFF_ROOT / "replay_evidence.py"
    replay_tree = _parse_python_file(replay_path)
    replay_text = replay_path.read_text(encoding="utf-8", errors="ignore")
    assignments = _module_assignment_nodes(replay_tree)
    requirements_node = assignments.get("REPLAY_ENVIRONMENT_REQUIREMENTS")
    if requirements_node is not None:
        hardcoded = _hardcoded_requirement_findings(requirements_node)
        findings.extend(
            hardcoded
            or ("replay_evidence:REPLAY_ENVIRONMENT_REQUIREMENTS_present",)
        )
    required_distributions = _literal_string_tuple(
        assignments.get("REPLAY_REQUIRED_DISTRIBUTIONS")
    )
    if not required_distributions:
        findings.append("replay_evidence:REPLAY_REQUIRED_DISTRIBUTIONS_missing")
    if "python-build" in required_distributions:
        findings.append("replay_evidence:forbidden_distribution:python-build")
    for token in (
        "tomllib",
        "_replay_requirement_specs",
        "requires-python",
        "importlib_metadata.version",
    ):
        if token not in replay_text:
            findings.append(f"replay_evidence:pyproject_source_missing:{token}")
    pyproject = _load_pyproject(repo_root)
    required = _replay_required_specs_from_pyproject(pyproject)
    for distribution in required_distributions:
        if distribution not in required:
            findings.append(f"replay_evidence:contract_source_missing:{distribution}")
    return _check("replay_environment_requirements_single_source", tuple(findings))


def _environment_dependency_sync(repo_root: Path) -> AuditCheck:
    pyproject_path = repo_root / "pyproject.toml"
    environment_path = repo_root / "environment.yml"
    if not pyproject_path.exists():
        return _check("environment_dependency_sync", ("pyproject.toml:missing",))
    if not environment_path.exists():
        return _check("environment_dependency_sync", ("environment.yml:missing",))
    pyproject = _load_pyproject(repo_root)
    environment_specs = _environment_dependency_specs(
        environment_path.read_text(encoding="utf-8")
    )
    project = pyproject.get("project", {})
    build_system = pyproject.get("build-system", {})
    tool = pyproject.get("tool", {})
    runtime = _dependency_specs(
        project.get("dependencies", []) if isinstance(project, Mapping) else []
    )
    dev = _dependency_specs(
        project.get("optional-dependencies", {}).get("dev", [])
        if isinstance(project, Mapping)
        and isinstance(project.get("optional-dependencies"), Mapping)
        else []
    )
    build_requires = _dependency_specs(
        build_system.get("requires", []) if isinstance(build_system, Mapping) else []
    )
    findings: list[str] = []
    expected_specs = {
        "build": ("python-build", dev.get("build", "")),
        "setuptools": ("setuptools", build_requires.get("setuptools", "")),
        "wheel": ("wheel", build_requires.get("wheel", "")),
        "pyarrow": ("pyarrow", runtime.get("pyarrow", "")),
        "pytest": ("pytest", dev.get("pytest", "")),
        "pytest-xdist": ("pytest-xdist", dev.get("pytest-xdist", "")),
        "ruff": ("ruff", dev.get("ruff", "")),
    }
    for pyproject_name, (conda_name, expected_spec) in expected_specs.items():
        conda_spec = environment_specs.get(conda_name, "")
        if not expected_spec or not conda_spec:
            findings.append(
                f"environment.yml:dependency_missing:{pyproject_name}->{conda_name}"
            )
            continue
        if not _conda_spec_is_same_or_stricter(conda_spec, expected_spec):
            findings.append(
                f"environment.yml:dependency_mismatch:{pyproject_name}->{conda_name}"
            )
    requires_python = (
        str(project.get("requires-python", ""))
        if isinstance(project, Mapping)
        else ""
    )
    expected_python = _python_minor_from_requires(requires_python)
    actual_python = _environment_python_pin(environment_specs)
    if expected_python and actual_python != expected_python:
        findings.append(f"environment.yml:python_pin_mismatch:expected={expected_python}")
    ruff_config = tool.get("ruff", {}) if isinstance(tool, Mapping) else {}
    actual_ruff_target = (
        str(ruff_config.get("target-version", ""))
        if isinstance(ruff_config, Mapping)
        else ""
    )
    expected_ruff_target = _ruff_target_from_python_minor(expected_python)
    if expected_ruff_target and actual_ruff_target != expected_ruff_target:
        findings.append(
            f"pyproject.toml:ruff_target_mismatch:expected={expected_ruff_target}"
        )
    return _check("environment_dependency_sync", tuple(findings))


def _no_duplicate_source_tree_scanner_regex(repo_root: Path) -> AuditCheck:
    verifier = (repo_root / HANDOFF_ROOT / "verify_handoff_package.py").read_text(
        encoding="utf-8"
    )
    findings = []
    if "def text_file_failures(" in verifier:
        findings.append("verifier:text_file_failures")
    if "SourceTreeSensitiveNegativeRule" in verifier:
        findings.append("verifier:duplicate_sensitive_rule")
    return _check("no_duplicate_source_tree_scanner_regex", tuple(findings))


def _negative_field_specs(repo_root: Path) -> AuditCheck:
    handoff_contract = _load_module(repo_root / HANDOFF_ROOT / "handoff_contract.py")
    required = {"field_name", "owner", "risk", "lifecycle", "acceptance_gate"}
    findings = []
    for index, spec in enumerate(
        handoff_contract.ACTIVE_DENIED_HANDOFF_EVIDENCE_FIELDS
    ):
        if not isinstance(spec, Mapping) or not required <= set(spec):
            findings.append(f"denied_field:{index}")
            continue
        missing_values = [key for key in required if not str(spec.get(key, "")).strip()]
        findings.extend(f"denied_field:{index}:{key}" for key in missing_values)
    return _check("negative_field_specs", tuple(findings))


def _active_open_issues_current_only(repo_root: Path) -> AuditCheck:
    text = (repo_root / "docs" / "architecture" / "open-issues.md").read_text(
        encoding="utf-8"
    )
    findings: list[str] = []
    if re.search(r"^Status:\s*closed\s*$", text, flags=re.MULTILINE):
        findings.append("closed_issue_recorded")
    if "handoff-validation-owner-closure" in text:
        findings.append("completed_issue_family_recorded")
    return _check("active_open_issues_current_only", tuple(findings))


def _old_helper_references_absent(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    for path in _tracked_text_files(repo_root):
        text = (repo_root / path).read_text(encoding="utf-8", errors="ignore")
        for token in OLD_HELPERS:
            if token in text and path != STRUCTURE_AUDIT_PATH:
                findings.append(f"{path.as_posix()}:{token}")
    return _check("old_helper_references_absent", tuple(findings))


def _no_duplicate_validation_or_command_ids(repo_root: Path) -> AuditCheck:
    handoff_contract = _load_handoff_script_module(repo_root, "handoff_contract")
    receipt_contract = _load_handoff_script_module(repo_root, "receipt_contract")
    row_counts = Counter(handoff_contract.MANDATORY_VALIDATION_ROWS)
    command_counts = Counter(receipt_contract.REQUIRED_CHECK_IDS)
    findings = [
        f"validation_row:{name}" for name, count in row_counts.items() if count > 1
    ]
    findings.extend(
        f"command_id:{name}" for name, count in command_counts.items() if count > 1
    )
    if set(handoff_contract.VALIDATION_ROW_SOURCE_CONTRACT) != set(
        handoff_contract.MANDATORY_VALIDATION_ROWS
    ):
        findings.append("validation_row_source_contract_mismatch")
    return _check("no_duplicate_validation_or_command_ids", tuple(findings))


def _no_repo_state_hardcoding(repo_root: Path) -> AuditCheck:
    findings: list[str] = []
    for path in _tracked_text_files(repo_root):
        if path == Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "governance_contract.py"
        ):
            continue
        if path.parts[:1] == ("tests",):
            continue
        text = (repo_root / path).read_text(encoding="utf-8", errors="ignore")
        if LOCAL_PATH_PATTERN.search(text):
            findings.append(f"{path.as_posix()}:local_path")
        if COMMIT_HASH_PATTERN.search(text):
            findings.append(f"{path.as_posix()}:commit_hash_like")
    return _check("no_repo_state_hardcoding", tuple(findings))


def _tracked_text_files(repo_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in governance_contract.source_tree_tracked_text_files(repo_root)
        if (repo_root / path).is_file()
    )


def _handoff_source(repo_root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (repo_root / HANDOFF_ROOT).glob("*.py")
    )


def _handoff_file_text(repo_root: Path, name: str) -> str:
    return (repo_root / HANDOFF_ROOT / name).read_text(
        encoding="utf-8", errors="ignore"
    )


def _environment_dependency_specs(text: str) -> dict[str, str]:
    specs: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        spec = stripped[2:].strip()
        if not spec or spec in {"conda-forge", "nodefaults"}:
            continue
        name = re.split(r"[<>=!~; ]", spec, maxsplit=1)[0]
        specs[name] = spec
    return specs


def _load_pyproject(repo_root: Path) -> dict[str, object]:
    try:
        with (repo_root / "pyproject.toml").open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _dependency_specs(items: Iterable[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items:
        name = re.split(r"[<>=!~; ]", str(item), maxsplit=1)[0]
        specs[name] = str(item)
    return specs


def _replay_required_specs_from_pyproject(
    pyproject: Mapping[str, object],
) -> dict[str, str]:
    project = pyproject.get("project", {})
    build_system = pyproject.get("build-system", {})
    runtime = _dependency_specs(
        project.get("dependencies", []) if isinstance(project, Mapping) else []
    )
    dev = _dependency_specs(
        project.get("optional-dependencies", {}).get("dev", [])
        if isinstance(project, Mapping)
        and isinstance(project.get("optional-dependencies"), Mapping)
        else []
    )
    build_requires = _dependency_specs(
        build_system.get("requires", []) if isinstance(build_system, Mapping) else []
    )
    specs: dict[str, str] = {}
    for name, source in (
        ("build", dev),
        ("pytest", dev),
        ("setuptools", build_requires),
        ("wheel", build_requires),
        ("pyarrow", runtime),
    ):
        if name in source:
            specs[name] = source[name]
    return specs


def _conda_spec_is_same_or_stricter(conda_spec: str, pyproject_spec: str) -> bool:
    pyproject_bounds = _bounds(pyproject_spec)
    conda_bounds = _bounds(conda_spec)
    return bool(pyproject_bounds) and pyproject_bounds <= conda_bounds


def _bounds(spec: str) -> set[tuple[str, str]]:
    return {
        (operator, version)
        for operator, version in re.findall(r"(>=|<=|<|>|==)\s*([0-9][^,;\s]*)", spec)
    }


def _environment_python_pin(environment_specs: Mapping[str, str]) -> str:
    match = re.fullmatch(r"python=(\d+\.\d+)", environment_specs.get("python", ""))
    return match.group(1) if match else ""


def _python_minor_from_requires(spec: str) -> str:
    match = re.search(r">=\s*(\d+\.\d+)", spec)
    return match.group(1) if match else ""


def _ruff_target_from_python_minor(python_minor: str) -> str:
    return "py" + python_minor.replace(".", "") if python_minor else ""


def _load_module(path: Path):
    module_name = f"structure_audit_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _imports_handoff_integration_helper(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute) and func.attr == "spec_from_file_location"
        ):
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and "test_build_handoff_package_integration" in arg.value
            ):
                return True
    return False


def _module_has_excluded_marker(source: str) -> bool:
    return any(
        token in source
        for token in (
            "pytestmark = pytest.mark.integration",
            "pytestmark = pytest.mark.mutation",
            "pytestmark = [pytest.mark.integration",
        )
    )


def _unmarked_valid_handoff_package_tests(source: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ("syntax_error",)
    findings: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        arg_names = {arg.arg for arg in node.args.args}
        if "valid_handoff_package" not in arg_names:
            continue
        if _function_has_excluded_marker(node):
            continue
        findings.append(node.name)
    return tuple(findings)


def _function_has_excluded_marker(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        text = ast.unparse(decorator)
        if text in {
            "pytest.mark.integration",
            "pytest.mark.mutation",
            "pytest.mark.slow",
        }:
            return True
    return False


def _load_handoff_script_module(repo_root: Path, name: str):
    path = repo_root / HANDOFF_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_python_file(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return ast.Module(body=[], type_ignores=[])


def _top_level_definition_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _top_level_assignment_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names.update(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    return names


def _module_assignment_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


def _literal_string_tuple(node: ast.AST | None) -> tuple[str, ...]:
    if not isinstance(node, ast.Tuple | ast.List):
        return ()
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
        else:
            return ()
    return tuple(values)


def _hardcoded_requirement_findings(node: ast.AST) -> tuple[str, ...]:
    findings: list[str] = []
    if not isinstance(node, ast.Tuple | ast.List):
        return ("replay_evidence:REPLAY_ENVIRONMENT_REQUIREMENTS_present",)
    for item in node.elts:
        if not isinstance(item, ast.Tuple | ast.List) or len(item.elts) != 2:
            continue
        name_node, spec_node = item.elts
        if (
            isinstance(name_node, ast.Constant)
            and isinstance(name_node.value, str)
            and isinstance(spec_node, ast.Constant)
            and isinstance(spec_node.value, str)
            and _bounds(spec_node.value)
        ):
            findings.append(
                "replay_evidence:hardcoded_requirement:"
                f"{name_node.value}:{spec_node.value}"
            )
    return tuple(findings)


def _imported_or_defined_names(tree: ast.Module) -> set[str]:
    names = set(_top_level_definition_names(tree))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _function_calls_name(
    tree: ast.Module,
    function_name: str,
    called_name: str,
) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name) and func.id == called_name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == called_name:
                return True
    return False


def _required_handoff_entries_contain(
    repo_root: Path,
    terms: tuple[str, ...],
) -> bool:
    handoff_contract = _load_handoff_script_module(repo_root, "handoff_contract")
    entries = getattr(handoff_contract, "REQUIRED_HANDOFF_ENTRIES", ())
    return any(any(term in str(entry) for term in terms) for entry in entries)


def _required_handoff_entry_text(path: Path) -> str:
    try:
        tree = _parse_python_file(path)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            targets = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            if "REQUIRED_HANDOFF_ENTRIES" not in targets:
                continue
            return ast.unparse(node)
    except Exception:
        return ""
    return ""


def _receipt_schema_field_names(tree: ast.Module) -> set[str]:
    fields: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not any(name.endswith("_FIELD_SPECS") for name in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                fields.add(key.value)
    return fields


def _imported_modules(path: Path) -> set[str]:
    tree = _parse_python_file(path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and path.parent.name:
                module = f"tests.{path.parent.name}.{module}".rstrip(".")
            imports.add(module)
    return imports


def _test_owner(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root / "tests").parts[0]
    except ValueError:
        return ""


def _check(name: str, findings: Iterable[str]) -> AuditCheck:
    items = tuple(findings)
    return AuditCheck(name=name, status="PASS" if not items else "FAIL", findings=items)


if __name__ == "__main__":
    raise SystemExit(main())
