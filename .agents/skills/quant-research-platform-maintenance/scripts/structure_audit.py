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


def _dependency_specs(items: Iterable[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items:
        name = re.split(r"[<>=!~; ]", str(item), maxsplit=1)[0]
        specs[name] = str(item)
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


def _check(name: str, findings: Iterable[str]) -> AuditCheck:
    items = tuple(findings)
    return AuditCheck(name=name, status="PASS" if not items else "FAIL", findings=items)


if __name__ == "__main__":
    raise SystemExit(main())
