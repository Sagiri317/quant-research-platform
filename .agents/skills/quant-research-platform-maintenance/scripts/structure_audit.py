"""Internal structural audit gate for QRP maintenance work."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
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
        _no_executable_pip_package_command(repo_root),
        _package_smoke_schema_closed(repo_root),
        _verifier_has_no_root_wide_exemptions(repo_root),
        _source_tree_scan_roots_configured(),
        _test_owner_placement(repo_root),
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


def _verifier_has_no_root_wide_exemptions(repo_root: Path) -> AuditCheck:
    verifier = (repo_root / HANDOFF_ROOT / "verify_handoff_package.py").read_text(
        encoding="utf-8"
    )
    findings = tuple(token for token in OLD_HELPERS if token in verifier)
    return _check("verifier_no_root_wide_exemptions", findings)


def _source_tree_scan_roots_configured() -> AuditCheck:
    configured = {
        Path(root).as_posix() for root in governance_contract.SOURCE_TREE_TEXT_ROOTS
    }
    return _check(
        "source_tree_scan_roots",
        tuple(sorted(REQUIRED_SOURCE_TREE_TEXT_ROOTS - configured)),
    )


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
    handoff_contract = _load_handoff_module(repo_root, "handoff_contract")
    receipt_contract = _load_handoff_module(repo_root, "receipt_contract")
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


def _load_handoff_module(repo_root: Path, name: str):
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
