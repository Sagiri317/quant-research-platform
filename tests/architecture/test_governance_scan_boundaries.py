from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.architecture.governance_scan_helpers import (
    RUNTIME_WORKSPACE_SOURCE_ALLOWLIST,
    RUNTIME_WORKSPACE_TEST_ALLOWLIST,
    SCAN_ROOTS,
)
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

REPO_ROOT = Path(".")
PROJECT_SRC = Path("src/quant_research_platform")
GOVERNANCE_CONTRACT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/governance_contract.py"
)
HANDOFF_VERIFIER = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
    "verify_handoff_package.py"
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
BLOCKED_PACKAGE_PATH_PARTS = {
    ".run",
}
BLOCKED_ROW_LEVEL_SUFFIXES = {
    ".arrow",
    ".csv",
    ".feather",
    ".parquet",
    ".pickle",
    ".pkl",
}

def test_formal_scan_roots_are_clean_checkout_stable_roots() -> None:
    runtime_workspace = Path(".run")

    assert Path("reports") not in SCAN_ROOTS
    assert runtime_workspace not in SCAN_ROOTS
    assert all(runtime_workspace.name not in root.parts for root in SCAN_ROOTS)

def test_tests_do_not_depend_on_runtime_workspace_path() -> None:
    runtime_workspace_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape('.run')}(?![A-Za-z0-9_])"
    )
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for path in Path("tests").rglob("*.py"):
        if path.resolve() == current_file or path in RUNTIME_WORKSPACE_TEST_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if runtime_workspace_pattern.search(line):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []

def test_source_and_tests_do_not_depend_on_runtime_workspace_path() -> None:
    runtime_workspace_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape('.run')}(?![A-Za-z0-9_])"
    )
    current_file = Path(__file__).resolve()
    hits: list[str] = []
    for root in (Path("src"), Path("tests")):
        for path in root.rglob("*.py"):
            if (
                path.resolve() == current_file
                or (root == Path("tests") and path in RUNTIME_WORKSPACE_TEST_ALLOWLIST)
                or (root == Path("src") and path in RUNTIME_WORKSPACE_SOURCE_ALLOWLIST)
            ):
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if runtime_workspace_pattern.search(line):
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []

def test_verify_governance_uses_current_local_only_roots() -> None:
    verify_governance = Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py"
    ).read_text(encoding="utf-8")

    assert "run_outputs/local/" not in verify_governance
    assert '"outputs/"' in verify_governance

def test_gitignore_is_active_doc_scanned_for_retired_result_paths(
    tmp_path: Path,
) -> None:
    import verify_governance

    (tmp_path / ".gitignore").write_text("run_outputs/local/\n", encoding="utf-8")
    result = verify_governance.Result()

    verify_governance.check_text_safety(result, tmp_path, [".gitignore"])

    assert any("retired_result_run_outputs" in error for error in result.errors)

def test_gitignore_rejects_outputs_local_layout(tmp_path: Path) -> None:
    import verify_governance

    (tmp_path / ".gitignore").write_text("outputs/local/\n", encoding="utf-8")
    result = verify_governance.Result()

    verify_governance.check_text_safety(result, tmp_path, [".gitignore"])

    assert any(
        "retired_result_outputs_local_layout" in error for error in result.errors
    )


def iter_source_files() -> tuple[Path, ...]:
    return tuple(sorted(PROJECT_SRC.rglob("*.py")))


def test_all_source_files_are_discoverable_and_parseable() -> None:
    files = iter_source_files()
    parse_failed: list[str] = []
    assert files, "source_scan_found_no_python_files"
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            parse_failed.append(f"{path}:{exc.lineno}:{exc.msg}")

    assert not parse_failed, "source_scan_parse_failed: " + "; ".join(parse_failed)


def test_top_level_packages_are_accounted_for() -> None:
    packages = {p.name for p in PROJECT_SRC.iterdir() if p.is_dir()}
    expected_minimum = {
        "agents",
        "catalog",
        "compute",
        "contracts",
        "interfaces",
        "markets",
        "reports",
        "research",
        "runtime",
        "sources",
        "storage",
        "workflows",
    }
    missing = expected_minimum - packages

    assert not missing, f"expected_top_level_packages_missing: {sorted(missing)}"


def test_packaged_source_tree_excludes_runtime_data_artifacts() -> None:
    blocked_paths = []
    for path in sorted(PROJECT_SRC.rglob("*")):
        if path.is_dir():
            continue
        relative_parts = path.relative_to(PROJECT_SRC).parts
        relative_part_set = set(relative_parts)
        if relative_part_set.intersection(BLOCKED_PACKAGE_PATH_PARTS):
            blocked_paths.append(path.as_posix())
            continue
        if _contains_data_artifact_path(relative_parts):
            blocked_paths.append(path.as_posix())
            continue
        if path.suffix.lower() in BLOCKED_ROW_LEVEL_SUFFIXES:
            blocked_paths.append(path.as_posix())

    assert blocked_paths == []


def test_pyproject_does_not_add_runtime_data_to_package() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    forbidden_sections = (
        "[tool.setuptools.package-data]",
        "[tool.setuptools.data-files]",
    )
    hits = [token for token in forbidden_sections if token in pyproject]

    assert hits == []


def test_source_tree_sensitive_scan_declares_required_tracked_text_roots() -> None:
    contract = load_maintenance_script_module("governance_contract")

    configured = {Path(root).as_posix() for root in contract.SOURCE_TREE_TEXT_ROOTS}

    assert REQUIRED_SOURCE_TREE_TEXT_ROOTS <= configured


def test_source_tree_sensitive_scan_uses_configured_roots() -> None:
    source = GOVERNANCE_CONTRACT.read_text(encoding="utf-8")

    assert "for root in SOURCE_TREE_TEXT_ROOTS" in source
    assert 'relative.parts[:1] not in {(".agents",), ("tests",)}' not in source


def test_handoff_verifier_has_no_root_wide_bundle_content_exemptions() -> None:
    source = HANDOFF_VERIFIER.read_text(encoding="utf-8")
    root_exemptions = "_".join(("BUNDLE", "CONTENT", "SCAN", "EXEMPT", "ROOTS"))
    file_exemptions = "_".join(("BUNDLE", "CONTENT", "SCAN", "EXEMPT", "FILES"))
    function_name = "_".join(("is", "bundle", "content", "scan", "exempt"))

    assert root_exemptions not in source
    assert file_exemptions not in source
    assert function_name not in source


def _contains_data_artifact_path(parts: tuple[str, ...]) -> bool:
    for index, part in enumerate(parts[:-1]):
        if part == "data" and parts[index + 1] in {
            "raw",
            "interim",
            "processed",
            "research",
        }:
            return True
    return False
