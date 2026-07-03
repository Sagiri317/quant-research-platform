from __future__ import annotations

import ast
from pathlib import Path

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
