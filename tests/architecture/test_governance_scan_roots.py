from __future__ import annotations

import re
from pathlib import Path

from tests.architecture.governance_scan_helpers import (
    RUNTIME_WORKSPACE_SOURCE_ALLOWLIST,
    RUNTIME_WORKSPACE_TEST_ALLOWLIST,
    SCAN_ROOTS,
)


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
