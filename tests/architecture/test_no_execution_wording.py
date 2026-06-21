from __future__ import annotations

import re
from pathlib import Path


def test_workflow_run_mode_replaces_execution_mode_contract() -> None:
    from quant_research_platform.contracts import execution

    assert not hasattr(execution, "Execution" + "Mode")
    assert execution.WorkflowRunMode.VALIDATE_ONLY.value == "validate_only"
    assert execution.WorkflowRunMode.WRITE_ARTIFACTS.value == "write_artifacts"


def test_runtime_workflow_outputs_do_not_use_execute_wording() -> None:
    forbidden = re.compile(
        r"\b(?:execute[A-Za-z0-9_]*|[A-Za-z0-9_]*dry_run[A-Za-z0-9_]*|"
        r"[A-Za-z0-9_]*dry-run[A-Za-z0-9_-]*)\b",
        re.IGNORECASE,
    )
    allowed_paths = {
        Path("tests/architecture/test_no_execution_wording.py"),
        Path("docs/architecture/open-issues-register.md"),
    }
    hits: list[str] = []
    for root in (
        Path("src/quant_research_platform/workflows"),
        Path("src/quant_research_platform/contracts"),
        Path("tests/workflows"),
    ):
        for path in root.rglob("*.py"):
            if path in allowed_paths:
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if forbidden.search(line):
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []
