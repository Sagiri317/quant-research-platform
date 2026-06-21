from __future__ import annotations

import re
from pathlib import Path


def test_workflow_run_policy_replaces_retired_action_contract() -> None:
    from quant_research_platform.contracts.workflows import WorkflowRunPolicy

    assert WorkflowRunPolicy.VALIDATE_ONLY.value == "validate_only"
    assert WorkflowRunPolicy.WRITE_ARTIFACTS.value == "write_artifacts"
    assert not Path("src/quant_research_platform/contracts/execution.py").exists()
    assert not Path("src/quant_research_platform/contracts/workflow_run.py").exists()


def test_runtime_workflow_outputs_do_not_use_action_wording() -> None:
    action_word = "exec" + "ute"
    dry_run = "dry" + "-run"
    forbidden = re.compile(
        r"\b(?:"
        + re.escape(action_word)
        + r"[A-Za-z0-9_]*|[A-Za-z0-9_]*dry_run[A-Za-z0-9_]*|"
        r"[A-Za-z0-9_]*" + re.escape(dry_run) + r"[A-Za-z0-9_-]*)\b",
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
                if forbidden.search(line) or "WorkflowRun" + "Mode" in line:
                    hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []
