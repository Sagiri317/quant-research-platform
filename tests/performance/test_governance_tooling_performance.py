from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pytest

pytestmark = pytest.mark.performance

GOVERNANCE_TOOLING_RUNTIME_BUDGET_SECONDS = 120.0


@dataclass(frozen=True)
class ToolRuntime:
    command: tuple[str, ...]
    duration_seconds: float
    stdout: str
    stderr: str


def test_structure_audit_runtime_guardrail() -> None:
    result = _run_tool(
        "python",
        ".agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py",
        "--repo-root",
        ".",
        "--json",
    )

    assert result.duration_seconds <= GOVERNANCE_TOOLING_RUNTIME_BUDGET_SECONDS
    assert '"status": "PASS"' in result.stdout


def test_verify_governance_active_docs_runtime_guardrail() -> None:
    result = _run_tool(
        "python",
        ".agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py",
        "--repo-root",
        ".",
        "--governance-only",
        "--scan-scope",
        "active-docs",
        "--json",
    )

    assert result.duration_seconds <= GOVERNANCE_TOOLING_RUNTIME_BUDGET_SECONDS
    assert '"status": "PASS"' in result.stdout


def test_verify_code_map_flow_runtime_guardrail() -> None:
    result = _run_tool(
        "python",
        ".agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py",
        "--repo-root",
        ".",
    )

    assert result.duration_seconds <= GOVERNANCE_TOOLING_RUNTIME_BUDGET_SECONDS
    assert result.stdout


def test_source_tree_scan_runtime_guardrail() -> None:
    governance_contract_path = (
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "governance_contract.py"
    )
    code = (
        "from pathlib import Path; "
        "import importlib.util; "
        f"p=Path({governance_contract_path!r}); "
        "s=importlib.util.spec_from_file_location('governance_contract', p); "
        "m=importlib.util.module_from_spec(s); "
        "import sys; "
        "sys.modules[s.name]=m; "
        "s.loader.exec_module(m); "
        "groups=m.source_tree_scan_groups(Path('.')); "
        "print(sum(len((v.get('findings') or [])) for v in groups.values()))"
    )
    result = _run_tool("python", "-c", code)

    assert result.duration_seconds <= GOVERNANCE_TOOLING_RUNTIME_BUDGET_SECONDS
    assert result.stdout.strip().isdigit()


def _run_tool(*command: str) -> ToolRuntime:
    run_command = tuple(
        sys.executable if item == "python" else item for item in command
    )
    started = perf_counter()
    result = subprocess.run(
        run_command,
        cwd=Path("."),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration_seconds = perf_counter() - started
    assert result.returncode == 0, result.stdout + result.stderr
    return ToolRuntime(
        command=command,
        duration_seconds=duration_seconds,
        stdout=result.stdout,
        stderr=result.stderr,
    )
