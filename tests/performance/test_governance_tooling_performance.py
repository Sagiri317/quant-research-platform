from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pytest

from tests.helpers.governance_git_repos import run_governance_script

pytestmark = pytest.mark.performance


@dataclass(frozen=True)
class ToolRuntime:
    command: tuple[str, ...]
    duration_seconds: float
    stdout: str
    stderr: str


def test_structure_audit_runtime_evidence(record_property) -> None:
    result = _run_tool(
        "python",
        ".agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py",
        "--repo-root",
        ".",
        "--json",
    )

    _record_runtime(record_property, "structure_audit", result)
    assert '"status": "PASS"' in result.stdout


def test_verify_governance_active_docs_runtime_evidence(record_property) -> None:
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

    _record_runtime(record_property, "verify_governance_active_docs", result)
    assert '"status": "PASS"' in result.stdout


def test_source_tree_scan_runtime_evidence(record_property) -> None:
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

    _record_runtime(record_property, "source_tree_scan_groups", result)
    assert result.stdout.strip().isdigit()


def _record_runtime(record_property, name: str, result: ToolRuntime) -> None:
    record_property(f"{name}_duration_seconds", result.duration_seconds)
    print(f"{name}_duration_seconds={result.duration_seconds:.6f}")


def _run_tool(*command: str) -> ToolRuntime:
    assert command[0] == "python"
    started = perf_counter()
    result = run_governance_script(Path("."), *command[1:])
    duration_seconds = perf_counter() - started
    assert result.returncode == 0, result.stdout + result.stderr
    return ToolRuntime(
        command=command,
        duration_seconds=duration_seconds,
        stdout=result.stdout,
        stderr=result.stderr,
    )
