from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")
WORKFLOW_ACTIONS = (
    "data_materialize",
    "dataset_build",
    "research_run",
    "report_build",
    "status",
    "inspect",
)


def test_public_workflow_actions_have_one_owner_and_direct_callers() -> None:
    workflow_root = SOURCE_ROOT / "workflows"
    owners = {action: [] for action in WORKFLOW_ACTIONS}
    for path in workflow_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in owners:
                    owners[node.name].append(path)

    assert owners == {
        action: [workflow_root / "research.py"] for action in WORKFLOW_ACTIONS
    }

    cli_path = SOURCE_ROOT / "interfaces" / "cli" / "commands.py"
    smoke_path = workflow_root / "smoke.py"
    assert _imports_research_workflow(cli_path)
    assert _imports_research_workflow(smoke_path)
    assert _direct_workflow_calls(cli_path) == set(WORKFLOW_ACTIONS)
    assert _direct_workflow_calls(smoke_path) == {
        "data_materialize",
        "dataset_build",
        "research_run",
        "report_build",
    }


def test_research_and_profile_owners_have_no_provider_traversal() -> None:
    research_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (SOURCE_ROOT / "research").rglob("*.py")
    )
    assert "markets.implementations" not in research_sources
    assert "quant_research_platform.sources" not in research_sources

    profile = (SOURCE_ROOT / "profiles" / "research.py").read_text(encoding="utf-8")
    assert "write_dataset" not in profile
    assert "pandas" not in profile
    assert "memory_budget_bytes" not in profile


def test_tracked_surfaces_have_no_retired_naming() -> None:
    violations: list[str] = []
    repository_root = Path(__file__).resolve().parents[2]
    for relative_path in _tracked_surface_paths():
        if _contains_retired_token(relative_path.as_posix()):
            violations.append(f"{relative_path}:path")
        path = repository_root / relative_path
        text = path.read_text(encoding="utf-8")
        if _contains_retired_token(text):
            violations.append(f"{relative_path}:text")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=relative_path.as_posix())
            for symbol in _defined_symbols(tree):
                if _is_retired_symbol(symbol):
                    violations.append(f"{relative_path}:{symbol}")

    assert violations == []


_RETIRED_TEXT = re.compile(
    r"(?i)(?<![a-z0-9])(?:phase\s*[_-]?\s*[12]|"
    "current" + r"[_-]" + "contract|"
    "unified" + r"[_-]" + "research|"
    "leg" + "acy|"
    "com" + r"pat(?:ibility)?|"
    r"_" + "v2)(?![a-z0-9])"
)


def _defined_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            symbols.add(node.arg)
    return symbols


def _is_retired_symbol(symbol: str) -> bool:
    return bool(_RETIRED_TEXT.fullmatch(symbol))


def _contains_retired_token(value: str) -> bool:
    return _RETIRED_TEXT.search(value) is not None


def _tracked_surface_paths() -> tuple[Path, ...]:
    repository_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    tracked = (Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw)
    selected = tuple(
        path
        for path in tracked
        if path.name == "README.md"
        or path.parts[:1] in (("docs",), ("src",), ("tests",))
    )
    return selected


def _imports_research_workflow(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "quant_research_platform.workflows"
        and any(
            alias.name == "research" and alias.asname is None for alias in node.names
        )
        for node in ast.walk(tree)
    )


def _direct_workflow_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "research"
        and node.func.attr in WORKFLOW_ACTIONS
    }
