from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")


def test_research_workflow_cutover_has_no_forwarding_module() -> None:
    research_path = SOURCE_ROOT / "workflows" / "research.py"
    assert research_path.is_file()
    assert not (SOURCE_ROOT / "workflows" / "staged.py").exists()

    tree = ast.parse(research_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert all(not module.endswith(".staged") for module in imported_modules)
    assert "staged" not in _defined_symbols(tree)


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


def test_one_common_processed_traversal_and_common_math_owner() -> None:
    processed_materializers = tuple(
        path
        for path in (SOURCE_ROOT / "markets" / "implementations").rglob(
            "*materializ*.py"
        )
    )
    assert processed_materializers == ()
    assert (SOURCE_ROOT / "research" / "factors" / "fundamental").is_dir()
    assert (SOURCE_ROOT / "research" / "factors" / "expectations").is_dir()
    assert (SOURCE_ROOT / "research" / "labels" / "engine.py").is_file()
    assert (SOURCE_ROOT / "research" / "portfolios" / "optimizer.py").is_file()


def test_production_source_has_no_retired_contract_symbols() -> None:
    forbidden = {
        "purge_gap_steps",
        "standardized_ridge_regression",
        "execution_lag_steps",
        "return_source_column",
        "rebalance_frequency",
        "financial_disclosures",
        "DEFAULT_UNIVERSE",
        "AS_OF_MASK",
        "EVALUATION_MASK",
    }
    violations: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if any(_contains_retired_token(part) for part in path.parts):
            violations.append(f"{path}:path")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        for symbol in _defined_symbols(tree):
            if symbol in forbidden or _is_retired_symbol(symbol):
                violations.append(f"{path}:{symbol}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in forbidden or _contains_retired_token(node.value):
                    violations.append(f"{path}:{node.lineno}:string")

    assert violations == []


_RETIRED_TEXT = re.compile(
    r"(?i)(?<![a-z0-9])(?:phase\s*[_-]?\s*1|"
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
