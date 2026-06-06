from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
SCAN_ROOTS = (Path("src"), Path("tests"))
OLD_PANEL_NAME = "Panel" + "Spec"
OLD_VALIDATE_NAME = "validate" + "_panel"
OLD_PROTOCOL_MODULE = ".".join(
    ("quant_research_platform", "compute", "protocols")
)
OLD_STORAGE_PROTOCOL = "Data" + "Spec" + "Catalog"


def test_compute_protocols_module_is_removed() -> None:
    assert not (PROJECT_SRC / "compute" / "protocols.py").exists()


def test_compute_ownership_symbols_live_in_owner_modules() -> None:
    panels_tree = _parse(PROJECT_SRC / "compute" / "panels.py")
    config_tree = _parse(PROJECT_SRC / "compute" / "config.py")
    backends_tree = _parse(PROJECT_SRC / "compute" / "backends.py")

    assert "ComputePanelSpec" in _top_level_classes(panels_tree)
    assert "validate_compute_panel" in _top_level_functions(panels_tree)
    assert {
        "WeightConfig",
        "TradabilityConfig",
        "CostConfig",
    }.issubset(_top_level_classes(config_tree))
    assert {
        "DATAFRAME_BACKEND",
        "TENSOR_BACKEND",
        "SUPPORTED_BACKENDS",
    }.issubset(_top_level_assignments(backends_tree))


def test_active_code_does_not_import_compute_protocols_or_old_exact_names() -> None:
    hits: list[str] = []
    for path in _python_files(SCAN_ROOTS):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == OLD_PROTOCOL_MODULE:
                hits.append(f"{path}:{node.lineno}: import {OLD_PROTOCOL_MODULE}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == OLD_PROTOCOL_MODULE:
                        hits.append(
                            f"{path}:{node.lineno}: import {OLD_PROTOCOL_MODULE}"
                        )
            elif isinstance(node, ast.Name) and node.id in {
                OLD_PANEL_NAME,
                OLD_VALIDATE_NAME,
                OLD_STORAGE_PROTOCOL,
            }:
                hits.append(f"{path}:{node.lineno}: {node.id}")

    assert not hits, "old_compute_or_storage_symbol_found: " + "; ".join(hits)


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _python_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return tuple(sorted(files))


def _top_level_classes(tree: ast.AST) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _top_level_functions(tree: ast.AST) -> set[str]:
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _top_level_assignments(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return names
