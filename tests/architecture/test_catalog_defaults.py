from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
CATALOG_INIT = PROJECT_SRC / "catalog" / "__init__.py"
CATALOG_DEFAULTS = PROJECT_SRC / "catalog" / "defaults.py"
SCAN_ROOTS = (Path("src"), Path("tests"))


def test_catalog_package_init_does_not_export_platform_defaults() -> None:
    source = CATALOG_INIT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CATALOG_INIT))
    all_names = _module_all_names(tree)

    assert "quant_research_platform.catalog.defaults" not in source
    assert "catalog.defaults" not in source
    assert "_DEFAULT_EXPORTS" not in source
    assert "__getattr__" not in {node.name for node in tree.body if _is_def(node)}
    assert "default_catalogs" not in all_names
    assert not [name for name in all_names if name.startswith("default_")]


def test_platform_defaults_are_imported_from_explicit_defaults_module() -> None:
    root_import_hits: list[str] = []
    for path in _python_files(SCAN_ROOTS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "quant_research_platform.catalog":
                continue
            default_names = [
                alias.name
                for alias in node.names
                if alias.name == "default_catalogs"
                or alias.name.startswith("default_")
            ]
            if default_names:
                root_import_hits.append(f"{path}:{node.lineno}")

    assert not root_import_hits, (
        "catalog_defaults_must_use_explicit_module: "
        + "; ".join(root_import_hits)
    )


def test_catalog_defaults_remains_platform_composition_root() -> None:
    source = CATALOG_DEFAULTS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CATALOG_DEFAULTS))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    all_names = _module_all_names(tree)

    assert "quant_research_platform.markets.implementations.a_share.catalog" in modules
    assert "quant_research_platform.sources.csmar.catalog" in modules
    assert "default_catalogs" in all_names


def _module_all_names(tree: ast.AST) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    names.append(item.value)
    return tuple(names)


def _is_def(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _python_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return tuple(sorted(files))
