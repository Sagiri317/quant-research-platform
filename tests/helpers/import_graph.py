from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_SRC = Path("src/quant_research_platform")
ARCHITECTURE_IMPORT_GRAPH_ROOTS = (
    Path("src"),
    Path("tests/runtime"),
    Path("tests/workflows"),
)


@dataclass(frozen=True)
class ImportEdge:
    path: Path
    lineno: int
    module: str


@pytest.fixture(scope="session")
def repo_python_files() -> tuple[Path, ...]:
    return python_files_under(ARCHITECTURE_IMPORT_GRAPH_ROOTS, missing_ok=True)


@pytest.fixture(scope="session")
def ast_trees(repo_python_files: tuple[Path, ...]) -> dict[Path, ast.AST]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in repo_python_files
    }


@pytest.fixture(scope="session")
def import_edges(ast_trees: dict[Path, ast.AST]) -> tuple[ImportEdge, ...]:
    return _import_edges_from_trees(ast_trees)


def python_files_under(
    roots: Iterable[Path],
    *,
    missing_ok: bool = False,
) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            if missing_ok:
                continue
            raise AssertionError(f"python_scan_root_missing: {root}")
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    return tuple(sorted(files))


def import_edges_for_roots(roots: Iterable[Path]) -> tuple[ImportEdge, ...]:
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in python_files_under(roots)
    }
    return _import_edges_from_trees(trees)


def read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def parse_source(path: Path) -> ast.Module:
    return ast.parse(read_source(path), filename=str(path))


def top_level_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def module_all_names(tree: ast.AST) -> tuple[str, ...]:
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
            names.extend(
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return tuple(names)


def imported_modules_from_tree(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def imported_modules_from_path(path: Path) -> set[str]:
    return imported_modules_from_tree(parse_source(path))


def ast_calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _import_edges_from_trees(ast_trees: dict[Path, ast.AST]) -> tuple[ImportEdge, ...]:
    edges: list[ImportEdge] = []
    for path, tree in ast_trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                edges.append(ImportEdge(path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                edges.extend(
                    ImportEdge(path, node.lineno, alias.name) for alias in node.names
                )
    return tuple(edges)


@pytest.fixture(scope="session")
def module_to_path(repo_python_files: tuple[Path, ...]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in repo_python_files:
        if not path.as_posix().startswith("src/"):
            continue
        relative = path.relative_to("src").with_suffix("")
        if relative.name == "__init__":
            relative = relative.parent
        mapping[".".join(relative.parts)] = path
    return mapping


@pytest.fixture(scope="session")
def path_to_layer(repo_python_files: tuple[Path, ...]) -> dict[Path, str]:
    layers: dict[Path, str] = {}
    for path in repo_python_files:
        parts = path.parts
        if (
            len(parts) >= 3
            and parts[0] == "src"
            and parts[1] == "quant_research_platform"
        ):
            layers[path] = parts[2]
        elif parts:
            layers[path] = parts[0]
    return layers


def import_edges_under(
    import_edges: tuple[ImportEdge, ...],
    root: Path,
) -> tuple[ImportEdge, ...]:
    if root.is_file():
        return tuple(edge for edge in import_edges if edge.path == root)
    root_text = root.as_posix().rstrip("/") + "/"
    return tuple(
        edge
        for edge in import_edges
        if edge.path.as_posix().startswith(root_text)
    )
