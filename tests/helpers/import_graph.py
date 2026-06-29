from __future__ import annotations

import ast
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
    files: list[Path] = []
    for root in ARCHITECTURE_IMPORT_GRAPH_ROOTS:
        if root.exists():
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    return tuple(sorted(files))


@pytest.fixture(scope="session")
def ast_trees(repo_python_files: tuple[Path, ...]) -> dict[Path, ast.AST]:
    return {
        path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in repo_python_files
    }


@pytest.fixture(scope="session")
def import_edges(ast_trees: dict[Path, ast.AST]) -> tuple[ImportEdge, ...]:
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
