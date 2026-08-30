from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path


@dataclass(frozen=True)
class ImportEdge:
    path: Path
    lineno: int
    module: str


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
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    parts = path.with_suffix("").parts
                    try:
                        package_root = parts.index("quant_research_platform")
                    except ValueError:
                        continue
                    package_end = (
                        len(parts) if path.stem == "__init__" else len(parts) - 1
                    )
                    package = ".".join(parts[package_root:package_end])
                    try:
                        module = resolve_name("." * node.level + module, package)
                    except ImportError:
                        continue
                edges.extend(
                    ImportEdge(
                        path,
                        node.lineno,
                        module if alias.name == "*" else f"{module}.{alias.name}",
                    )
                    for alias in node.names
                )
            elif isinstance(node, ast.Import):
                edges.extend(
                    ImportEdge(path, node.lineno, alias.name) for alias in node.names
                )
    return tuple(edges)


def import_edges_under(
    import_edges: tuple[ImportEdge, ...],
    root: Path,
) -> tuple[ImportEdge, ...]:
    if root.is_file():
        return tuple(edge for edge in import_edges if edge.path == root)
    root_text = root.as_posix().rstrip("/") + "/"
    return tuple(
        edge for edge in import_edges if edge.path.as_posix().startswith(root_text)
    )
