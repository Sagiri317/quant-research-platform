from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")


def test_reports_do_not_write_files() -> None:
    forbidden_calls = {
        "Path.write_text",
        "json.dump(",
        "atomic_write_json",
        "atomic_write_text",
        "atomic_write_csv",
        "to_parquet",
        "ParquetWriter",
    }
    hits: list[str] = []
    for path in sorted((PROJECT_SRC / "reports").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for token in forbidden_calls:
            if token in source:
                hits.append(f"{path.as_posix()}:{token}")

    assert hits == []


def test_artifacts_write_given_payloads_without_building_report_content() -> None:
    artifacts_root = PROJECT_SRC / "artifacts"
    hits: list[str] = []
    for path, lineno, module in _iter_imports(artifacts_root):
        if module.startswith(
            (
                "quant_research_platform.reports",
                "quant_research_platform.research",
                "quant_research_platform.markets.implementations",
            )
        ):
            hits.append(f"{path.as_posix()}:{lineno}:{module}")
    for path in sorted(artifacts_root.rglob("*.py")) if artifacts_root.exists() else ():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ) and node.name.startswith(("build_", "render_")):
                hits.append(f"{path.as_posix()}:{node.lineno}:{node.name}")

    assert hits == []


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    if not root.exists():
        return ((root, 0, "missing-artifacts-package"),)
    paths = (root,) if root.is_file() else tuple(sorted(root.rglob("*.py")))
    imports: list[tuple[Path, int, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)
