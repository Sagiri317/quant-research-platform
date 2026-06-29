from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import ImportEdge, import_edges_under

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


def test_artifacts_write_given_payloads_without_building_report_content(
    ast_trees: dict[Path, ast.AST],
    import_edges: tuple[ImportEdge, ...],
) -> None:
    artifacts_root = PROJECT_SRC / "artifacts"
    hits: list[str] = []
    for edge in import_edges_under(import_edges, artifacts_root):
        if edge.module.startswith(
            (
                "quant_research_platform.reports",
                "quant_research_platform.research",
                "quant_research_platform.markets.implementations",
            )
        ):
            hits.append(f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}")
    for path in sorted(artifacts_root.rglob("*.py")) if artifacts_root.exists() else ():
        tree = ast_trees[path]
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ) and node.name.startswith(("build_", "render_")):
                hits.append(f"{path.as_posix()}:{node.lineno}:{node.name}")

    assert hits == []
