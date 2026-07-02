from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import ImportEdge, import_edges_under

PROJECT_SRC = Path("src/quant_research_platform")
ARTIFACT_OUTPUTS = PROJECT_SRC / "artifacts" / "outputs.py"
ARTIFACT_OUTPUT_OWNER_MODULES = (
    PROJECT_SRC / "artifacts" / "_output_manifest.py",
    PROJECT_SRC / "artifacts" / "_output_common_validation.py",
    PROJECT_SRC / "artifacts" / "_prediction_output_validation.py",
    PROJECT_SRC / "artifacts" / "_evaluation_output_validation.py",
    PROJECT_SRC / "artifacts" / "_portfolio_output_validation.py",
    PROJECT_SRC / "artifacts" / "_stage_profile_outputs.py",
    PROJECT_SRC / "artifacts" / "_prediction_outputs.py",
    PROJECT_SRC / "artifacts" / "_evaluation_outputs.py",
    PROJECT_SRC / "artifacts" / "_portfolio_outputs.py",
    PROJECT_SRC / "artifacts" / "_parquet_writers.py",
)
ARTIFACT_OUTPUT_PUBLIC_WRITERS = {
    "write_stage_profile_outputs",
    "write_prediction_outputs",
    "write_evaluation_outputs",
    "write_portfolio_outputs",
}


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


def test_artifacts_outputs_root_delegates_to_owner_modules() -> None:
    source = ARTIFACT_OUTPUTS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ARTIFACT_OUTPUTS))
    function_names = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    missing_modules = [
        path.as_posix()
        for path in ARTIFACT_OUTPUT_OWNER_MODULES
        if not path.is_file()
    ]
    non_delegating_writers = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ARTIFACT_OUTPUT_PUBLIC_WRITERS
        and not _function_body_is_single_call(node)
    ]

    assert missing_modules == []
    assert function_names == ARTIFACT_OUTPUT_PUBLIC_WRITERS
    assert class_names == set()
    assert non_delegating_writers == []


def test_output_writer_uses_local_catalog_not_profile_bundle() -> None:
    source = ARTIFACT_OUTPUTS.read_text(encoding="utf-8")
    common_source = (
        PROJECT_SRC / "artifacts" / "_output_common_validation.py"
    ).read_text(encoding="utf-8")

    assert "quant_research_platform.profiles.research" not in source
    assert "local_a_share_csmar_catalogs" not in source
    assert "quant_research_platform.artifacts.definitions" not in source
    assert "output_catalog(" not in source
    assert "output_catalog(" in common_source


def _function_body_is_single_call(function: ast.FunctionDef) -> bool:
    body = [
        node
        for node in function.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Return):
        return isinstance(statement.value, ast.Call)
    if isinstance(statement, ast.Expr):
        return isinstance(statement.value, ast.Call)
    return False
