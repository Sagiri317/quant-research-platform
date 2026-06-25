from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
SCAN_ROOTS = (Path("src"), Path("tests"))
OLD_PANEL_NAME = "Panel" + "Spec"
OLD_VALIDATE_NAME = "validate" + "_panel"
OLD_PROTOCOL_MODULE = ".".join(("quant_research_platform", "compute", "protocols"))
OLD_STORAGE_PROTOCOL = "Data" + "Spec" + "Catalog"


def test_compute_protocols_module_is_removed() -> None:
    assert not (PROJECT_SRC / "compute" / "protocols.py").exists()
    assert not (PROJECT_SRC / "compute" / "config.py").exists()


def test_compute_ownership_symbols_live_in_owner_modules() -> None:
    panels_tree = _parse(PROJECT_SRC / "compute" / "panels.py")
    backends_tree = _parse(PROJECT_SRC / "compute" / "backends.py")
    portfolios_tree = _parse(PROJECT_SRC / "contracts" / "portfolios.py")

    assert "ComputePanelSpec" in _top_level_classes(panels_tree)
    assert "validate_compute_panel" in _top_level_functions(panels_tree)
    assert {
        "WeightConfig",
        "TradabilityConfig",
        "CostConfig",
    }.issubset(_top_level_classes(portfolios_tree))
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


def test_compute_modules_do_not_import_platform_layers() -> None:
    forbidden_prefixes = (
        "quant_research_platform.research",
        "quant_research_platform.reports",
        "quant_research_platform.workflows",
        "quant_research_platform.catalog",
        "quant_research_platform.storage",
    )
    hits: list[str] = []
    for path in _python_files((PROJECT_SRC / "compute",)):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(forbidden_prefixes):
                    hits.append(f"{path}:{node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        hits.append(f"{path}:{node.lineno}: {alias.name}")

    assert not hits, "compute_layer_import_boundary_violation: " + "; ".join(hits)


def test_compute_cross_section_public_kernels_use_compute_names() -> None:
    hits: list[str] = []
    for path in (
        PROJECT_SRC / "compute/dataframe/cross_section.py",
        PROJECT_SRC / "compute/tensor/cross_section.py",
    ):
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            if node.name.startswith(("evaluate_", "analyze_")):
                hits.append(f"{path}:{node.lineno}: {node.name}")

    assert not hits, "compute_public_kernel_name_not_neutral: " + "; ".join(hits)


def test_compute_cross_section_output_columns_use_label_value_names() -> None:
    retired_column = "mean_forward" + "_return"
    hits: list[str] = []
    for path in (
        PROJECT_SRC / "compute/dataframe/cross_section.py",
        PROJECT_SRC / "compute/tensor/cross_section.py",
    ):
        text = path.read_text(encoding="utf-8")
        if retired_column in text:
            hits.append(f"{path}: {retired_column}")

    assert not hits, "compute_grouped_output_name_not_neutral: " + "; ".join(hits)


def test_evaluation_group_artifact_names_are_label_value_neutral() -> None:
    hits: list[str] = []
    roots = (
        PROJECT_SRC / "reports",
        PROJECT_SRC / "research/evaluations",
    )
    forbidden_terms = (
        "grouped_returns",
        "evaluation_grouped_returns",
        "grouped_returns.parquet",
    )
    for path in _python_files(roots):
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            if term in text:
                hits.append(f"{path}: {term}")

    assert not hits, "evaluation_group_artifact_name_not_neutral: " + "; ".join(hits)


def test_evaluation_result_and_report_writer_do_not_use_generic_metadata() -> None:
    engine_path = PROJECT_SRC / "research/evaluations/engine.py"
    report_path = PROJECT_SRC / "artifacts/outputs.py"
    engine_tree = _parse(engine_path)
    report_text = report_path.read_text(encoding="utf-8")
    result_class = next(
        node
        for node in engine_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EvaluationResult"
    )
    field_names = {
        child.target.id
        for child in result_class.body
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
    }

    assert "metadata" not in field_names
    assert "diagnostics" in field_names
    assert 'attrs.get("metadata"' not in report_text
    assert 'attrs["metadata"]' not in report_text
    assert "metrics.attrs" not in report_text


def test_prediction_outputs_report_writer_use_specific_metadata_attrs() -> None:
    outputs_text = (PROJECT_SRC / "research/predictions/outputs.py").read_text(
        encoding="utf-8"
    )
    report_text = (PROJECT_SRC / "artifacts/outputs.py").read_text(encoding="utf-8")

    assert 'attrs["metadata"]' not in outputs_text
    assert 'attrs.get("metadata"' not in report_text
    assert "prediction_metadata_factory" in outputs_text
    assert '"prediction_metadata"' in report_text


def test_tensor_modules_do_not_import_dataframe_private_helpers() -> None:
    hits: list[str] = []
    for path in _python_files((PROJECT_SRC / "compute/tensor",)):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.startswith("quant_research_platform.compute.dataframe"):
                continue
            private_names = [
                alias.name for alias in node.names if alias.name.startswith("_")
            ]
            if private_names:
                hits.append(f"{path}:{node.lineno}: {node.module}.{private_names}")

    assert not hits, "tensor_imports_dataframe_private_helper: " + "; ".join(hits)


def test_compute_docstrings_do_not_contain_garbled_question_mark_terms() -> None:
    hits: list[str] = []
    for path in _python_files((PROJECT_SRC / "compute",)):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(
                node,
                ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
            ):
                continue
            docstring = ast.get_docstring(node, clean=False)
            if docstring and "?" in docstring:
                hits.append(f"{path}:{getattr(node, 'lineno', 1)}: docstring")
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and "?" in token.string:
                hits.append(f"{path}:{token.start[0]}: comment")

    assert not hits, "garbled_compute_docstring_terms: " + "; ".join(hits)


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
