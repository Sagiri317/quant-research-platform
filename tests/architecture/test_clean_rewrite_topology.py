from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")


def test_clean_rewrite_topology_has_required_packages() -> None:
    required = {
        "contracts",
        "storage",
        "catalog",
        "sources",
        "markets",
        "research",
        "workflows",
        "compute",
        "agents",
        "interfaces",
        "reports",
        "runtime",
    }

    existing = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert existing == required


def test_retired_top_level_source_packages_are_absent() -> None:
    retired = {
        "artifacts",
        "datasets",
        "pipelines",
        "kernels",
        "core",
        "data",
        "data_layers",
        "data_artifacts",
        "research_datasets",
    }

    existing = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert existing.isdisjoint(retired)


def test_clean_rewrite_target_files_exist() -> None:
    required_files = {
        "contracts/ids.py",
        "contracts/time.py",
        "contracts/columns.py",
        "contracts/data.py",
        "contracts/execution.py",
        "contracts/factors.py",
        "contracts/labels.py",
        "contracts/masks.py",
        "contracts/universes.py",
        "contracts/datasets.py",
        "contracts/evaluations.py",
        "contracts/models.py",
        "contracts/predictions.py",
        "contracts/portfolios.py",
        "contracts/privacy.py",
        "contracts/workflows.py",
        "contracts/validation.py",
        "storage/layout.py",
        "storage/parquet.py",
        "storage/manifests.py",
        "storage/partitions.py",
        "storage/checksums.py",
        "storage/privacy.py",
        "catalog/processed.py",
        "catalog/factors.py",
        "catalog/labels.py",
        "catalog/masks.py",
        "catalog/universes.py",
        "catalog/datasets.py",
        "catalog/evaluations.py",
        "catalog/models.py",
        "catalog/predictions.py",
        "catalog/workflows.py",
        "markets/base.py",
        "markets/calendars.py",
        "workflows/specs.py",
        "workflows/planner.py",
        "workflows/executor.py",
        "workflows/checkpoints.py",
        "agents/factor_factory.py",
        "agents/label_factory.py",
        "agents/dataset_planner.py",
        "agents/workflow_planner.py",
        "agents/proposals.py",
        "agents/safety.py",
        "reports/bundles.py",
        "reports/manifests.py",
        "reports/summaries.py",
        "runtime/privacy.py",
        "runtime/sanitization.py",
        "runtime/logging.py",
        "runtime/profiling.py",
        "runtime/errors.py",
        "compute/dataframe/construction.py",
        "compute/dataframe/constraints.py",
        "compute/dataframe/turnover.py",
        "compute/dataframe/exposure.py",
        "compute/dataframe/reporting.py",
        "compute/dataframe/portfolio.py",
        "compute/tensor/construction.py",
        "compute/tensor/constraints.py",
        "compute/tensor/turnover.py",
        "compute/tensor/exposure.py",
        "compute/tensor/reporting.py",
        "compute/tensor/portfolio.py",
        "sources/csmar/raw_schemas.py",
        "sources/csmar/field_mappings.py",
        "sources/csmar/checksums.py",
        "sources/csmar/contracts.py",
    }

    missing = [
        path for path in sorted(required_files) if not (SOURCE_ROOT / path).is_file()
    ]

    assert missing == []


def test_removed_root_directories_are_absent() -> None:
    retired_roots = {"config", "scripts", "web"}
    existing = {path.name for path in Path(".").iterdir() if path.is_dir()}

    assert existing.isdisjoint(retired_roots)


def test_compute_portfolio_modules_are_facades_only() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "compute/dataframe/portfolio.py",
        SOURCE_ROOT / "compute/tensor/portfolio.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                hits.append(f"{path}:{node.lineno}: {node.name}")

    assert hits == []


def test_compute_split_modules_own_implementations() -> None:
    split_modules = (
        SOURCE_ROOT / "compute/dataframe/construction.py",
        SOURCE_ROOT / "compute/dataframe/constraints.py",
        SOURCE_ROOT / "compute/dataframe/turnover.py",
        SOURCE_ROOT / "compute/dataframe/exposure.py",
        SOURCE_ROOT / "compute/dataframe/reporting.py",
        SOURCE_ROOT / "compute/tensor/construction.py",
        SOURCE_ROOT / "compute/tensor/constraints.py",
        SOURCE_ROOT / "compute/tensor/turnover.py",
        SOURCE_ROOT / "compute/tensor/exposure.py",
        SOURCE_ROOT / "compute/tensor/reporting.py",
    )
    missing: list[str] = []
    reverse_imports: list[str] = []
    for path in split_modules:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        has_impl = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        if not has_impl:
            missing.append(str(path))
        if ".portfolio import" in text:
            reverse_imports.append(str(path))

    assert missing == []
    assert reverse_imports == []


def test_compute_constraint_helpers_are_not_owned_by_portfolio_helpers() -> None:
    forbidden = {
        SOURCE_ROOT / "compute/dataframe/portfolio_helpers.py": {
            "_apply_tradability_constraints",
            "_build_tradability_context",
            "_is_entry_blocked",
            "_is_exit_blocked",
            "_tradability_diagnostic_counts",
            "_validate_tradability_columns",
        },
        SOURCE_ROOT / "compute/tensor/portfolio_helpers.py": {
            "_apply_array_tradability_constraints",
            "_optional_bool_array",
            "_previous_position_vector",
            "_ranked_asset_indices",
        },
    }
    hits: list[str] = []
    for path, names in forbidden.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits.extend(
            f"{path}:{node.lineno}: {node.name}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )

    assert hits == []


def test_csmar_contracts_module_is_facade_only() -> None:
    path = SOURCE_ROOT / "sources/csmar/contracts.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        f"{path}:{node.lineno}: {node.name}"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    star_imports = [
        f"{path}:{node.lineno}: {alias.name}"
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "*"
    ]

    assert definitions == []
    assert star_imports == []


def test_csmar_mapping_and_checksum_owners_are_split_from_raw_schemas() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    raw_tree = ast.parse(raw_path.read_text(encoding="utf-8"), filename=str(raw_path))
    forbidden_raw_owners = {
        "map_csmar_columns",
        "normalize_stock_code",
        "normalize_trdnew_capchg",
        "normalize_trdnew_daily_market",
        "_count_csv_rows",
        "_file_sha256",
        "build_raw_export_metadata",
    }
    raw_hits = [
        f"{raw_path}:{node.lineno}: {node.name}"
        for node in raw_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in forbidden_raw_owners
    ]

    assert raw_hits == []


def test_csmar_split_modules_own_mapping_and_checksum_functions() -> None:
    expected = {
        SOURCE_ROOT / "sources/csmar/field_mappings.py": {
            "map_csmar_columns",
            "normalize_stock_code",
            "normalize_trdnew_capchg",
            "normalize_trdnew_daily_market",
        },
        SOURCE_ROOT / "sources/csmar/checksums.py": {
            "_count_csv_rows",
            "_file_sha256",
            "build_raw_export_metadata",
            "raw_export_sha256",
        },
    }
    missing: list[str] = []
    reverse_imports: list[str] = []
    for path, names in expected.items():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        defined = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        for name in sorted(names - defined):
            missing.append(f"{path}: {name}")
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            if not node.module.endswith(".raw_schemas"):
                continue
            imported_names = {alias.name for alias in node.names}
            forbidden = names & imported_names
            for name in sorted(forbidden):
                reverse_imports.append(f"{path}:{node.lineno}: {name}")

    assert missing == []
    assert reverse_imports == []
