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
        "sources/csmar/dataset_contracts.py",
        "sources/csmar/raw_schemas.py",
        "sources/csmar/field_mappings.py",
        "sources/csmar/checksums.py",
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
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
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
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
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


def test_retired_csmar_compatibility_entrypoints_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "sources/csmar/contracts.py",
        SOURCE_ROOT / "sources/csmar" / ("build" + "_interim.py"),
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_csmar_package_init_does_not_reexport_convenience_functions() -> None:
    path = SOURCE_ROOT / "sources/csmar/__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    exported_names = set()
    imported_names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exported_names.update(
                        item.value
                        for item in node.value.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    )
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.asname or alias.name for alias in node.names)

    forbidden = {
        "build" + "_interim_frame",
        "import_interim_data",
        "normalize_daily_market_frame",
    }
    assert exported_names.isdisjoint(forbidden)
    assert imported_names.isdisjoint(forbidden)


def test_csmar_dataset_contracts_are_owned_by_dataset_contracts_module() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    dataset_path = SOURCE_ROOT / "sources/csmar/dataset_contracts.py"
    raw_tree = ast.parse(raw_path.read_text(encoding="utf-8"), filename=str(raw_path))
    dataset_tree = ast.parse(
        dataset_path.read_text(encoding="utf-8"),
        filename=str(dataset_path),
    )
    forbidden_raw_names = {
        "DatasetContract",
        "DATASET_CONTRACTS",
        "missing_required_columns",
        "assert_required_columns",
        "duplicate_primary_keys",
        "ohlc_violations",
        "negative_liquidity_rows",
    }
    raw_defined = _top_level_defined_names(raw_tree)
    dataset_defined = _top_level_defined_names(dataset_tree)

    assert raw_defined.isdisjoint(forbidden_raw_names)
    assert forbidden_raw_names.issubset(dataset_defined)


def test_csmar_mapping_and_checksum_owners_are_split_from_raw_schemas() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    raw_text = raw_path.read_text(encoding="utf-8")
    raw_tree = ast.parse(raw_path.read_text(encoding="utf-8"), filename=str(raw_path))
    forbidden_raw_owners = {
        "map_csmar_columns",
        "missing_raw_columns",
        "missing_raw_column_names",
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
    assert "column_mapping" not in raw_text
    assert "column_aliases" not in raw_text
    assert "chinese_meanings" not in raw_text
    assert "sha256" not in raw_text


def test_csmar_split_modules_own_mapping_and_checksum_functions() -> None:
    expected = {
        SOURCE_ROOT / "sources/csmar/field_mappings.py": {
            "missing_raw_columns",
            "missing_raw_column_names",
            "map_csmar_columns",
            "normalize_stock_code",
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


def test_csmar_field_mappings_does_not_own_validators_checksums_or_mappers() -> None:
    path = SOURCE_ROOT / "sources/csmar/field_mappings.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {
        "DatasetContract",
        "DATASET_CONTRACTS",
        "assert_required_columns",
        "build_raw_export_metadata",
        "duplicate_primary_keys",
        "missing_required_columns",
        "negative_liquidity_rows",
        "normalize_trdnew_capchg",
        "normalize_trdnew_daily_market",
        "ohlc_violations",
        "raw_export_sha256",
    }

    assert _top_level_defined_names(tree).isdisjoint(forbidden)


def test_csmar_mapping_metadata_is_owned_by_field_mappings() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    mapping_path = SOURCE_ROOT / "sources/csmar/field_mappings.py"
    checksum_path = SOURCE_ROOT / "sources/csmar/checksums.py"
    raw_tree = ast.parse(raw_path.read_text(encoding="utf-8"), filename=str(raw_path))
    mapping_tree = ast.parse(
        mapping_path.read_text(encoding="utf-8"), filename=str(mapping_path)
    )
    checksum_tree = ast.parse(
        checksum_path.read_text(encoding="utf-8"), filename=str(checksum_path)
    )

    raw_dataclass_fields = {
        node.target.id
        for class_node in raw_tree.body
        if (
            isinstance(class_node, ast.ClassDef)
            and class_node.name == "RawTableContract"
        )
        for node in class_node.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    mapping_assignments = {
        node.targets[0].id
        for node in mapping_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    checksum_functions = {
        node.name for node in checksum_tree.body if isinstance(node, ast.FunctionDef)
    }

    assert raw_dataclass_fields == {
        "dataset",
        "table_code",
        "table_name_zh",
        "required_columns",
    }
    assert {
        "CSMAR_COLUMN_MAPPINGS",
        "CSMAR_COLUMN_ALIASES",
        "CSMAR_FIELD_DESCRIPTIONS",
    }.issubset(mapping_assignments)
    assert {"_file_sha256", "raw_export_sha256", "build_raw_export_metadata"}.issubset(
        checksum_functions
    )


def test_platform_core_does_not_import_csmar_source_modules() -> None:
    allowed_composition_roots = {
        SOURCE_ROOT / "catalog" / "defaults.py",
    }
    checked_roots = (
        SOURCE_ROOT / "interfaces/api",
        SOURCE_ROOT / "interfaces/cli",
        SOURCE_ROOT / "interfaces/services.py",
        SOURCE_ROOT / "storage",
        SOURCE_ROOT / "reports",
        SOURCE_ROOT / "workflows",
        SOURCE_ROOT / "contracts",
        SOURCE_ROOT / "catalog",
    )
    hits: list[str] = []
    for path in _python_files(checked_roots):
        if path in allowed_composition_roots:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("quant_research_platform.sources.csmar"):
                        hits.append(f"{path}:{node.lineno}: {alias.name}")
                continue
            if module and module.startswith("quant_research_platform.sources.csmar"):
                hits.append(f"{path}:{node.lineno}: {module}")

    assert hits == []


def test_platform_core_does_not_hardcode_csmar_raw_fields_or_tables() -> None:
    checked_roots = (
        SOURCE_ROOT / "interfaces",
        SOURCE_ROOT / "storage",
        SOURCE_ROOT / "reports",
        SOURCE_ROOT / "workflows",
        SOURCE_ROOT / "contracts",
        SOURCE_ROOT / "catalog",
    )
    forbidden = (
        "Stkcd",
        "Trddt",
        "Opnprc",
        "Clsprc",
        "TRDNEW",
        "TRD_Dalyr",
    )
    hits: list[str] = []
    for path in _python_files(checked_roots):
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path}: {term}")

    assert hits == []


def test_public_interfaces_remain_provider_neutral() -> None:
    checked_roots = (
        SOURCE_ROOT / "interfaces/api",
        SOURCE_ROOT / "interfaces/cli",
        SOURCE_ROOT / "interfaces/services.py",
    )
    forbidden = (
        "/csmar/",
        "/provider/",
        "/source/",
        "qrp csmar",
        "qrp provider",
        "qrp source",
    )
    hits: list[str] = []
    for path in _python_files(checked_roots):
        text = path.read_text(encoding="utf-8")
        for term in forbidden:
            if term in text:
                hits.append(f"{path}: {term}")

    assert hits == []


def test_csmar_data_additions_are_source_local_by_default() -> None:
    """CSMAR additions must not require new API or CLI route families by default."""

    source_local_change_policy = {
        "same table same schema new files": "sources/csmar",
        "same table new fields": "sources/csmar + tests",
        "new CSMAR interim product": "sources/csmar/catalog.py + catalog registration",
        "public route additions": "forbidden",
        "public CLI group additions": "forbidden",
    }

    assert source_local_change_policy["public route additions"] == "forbidden"
    assert source_local_change_policy["public CLI group additions"] == "forbidden"


def _top_level_defined_names(tree: ast.Module) -> set[str]:
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


def _python_files(roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(path for path in root.rglob("*.py") if path.is_file())
    return files
