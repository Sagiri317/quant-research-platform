from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers import frequency_ids

SOURCE_ROOT = Path("src/quant_research_platform")


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse_source(path: Path) -> ast.Module:
    return ast.parse(_read_source(path), filename=str(path))


def test_platform_topology_has_required_packages() -> None:
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


def test_platform_target_files_exist() -> None:
    required_files = {
        "contracts/ids.py",
        "contracts/time.py",
        "contracts/columns.py",
        "contracts/data.py",
        "contracts/factors.py",
        "contracts/labels.py",
        "contracts/datasets.py",
        "contracts/evaluations.py",
        "contracts/models.py",
        "contracts/predictions.py",
        "contracts/portfolios.py",
        "contracts/privacy.py",
        "contracts/workflows.py",
        "contracts/validation.py",
        "storage/layout.py",
        "storage/datasets.py",
        "storage/manifests.py",
        "storage/checksums.py",
        "catalog/processed.py",
        "catalog/factors.py",
        "catalog/labels.py",
        "catalog/datasets.py",
        "catalog/evaluations.py",
        "catalog/models.py",
        "catalog/predictions.py",
        "catalog/workflows.py",
        "markets/base.py",
        "markets/calendars.py",
        "workflows/planner.py",
        "workflows/executor.py",
        "workflows/checkpoints.py",
        "workflows/verification_evidence.py",
        "agents/factor_factory.py",
        "agents/label_factory.py",
        "agents/dataset_planner.py",
        "agents/workflow_planner.py",
        "agents/proposals.py",
        "agents/safety.py",
        "reports/report_artifacts.py",
        "reports/schema_inventory.py",
        "reports/verification_evidence.py",
        "runtime/privacy.py",
        "runtime/logging.py",
        "runtime/profiling.py",
        "runtime/errors.py",
        "runtime/execution_metrics.py",
        "compute/dataframe/construction.py",
        "compute/dataframe/constraints.py",
        "compute/dataframe/turnover.py",
        "compute/dataframe/exposure.py",
        "compute/dataframe/portfolio_returns.py",
        "compute/tensor/construction.py",
        "compute/tensor/constraints.py",
        "compute/tensor/turnover.py",
        "compute/tensor/exposure.py",
        "compute/tensor/portfolio_returns.py",
        "sources/csmar/ids.py",
        "sources/csmar/raw_schemas.py",
        "sources/csmar/field_mappings.py",
        "sources/csmar/checksums.py",
        "sources/csmar/interim.py",
        "markets/implementations/a_share/processed/annotations.py",
        "markets/implementations/a_share/processed/materializer.py",
        "markets/implementations/a_share/processed/panels.py",
        "markets/implementations/a_share/factors/definitions.py",
        "markets/implementations/a_share/factors/calculations.py",
        "markets/implementations/a_share/labels/definitions.py",
        "markets/implementations/a_share/labels/calculations.py",
        "markets/implementations/a_share/verification/private_aggregate.py",
        "research/labels/forward_returns.py",
        "interfaces/private_cli.py",
    }

    missing = [
        path for path in sorted(required_files) if not (SOURCE_ROOT / path).is_file()
    ]

    assert missing == []


def test_removed_root_directories_are_absent() -> None:
    retired_roots = {"config", "scripts", "web"}
    existing = {path.name for path in Path(".").iterdir() if path.is_dir()}

    assert existing.isdisjoint(retired_roots)


def test_compute_portfolio_compatibility_facades_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "compute/dataframe/portfolio.py",
        SOURCE_ROOT / "compute/dataframe/helpers.py",
        SOURCE_ROOT / "compute/dataframe" / ("report" + "ing.py"),
        SOURCE_ROOT / "compute/tensor/portfolio.py",
        SOURCE_ROOT / "compute/tensor" / ("report" + "ing.py"),
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_report_placeholder_and_retired_research_modules_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "reports/bundles.py",
        SOURCE_ROOT / "reports/index.py",
        SOURCE_ROOT / "reports/manifests.py",
        SOURCE_ROOT / "reports/summaries.py",
        SOURCE_ROOT / "research/models" / ("base" + "line.py"),
        SOURCE_ROOT / "research/predictions/bundle.py",
        SOURCE_ROOT / "research/predictions/handoff.py",
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_active_source_has_no_broad_artifact_report_diagnostics_abstractions() -> None:
    forbidden_names = {
        "Artifact" + "Service",
        "Artifact" + "Store",
        "Artifact" + "Registry",
        "Run" + "Repository",
        "Report" + "Repository",
        "Report" + "Store",
        "Evidence" + "Registry",
        "Diagnostics" + "Manager",
        "Diagnostics" + "Store",
        "Diagnostics" + "Registry",
        "Evaluation" + "Facade",
        "Report" + "Facade",
        "Artifact" + "Facade",
    }
    hits: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = _parse_source(path)
        defined = _top_level_defined_names(tree)
        for name in sorted(forbidden_names & defined):
            hits.append(f"{path}: {name}")

    assert hits == []


def test_compute_split_modules_own_implementations() -> None:
    split_modules = (
        SOURCE_ROOT / "compute/dataframe/construction.py",
        SOURCE_ROOT / "compute/dataframe/constraints.py",
        SOURCE_ROOT / "compute/dataframe/turnover.py",
        SOURCE_ROOT / "compute/dataframe/exposure.py",
        SOURCE_ROOT / "compute/dataframe/portfolio_returns.py",
        SOURCE_ROOT / "compute/tensor/construction.py",
        SOURCE_ROOT / "compute/tensor/constraints.py",
        SOURCE_ROOT / "compute/tensor/turnover.py",
        SOURCE_ROOT / "compute/tensor/exposure.py",
        SOURCE_ROOT / "compute/tensor/portfolio_returns.py",
    )
    missing: list[str] = []
    reverse_imports: list[str] = []
    for path in split_modules:
        text = _read_source(path)
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
        tree = _parse_source(path)
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
        SOURCE_ROOT / "sources/csmar/columns.py",
        SOURCE_ROOT / "sources/csmar/importer.py",
        SOURCE_ROOT / "sources/csmar/mappers.py",
        SOURCE_ROOT / "sources/csmar/dataset_contracts.py",
        SOURCE_ROOT / "markets/implementations/a_share/processed/migration.py",
        SOURCE_ROOT / "reports/private_aggregate_verification.py",
        SOURCE_ROOT / "research/evaluations/parity.py",
        SOURCE_ROOT / "runtime/sanitization.py",
        SOURCE_ROOT / "storage/privacy.py",
        SOURCE_ROOT / "workflows/specs.py",
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_a_share_factor_builder_has_no_legacy_partition_fallback() -> None:
    path = SOURCE_ROOT / "markets/implementations/a_share/factors/build_factors.py"
    assert not path.exists()


def test_a_share_source_to_dataset_old_helpers_are_absent() -> None:
    retired_functions = {
        SOURCE_ROOT / "markets/implementations/a_share/processed/panels.py": {
            "_legacy_build_panel_frame_from_interim",
            "build_panel_frame_from_interim",
            "write_panel_processed_data",
            "build_panel_from_interim",
            "_filter_panel_years",
            "_filter_panel_end_date",
            "_is_missing_dataset_artifact_error",
        },
        SOURCE_ROOT / "markets/implementations/a_share/processed/contexts.py": {
            "_context_frame_from_panel",
            "_select_context_columns",
            "select_context_columns",
            "write_contexts",
        },
        SOURCE_ROOT / "markets/implementations/a_share/universes/build.py": {
            "_universe_frame_from_panel",
            "select_universe_columns",
            "write_universe",
        },
        SOURCE_ROOT / "markets/implementations/a_share/masks/build.py": {
            "_mask_frame_from_panel",
            "_select_mask_columns",
            "write_masks",
        },
        SOURCE_ROOT / "markets/implementations/a_share/factors/build_factors.py": {
            "calculate_panel_factors",
            "calculate_price_volume_factor_full",
            "_factor_output_frame",
            "_factor_panel_input_columns",
            "_is_missing_dataset_artifact_error",
            "read_panel",
            "read_full_panel_for_factor_build",
            "select_slim_factor_columns",
        },
        SOURCE_ROOT / "markets/implementations/a_share/labels/build_labels.py": {
            "calculate_forward_returns_full",
            "calculate_panel_labels",
            "_label_frame_with_forward_returns",
            "_label_output_frame",
            "select_existing_columns",
            "select_label_columns",
            "write_forward_labels",
        },
        SOURCE_ROOT / "markets/implementations/a_share/processed/materializer.py": {
            "_read_processed_panel",
        },
    }
    hits: list[str] = []
    for path, names in retired_functions.items():
        if not path.exists():
            continue
        tree = _parse_source(path)
        hits.extend(
            f"{path}:{node.lineno}: {node.name}"
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )

    assert hits == []


def test_csmar_package_init_does_not_reexport_convenience_functions() -> None:
    path = SOURCE_ROOT / "sources/csmar/__init__.py"
    tree = _parse_source(path)
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

    forbidden = {"build" + "_interim_frame", "import_" + "interim_data"}
    assert exported_names.isdisjoint(forbidden)
    assert imported_names.isdisjoint(forbidden)


def test_package_inits_do_not_forward_storage_or_cli_owner_symbols() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "storage/__init__.py",
        SOURCE_ROOT / "interfaces/cli/__init__.py",
    ):
        tree = _parse_source(path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom | ast.Import):
                hits.append(f"{path}:{node.lineno}: import")
            elif isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                hits.append(f"{path}:{node.lineno}: __getattr__")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        hits.append(f"{path}:{node.lineno}: __all__")

    assert hits == []


def test_reviewed_package_inits_do_not_reexport_convenience_symbols() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "research/evaluations/__init__.py",
        SOURCE_ROOT / "compute/tensor/__init__.py",
    ):
        tree = _parse_source(path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom | ast.Import):
                hits.append(f"{path}:{node.lineno}: import")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        hits.append(f"{path}:{node.lineno}: __all__")

    assert hits == []


def test_non_interface_source_modules_do_not_define_hidden_cli_entrypoints() -> None:
    allowed_cli_root = SOURCE_ROOT / "interfaces/cli"
    allowed_private_cli = SOURCE_ROOT / "interfaces/private_cli.py"
    hits: list[str] = []
    for path in _python_files((SOURCE_ROOT,)):
        tree = _parse_source(path)
        if path == allowed_private_cli or _is_relative_to(path, allowed_cli_root):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "argparse":
                        hits.append(f"{path}:{node.lineno}: argparse")
            elif isinstance(node, ast.ImportFrom) and node.module == "argparse":
                hits.append(f"{path}:{node.lineno}: argparse")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and (
                node.name == "main"
            ):
                hits.append(f"{path}:{node.lineno}: main")
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Typer"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "typer"
                ):
                    hits.append(f"{path}:{node.lineno}: typer.Typer")
            elif isinstance(node, ast.If) and _is_dunder_main_guard(node.test):
                hits.append(f"{path}:{node.lineno}: __main__")

    assert hits == []


def test_csmar_source_package_does_not_own_market_interim_contracts() -> None:
    source_root = SOURCE_ROOT / "sources/csmar"
    forbidden_raw_names = {
        "DatasetContract",
        "DATASET_CONTRACTS",
        "missing_required_columns",
        "assert_required_columns",
        "duplicate_primary_keys",
        "ohlc_violations",
        "negative_liquidity_rows",
    }
    hits: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = _parse_source(path)
        defined = _top_level_defined_names(tree)
        overlap = sorted(forbidden_raw_names & defined)
        hits.extend(f"{path}: {name}" for name in overlap)

    assert hits == []


def test_csmar_source_package_does_not_own_a_share_interim_pruning() -> None:
    source_root = SOURCE_ROOT / "sources/csmar"
    forbidden_tokens = {
        "ASSET_TIME_INTERIM_KEEP_COLUMNS",
        "ASSET_REFERENCE_INTERIM_KEEP_COLUMNS",
        "INTERIM_KEEP_COLUMNS",
        "apply_interim_column_rules",
        "is_a_share_scope",
        "eligible_base",
    }
    hits: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in sorted(forbidden_tokens):
            if token in text:
                hits.append(f"{path}: {token}")

    assert hits == []


def test_a_share_csmar_binding_does_not_export_unprefixed_behavior() -> None:
    path = SOURCE_ROOT / "markets/implementations/a_share/source_bindings/csmar.py"
    assert not path.exists()


def test_csmar_mapping_and_checksum_owners_are_split_from_raw_schemas() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    raw_text = _read_source(raw_path)
    raw_tree = _parse_source(raw_path)
    forbidden_raw_owners = {
        "map_columns",
        "missing_raw_columns",
        "missing_raw_column_names",
        "normalize_stock_code",
        "normalize_trdnew_capchg",
        f"normalize_trdnew_{frequency_ids.MARKET_BARS_DATA_ID}",
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
            "map_columns",
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
        text = _read_source(path)
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


def test_csmar_source_raw_exports_do_not_write_metadata_manifests() -> None:
    path = SOURCE_ROOT / "sources/csmar/raw_exports.py"
    tree = _parse_source(path)
    text = path.read_text(encoding="utf-8")
    hits: list[str] = []
    forbidden_functions = {"write_raw_export_metadata"}
    forbidden_imports = {
        "write_manifest",
        "write_internal_artifact_manifest",
        "build_raw_export_metadata",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in forbidden_functions:
            hits.append(f"{path}:{node.lineno}: {node.name}")
        if isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            for name in sorted(imported & forbidden_imports):
                hits.append(f"{path}:{node.lineno}: {name}")

    assert "write_metadata" not in text
    assert "metadata_path" not in text
    assert hits == []


def test_csmar_field_mappings_does_not_own_validators_checksums_or_mappers() -> None:
    path = SOURCE_ROOT / "sources/csmar/field_mappings.py"
    tree = _parse_source(path)
    forbidden = {
        "DatasetContract",
        "DATASET_CONTRACTS",
        "assert_required_columns",
        "build_raw_export_metadata",
        "duplicate_primary_keys",
        "missing_required_columns",
        "negative_liquidity_rows",
        "normalize_trdnew_capchg",
        f"normalize_trdnew_{frequency_ids.MARKET_BARS_DATA_ID}",
        "ohlc_violations",
        "raw_export_sha256",
    }

    assert _top_level_defined_names(tree).isdisjoint(forbidden)


def test_csmar_mapping_metadata_is_owned_by_field_mappings() -> None:
    raw_path = SOURCE_ROOT / "sources/csmar/raw_schemas.py"
    mapping_path = SOURCE_ROOT / "sources/csmar/field_mappings.py"
    checksum_path = SOURCE_ROOT / "sources/csmar/checksums.py"
    raw_tree = _parse_source(raw_path)
    mapping_tree = _parse_source(mapping_path)
    checksum_tree = _parse_source(checksum_path)

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
        "optional_columns",
    }
    assert {
        "COLUMN_MAPPINGS",
        "COLUMN_ALIASES",
        "FIELD_DESCRIPTIONS",
    }.issubset(mapping_assignments)
    assert {"_file_sha256", "raw_export_sha256", "build_raw_export_metadata"}.issubset(
        checksum_functions
    )


def test_platform_core_does_not_import_csmar_source_modules() -> None:
    allowed_composition_roots = {
        SOURCE_ROOT / "catalog" / "profiles.py",
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
        tree = _parse_source(path)
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
        "new A-share CSMAR interim product": (
            "sources/csmar/interim.py + A-share processed materializer "
            "dependency wiring"
        ),
        "public route additions": "forbidden",
        "public CLI group additions": "forbidden",
    }

    assert source_local_change_policy["public route additions"] == "forbidden"
    assert source_local_change_policy["public CLI group additions"] == "forbidden"
    assert "sources/csmar/interim.py" in (
        source_local_change_policy["new A-share CSMAR interim product"]
    )


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


def _is_dunder_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    if not (
        isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
    ):
        return False
    comparator = node.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
