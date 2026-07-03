from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers import frequency_ids
from tests.helpers.import_graph import python_files_under

SOURCE_ROOT = Path("src/quant_research_platform")
API_APP = SOURCE_ROOT / "interfaces/api/app.py"
PUBLIC_CLI = SOURCE_ROOT / "interfaces/cli/commands.py"
PRIVATE_CLI = SOURCE_ROOT / "interfaces/private_cli.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _parse_source(path: Path) -> ast.Module:
    return ast.parse(_read_source(path), filename=str(path))


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


def _ast_calls(tree: ast.Module) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _is_app_route_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
        and node.func.attr in {"get", "post", "put", "patch", "delete"}
    )


def _is_add_typer_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_typer"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    )


def _is_typer_command_decorator(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "command"
        and isinstance(node.func.value, ast.Name)
    )


def _decorator_owner_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    raise AssertionError("decorator_owner_missing")


def _first_string_arg(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    raise AssertionError("first_string_arg_missing")


def _keyword_string(node: ast.Call, keyword_name: str) -> str:
    for keyword in node.keywords:
        if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    raise AssertionError(f"keyword_string_missing: {keyword_name}")


def _is_typer_echo_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "typer"
        and node.func.attr == "echo"
    )


def _is_sanitize_shared_text_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sanitize_shared_text"
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
def test_retired_top_level_source_packages_are_absent() -> None:
    retired = {
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


def test_retired_csmar_compatibility_entrypoints_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "sources/csmar/contracts.py",
        SOURCE_ROOT / "sources/csmar" / ("build" + "_interim.py"),
        SOURCE_ROOT / "sources/csmar/columns.py",
        SOURCE_ROOT / "sources/csmar/importer.py",
        SOURCE_ROOT / "sources/csmar/mappers.py",
        SOURCE_ROOT / "sources/csmar/dataset_contracts.py",
        SOURCE_ROOT / "markets/implementations/a_share/processed/migration.py",
        SOURCE_ROOT / "reports/aggregates_verification.py",
        SOURCE_ROOT / "research/evaluations/parity.py",
        SOURCE_ROOT / "runtime/sanitization.py",
        SOURCE_ROOT / "storage/privacy.py",
        SOURCE_ROOT / "workflows/specs.py",
    )

    assert [str(path) for path in retired if path.exists()] == []


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


def test_a_share_materialization_validation_owner_is_split_from_entrypoint() -> None:
    owner_path = (
        SOURCE_ROOT
        / "markets/implementations/a_share/processed/materialization_validation.py"
    )
    entrypoint_path = (
        SOURCE_ROOT / "markets/implementations/a_share/processed/materializer.py"
    )
    owner_required = {
        "_ProcessedInputContract",
        "validate_active_processed_profile_frequencies",
        "validate_processed_dependency_graph",
        "validate_processed_input_contract",
        "annotation_input_contract",
        "factor_input_contract",
        "label_input_contract",
        "require_source_input_columns",
        "require_source_input_columns_from_schema",
        "require_source_inputs_for_spec",
    }

    owner_defined = _top_level_defined_names(_parse_source(owner_path))
    entrypoint_defined = _top_level_defined_names(_parse_source(entrypoint_path))

    assert sorted(owner_required - owner_defined) == []
    assert sorted(owner_required & entrypoint_defined) == []


def test_a_share_materialization_io_owner_is_split_from_entrypoint() -> None:
    owner_path = (
        SOURCE_ROOT / "markets/implementations/a_share/processed/materialization_io.py"
    )
    entrypoint_path = (
        SOURCE_ROOT / "markets/implementations/a_share/processed/materializer.py"
    )
    owner_required = {
        "iter_sorted_output_chunks",
        "processed_root",
        "read_processed_inputs_for_spec",
        "write_and_inspect",
    }

    owner_defined = _top_level_defined_names(_parse_source(owner_path))
    entrypoint_defined = _top_level_defined_names(_parse_source(entrypoint_path))

    assert sorted(owner_required - owner_defined) == []
    assert sorted(owner_required & entrypoint_defined) == []


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
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
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
    for path in python_files_under(checked_roots):
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
    for path in python_files_under(checked_roots):
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
    assert (
        "sources/csmar/interim.py"
        in (source_local_change_policy["new A-share CSMAR interim product"])
    )


