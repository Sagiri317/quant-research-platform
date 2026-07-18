from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

from tests.architecture.topology_helpers import call_name
from tests.helpers.import_graph import import_edges_for_roots, is_relative_to

SOURCE_ROOT = Path("src/quant_research_platform")
PLATFORM_FLOW = Path("docs/code-map/platform-flow.md")
FORBIDDEN_PUBLIC_STORAGE_PARAMS = {
    "output_dir",
    "partition_filter",
    "force",
    "overwrite",
    "dependency_policy",
    "year_filter",
}
FORBIDDEN_INTERIM_FIELDS = {
    "market_id",
    "asset_id",
    "eligible",
    "can_enter",
    "can_exit",
}


def test_platform_flow_distinguishes_handle_checks_and_memory_diagnostics() -> None:
    text = PLATFORM_FLOW.read_text(encoding="utf-8")

    assert "same open Parquet handle" in text
    assert "session final verification reopens" in text
    assert "does not prove native memory, RSS, or the full working-set peak" in text
    assert "actual peak remains 5B" in text


DELETED_MODULE_PATHS = (
    SOURCE_ROOT / "markets/implementations/a_share/source_bindings",
    SOURCE_ROOT / "contracts/data_products.py",
    SOURCE_ROOT / "catalog/data_products.py",
    SOURCE_ROOT / "catalog/provider_bindings.py",
    SOURCE_ROOT / "contracts/masks.py",
    SOURCE_ROOT / "contracts/universes.py",
    SOURCE_ROOT / "catalog/masks.py",
    SOURCE_ROOT / "catalog/universes.py",
    SOURCE_ROOT / "markets/implementations/a_share/processed/_dataset_helpers.py",
    SOURCE_ROOT / "markets/implementations/a_share/processed/io.py",
)
DELETED_FUNCTIONS = {
    "calculate_price_volume_factor_asset_batches",
    "calculate_forward_returns_asset_batches",
    "build_panel_asset_batches_from_interim",
    "write_contexts_from_panel_dataset",
    "write_masks_from_panel_dataset",
    "write_universe_from_panel_dataset",
    "build_prediction_frame",
    "_build_prediction_frame_from_frames",
    "_collect_prediction_frame",
    "build_" + "interim_frame",
}

MATERIALIZED_COLLECTION_NAMES = {
    "artifact_parts",
    "batches",
    "buffers",
    "collected",
    "combined",
    "frames",
    "outputs",
    "parts",
    "pieces",
    "positions_for_summary",
    "records",
    "results",
    "returns_for_summary",
    "summary_frames",
    "tables",
    "materialized",
}

UNKNOWN_CONCAT_NAMES = {
    "artifact_parts",
    "batches",
    "buffers",
    "chunks",
    "collected",
    "combined",
    "frames",
    "outputs",
    "parts",
    "pieces",
    "positions_for_summary",
    "records",
    "results",
    "returns_for_summary",
    "summary_frames",
    "tables",
    "materialized",
}

RUN_RESULT_AND_RUN_ID_DERIVED_NAMES = {
    "artifact_root",
    "output_dir",
    "report_root",
    "result_dir",
    "root_name",
    "run_name",
    "verification_dir",
}

RUN_RESULT_AND_RUN_ID_SCAN_ROOTS = (
    SOURCE_ROOT / "workflows",
    SOURCE_ROOT / "reports",
    SOURCE_ROOT / "interfaces",
)

MATERIALIZED_SCAN_ALLOWLIST = {
    (
        Path("src/quant_research_platform/storage/_dataset_reading.py"),
        "materialize_dataset_session",
    ): (
        "the single internal storage materialization boundary consumes only the "
        "bounded physical iterator and is shared by every materialized caller"
    ),
    (
        Path("src/quant_research_platform/research/models/validation_split.py"),
        "trim_training_buffer",
    ): (
        "bounded training buffer combines only max_train_rows plus purge context"
    ),
    (
        Path("src/quant_research_platform/research/models/validation_split.py"),
        "apply_training_controls",
    ): "bounded in-memory training controls on an already capped frame",
    (
        Path("src/quant_research_platform/research/portfolios/runner.py"),
        "_combine_context_and_chunk",
    ): "bounded lag context carries only recent timestamps",
    (
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/panels.py"
        ),
        "compute_market_panel",
    ): (
        "per-window panel computation groups one bounded window already selected "
        "upstream"
    ),
    (
        Path(
            "src/quant_research_platform/markets/implementations/a_share/"
            "processed/panels.py"
        ),
        "compute_index_membership_frame",
    ): "per-window index membership calculation for reference-sized inputs",
    (
        Path("src/quant_research_platform/research/_memory.py"),
        "_merge_time_boundary_chunks",
    ): "bounded one-boundary merge to keep complete timestamps together",
    (
        Path("src/quant_research_platform/research/evaluations/engine.py"),
        "_combine_metric_frames",
    ): "evaluation metric summaries are small aggregate frames",
}


def test_data_spec_contract_fields_are_contracted() -> None:
    data_contracts = importlib.import_module("quant_research_platform.contracts.data")
    dataset_contracts = importlib.import_module(
        "quant_research_platform.contracts.datasets"
    )

    data_spec_fields = _dataclass_field_names(data_contracts.DataSpec)
    interim_fields = _dataclass_field_names(data_contracts.InterimDataSpec)
    processed_fields = _dataclass_field_names(data_contracts.ProcessedDataSpec)
    research_fields = _dataclass_field_names(dataset_contracts.ResearchDatasetSpec)

    assert "data_id" in data_spec_fields
    assert "market_id" not in data_spec_fields
    assert "market_id" not in interim_fields
    assert "source_id" in interim_fields
    assert "interim_data_id" not in interim_fields
    assert "processed_data_id" not in processed_fields
    assert "dataset_id" not in research_fields
    assert "processed_data_ids" not in research_fields
    assert "input_data_ids" in research_fields


def test_processed_spec_uses_kind_not_category() -> None:
    data_contracts = importlib.import_module("quant_research_platform.contracts.data")
    processed_fields = _dataclass_field_names(data_contracts.ProcessedDataSpec)

    assert "kind" in processed_fields
    assert "category" not in processed_fields
    assert {item.name for item in data_contracts.ProcessedDataKind} == {
        "PANEL",
        "ANNOTATION",
        "FACTOR",
        "LABEL",
    }
    assert {item.value for item in data_contracts.ProcessedDataKind} == {
        "panel",
        "annotation",
        "factor",
        "label",
    }


def test_time_window_contract_is_closed_and_required() -> None:
    time_contracts = importlib.import_module("quant_research_platform.contracts.time")
    signature = inspect.signature(time_contracts.TimeWindow)

    assert tuple(signature.parameters) == ("start", "end")
    window = time_contracts.TimeWindow("2014-01-01", "2016-12-31")
    assert window.start.isoformat() == "2014-01-01"
    assert window.end.isoformat() == "2016-12-31"


def test_public_storage_signatures_hide_physical_and_execution_knobs() -> None:
    storage = importlib.import_module("quant_research_platform.storage.datasets")
    expected = {
        "write_dataset": ("dataset_root", "data", "spec", "instance_metadata"),
        "read_dataset": ("dataset_root", "columns", "time_window", "mode"),
        "inspect_dataset": ("dataset_root", "time_window"),
        "fingerprint_dataset": ("dataset_root", "columns", "time_window"),
        "compare_datasets": ("left_root", "right_root", "columns", "time_window"),
    }

    for function_name, required_names in expected.items():
        parameters = inspect.signature(getattr(storage, function_name)).parameters
        assert FORBIDDEN_PUBLIC_STORAGE_PARAMS.isdisjoint(parameters), function_name
        if function_name == "write_dataset":
            assert tuple(parameters) == required_names
        else:
            for name in required_names:
                assert name in parameters, function_name


def test_research_dataset_assembler_does_not_mutate_storage_manifests() -> None:
    assembler_path = SOURCE_ROOT / "research/datasets/assembler.py"
    source = assembler_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(assembler_path))
    forbidden_imports = {
        "read_manifest",
        "replace_internal_artifact_manifest",
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert imported_names.isdisjoint(forbidden_imports)
    assert "_write_input_fingerprints" not in {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    string_constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "_manifest" not in string_constants
    assert "manifest.json" not in string_constants


def test_dataset_inspection_absorbs_planning_metadata() -> None:
    storage = importlib.import_module("quant_research_platform.storage.datasets")
    inspection_fields = _dataclass_field_names(storage.DatasetInspection)

    assert not hasattr(storage, "_dataset_planning_metadata")
    assert "_DatasetPlanningMetadata" not in vars(storage)
    assert {
        "root",
        "data_root",
        "spec",
        "columns",
        "schema_hash",
        "file_count",
        "row_count",
        "partitions",
        "partition_row_counts",
        "date_min",
        "date_max",
        "manifest",
    }.issubset(inspection_fields)


def test_csmar_source_layer_has_no_a_share_or_market_semantics() -> None:
    csmar_root = SOURCE_ROOT / "sources/csmar"
    bad_imports = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_for_roots((csmar_root,))
        if edge.module.startswith(
            "quant_research_platform.markets.implementations.a_share"
        )
    ]
    source_text_hits = []
    for path in sorted(csmar_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        source_text_hits.extend(
            f"{path}:{name}"
            for name in FORBIDDEN_INTERIM_FIELDS
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text)
        )
    interim = importlib.import_module("quant_research_platform.sources.csmar.interim")
    specs = tuple(interim.interim_specs())
    spec_hits = [
        f"{spec.data_id}:{field_name}"
        for spec in specs
        for field_name in FORBIDDEN_INTERIM_FIELDS
        if _spec_contains_field(spec, field_name)
    ]

    assert bad_imports == []
    assert source_text_hits == []
    assert specs
    assert spec_hits == []


def test_deleted_data_product_provider_mask_universe_modules_are_absent() -> None:
    existing = [path.as_posix() for path in DELETED_MODULE_PATHS if path.exists()]

    assert existing == []


def test_deleted_writer_and_chunk_functions_are_absent() -> None:
    hits = [
        f"{path}:{node.lineno}:{node.name}"
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name in DELETED_FUNCTIONS
    ]

    assert hits == []


def test_production_hot_paths_do_not_reintroduce_materialized_collectors() -> None:
    hits: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in _iter_function_defs(tree):
            if (path, function.name) in MATERIALIZED_SCAN_ALLOWLIST:
                continue
            assigned_lists = _assigned_empty_list_names(function)
            names_from_list_calls = _names_assigned_from_list_calls(function)
            names_from_iter_reads = _names_appended_from_dataset_iterators(function)
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or call_name(node.func) != "concat":
                    continue
                if not _is_pandas_concat(node):
                    continue
                target_name = _concat_argument_name(node)
                if target_name is None:
                    continue
                if target_name in names_from_list_calls:
                    hits.append(
                        f"{path}:{node.lineno}:{function.name}: "
                        f"list(iterator)->pd.concat({target_name})"
                    )
                elif target_name in names_from_iter_reads:
                    hits.append(
                        f"{path}:{node.lineno}:{function.name}: "
                        f"append(read/iter)->pd.concat({target_name})"
                    )
                elif target_name in assigned_lists.intersection(
                    MATERIALIZED_COLLECTION_NAMES
                ):
                    hits.append(
                        f"{path}:{node.lineno}:{function.name}: "
                        f"append/extend->pd.concat({target_name})"
                    )
                elif target_name in UNKNOWN_CONCAT_NAMES:
                    hits.append(
                        f"{path}:{node.lineno}:{function.name}: "
                        f"pd.concat({target_name}) has unbounded collector name"
                    )

    assert hits == []


def test_run_id_and_run_result_derived_names_are_contract_checked() -> None:
    hits: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if not any(
            is_relative_to(path, root)
            for root in RUN_RESULT_AND_RUN_ID_SCAN_ROOTS
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in _iter_function_defs(tree):
            derived_assignments = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Assign | ast.AnnAssign)
                for target in _assignment_targets(node)
                if target in RUN_RESULT_AND_RUN_ID_DERIVED_NAMES
            ]
            if not derived_assignments:
                continue
            called_names = {
                call_name(node.func)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            if called_names.isdisjoint(
                {"validate_run_id"}
            ):
                hits.extend(
                    f"{path}:{node.lineno}:{function.name}: {target_name}"
                    for node in derived_assignments
                    for target_name in _assignment_targets(node)
                )

    assert hits == []


def test_runtime_resource_contract_has_no_generic_planner_surface() -> None:
    resources = importlib.import_module("quant_research_platform.runtime.resources")

    assert not hasattr(resources, "ChunkAxis")
    assert not hasattr(resources, "ResourceChunkPlan")
    assert not hasattr(resources, "ResourceChunkPlanner")
    assert not Path(
        "src/quant_research_platform/runtime/_resource_accounting.py"
    ).exists()


def test_runtime_resource_contract_has_one_current_source_surface() -> None:
    resources = importlib.import_module("quant_research_platform.runtime.resources")

    assert _dataclass_field_names(resources.RuntimeResourcePolicy) == {
        "task_memory_budget_bytes",
        "payload_target_bytes",
    }
    assert resources.__all__ == [
        "RuntimeResourcePolicy",
        "active_runtime_resource_policy",
        "pandas_frame_payload_bytes",
        "require_free_disk_headroom",
        "runtime_resource_policy_scope",
    ]
    assert resources._RETIRED_RESOURCE_SETTINGS == {
        "QRP_MEMORY_BUDGET_FRACTION",
        "QRP_MEMORY_BUDGET_MB",
        "QRP_CHUNK_TARGET_MB",
        "QRP_MEMORY_SAFETY_FACTOR",
        "QRP_MIN_FREE_DISK_MB",
    }
    disk_parameters = inspect.signature(
        resources.require_free_disk_headroom
    ).parameters
    assert tuple(disk_parameters) == ("path", "reserve_bytes")
    assert disk_parameters["reserve_bytes"].kind is inspect.Parameter.KEYWORD_ONLY

    retired_symbols = (
        "ResourceChunkPlanner",
        "ResourceChunkPlan",
        "ChunkAxis",
        "effective_chunk_bytes",
        "memory_budget_fraction",
        "chunk_target_fraction",
        "chunk_target_cap_bytes",
        "memory_safety_factor",
        "min_free_disk_bytes",
        "_canonical_nested_value_bound",
        "_nested_json_bound",
        "getsizeof",
        "categorical_state_bytes",
        "dataset_categorical_state_exceeds_target_bytes",
    )
    runtime_source = SOURCE_ROOT / "runtime/resources.py"
    hits: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        if path == runtime_source:
            continue
        source = path.read_text(encoding="utf-8")
        for symbol in retired_symbols:
            if re.search(rf"\b{re.escape(symbol)}\b", source):
                hits.append(f"{path}:{symbol}")
        if path != runtime_source and re.search(r"\.memory_usage\s*\(", source):
            hits.append(f"{path}:direct_pandas_memory_usage")

    storage_reader_source = (
        SOURCE_ROOT / "storage/_dataset_reading.py"
    ).read_text(encoding="utf-8")
    if "target_state_bytes" in storage_reader_source:
        hits.append("storage/_dataset_reading.py:target_state_bytes")

    retired_setting_test_allowlist = {
        Path("tests/architecture/test_data_layer_boundaries.py"),
        Path("tests/runtime/test_resources.py"),
    }
    for path in Path("tests").rglob("*.py"):
        if path in retired_setting_test_allowlist:
            continue
        source = path.read_text(encoding="utf-8")
        for setting in resources._RETIRED_RESOURCE_SETTINGS:
            if setting in source:
                hits.append(f"{path}:positive_retired_setting:{setting}")

    assert hits == []


def _dataclass_field_names(cls: type[object]) -> set[str]:
    assert is_dataclass(cls)
    return {field.name for field in fields(cls)}

def _spec_contains_field(spec: object, field_name: str) -> bool:
    value = getattr(spec, field_name, None)
    if value is not None:
        return True
    for attr in ("column_roles", "primary_key", "partition_columns"):
        candidate = getattr(spec, attr, ())
        if field_name in candidate:
            return True
    return False


def _iter_function_defs(tree: ast.Module) -> tuple[ast.FunctionDef, ...]:
    return tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )


def _assigned_empty_list_names(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            if isinstance(node.value, ast.List) and not node.value.elts:
                names.update(_assignment_targets(node))
    return names


def _names_assigned_from_list_calls(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        if isinstance(node.value, ast.Call) and call_name(node.value.func) == "list":
            names.update(_assignment_targets(node))
    return names


def _names_appended_from_dataset_iterators(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"append", "extend"}:
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if not node.args or not _call_tree_reads_dataset_or_iterator(node.args[0]):
            continue
        names.add(node.func.value.id)
    return names


def _call_tree_reads_dataset_or_iterator(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            called_name = call_name(child.func)
            if called_name in {"read_dataset", "read_parquet"}:
                return True
            if called_name.startswith("iter_"):
                return True
    return False


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    if isinstance(node, ast.AnnAssign):
        targets = (node.target,)
    else:
        targets = tuple(node.targets)
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return tuple(names)


def _is_pandas_concat(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "concat"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pd"
    )


def _concat_argument_name(node: ast.Call) -> str | None:
    if not node.args:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Name):
        return argument.id
    return None
