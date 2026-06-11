from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")
FORBIDDEN_PUBLIC_STORAGE_PARAMS = {
    "partition_filter",
    "force",
    "overwrite",
    "dependency_policy",
}
FORBIDDEN_INTERIM_FIELDS = {
    "market_id",
    "asset_id",
    "eligible",
    "can_enter",
    "can_exit",
}
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
        "write_dataset": ("dataset_root", "data", "spec"),
        "read_dataset": ("dataset_root", "columns", "time_window", "mode"),
        "inspect_dataset": ("dataset_root", "time_window"),
        "fingerprint_dataset": ("dataset_root", "columns", "time_window"),
        "compare_datasets": ("left_root", "right_root", "columns", "time_window"),
    }

    for function_name, required_names in expected.items():
        parameters = inspect.signature(getattr(storage, function_name)).parameters
        assert FORBIDDEN_PUBLIC_STORAGE_PARAMS.isdisjoint(parameters), function_name
        for name in required_names:
            assert name in parameters, function_name


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
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(csmar_root)
        if module.startswith("quant_research_platform.markets.implementations.a_share")
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


def test_chunk_axis_contract_has_no_year_alias() -> None:
    resources = importlib.import_module("quant_research_platform.runtime.resources")

    assert {item.name for item in resources.ChunkAxis} == {
        "ASSET",
        "TIME",
        "PARTITION",
        "ROW_GROUP",
    }
    assert "YEAR" not in resources.ChunkAxis.__members__


def _dataclass_field_names(cls: type[object]) -> set[str]:
    assert is_dataclass(cls)
    return {field.name for field in fields(cls)}


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    imports: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)


def _spec_contains_field(spec: object, field_name: str) -> bool:
    value = getattr(spec, field_name, None)
    if value is not None:
        return True
    for attr in ("column_roles", "primary_key", "partition_columns"):
        candidate = getattr(spec, attr, ())
        if field_name in candidate:
            return True
    return False
