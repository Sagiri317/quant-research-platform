"""Research dataset assembly from processed datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.storage.datasets import (
    fingerprint_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_manifest,
    replace_internal_artifact_manifest,
)


def assemble_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    time_window: TimeWindow,
) -> Path:
    """Assemble one registered research dataset from processed inputs."""

    spec = dataset_catalog.get(dataset_id)
    _validate_research_input_specs(spec, processed_catalog)
    input_fingerprints = _input_fingerprints(
        layout=layout,
        dataset_spec=spec,
        processed_catalog=processed_catalog,
        time_window=time_window,
    )
    frame = _join_processed_inputs(
        layout=layout,
        dataset_spec=spec,
        processed_catalog=processed_catalog,
        time_window=time_window,
    )
    validate_research_dataset(frame, spec)
    output = frame.loc[:, list(spec.required_columns)].copy()
    artifact_root = layout.research_dataset_root(spec.market_id, spec.data_id)
    write_dataset(artifact_root, output, spec=spec)
    _write_input_fingerprints(artifact_root, input_fingerprints)
    return artifact_root


def _join_processed_inputs(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
) -> pd.DataFrame:
    assembled: pd.DataFrame | None = None
    for input_data_id in dataset_spec.input_data_ids:
        processed_spec = processed_catalog.get(input_data_id)
        input_frame = _read_processed_input(
            layout=layout,
            dataset_spec=dataset_spec,
            processed_spec=processed_spec,
            existing_columns=tuple(assembled.columns) if assembled is not None else (),
            time_window=time_window,
        )
        if assembled is None:
            assembled = input_frame
            continue
        assembled = assembled.merge(
            input_frame,
            on=list(dataset_spec.primary_key),
            how="inner",
            validate="one_to_one",
        )
    if assembled is None:
        raise ValueError("empty_research_dataset_inputs")
    return assembled.loc[:, list(dataset_spec.required_columns)]


def _read_processed_input(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    existing_columns: tuple[str, ...],
    time_window: TimeWindow,
) -> pd.DataFrame:
    columns = _processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=existing_columns,
    )
    data_root = layout.processed_data_root(
        processed_spec.market_id,
        processed_spec.kind,
        processed_spec.data_id,
    )
    return read_dataset(
        data_root,
        time_window=time_window,
        columns=columns,
    )


def _processed_input_columns(
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    *,
    existing_columns: tuple[str, ...],
) -> tuple[str, ...]:
    existing = set(existing_columns)
    output_columns = set(dataset_spec.required_columns)
    columns: list[str] = []
    for column in dataset_spec.primary_key:
        columns.append(column)
    for column in processed_spec.required_columns:
        if column in dataset_spec.primary_key:
            continue
        if column not in output_columns:
            continue
        if column in existing:
            continue
        columns.append(column)
    return tuple(columns)


def _validate_research_input_specs(
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
) -> None:
    processed_specs = {
        data_id: processed_catalog.get(data_id)
        for data_id in dataset_spec.input_data_ids
    }
    role_by_column: dict[str, set[ColumnRole]] = {}
    kind_by_column: dict[str, set[ProcessedDataKind]] = {}
    for processed_spec in processed_specs.values():
        for column, role in processed_spec.column_roles.items():
            role_by_column.setdefault(column, set()).add(role)
            kind_by_column.setdefault(column, set()).add(processed_spec.kind)

    _require_columns_from_roles_or_kinds(
        dataset_spec.feature_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FEATURE, ColumnRole.FACTOR},
        allowed_kinds={ProcessedDataKind.FACTOR},
        error_name="invalid_research_feature_columns",
    )
    _require_columns_from_roles_or_kinds(
        dataset_spec.label_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.LABEL},
        allowed_kinds=set(),
        error_name="invalid_research_label_columns",
    )
    _require_columns_from_roles_or_kinds(
        dataset_spec.filter_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.MASK, ColumnRole.UNIVERSE},
        allowed_kinds=set(),
        error_name="invalid_research_filter_columns",
    )
    _require_columns_from_roles_or_kinds(
        dataset_spec.context_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.CONTEXT},
        allowed_kinds=set(),
        error_name="invalid_research_context_columns",
    )


def _require_columns_from_roles_or_kinds(
    columns: tuple[str, ...],
    *,
    role_by_column: dict[str, set[ColumnRole]],
    kind_by_column: dict[str, set[ProcessedDataKind]],
    allowed_roles: set[ColumnRole],
    allowed_kinds: set[ProcessedDataKind],
    error_name: str,
) -> None:
    invalid = [
        column
        for column in columns
        if not role_by_column.get(column, set()).intersection(allowed_roles)
        and not kind_by_column.get(column, set()).intersection(allowed_kinds)
    ]
    if invalid:
        raise ValueError(f"{error_name}: {', '.join(invalid)}")


def _input_fingerprints(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
) -> dict[str, dict[str, object]]:
    fingerprints: dict[str, dict[str, object]] = {}
    for data_id in dataset_spec.input_data_ids:
        processed_spec = processed_catalog.get(data_id)
        root = layout.processed_data_root(
            processed_spec.market_id,
            processed_spec.kind,
            processed_spec.data_id,
        )
        fingerprints[data_id] = asdict(
            fingerprint_dataset(root, time_window=time_window)
        )
    return fingerprints


def _write_input_fingerprints(
    artifact_root: Path,
    input_fingerprints: dict[str, dict[str, object]],
) -> None:
    manifest_path = artifact_root / "_manifest" / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest["input_fingerprints"] = input_fingerprints
    replace_internal_artifact_manifest(manifest_path, manifest)


__all__ = ["assemble_research_dataset"]
