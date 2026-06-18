"""Research dataset assembly from processed datasets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import (
    SessionWindowPlanner,
    session_dates,
)
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    fingerprint_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20


def assemble_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy | None = None,
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
    chunks = _iter_joined_processed_chunks(
        layout=layout,
        dataset_spec=spec,
        processed_catalog=processed_catalog,
        time_window=time_window,
        resource_policy=resource_policy,
    )
    artifact_root = layout.research_dataset_root(spec.market_id, spec.data_id)
    write_dataset(
        artifact_root,
        chunks,
        spec=spec,
        instance_metadata={"input_fingerprints": input_fingerprints},
    )
    return artifact_root


def _iter_joined_processed_chunks(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterator[pd.DataFrame]:
    base_data_id = _base_input_data_id(dataset_spec)
    base_spec = processed_catalog.get(base_data_id)
    base_root = layout.processed_data_root(
        base_spec.market_id,
        base_spec.kind,
        base_spec.data_id,
    )
    sessions = _sessions_from_base_input(
        base_root,
        primary_key=dataset_spec.primary_key,
        time_window=time_window,
    )
    if not sessions:
        raise ValueError("no_research_dataset_sessions")
    estimated = _estimate_research_join_bytes_per_session(
        layout=layout,
        dataset_spec=dataset_spec,
        processed_catalog=processed_catalog,
        sessions=sessions,
        time_window=time_window,
        base_data_id=base_data_id,
    )
    policy = resource_policy or active_runtime_resource_policy()
    resource_plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=len(sessions),
        estimated_bytes_per_session=estimated,
    )
    windows = SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=resource_plan.target_core_session_count,
    ).windows(time_window, reason="research_assembly")

    for window in windows:
        chunk = _join_processed_inputs(
            layout=layout,
            dataset_spec=dataset_spec,
            processed_catalog=processed_catalog,
            time_window=window.core,
            base_data_id=base_data_id,
        )
        validate_research_dataset(chunk, dataset_spec)
        yield (
            chunk.loc[:, list(_ordered_output_columns(dataset_spec))]
            .sort_values(list(dataset_spec.primary_key))
            .reset_index(drop=True)
        )


def _join_processed_inputs(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    base_data_id: str | None = None,
) -> pd.DataFrame:
    ordered_input_ids = _ordered_input_data_ids(
        dataset_spec,
        base_data_id=base_data_id,
    )
    assembled: pd.DataFrame | None = None
    base_frame: pd.DataFrame | None = None
    for input_data_id in ordered_input_ids:
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
            base_frame = input_frame
            continue
        assert base_frame is not None
        _validate_input_key_coverage(
            base_frame,
            input_frame,
            dataset_spec=dataset_spec,
            input_data_id=input_data_id,
        )
        assembled = assembled.merge(
            input_frame,
            on=list(dataset_spec.primary_key),
            how="left",
            validate="one_to_one",
        )
    if assembled is None:
        raise ValueError("empty_research_dataset_inputs")
    assembled = _apply_feature_transforms(assembled, dataset_spec)
    return assembled.loc[:, list(_ordered_output_columns(dataset_spec))]


def _sessions_from_base_input(
    base_root: Path,
    *,
    primary_key: tuple[str, ...],
    time_window: TimeWindow,
) -> tuple[pd.Timestamp, ...]:
    chunks = read_dataset(
        base_root,
        columns=list(primary_key),
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    )
    timestamps: list[pd.Timestamp] = []
    for chunk in chunks:
        timestamps.extend(chunk["timestamp"].dropna().unique())
    return session_dates(timestamps)


def _base_input_data_id(dataset_spec: ResearchDatasetSpec) -> str:
    if not dataset_spec.input_data_ids:
        raise ValueError("empty_research_dataset_inputs")
    if dataset_spec.base_input_data_id not in dataset_spec.input_data_ids:
        raise ValueError(
            f"base_input_data_id_not_in_inputs: {dataset_spec.base_input_data_id}"
        )
    return dataset_spec.base_input_data_id


def _ordered_input_data_ids(
    dataset_spec: ResearchDatasetSpec,
    *,
    base_data_id: str | None,
) -> tuple[str, ...]:
    base = base_data_id or _base_input_data_id(dataset_spec)
    return (
        base,
        *(data_id for data_id in dataset_spec.input_data_ids if data_id != base),
    )


def _validate_input_key_coverage(
    base_frame: pd.DataFrame,
    input_frame: pd.DataFrame,
    *,
    dataset_spec: ResearchDatasetSpec,
    input_data_id: str,
) -> None:
    key_columns = list(dataset_spec.primary_key)
    base_keys = base_frame.loc[:, key_columns].drop_duplicates()
    input_keys = input_frame.loc[:, key_columns].drop_duplicates()
    extra = input_keys.merge(
        base_keys,
        on=key_columns,
        how="left",
        indicator=True,
    )["_merge"].eq("left_only")
    if bool(extra.any()):
        raise ValueError(
            f"extra_research_input_keys: {input_data_id}: {int(extra.sum())}"
        )
    policy = dataset_spec.key_coverage.get(input_data_id, "strict")
    if policy == "sparse":
        return
    if policy != "strict":
        raise ValueError(
            f"invalid_research_key_coverage: {input_data_id}={policy}"
        )
    missing = base_keys.merge(
        input_keys,
        on=key_columns,
        how="left",
        indicator=True,
    )["_merge"].eq("left_only")
    if bool(missing.any()):
        raise ValueError(
            f"missing_research_input_keys: {input_data_id}: {int(missing.sum())}"
        )


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
    try:
        return read_dataset(
            data_root,
            time_window=time_window,
            columns=columns,
        )
    except ValueError as exc:
        if str(exc) == "dataset_column_not_found":
            missing_keys = [
                column
                for column in dataset_spec.primary_key
                if column not in processed_spec.required_columns
            ]
            if missing_keys:
                raise ValueError(
                    f"missing_research_join_key: {processed_spec.data_id}: "
                    + ", ".join(missing_keys)
                ) from exc
        raise


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
    _base_input_data_id(dataset_spec)
    processed_specs = {
        data_id: processed_catalog.get(data_id)
        for data_id in dataset_spec.input_data_ids
    }
    for data_id, processed_spec in processed_specs.items():
        missing_key_columns = [
            column
            for column in dataset_spec.primary_key
            if column not in processed_spec.primary_key
        ]
        if missing_key_columns:
            raise ValueError(
                f"missing_research_join_key: {data_id}: "
                + ", ".join(missing_key_columns)
            )
        if processed_spec.primary_key != dataset_spec.primary_key:
            raise ValueError(f"incompatible_research_input_keys: {data_id}")
    _require_no_duplicate_output_columns(dataset_spec, processed_specs)
    _require_research_required_columns_available(dataset_spec, processed_specs)
    role_by_column: dict[str, set[ColumnRole]] = {}
    kind_by_column: dict[str, set[ProcessedDataKind]] = {}
    for processed_spec in processed_specs.values():
        for column, role in processed_spec.column_roles.items():
            role_by_column.setdefault(column, set()).add(role)
            kind_by_column.setdefault(column, set()).add(processed_spec.kind)

    _require_columns_from_roles_or_kinds(
        dataset_spec.factor_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FACTOR},
        allowed_kinds={ProcessedDataKind.FACTOR},
        error_name="invalid_research_factor_columns",
    )
    generated_feature_columns = set(_generated_feature_columns(dataset_spec))
    stored_feature_columns = tuple(
        column
        for column in dataset_spec.feature_columns
        if column not in generated_feature_columns
    )
    _require_columns_from_roles_or_kinds(
        stored_feature_columns,
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.FEATURE},
        allowed_kinds=set(),
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


def _require_no_duplicate_output_columns(
    dataset_spec: ResearchDatasetSpec,
    processed_specs: dict[str, ProcessedDataSpec],
) -> None:
    requested = (
        set(_ordered_output_columns(dataset_spec))
        .difference(dataset_spec.primary_key)
        .difference(_generated_feature_columns(dataset_spec))
    )
    producers: dict[str, list[str]] = {}
    for data_id, processed_spec in processed_specs.items():
        for column in processed_spec.required_columns:
            if column in dataset_spec.primary_key or column not in requested:
                continue
            producers.setdefault(column, []).append(data_id)
    duplicates = sorted(
        column for column, data_ids in producers.items() if len(data_ids) > 1
    )
    if duplicates:
        raise ValueError("duplicate_research_output_columns: " + ", ".join(duplicates))


def _require_research_required_columns_available(
    dataset_spec: ResearchDatasetSpec,
    processed_specs: dict[str, ProcessedDataSpec],
) -> None:
    available = {
        column
        for processed_spec in processed_specs.values()
        for column in processed_spec.required_columns
    }
    missing = sorted(
        set(_ordered_output_columns(dataset_spec))
        .difference(available)
        .difference(_generated_feature_columns(dataset_spec))
    )
    if missing:
        raise ValueError("missing_research_required_columns: " + ", ".join(missing))


def _ordered_output_columns(dataset_spec: ResearchDatasetSpec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *dataset_spec.primary_key,
                *dataset_spec.factor_columns,
                *dataset_spec.feature_columns,
                *dataset_spec.label_columns,
                *dataset_spec.filter_columns,
                *dataset_spec.context_columns,
            )
        )
    )


def _generated_feature_columns(dataset_spec: ResearchDatasetSpec) -> tuple[str, ...]:
    return tuple(
        column
        for transform in dataset_spec.feature_transforms
        for column in transform.output_columns
    )


def _apply_feature_transforms(
    frame: pd.DataFrame,
    dataset_spec: ResearchDatasetSpec,
) -> pd.DataFrame:
    if not dataset_spec.feature_transforms:
        return frame
    return apply_feature_transforms(frame, dataset_spec.feature_transforms)


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


def _estimate_research_join_bytes_per_session(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
    base_data_id: str,
) -> int:
    sample_session_count = min(len(sessions), SIZING_SAMPLE_SESSION_CAP)
    sample = SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=sample_session_count,
    ).windows(time_window, reason="research_sizing")[0]
    frame = _join_processed_inputs(
        layout=layout,
        dataset_spec=dataset_spec,
        processed_catalog=processed_catalog,
        time_window=sample.core,
        base_data_id=base_data_id,
    )
    return _measured_bytes_per_session(
        frame,
        sample_sessions=sample.core_sessions,
        target_session_count=len(sessions),
    )


def _measured_bytes_per_session(
    frame: pd.DataFrame,
    *,
    sample_sessions: tuple[pd.Timestamp, ...],
    target_session_count: int,
) -> int:
    if frame.empty:
        return 1
    total = int(frame.memory_usage(index=True, deep=True).sum())
    sample_bytes_per_session = total / max(1, len(sample_sessions))
    target_bytes_per_session = total / max(1, target_session_count)
    return max(1, ceil(max(sample_bytes_per_session, target_bytes_per_session)))


__all__ = ["assemble_research_dataset"]
