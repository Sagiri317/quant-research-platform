"""Research dataset assembly from processed datasets."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import TimeWindow, is_intraday_time_window
from quant_research_platform.markets.session_windows import (
    SessionWindowPlanner,
)
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.runtime._resource_accounting import (
    python_list_container_bytes,
    python_tuple_container_bytes,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_reading import (
    build_dataset_read_context,
    iter_dataset_chunks,
    materialize_dataset_context,
)
from quant_research_platform.storage._dataset_schema import dataset_identity_payload
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20
# Research assembly temporarily holds merge/sort/write copies of the joined frame,
# so session sizing must reserve peak headroom beyond the steady-state frame size.
RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER = 4


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
    policy = resource_policy or active_runtime_resource_policy()
    policy.require_valid_chunk_budget()
    require_free_disk_headroom(
        policy,
        layout.project_root,
        reserve_bytes=policy.effective_chunk_bytes * 2,
    )
    _validate_research_input_specs(spec, processed_catalog)
    input_identities = _input_dataset_identities(
        layout=layout,
        dataset_spec=spec,
        processed_catalog=processed_catalog,
    )
    artifact_root = layout.research_dataset_root(spec.market_id, spec.data_id)
    require_free_disk_headroom(
        policy,
        artifact_root.parent,
        reserve_bytes=2 * policy.effective_chunk_bytes,
    )
    chunks = _iter_disk_guarded_output_chunks(
        _iter_joined_processed_chunks(
            layout=layout,
            dataset_spec=spec,
            processed_catalog=processed_catalog,
            time_window=time_window,
            resource_policy=policy,
        ),
        policy=policy,
        scratch_parent=artifact_root.parent,
    )
    write_dataset(
        artifact_root,
        chunks,
        spec=spec,
        instance_metadata={"input_dataset_identities": input_identities},
    )
    return artifact_root


def _iter_disk_guarded_output_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    policy: RuntimeResourcePolicy,
    scratch_parent: Path,
) -> Iterator[pd.DataFrame]:
    for chunk in chunks:
        chunk_bytes = int(chunk.memory_usage(index=True, deep=True).sum())
        require_free_disk_headroom(
            policy,
            scratch_parent,
            reserve_bytes=(
                2 * policy.effective_chunk_bytes
                + RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER * chunk_bytes
            ),
        )
        yield chunk
    require_free_disk_headroom(
        policy,
        scratch_parent,
        reserve_bytes=2 * policy.effective_chunk_bytes,
    )


def _iter_joined_processed_chunks(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy | None,
) -> Iterator[pd.DataFrame]:
    policy = resource_policy or active_runtime_resource_policy()
    base_data_id = _base_input_data_id(dataset_spec)
    base_spec = processed_catalog.get(base_data_id)
    base_root = layout.processed_data_root(
        base_spec.market_id,
        base_spec.kind,
        base_spec.data_id,
    )
    time_keys = _time_keys_from_base_input(
        base_root,
        primary_key=dataset_spec.primary_key,
        time_window=time_window,
        resource_policy=policy,
    )
    if not time_keys:
        raise ValueError("no_research_dataset_sessions")
    estimated = _estimate_research_join_bytes_per_session(
        layout=layout,
        dataset_spec=dataset_spec,
        processed_catalog=processed_catalog,
        time_keys=time_keys,
        time_window=time_window,
        base_data_id=base_data_id,
        resource_policy=policy,
    )
    resource_plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=len(time_keys),
        estimated_bytes_per_session=estimated,
    )
    chunk_windows = _research_chunk_windows(
        time_keys,
        time_window=time_window,
        target_core_count=resource_plan.target_core_session_count,
        reason="research_assembly",
    )

    for chunk_window in chunk_windows:
        chunk = _join_processed_inputs(
            layout=layout,
            dataset_spec=dataset_spec,
            processed_catalog=processed_catalog,
            time_window=chunk_window,
            base_data_id=base_data_id,
            resource_policy=policy,
        )
        validate_research_dataset(chunk, dataset_spec)
        output = (
            chunk.loc[:, list(_ordered_output_columns(dataset_spec))]
            .sort_values(list(dataset_spec.primary_key))
            .reset_index(drop=True)
        )
        _require_research_final_write_working_set(
            output,
            resource_policy=policy,
        )
        yield output


def _join_processed_inputs(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    base_data_id: str | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame:
    ordered_input_ids = _ordered_input_data_ids(
        dataset_spec,
        base_data_id=base_data_id,
    )
    assembled: pd.DataFrame | None = None
    base_keys: pd.DataFrame | None = None
    for input_data_id in ordered_input_ids:
        processed_spec = processed_catalog.get(input_data_id)
        input_frame = _read_processed_input(
            layout=layout,
            dataset_spec=dataset_spec,
            processed_spec=processed_spec,
            existing_columns=tuple(assembled.columns) if assembled is not None else (),
            time_window=time_window,
            resource_policy=resource_policy,
        )
        if assembled is None:
            _require_research_join_working_set(
                retained_bytes=0,
                input_bytes=_frame_bytes(input_frame),
                base_key_bytes=_frame_bytes(
                    input_frame.loc[:, list(dataset_spec.primary_key)]
                ),
                resource_policy=resource_policy,
            )
            assembled = input_frame
            base_keys = input_frame.loc[:, list(dataset_spec.primary_key)].copy()
            continue
        assert base_keys is not None
        retained_bytes = _frame_bytes(assembled)
        input_bytes = _frame_bytes(input_frame)
        base_key_bytes = _frame_bytes(base_keys)
        _require_research_join_working_set(
            retained_bytes=retained_bytes,
            input_bytes=input_bytes,
            base_key_bytes=base_key_bytes,
            include_coverage_working_set=True,
            resource_policy=resource_policy,
        )
        _validate_input_key_coverage(
            base_keys,
            input_frame,
            dataset_spec=dataset_spec,
            input_data_id=input_data_id,
        )
        _require_research_join_working_set(
            retained_bytes=retained_bytes,
            input_bytes=input_bytes,
            base_key_bytes=base_key_bytes,
            resource_policy=resource_policy,
        )
        assembled = assembled.merge(
            input_frame,
            on=list(dataset_spec.primary_key),
            how="left",
            validate="one_to_one",
        )
    if assembled is None:
        raise ValueError("empty_research_dataset_inputs")
    del base_keys
    if 2 * _frame_bytes(assembled) > resource_policy.effective_chunk_bytes:
        raise ValueError("research_join_working_set_exceeds_memory_budget")
    assembled = _apply_feature_transforms(assembled, dataset_spec)
    if 2 * _frame_bytes(assembled) > resource_policy.effective_chunk_bytes:
        raise ValueError("research_join_working_set_exceeds_memory_budget")
    return assembled.loc[:, list(_ordered_output_columns(dataset_spec))]


def _require_research_join_working_set(
    *,
    retained_bytes: int,
    input_bytes: int,
    base_key_bytes: int,
    include_coverage_working_set: bool = False,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    join_peak_bytes = (
        2 * (retained_bytes + input_bytes)
        + base_key_bytes
    )
    coverage_peak_bytes = (
        retained_bytes
        + 3 * input_bytes
        + 2 * base_key_bytes
    )
    required_bytes = (
        max(join_peak_bytes, coverage_peak_bytes)
        if include_coverage_working_set
        else join_peak_bytes
    )
    if required_bytes > resource_policy.effective_chunk_bytes:
        raise ValueError("research_join_working_set_exceeds_memory_budget")


def _frame_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(index=True, deep=True).sum())


def _require_research_final_write_working_set(
    output: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER * _frame_bytes(output)
        > resource_policy.effective_chunk_bytes
    ):
        raise ValueError(
            "research_final_write_working_set_exceeds_memory_budget"
        )


def _time_keys_from_base_input(
    base_root: Path,
    *,
    primary_key: tuple[str, ...],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> tuple[pd.Timestamp, ...]:
    context = build_dataset_read_context(
        base_root,
        columns=primary_key,
        time_window=time_window,
    )
    timestamps: set[pd.Timestamp] = set()
    state_bytes = sys.getsizeof(timestamps)
    intraday = is_intraday_time_window(time_window)
    physical_target_bytes = max(1, resource_policy.effective_chunk_bytes // 2)
    target_state_bytes = resource_policy.effective_chunk_bytes - physical_target_bytes
    for chunk in iter_dataset_chunks(
        context,
        target_chunk_bytes=physical_target_bytes,
    ):
        for value in chunk["timestamp"].dropna().unique():
            timestamp = pd.Timestamp(value)
            if not intraday:
                timestamp = timestamp.normalize()
            if timestamp in timestamps:
                continue
            previous_container_bytes = sys.getsizeof(timestamps)
            timestamps.add(timestamp)
            state_bytes += (
                sys.getsizeof(timestamps)
                - previous_container_bytes
                + sys.getsizeof(timestamp)
            )
        finalization_bytes = python_list_container_bytes(len(timestamps))
        finalization_bytes += python_tuple_container_bytes(len(timestamps))
        if state_bytes + finalization_bytes > target_state_bytes:
            raise ValueError("research_time_key_state_exceeds_chunk_budget")
    return tuple(sorted(timestamps))


def _research_chunk_windows(
    time_keys: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    target_core_count: int,
    reason: str,
) -> tuple[TimeWindow, ...]:
    if not is_intraday_time_window(time_window):
        return tuple(
            window.core
            for window in SessionWindowPlanner.from_sessions(
                time_keys,
                target_core_session_count=target_core_count,
            ).windows(time_window, reason=reason)
        )
    return tuple(
        TimeWindow(chunk[0].to_pydatetime(), chunk[-1].to_pydatetime())
        for start_index in range(0, len(time_keys), target_core_count)
        if (chunk := time_keys[start_index : start_index + target_core_count])
    )


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
    base_keys: pd.DataFrame,
    input_frame: pd.DataFrame,
    *,
    dataset_spec: ResearchDatasetSpec,
    input_data_id: str,
) -> None:
    key_columns = list(dataset_spec.primary_key)
    input_keys = input_frame.loc[:, key_columns].copy()
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
    resource_policy: RuntimeResourcePolicy,
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
        context = build_dataset_read_context(
            data_root,
            time_window=time_window,
            columns=tuple(columns),
        )
        return materialize_dataset_context(
            context,
            target_chunk_bytes=resource_policy.effective_chunk_bytes,
            materialized_budget_bytes=(
                resource_policy.effective_chunk_bytes
                // RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER
            ),
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


def _input_dataset_identities(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for data_id in dataset_spec.input_data_ids:
        processed_spec = processed_catalog.get(data_id)
        root = layout.processed_data_root(
            processed_spec.market_id,
            processed_spec.kind,
            processed_spec.data_id,
        )
        inspection = inspect_dataset(root)
        identities[data_id] = dataset_identity_payload(
            inspection.spec,
            inspection.manifest,
        )
    return dict(sorted(identities.items()))


def _estimate_research_join_bytes_per_session(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_keys: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
    base_data_id: str,
    resource_policy: RuntimeResourcePolicy,
) -> int:
    sample_windows = _research_chunk_windows(
        time_keys,
        time_window=time_window,
        target_core_count=1,
        reason="research_sizing",
    )[:SIZING_SAMPLE_SESSION_CAP]
    estimates = []
    for sample_window, sample_time_key in zip(
        sample_windows,
        time_keys[: len(sample_windows)],
        strict=True,
    ):
        frame = _join_processed_inputs(
            layout=layout,
            dataset_spec=dataset_spec,
            processed_catalog=processed_catalog,
            time_window=sample_window,
            base_data_id=base_data_id,
            resource_policy=resource_policy,
        )
        estimates.append(
            _measured_bytes_per_session(
                frame,
                sample_sessions=(sample_time_key,),
                target_session_count=len(time_keys),
            )
        )
    return max(estimates, default=1)


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
    measured = max(1, ceil(max(sample_bytes_per_session, target_bytes_per_session)))
    return max(1, measured * RESEARCH_ASSEMBLY_PEAK_MEMORY_MULTIPLIER)


__all__ = ["assemble_research_dataset"]
