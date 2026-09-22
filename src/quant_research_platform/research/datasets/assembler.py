"""Research dataset assembly from processed datasets."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import TimeWindow, is_datetime_window
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.validation import (
    validate_research_dataset,
)
from quant_research_platform.research.features.transforms import (
    _feature_transform_working_bytes,
    apply_feature_transforms,
)
from quant_research_platform.research.filtering import (
    A_SHARE_ELIGIBILITY_FACTS,
    a_share_eligibility,
)
from quant_research_platform.research.protocols.a_share import SCENARIO_IDS
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    materialize_dataset_session_owned,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import (
    COMMON_MANIFEST_METADATA_KEY,
    dataset_identity_payload,
)
from quant_research_platform.storage.datasets.writing import (
    DatasetWriteReceipt,
    write_dataset,
)
from quant_research_platform.storage.identity import require_artifact_root
from quant_research_platform.storage.layout import StorageLayout


def assemble_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy | None = None,
    instance_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Assemble one registered research dataset from processed inputs."""
    with runtime_resource_policy_scope(resource_policy):
        return _assemble_research_dataset_impl(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id=dataset_id,
            time_window=time_window,
            instance_metadata=instance_metadata,
        )


def _assemble_research_dataset_impl(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    processed_catalog: ProcessedDataCatalog,
    dataset_id: str,
    time_window: TimeWindow,
    instance_metadata: Mapping[str, object] | None,
) -> Path:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    spec = dataset_catalog.get(dataset_id)
    _validate_research_input_specs(spec, processed_catalog)
    with ExitStack() as stack:
        input_sessions = _open_input_dataset_sessions(
            layout=layout,
            dataset_spec=spec,
            processed_catalog=processed_catalog,
            time_window=time_window,
            stack=stack,
        )
        return _assemble_research_dataset_from_sessions(
            layout=layout,
            spec=spec,
            processed_catalog=processed_catalog,
            time_window=time_window,
            input_sessions=input_sessions,
            instance_metadata=instance_metadata,
        )


def _assemble_research_dataset_from_sessions(
    *,
    layout: StorageLayout,
    spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    input_sessions: Mapping[str, DatasetSession],
    instance_metadata: Mapping[str, object] | None,
) -> Path:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    artifact_root = require_artifact_root(
        layout.target_root(
            spec.data_id,
            layout.research_dataset_root(spec.market_id, spec.data_id),
        )
    )
    chunks = _iter_joined_processed_chunks(
        dataset_spec=spec,
        processed_catalog=processed_catalog,
        sessions=input_sessions,
        time_window=time_window,
    )

    def verify_input_sessions(_receipt: DatasetWriteReceipt) -> None:
        for session in input_sessions.values():
            session.verify()

    metadata = dict(instance_metadata or {})
    if COMMON_MANIFEST_METADATA_KEY not in metadata:
        metadata["input_dataset_identities"] = _input_dataset_identities(
            dataset_spec=spec,
            sessions=input_sessions,
        )
        base_manifest = input_sessions[spec.base_input_data_id].manifest
        if "source_artifact_identity" in base_manifest:
            metadata["source_artifact_identity"] = base_manifest[
                "source_artifact_identity"
            ]
    write_dataset(
        artifact_root,
        chunks,
        spec=spec,
        instance_metadata=metadata,
        staged_validator=verify_input_sessions,
        resource_policy=policy,
    )
    return artifact_root


def _iter_joined_processed_chunks(
    *,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    sessions: Mapping[str, DatasetSession],
    time_window: TimeWindow,
) -> Iterator[object]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    base_data_id = _base_input_data_id(dataset_spec)
    producer_bound = max(1, policy.operation_headroom_bytes // 2)
    lease = yield producer_bound
    chunk_windows = _iter_research_chunk_windows_from_base_input(
        sessions[base_data_id],
        time_window=time_window,
    )
    with closing_iterator(iter(chunk_windows)) as active_chunk_windows:
        for window_index, chunk_window in enumerate(active_chunk_windows):
            if window_index:
                lease = yield producer_bound
            chunk = _join_processed_inputs(
                dataset_spec=dataset_spec,
                processed_catalog=processed_catalog,
                sessions=sessions,
                time_window=chunk_window,
                base_data_id=base_data_id,
            )
            chunk_bytes = pandas_frame_payload_bytes(chunk)
            if chunk_bytes + 256 * 1024 > lease.reserved_bytes:
                raise ValueError("research_joined_bound_invalid")
            validate_research_dataset(chunk, dataset_spec)
            sort_index_bytes = len(chunk) * np.dtype(np.intp).itemsize
            with policy.lease(
                "research.assembly.sort",
                sort_index_bytes + sort_index_bytes,
            ):
                physical_order = (
                    "timestamp",
                    *(key for key in dataset_spec.primary_key if key != "timestamp"),
                )
                output = chunk.sort_values(
                    list(physical_order),
                    kind="mergesort",
                    ignore_index=True,
                )
            output_bytes = pandas_frame_payload_bytes(output)
            actual_bound = chunk_bytes + output_bytes + 256 * 1024
            if actual_bound > lease.reserved_bytes:
                raise ValueError("research_joined_bound_invalid")
            lease.shrink_to(actual_bound)
            del chunk
            yield OwnedValue(output, lease)
            del output
            lease = None


def _join_processed_inputs(
    *,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    sessions: Mapping[str, DatasetSession],
    time_window: TimeWindow,
    base_data_id: str | None = None,
) -> pd.DataFrame:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    ordered_input_ids = _ordered_input_data_ids(
        dataset_spec,
        base_data_id=base_data_id,
    )
    assembled: pd.DataFrame | None = None
    base_keys: pd.DataFrame | None = None
    assembled_state = ExitStack()
    base_key_state = ExitStack()
    try:
        for input_data_id in ordered_input_ids:
            processed_spec = processed_catalog.get(input_data_id)
            session = sessions[input_data_id]
            owned_input = _read_processed_input(
                dataset_spec=dataset_spec,
                processed_spec=processed_spec,
                session=session,
                existing_columns=(
                    tuple(assembled.columns) if assembled is not None else ()
                ),
                time_window=time_window,
            )
            input_frame = owned_input.value
            input_bytes = pandas_frame_payload_bytes(input_frame)
            input_state = ExitStack()
            input_state.callback(
                owned_input.release,
                owner="research.assembly.input",
            )
            try:
                if assembled is None:
                    next_base_state = ExitStack()
                    try:
                        next_base_state.enter_context(
                            policy.lease(
                                "research.assembly.base_keys",
                                input_bytes,
                            )
                        )
                        next_base_keys = input_frame.loc[
                            :, list(dataset_spec.primary_key)
                        ].copy()
                        if pandas_frame_payload_bytes(next_base_keys) > input_bytes:
                            raise ValueError(
                                "research_known_join_payloads_exceed_task_memory_budget"
                            )
                    except BaseException as primary:
                        try:
                            next_base_state.close()
                        except BaseException as cleanup_error:
                            primary.add_note(
                                "research base-key state release also failed: "
                                f"{type(cleanup_error).__name__}"
                            )
                        raise
                    assembled = input_frame
                    assembled_state.close()
                    assembled_state = input_state.pop_all()
                    base_keys = next_base_keys
                    base_key_state.close()
                    base_key_state = next_base_state.pop_all()
                    continue

                assert base_keys is not None
                assembled_bytes = pandas_frame_payload_bytes(assembled)
                base_key_bytes = pandas_frame_payload_bytes(base_keys)
                output_bound = assembled_bytes + input_bytes
                output_state = ExitStack()
                try:
                    output_state.enter_context(
                        policy.lease(
                            "research.assembly.merge.output",
                            output_bound,
                        )
                    )
                    coverage_bound = sum(
                        (
                            input_bytes,  # input key projection
                            base_key_bytes + input_bytes,  # base index and lookup
                            base_key_bytes + input_bytes,  # input index and lookup
                            len(base_keys) + len(input_frame),  # result masks
                        )
                    )
                    with policy.lease(
                        "research.assembly.coverage",
                        coverage_bound,
                    ):
                        _validate_input_key_coverage(
                            base_keys,
                            input_frame,
                            dataset_spec=dataset_spec,
                            input_data_id=input_data_id,
                        )
                    join_index_bytes = (len(assembled) + len(input_frame)) * np.dtype(
                        np.intp
                    ).itemsize
                    with policy.lease(
                        "research.assembly.merge.workspace",
                        base_key_bytes + input_bytes + join_index_bytes,
                    ):
                        merged = assembled.merge(
                            input_frame,
                            on=list(dataset_spec.primary_key),
                            how="left",
                            validate="one_to_one",
                        )
                    if pandas_frame_payload_bytes(merged) > output_bound:
                        raise ValueError(
                            "research_known_join_payloads_exceed_task_memory_budget"
                        )
                except BaseException as primary:
                    try:
                        output_state.close()
                    except BaseException as cleanup_error:
                        primary.add_note(
                            "research merge state release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    raise
                assembled_state.close()
                assembled = merged
                assembled_state = output_state.pop_all()
            finally:
                primary = sys.exception()
                try:
                    input_state.close()
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "research input state release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )

        if assembled is None:
            raise ValueError("empty_research_dataset_inputs")
        base_key_state.close()
        base_keys = None
        assembled = _add_a_share_semantic_columns(assembled, dataset_spec)
        if dataset_spec.feature_transforms:
            transformed_state = ExitStack()
            try:
                transformed_lease = transformed_state.enter_context(
                    policy.lease(
                        "research.assembly.transformed",
                        _feature_transform_working_bytes(
                            assembled,
                            dataset_spec.feature_transforms,
                        ),
                    )
                )
                transformed = _apply_feature_transforms(
                    assembled,
                    dataset_spec,
                    resource_lease=transformed_lease,
                )
                transformed_lease.shrink_to(pandas_frame_payload_bytes(transformed))
                transformed_lease.transfer("research.assembly.transformed")
                assembled_state.close()
            except BaseException as primary:
                try:
                    transformed_state.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        "research transformed state release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                raise
            assembled = transformed
            assembled_state = transformed_state.pop_all()

        projection_bound = pandas_frame_payload_bytes(assembled)
        with policy.lease(
            "research.assembly.projection",
            projection_bound,
        ):
            output = assembled.loc[:, list(_ordered_output_columns(dataset_spec))]
            if pandas_frame_payload_bytes(output) > projection_bound:
                raise ValueError(
                    "research_known_join_payloads_exceed_task_memory_budget"
                )
            return output
    finally:
        primary = sys.exception()
        first_failure: BaseException | None = None
        try:
            base_key_state.close()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "research base-key state release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            else:
                first_failure = cleanup_error
        try:
            assembled_state.close()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "research assembled state release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            elif first_failure is None:
                first_failure = cleanup_error
            else:
                first_failure.add_note(
                    "research assembled state release also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if first_failure is not None:
            raise first_failure


def _iter_research_chunk_windows_from_base_input(
    base_session: DatasetSession,
    *,
    time_window: TimeWindow,
) -> Iterator[TimeWindow]:
    """Partition the complete request domain using ordered base-chunk boundaries."""

    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    intraday = is_datetime_window(time_window)
    last_timestamp: pd.Timestamp | None = None
    pending: TimeWindow | None = None
    with base_session.filtered_view(time_window) as session:
        with closing_iterator(
            iter_dataset_chunks(
                session,
                target_chunk_bytes=policy.operation_headroom_bytes,
            )
        ) as chunks:
            for chunk in chunks:
                timestamps = chunk["timestamp"]
                timestamp_bytes = int(timestamps.memory_usage(index=True, deep=True))
                with policy.lease(
                    "research.assembler.timestamp.windows",
                    max(1, timestamp_bytes + len(timestamps) * 32),
                ):
                    values = timestamps.dropna().unique()
                    chunk_timestamps = pd.DatetimeIndex(values).sort_values()
                    if not intraday:
                        chunk_timestamps = chunk_timestamps.normalize()
                    chunk_timestamps = chunk_timestamps.drop_duplicates()
                    if chunk_timestamps.empty:
                        continue
                    if last_timestamp is not None:
                        current = pd.Timestamp(chunk_timestamps[0])
                        if current < last_timestamp:
                            raise ValueError("non_monotonic_research_time_keys")
                        chunk_timestamps = chunk_timestamps[
                            chunk_timestamps > last_timestamp
                        ]
                    if chunk_timestamps.empty:
                        continue
                    start = (
                        pd.Timestamp(time_window.start)
                        if last_timestamp is None
                        else last_timestamp
                        + pd.Timedelta(1, unit="ns" if intraday else "D")
                    )
                    end = pd.Timestamp(chunk_timestamps[-1])
                    last_timestamp = end
                    if intraday:
                        window = TimeWindow(
                            start,
                            end,
                        )
                    else:
                        window = TimeWindow(start.date(), end.date())
                if pending is not None:
                    yield pending
                pending = window
    if pending is None:
        raise ValueError("no_research_dataset_sessions")
    yield TimeWindow(pending.start, time_window.end)


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
    input_keys = input_frame.loc[:, key_columns]
    policy = dataset_spec.key_coverage.get(input_data_id, "strict")
    if input_keys.equals(base_keys) and policy in {"strict", "sparse"}:
        return
    base_index = pd.MultiIndex.from_frame(base_keys)
    input_index = pd.MultiIndex.from_frame(input_keys)
    extra = ~input_index.isin(base_index)
    if bool(extra.any()):
        raise ValueError(
            f"extra_research_input_keys: {input_data_id}: {int(extra.sum())}"
        )
    if policy == "sparse":
        return
    if policy != "strict":
        raise ValueError(f"invalid_research_key_coverage: {input_data_id}={policy}")
    missing = ~base_index.isin(input_index)
    if bool(missing.any()):
        raise ValueError(
            f"missing_research_input_keys: {input_data_id}: {int(missing.sum())}"
        )


def _read_processed_input(
    *,
    dataset_spec: ResearchDatasetSpec,
    processed_spec: ProcessedDataSpec,
    session: DatasetSession,
    existing_columns: tuple[str, ...],
    time_window: TimeWindow,
) -> OwnedValue[pd.DataFrame]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    columns = _processed_input_columns(
        dataset_spec,
        processed_spec,
        existing_columns=existing_columns,
    )
    if tuple(columns) != session.logical_columns:
        raise ValueError("research_input_session_columns_mismatch")
    with session.filtered_view(time_window) as view:
        return materialize_dataset_session_owned(
            view,
            target_chunk_bytes=policy.operation_headroom_bytes,
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
        tuple(
            column
            for column in dataset_spec.label_columns
            if column not in _generated_semantic_columns(dataset_spec)
        ),
        role_by_column=role_by_column,
        kind_by_column=kind_by_column,
        allowed_roles={ColumnRole.LABEL},
        allowed_kinds=set(),
        error_name="invalid_research_label_columns",
    )
    _require_columns_from_roles_or_kinds(
        tuple(
            column
            for column in dataset_spec.filter_columns
            if column not in _generated_semantic_columns(dataset_spec)
        ),
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
        .difference(
            _generated_feature_columns(dataset_spec)
            + _generated_semantic_columns(dataset_spec)
        )
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


def _generated_semantic_columns(
    dataset_spec: ResearchDatasetSpec,
) -> tuple[str, ...]:
    if dataset_spec.data_id not in SCENARIO_IDS:
        return ()
    return tuple(
        column
        for column in ("eligible", "model_target")
        if column in _ordered_output_columns(dataset_spec)
    )


def _add_a_share_semantic_columns(
    frame: pd.DataFrame,
    dataset_spec: ResearchDatasetSpec,
) -> pd.DataFrame:
    if dataset_spec.data_id not in SCENARIO_IDS:
        return frame
    missing = sorted(set(A_SHARE_ELIGIBILITY_FACTS).difference(frame.columns))
    if missing:
        raise ValueError("a_share_eligibility_facts_missing: " + ", ".join(missing))
    result = frame.copy()
    result["eligible"] = a_share_eligibility(result, dataset_spec.data_id)
    if "model_target" in dataset_spec.label_columns:
        if "forward_return_20d" not in result.columns:
            raise ValueError("a_share_model_target_label_missing")
        result["model_target"] = _a_share_model_target(result)
    return result


def _a_share_model_target(frame: pd.DataFrame) -> pd.Series:
    values = strict_finite_numeric_series(
        frame["forward_return_20d"],
        error="invalid_a_share_model_target_label",
        allow_null=True,
    ).astype("float64")
    eligible = strict_bool_series(
        frame["eligible"],
        error="invalid_a_share_eligibility",
    )
    target = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _timestamp, positions in frame.groupby("timestamp", sort=False).groups.items():
        valid = positions[eligible.loc[positions] & values.loc[positions].notna()]
        if len(valid) < 2:
            continue
        valid_values = values.loc[valid].to_numpy(dtype="float64", copy=False)
        if (valid_values == valid_values[0]).all():
            continue
        target.loc[valid] = _stable_zscore(values.loc[valid])
    return target


def _apply_feature_transforms(
    frame: pd.DataFrame,
    dataset_spec: ResearchDatasetSpec,
    *,
    resource_lease: MemoryLease | None = None,
) -> pd.DataFrame:
    if not dataset_spec.feature_transforms:
        return frame
    return apply_feature_transforms(
        frame,
        dataset_spec.feature_transforms,
        resource_lease=resource_lease,
    )


def _input_dataset_identities(
    *,
    dataset_spec: ResearchDatasetSpec,
    sessions: Mapping[str, DatasetSession],
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for data_id in dataset_spec.input_data_ids:
        session = sessions[data_id]
        if session.spec.get("data_id") != data_id:
            raise ValueError("invalid_research_input_dataset_identity")
        identities[data_id] = dataset_identity_payload(
            session.spec,
            session.manifest,
        )
    return dict(sorted(identities.items()))


def _open_input_dataset_sessions(
    *,
    layout: StorageLayout,
    dataset_spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow,
    stack: ExitStack,
) -> dict[str, DatasetSession]:
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    sessions: dict[str, DatasetSession] = {}
    existing_columns: tuple[str, ...] = ()
    for data_id in _ordered_input_data_ids(dataset_spec, base_data_id=None):
        processed_spec = processed_catalog.get(data_id)
        columns = _processed_input_columns(
            dataset_spec,
            processed_spec,
            existing_columns=existing_columns,
        )
        root = layout.target_root(
            data_id,
            layout.processed_data_root(
                processed_spec.market_id,
                processed_spec.kind,
                processed_spec.data_id,
            ),
        )
        try:
            session = stack.enter_context(
                open_dataset_session(
                    root,
                    columns=columns,
                    time_window=time_window,
                    resource_policy=policy,
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
        sessions[data_id] = session
        existing_columns = tuple(dict.fromkeys((*existing_columns, *columns)))
    return sessions
