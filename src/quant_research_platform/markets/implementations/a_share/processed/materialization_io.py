"""A-share processed materialization IO helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    bind_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
    inspect_dataset,
)
from quant_research_platform.storage._dataset_paths import require_artifact_root
from quant_research_platform.storage._dataset_reading import (
    closing_dataset_iterator,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage._dataset_writing import (
    write_dataset as _write_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.layout import StorageLayout


class MaterializationCapacityExceeded(ValueError):
    """Typed owner-local signal for adaptive materialization capacity."""


def write_and_inspect(
    spec: ProcessedDataSpec,
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    retained_written_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in written.values()
    )
    root = processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = iter_sorted_output_chunks(
        frame,
        spec,
        resource_policy=resource_policy,
    )
    with runtime_resource_policy_scope(resource_policy):
        _write_dataset(
            root,
            output,
            spec=spec,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_written_bytes,
        )
        inspection = inspect_dataset(
            root,
            time_window=time_window,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_written_bytes,
        )
    written[spec.data_id] = inspection
    return inspection


def processed_root(layout: StorageLayout, spec: ProcessedDataSpec) -> Path:
    return require_artifact_root(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    )


def read_processed_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow | None = None,
    columns_by_data_id: dict[str, Sequence[str]] | None = None,
    retained_payload_bytes: int = 0,
    asset_ids: tuple[object, ...] | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    with (
        runtime_resource_policy_scope(resource_policy),
        ExitStack() as retained_inputs,
    ):
        for data_id in spec.processed_input_data_ids:
            dependency = processed_catalog.get(data_id)
            inputs[data_id] = read_materialized_dataset(
                processed_root(layout, dependency),
                resource_policy=resource_policy,
                time_window=time_window,
                columns=(
                    None
                    if columns_by_data_id is None
                    else columns_by_data_id.get(data_id)
                ),
                retained_payload_bytes=retained_payload_bytes,
                asset_ids=asset_ids,
            )
            retained_inputs.enter_context(
                resource_policy.reserve(
                    "a_share.processed_input",
                    _frame_bytes(inputs[data_id]),
                )
            )
    return inputs


def iter_sorted_output_chunks(
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    spec: ProcessedDataSpec,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> Iterable[pd.DataFrame]:
    bound_policy = bind_runtime_resource_policy(resource_policy)

    def activate() -> Iterator[pd.DataFrame]:
        assert bound_policy is not None
        with runtime_resource_policy_scope(bound_policy):
            frames = iter((frame,)) if isinstance(frame, pd.DataFrame) else iter(frame)
            previous_last_key: tuple[object, ...] | None = None
            with closing_dataset_iterator(frames):
                while True:
                    try:
                        chunk = next(frames)
                    except StopIteration:
                        break
                    if tuple(chunk.columns) != spec.required_columns:
                        raise ValueError("a_share_output_columns_mismatch")
                    output = chunk
                    if output.empty:
                        del output, chunk
                        continue
                    output_bytes = pandas_frame_payload_bytes(output)
                    key_temporary_bytes = len(output) * (
                        len(spec.primary_key) * np.dtype(object).itemsize
                        + 2 * np.dtype(np.intp).itemsize
                    )
                    if (
                        bound_policy.reserved_memory_bytes
                        + output_bytes
                        + key_temporary_bytes
                        > bound_policy.task_memory_budget_bytes
                    ):
                        raise MaterializationCapacityExceeded(
                            "a_share_known_payloads_exceed_task_memory_budget"
                        )
                    with bound_policy.reserve(
                        "a_share.output.order",
                        key_temporary_bytes,
                    ):
                        key_arrays = tuple(
                            output[column].to_numpy(copy=False)
                            for column in reversed(spec.primary_key)
                        )
                        try:
                            order = np.lexsort(key_arrays)
                        except (TypeError, ValueError):
                            raise ValueError(
                                "a_share_output_primary_key_order_regression"
                            ) from None
                        expected = np.arange(len(output), dtype=np.intp)
                        if not np.array_equal(order, expected):
                            raise ValueError(
                                "a_share_output_primary_key_order_regression"
                            )
                        first_key = tuple(array[0] for array in reversed(key_arrays))
                        try:
                            cross_chunk_regression = (
                                previous_last_key is not None
                                and bool(first_key < previous_last_key)
                            )
                        except (TypeError, ValueError):
                            cross_chunk_regression = True
                        if cross_chunk_regression:
                            raise ValueError(
                                "a_share_output_primary_key_order_regression"
                            )
                        previous_last_key = tuple(
                            array[-1] for array in reversed(key_arrays)
                        )
                        del expected, key_arrays, order
                    del chunk
                    yield output
                    del output

    return activate()


def read_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    market_bars_data_id: str | None = None,
    columns_by_data_id: dict[str, Sequence[str]] | None = None,
    asset_ids_by_data_id: dict[str, tuple[object, ...]] | None = None,
    asset_columns_by_data_id: dict[str, str] | None = None,
    retained_payload_bytes: int = 0,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    with (
        runtime_resource_policy_scope(resource_policy),
        ExitStack() as retained_inputs,
    ):
        for data_id in spec.source_input_data_ids:
            if spec.kind is ProcessedDataKind.PANEL and data_id == market_bars_data_id:
                continue
            root = layout.interim_data_root(source_id, data_id)
            inputs[data_id] = read_materialized_dataset(
                root,
                resource_policy=resource_policy,
                columns=(
                    None
                    if columns_by_data_id is None
                    else columns_by_data_id.get(data_id)
                ),
                retained_payload_bytes=retained_payload_bytes,
                asset_ids=(
                    None
                    if asset_ids_by_data_id is None
                    else asset_ids_by_data_id.get(data_id)
                ),
                asset_column=(
                    "asset_id"
                    if asset_columns_by_data_id is None
                    else asset_columns_by_data_id.get(data_id, "asset_id")
                ),
            )
            retained_inputs.enter_context(
                resource_policy.reserve(
                    "a_share.source_input",
                    _frame_bytes(inputs[data_id]),
                )
            )
    return inputs


def read_materialized_dataset(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    retained_payload_bytes: int = 0,
    asset_ids: tuple[object, ...] | None = None,
    asset_column: str = "asset_id",
) -> pd.DataFrame:
    if retained_payload_bytes < 0:
        raise ValueError("invalid_a_share_retained_payload_bytes")
    available_bytes = (
        resource_policy.task_memory_budget_bytes
        - resource_policy.reserved_memory_bytes
        - retained_payload_bytes
    )
    if available_bytes <= 0:
        raise MaterializationCapacityExceeded(
            "a_share_known_payloads_exceed_task_memory_budget"
        )
    with open_dataset_session(
        root,
        columns=None if columns is None else tuple(columns),
        time_window=time_window,
        resource_policy=resource_policy,
        row_filters=(() if asset_ids is None else ((asset_column, "in", asset_ids),)),
    ) as session:
        return materialize_dataset_session(
            session,
            target_chunk_bytes=resource_policy.remaining_memory_bytes,
            materialized_budget_bytes=available_bytes,
        )


def iter_dataset_chunks_with_policy(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    target_chunk_bytes: int | None = None,
    asset_ids: tuple[object, ...] | None = None,
    asset_column: str = "asset_id",
    retained_payload_bytes: int = 0,
    _retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    bound_policy = bind_runtime_resource_policy(resource_policy)
    if retained_payload_bytes < 0:
        raise ValueError("invalid_a_share_retained_payload_bytes")
    retained_batch_bytes = _retained_payload_bytes

    def observe_batch(batch, _frame: pd.DataFrame) -> None:  # noqa: ANN001
        assert retained_batch_bytes is not None
        retained_batch_bytes[0] = batch.get_total_buffer_size()

    def activate() -> Iterator[pd.DataFrame]:
        assert bound_policy is not None
        target_bytes = (
            bound_policy.remaining_memory_bytes
            if target_chunk_bytes is None
            else target_chunk_bytes
        )
        if type(target_bytes) is not int or target_bytes <= 0:
            raise ValueError("invalid_target_chunk_bytes")
        operation_headroom_bytes = (
            bound_policy.task_memory_budget_bytes
            - bound_policy.reserved_memory_bytes
            - retained_payload_bytes
        )
        if operation_headroom_bytes <= 0:
            raise MaterializationCapacityExceeded(
                "a_share_known_payloads_exceed_task_memory_budget"
            )
        with open_dataset_session(
            root,
            columns=None if columns is None else tuple(columns),
            time_window=time_window,
            resource_policy=bound_policy,
            row_filters=(
                () if asset_ids is None else ((asset_column, "in", asset_ids),)
            ),
        ) as session:
            yield from iter_dataset_chunks(
                session,
                target_chunk_bytes=target_bytes,
                operation_headroom_bytes=operation_headroom_bytes,
                batch_observer=(
                    observe_batch if retained_batch_bytes is not None else None
                ),
            )

    return activate()


def _frame_bytes(frame: pd.DataFrame) -> int:
    return pandas_frame_payload_bytes(frame)


__all__ = [
    "iter_sorted_output_chunks",
    "processed_root",
    "read_processed_inputs_for_spec",
    "read_source_inputs_for_spec",
    "write_and_inspect",
]
