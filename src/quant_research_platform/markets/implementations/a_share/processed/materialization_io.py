"""A-share processed materialization IO helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import (
    closing_dataset_iterator,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def write_and_inspect(
    spec: ProcessedDataSpec,
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    root = processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = _iter_disk_guarded_output_chunks(
        iter_sorted_output_chunks(frame, spec),
        scratch_parent=root.parent,
    )
    with runtime_resource_policy_scope(resource_policy):
        write_dataset(root, output, spec=spec)
        inspection = inspect_dataset(root, time_window=time_window)
    written[spec.data_id] = inspection
    return inspection


def processed_root(layout: StorageLayout, spec: ProcessedDataSpec) -> Path:
    return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)


def read_processed_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow | None = None,
    columns_by_data_id: dict[str, Sequence[str]] | None = None,
    retained_payload_bytes: int = 0,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    with runtime_resource_policy_scope(resource_policy):
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
                retained_payload_bytes=(
                    retained_payload_bytes
                    + sum(_frame_bytes(frame) for frame in inputs.values())
                ),
            )
    return inputs


def iter_sorted_output_chunks(
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    spec: ProcessedDataSpec,
) -> Iterable[pd.DataFrame]:
    frames = iter((frame,)) if isinstance(frame, pd.DataFrame) else iter(frame)
    with closing_dataset_iterator(frames):
        for chunk in frames:
            output = chunk.loc[:, list(spec.required_columns)]
            if output.empty:
                del output, chunk
                continue
            output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
            del chunk
            yield output
            del output


def _iter_disk_guarded_output_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    scratch_parent: Path,
) -> Iterator[pd.DataFrame]:
    chunks = iter(chunks)
    with closing_dataset_iterator(chunks):
        for chunk in chunks:
            chunk_bytes = pandas_frame_payload_bytes(chunk)
            require_free_disk_headroom(
                scratch_parent,
                reserve_bytes=chunk_bytes,
            )
            del chunk_bytes
            yield chunk
            del chunk


def read_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    market_bars_data_id: str | None = None,
    columns_by_data_id: dict[str, Sequence[str]] | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    with runtime_resource_policy_scope(resource_policy):
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
                retained_payload_bytes=sum(
                    _frame_bytes(frame) for frame in inputs.values()
                ),
            )
    return inputs


def read_materialized_dataset(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    retained_payload_bytes: int = 0,
) -> pd.DataFrame:
    if retained_payload_bytes < 0:
        raise ValueError("invalid_a_share_retained_payload_bytes")
    available_bytes = resource_policy.task_memory_budget_bytes - retained_payload_bytes
    if available_bytes <= 0:
        raise ValueError("a_share_known_payloads_exceed_task_memory_budget")
    with open_dataset_session(
        root,
        columns=None if columns is None else tuple(columns),
        time_window=time_window,
        resource_policy=resource_policy,
    ) as session:
        return materialize_dataset_session(
            session,
            target_chunk_bytes=resource_policy.payload_target_bytes,
            materialized_budget_bytes=available_bytes,
        )


def iter_dataset_chunks_with_policy(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    target_chunk_bytes: int | None = None,
    _retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = (
        resource_policy.payload_target_bytes
        if target_chunk_bytes is None
        else target_chunk_bytes
    )
    if type(target_bytes) is not int or target_bytes <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    if target_bytes > resource_policy.payload_target_bytes:
        raise ValueError("a_share_indivisible_market_unit_exceeds_payload_target")
    retained_payload_bytes = _retained_payload_bytes

    def observe_batch(batch, _frame: pd.DataFrame) -> None:  # noqa: ANN001
        assert retained_payload_bytes is not None
        retained_payload_bytes[0] = batch.get_total_buffer_size()

    with open_dataset_session(
        root,
        columns=None if columns is None else tuple(columns),
        time_window=time_window,
        resource_policy=resource_policy,
    ) as session:
        yield from iter_dataset_chunks(
            session,
            target_chunk_bytes=target_bytes,
            batch_observer=(
                observe_batch if retained_payload_bytes is not None else None
            ),
        )


def _frame_bytes(frame: pd.DataFrame) -> int:
    return pandas_frame_payload_bytes(frame)


__all__ = [
    "iter_sorted_output_chunks",
    "processed_root",
    "read_processed_inputs_for_spec",
    "read_source_inputs_for_spec",
    "write_and_inspect",
]
