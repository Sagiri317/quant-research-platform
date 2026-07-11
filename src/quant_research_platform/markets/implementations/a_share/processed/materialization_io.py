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
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_reading import (
    build_dataset_read_context,
    iter_dataset_chunks,
    materialize_dataset_context,
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
    require_free_disk_headroom(
        resource_policy,
        root.parent,
        reserve_bytes=2 * resource_policy.effective_chunk_bytes,
    )
    output = _iter_disk_guarded_output_chunks(
        iter_sorted_output_chunks(frame, spec),
        policy=resource_policy,
        scratch_parent=root.parent,
    )
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
    retained_bytes: int = 0,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
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
            retained_bytes=(
                retained_bytes + sum(_frame_bytes(frame) for frame in inputs.values())
            ),
        )
    return inputs


def iter_sorted_output_chunks(
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    spec: ProcessedDataSpec,
) -> Iterable[pd.DataFrame]:
    frames = [frame] if isinstance(frame, pd.DataFrame) else frame
    for chunk in frames:
        output = chunk.loc[:, list(spec.required_columns)].copy()
        if output.empty:
            continue
        yield output.sort_values(list(spec.primary_key)).reset_index(drop=True)


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
                + 4 * chunk_bytes
            ),
        )
        yield chunk
    require_free_disk_headroom(
        policy,
        scratch_parent,
        reserve_bytes=2 * policy.effective_chunk_bytes,
    )


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
            retained_bytes=sum(_frame_bytes(frame) for frame in inputs.values()),
        )
    return inputs


def read_materialized_dataset(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    retained_bytes: int = 0,
) -> pd.DataFrame:
    if retained_bytes < 0:
        raise ValueError("invalid_a_share_retained_bytes")
    available_bytes = resource_policy.effective_chunk_bytes - retained_bytes
    if available_bytes <= 0:
        raise ValueError("a_share_retained_state_exceeds_resource_budget")
    context = build_dataset_read_context(
        root,
        columns=None if columns is None else tuple(columns),
        time_window=time_window,
    )
    return materialize_dataset_context(
        context,
        target_chunk_bytes=available_bytes,
        materialized_budget_bytes=available_bytes,
    )


def iter_dataset_chunks_with_policy(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    time_window: TimeWindow | None = None,
    columns: Sequence[str] | None = None,
    target_chunk_bytes: int | None = None,
) -> Iterator[pd.DataFrame]:
    context = build_dataset_read_context(
        root,
        columns=None if columns is None else tuple(columns),
        time_window=time_window,
    )
    target_bytes = (
        resource_policy.effective_chunk_bytes
        if target_chunk_bytes is None
        else target_chunk_bytes
    )
    if target_bytes > resource_policy.effective_chunk_bytes:
        raise ValueError("a_share_physical_target_exceeds_resource_budget")
    yield from iter_dataset_chunks(
        context,
        target_chunk_bytes=target_bytes,
    )


def _frame_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(index=True, deep=True).sum())


__all__ = [
    "iter_sorted_output_chunks",
    "processed_root",
    "read_processed_inputs_for_spec",
    "read_source_inputs_for_spec",
    "write_and_inspect",
]
