"""A-share processed materialization IO helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
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
) -> DatasetInspection:
    root = processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = iter_sorted_output_chunks(frame, spec)
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
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    for data_id in spec.processed_input_data_ids:
        dependency = processed_catalog.get(data_id)
        inputs[data_id] = read_dataset(
            processed_root(layout, dependency),
            time_window=time_window,
            columns=(
                None
                if columns_by_data_id is None
                else columns_by_data_id.get(data_id)
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


def read_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    market_bars_data_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    market_bars_data_id = (
        market_bars_data_id
    )
    for data_id in spec.source_input_data_ids:
        if spec.kind is ProcessedDataKind.PANEL and data_id == market_bars_data_id:
            continue
        root = layout.interim_data_root(source_id, data_id)
        inputs[data_id] = read_dataset(root)
    return inputs


__all__ = [
    "iter_sorted_output_chunks",
    "processed_root",
    "read_processed_inputs_for_spec",
    "read_source_inputs_for_spec",
    "write_and_inspect",
]
