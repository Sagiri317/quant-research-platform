"""Research dataset IO helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def read_research_dataset(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
    time_window: TimeWindow | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    artifact_root = layout.research_dataset_root(market_id, dataset_id)
    return read_dataset(
        artifact_root,
        time_window=time_window,
        columns=columns,
    )


def iter_research_dataset_chunks(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
    time_window: TimeWindow | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    artifact_root = layout.research_dataset_root(market_id, dataset_id)
    chunks = read_dataset(
        artifact_root,
        time_window=time_window,
        columns=columns,
        mode=DatasetReadMode.CHUNKED,
    )
    if isinstance(chunks, pd.DataFrame):
        raise TypeError("research_dataset_chunk_reader_not_iterable")
    yield from chunks


def inspect_research_dataset(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
    time_window: TimeWindow | None = None,
) -> DatasetInspection:
    return inspect_dataset(
        layout.research_dataset_root(market_id, dataset_id),
        time_window=time_window,
    )


def read_research_dataset_spec_payload(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
) -> dict[str, Any]:
    return inspect_research_dataset(
        layout,
        market_id=market_id,
        dataset_id=dataset_id,
    ).spec


def read_research_dataset_manifest(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
) -> dict[str, Any]:
    return inspect_research_dataset(
        layout,
        market_id=market_id,
        dataset_id=dataset_id,
    ).manifest


__all__ = [
    "inspect_research_dataset",
    "read_research_dataset",
    "read_research_dataset_manifest",
    "read_research_dataset_spec_payload",
]
