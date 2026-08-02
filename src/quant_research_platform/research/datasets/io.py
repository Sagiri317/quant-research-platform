"""Research dataset IO helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


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
    return chunks
