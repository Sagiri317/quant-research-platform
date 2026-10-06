"""Research dataset IO helpers."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.layout import StorageLayout


def iter_research_dataset_chunks(
    layout: StorageLayout,
    *,
    market_id: MarketId,
    dataset_id: str,
    time_window: TimeWindow | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
    access_window: str = "development",
) -> Iterator[pd.DataFrame]:
    time_window = _research_dataset_access_window(market_id, time_window, access_window)
    artifact_root = layout.target_root(
        dataset_id,
        layout.research_dataset_root(market_id, dataset_id),
    )
    chunks = read_dataset(
        artifact_root,
        time_window=time_window,
        columns=columns,
        mode=DatasetReadMode.CHUNKED,
    )
    if isinstance(chunks, pd.DataFrame):
        raise TypeError("research_dataset_chunk_reader_not_iterable")
    return chunks


def _research_dataset_access_window(
    market_id: MarketId,
    time_window: TimeWindow | None,
    access_window: str = "development",
) -> TimeWindow | None:
    if str(market_id) == "a_share":
        from quant_research_platform.research.protocols.a_share import (
            DEVELOPMENT_END,
            FORMAL_DATASET_START,
            HOLDOUT_END,
            HOLDOUT_START,
        )

        if access_window not in {"development", "holdout"}:
            raise ValueError("unknown_a_share_access_window")
        allowed = TimeWindow(
            FORMAL_DATASET_START if access_window == "development" else HOLDOUT_START,
            DEVELOPMENT_END if access_window == "development" else HOLDOUT_END,
        )
        if time_window is None:
            time_window = allowed
        elif pd.Timestamp(time_window.start) < pd.Timestamp(
            allowed.start
        ) or pd.Timestamp(time_window.end) > pd.Timestamp(allowed.end):
            raise PermissionError("research_dataset_access_window_forbidden")
    return time_window
