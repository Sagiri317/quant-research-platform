"""Crypto forward-return label materialization."""

from __future__ import annotations

from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
) -> DatasetInspection:
    panel = read_dataset(
        layout.processed_data_root(
            MARKET_ID,
            ProcessedDataKind.PANEL,
            MARKET_PANEL_DATA_ID,
        ),
        time_window=time_window,
    )
    labels = add_forward_return_labels(
        panel,
        horizons=CRYPTO_FORWARD_RETURN_HORIZONS,
        frequency=DataFrequency.MINUTE,
        execution_price_column="close",
    ).loc[:, spec.required_columns]
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    write_dataset(root, labels, spec=spec)
    return inspect_dataset(root, time_window=time_window)


__all__ = ["write_labels"]
