"""Crypto microstructure factor materialization."""

from __future__ import annotations

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.research.factors.microstructure import (
    calculate_microstructure_factors,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def write_factors(
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
    factors = calculate_microstructure_factors(panel).loc[:, spec.required_columns]
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    write_dataset(root, factors, spec=spec)
    return inspect_dataset(root, time_window=time_window)


__all__ = ["write_factors"]
