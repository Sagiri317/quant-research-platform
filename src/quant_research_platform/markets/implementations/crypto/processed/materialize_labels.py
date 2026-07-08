"""Crypto forward-return label materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

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
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage._dataset_reading import filter_frame_by_time_window
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

LABEL_LOOKAHEAD_BARS = max(CRYPTO_FORWARD_RETURN_HORIZONS) + 1
_CORE_ROW = "_qrp_core_row"


def write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    resource_policy.require_valid_chunk_budget()
    panel_root = layout.processed_data_root(
        MARKET_ID,
        ProcessedDataKind.PANEL,
        MARKET_PANEL_DATA_ID,
    )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    write_dataset(
        root,
        _iter_label_chunks(
            spec,
            panel_root=panel_root,
            time_window=time_window,
        ),
        spec=spec,
    )
    return inspect_dataset(root, time_window=time_window)


def _iter_label_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_root,
    time_window: TimeWindow,
) -> Iterator[pd.DataFrame]:
    buffer = pd.DataFrame()
    for panel in read_dataset(
        panel_root,
        time_window=materialization_dependencies.label_read_window(time_window),
        mode=DatasetReadMode.CHUNKED,
    ):
        if panel.empty:
            continue
        panel = panel.copy()
        panel[_CORE_ROW] = _core_mask(panel, time_window)
        buffer = _ordered_panel(
            panel if buffer.empty else pd.concat((buffer, panel), ignore_index=True)
        )
        emit_mask = _label_emit_mask(buffer) & buffer[_CORE_ROW].astype("boolean")
        if not emit_mask.any():
            continue
        labeled = _label_frame(buffer)
        yield labeled.loc[emit_mask].loc[:, spec.required_columns]
        buffer = buffer.loc[~emit_mask].copy()
    if not buffer.empty:
        labeled = _label_frame(buffer)
        core_mask = buffer[_CORE_ROW].astype("boolean")
        if core_mask.any():
            yield labeled.loc[core_mask].loc[:, spec.required_columns]


def _label_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return add_forward_return_labels(
        panel,
        horizons=CRYPTO_FORWARD_RETURN_HORIZONS,
        frequency=DataFrequency.MINUTE,
        execution_price_column="close",
        expected_interval=pd.Timedelta(minutes=1),
        require_contiguous=True,
    )


def _label_emit_mask(panel: pd.DataFrame) -> pd.Series:
    positions = panel.groupby("asset_id", sort=False).cumcount()
    group_sizes = panel.groupby("asset_id", sort=False)["asset_id"].transform("size")
    return positions.lt(group_sizes - LABEL_LOOKAHEAD_BARS)


def _ordered_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)


def _core_mask(frame: pd.DataFrame, time_window: TimeWindow) -> pd.Series:
    spec = {
        "column_roles": {"timestamp": "timestamp"},
        "primary_key": ("timestamp",),
    }
    work = frame.loc[:, ["timestamp"]].copy()
    work["_qrp_original_position"] = range(len(frame))
    filtered = filter_frame_by_time_window(
        work,
        spec=spec,
        time_window=time_window,
    )
    positions = pd.Series(False, index=range(len(frame)))
    positions.iloc[filtered["_qrp_original_position"].to_numpy(dtype=int)] = True
    return pd.Series(positions.to_numpy(dtype=bool), index=frame.index)


__all__ = ["write_labels"]
