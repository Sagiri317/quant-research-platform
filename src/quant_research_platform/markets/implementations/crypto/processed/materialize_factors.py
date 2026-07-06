"""Crypto microstructure factor materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
)
from quant_research_platform.research.factors.microstructure import (
    calculate_microstructure_factors,
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

MICROSTRUCTURE_ROLLING_WINDOW_MINUTES = 5
MICROSTRUCTURE_ROLLING_CARRY_BARS = MICROSTRUCTURE_ROLLING_WINDOW_MINUTES - 1
_CORE_ROW = "_qrp_core_row"


def write_factors(
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
        _iter_factor_chunks(
            spec,
            panel_root=panel_root,
            time_window=time_window,
        ),
        spec=spec,
    )
    return inspect_dataset(root, time_window=time_window)


def _iter_factor_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_root,
    time_window: TimeWindow,
) -> Iterator[pd.DataFrame]:
    carry_by_asset: dict[str, pd.DataFrame] = {}
    for panel in read_dataset(
        panel_root,
        time_window=materialization_dependencies.factor_read_window(time_window),
        mode=DatasetReadMode.CHUNKED,
    ):
        if panel.empty:
            continue
        core = panel.copy()
        core[_CORE_ROW] = _core_mask(core, time_window)
        combined = _with_factor_carry(core, carry_by_asset)
        factors = calculate_microstructure_factors(
            combined,
            expected_interval=pd.Timedelta(minutes=1),
            min_periods=MICROSTRUCTURE_ROLLING_WINDOW_MINUTES,
        )
        output = factors.loc[factors[_CORE_ROW]].copy()
        yield output.loc[:, spec.required_columns]
        _update_factor_carry(combined, carry_by_asset)


def _with_factor_carry(
    core: pd.DataFrame,
    carry_by_asset: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not carry_by_asset:
        return core
    return pd.concat(
        (
            *(
                carry.assign(**{_CORE_ROW: False})
                for carry in carry_by_asset.values()
            ),
            core,
        ),
        ignore_index=True,
    )


def _update_factor_carry(
    frame: pd.DataFrame,
    carry_by_asset: dict[str, pd.DataFrame],
) -> None:
    ordered = frame.sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    )
    for asset_id, asset_rows in ordered.groupby("asset_id", sort=False):
        carry_by_asset[str(asset_id)] = (
            asset_rows.drop(columns=[_CORE_ROW])
            .tail(MICROSTRUCTURE_ROLLING_CARRY_BARS)
            .copy()
        )


def _core_mask(frame: pd.DataFrame, time_window: TimeWindow) -> pd.Series:
    spec = {
        "column_roles": {"timestamp": "timestamp"},
        "primary_key": ("timestamp",),
    }
    filtered = filter_frame_by_time_window(
        frame.loc[:, ["timestamp"]].assign(_qrp_index=frame.index),
        spec=spec,
        time_window=time_window,
    )
    return frame.index.isin(filtered["_qrp_index"])


__all__ = ["write_factors"]
