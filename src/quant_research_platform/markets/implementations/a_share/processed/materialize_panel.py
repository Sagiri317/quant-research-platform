"""A-share processed panel materialization."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io,
    materialization_planning,
    materialization_validation,
    panels,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    market_bars_data_id: str,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    source_inputs = materialization_io.read_source_inputs_for_spec(
        spec,
        layout=layout,
        source_id=source_id,
        market_bars_data_id=market_bars_data_id,
    )
    materialization_validation.require_source_inputs_for_spec(spec, source_inputs)
    stock_master = source_inputs["stock_master"]
    market_cap = source_inputs["market_cap"]
    trading_calendar = source_inputs["trading_calendar"]
    market_bars_root = layout.interim_data_root(source_id, market_bars_data_id)
    market_bars_inspection = inspect_dataset(market_bars_root)
    materialization_validation.require_source_input_columns_from_schema(
        market_bars_data_id,
        market_bars_inspection.columns,
        columns=("open", "high", "low", "close", "volume", "amount"),
    )
    estimated = materialization_planning.estimate_panel_bytes_per_session(
        sessions,
        time_window=time_window,
        market_bars_root=market_bars_root,
        inspection=market_bars_inspection,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
        required_columns=spec.required_columns,
    )
    windows = materialization_planning.planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=market_bars_inspection,
        estimated_bytes_per_session=estimated,
        lookahead_sessions=1,
        reason="panel_next_session",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            market_bars = read_dataset(market_bars_root, time_window=window.read)
            materialization_validation.require_source_input_columns(
                market_bars_data_id,
                market_bars,
                columns=("open", "high", "low", "close", "volume", "amount"),
            )
            panel = panels.compute_market_panel(
                market_bars,
                stock_master=stock_master,
                market_cap=market_cap,
                trading_calendar=trading_calendar,
            )
            output = materialization_planning.filter_output_sessions(
                panel,
                window.core_sessions,
            )
            yield output.loc[:, list(spec.required_columns)].copy()

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


__all__ = ["write_panel"]
