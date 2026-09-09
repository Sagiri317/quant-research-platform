"""Common factor dispatch and dependency-window calculations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)

from .fundamental.calculations import calculate_fundamental_factors
from .price_volume.calculations import calculate_price_volume_factors
from .price_volume.definitions import PRICE_VOLUME_FACTOR_FAMILY


def calculate_factors_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[FactorDefinition],
    output_window: TimeWindow | None = None,
) -> pd.DataFrame:
    requested = tuple(definitions)
    _require_factor_input_columns(panel_frame, requested)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    output_source = panel_frame
    if output_window is not None:
        start, end, closed = time_window_timestamp_interval(output_window)
        timestamps = pd.to_datetime(panel_frame["timestamp"])
        output_source = panel_frame.loc[
            timestamps.ge(start)
            & (timestamps.le(end) if closed else timestamps.lt(end))
        ]
    if not requested:
        return output_source.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    if families == ("fundamental",):
        return _with_keys(
            output_source,
            calculate_fundamental_factors(
                panel_frame,
                factor_columns=tuple(
                    definition.output_column for definition in requested
                ),
            ),
            output_columns=tuple(definition.output_column for definition in requested),
        )
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        raise ValueError("mixed_or_unknown_factor_families")
    calculated = calculate_price_volume_factors(
        panel_frame,
        factor_columns=tuple(definition.output_column for definition in requested),
        output_window=output_window,
    )
    return _with_keys(
        output_source,
        calculated,
        output_columns=tuple(definition.output_column for definition in requested),
    )


def factor_panel_input_columns(
    definitions: Sequence[FactorDefinition],
) -> tuple[str, ...]:
    requested = tuple(definitions)
    if not requested:
        return ("market_id", "asset_id", "timestamp")
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(
                    column
                    for definition in requested
                    for column in definition.input_columns
                ),
            )
        )
    )


def factor_lookback_bars(definitions: Sequence[FactorDefinition]) -> int:
    return max(
        (
            value
            for definition in definitions
            for value in (
                definition.lookback_window,
                definition.window,
                *definition.required_windows,
            )
            if value is not None
        ),
        default=0,
    )


def _require_factor_input_columns(
    panel_frame: pd.DataFrame,
    definitions: Sequence[FactorDefinition],
) -> None:
    missing = [
        column
        for column in factor_panel_input_columns(definitions)
        if column not in panel_frame.columns
    ]
    if missing:
        raise ValueError("missing_factor_input_columns: " + ", ".join(missing))


def _with_keys(
    source: pd.DataFrame,
    calculated: pd.DataFrame,
    *,
    output_columns: tuple[str, ...],
) -> pd.DataFrame:
    keys = source.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    values = calculated.loc[:, ["asset_id", "timestamp", *output_columns]]
    output = keys.merge(
        values,
        on=["asset_id", "timestamp"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_calculated"),
    )
    return output.loc[:, ["market_id", "asset_id", "timestamp", *output_columns]]
