"""Common factor dispatch and dependency-window calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.time import TimeWindow

from .expectations.calculations import build_expectations_features
from .fundamental.calculations import calculate_fundamental_factors
from .price_volume.calculations import calculate_price_volume_factors
from .price_volume.definitions import PRICE_VOLUME_FACTOR_FAMILY
from .price_volume.windows import price_volume_valid_bar_read_window


def calculate_factors_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[FactorDefinition],
) -> pd.DataFrame:
    requested = tuple(definitions)
    _require_factor_input_columns(panel_frame, requested)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    if families == ("fundamental",):
        return _with_keys(
            panel_frame,
            calculate_fundamental_factors(
                panel_frame,
                factor_columns=tuple(
                    definition.output_column for definition in requested
                ),
            ),
            output_columns=tuple(definition.output_column for definition in requested),
        )
    if families == ("expectations",):
        return _with_keys(
            panel_frame,
            build_expectations_features(
                panel_frame,
                panel_frame,
                trading_sessions=tuple(
                    sorted(pd.to_datetime(panel_frame["timestamp"]).unique())
                ),
            ),
            output_columns=tuple(definition.output_column for definition in requested),
        )
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        raise ValueError("mixed_or_unknown_factor_families")
    calculated = calculate_price_volume_factors(
        panel_frame,
        factor_columns=tuple(definition.output_column for definition in requested),
    )
    return _with_keys(
        panel_frame,
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


def factor_read_window_for_panel(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    definitions: Sequence[FactorDefinition],
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
) -> TimeWindow:
    lookback = factor_lookback_bars(definitions)
    if lookback == 0:
        return read_window
    return price_volume_valid_bar_read_window(
        panel_keys,
        read_window=read_window,
        core_sessions=core_sessions,
        lookback_bars=lookback,
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
