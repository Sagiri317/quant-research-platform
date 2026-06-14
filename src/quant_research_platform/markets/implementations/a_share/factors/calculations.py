"""A-share-specific factor calculations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.price_volume import (
    PRICE_VOLUME_FACTOR_FAMILY,
    calculate_price_volume_factors,
    price_volume_valid_bar_read_window,
)


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
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_a_share_factor_family: {unsupported}")
    return calculate_price_volume_factors(
        panel_frame.loc[:, list(factor_panel_input_columns(requested))],
        factor_columns=tuple(definition.output_column for definition in requested),
    )


def factor_panel_input_columns(
    definitions: Sequence[FactorDefinition],
) -> tuple[str, ...]:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return ("market_id", "asset_id", "timestamp")
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_a_share_factor_family: {unsupported}")
    return _factor_panel_columns(requested)


def factor_lookback_bars(definitions: Sequence[FactorDefinition]) -> int:
    values = [
        value
        for definition in definitions
        for value in (
            definition.lookback_window,
            definition.window,
            *definition.required_windows,
        )
        if value is not None
    ]
    return max(values, default=0)


def factor_read_window_for_panel(
    panel_keys: pd.DataFrame,
    *,
    definitions: Sequence[FactorDefinition],
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
) -> TimeWindow:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return read_window
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_a_share_factor_family: {unsupported}")
    return price_volume_valid_bar_read_window(
        panel_keys,
        read_window=read_window,
        core_sessions=core_sessions,
        lookback_bars=factor_lookback_bars(requested),
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
        family = definitions[0].family if definitions else PRICE_VOLUME_FACTOR_FAMILY
        raise ValueError(
            "missing_a_share_factor_input_columns: "
            f"{family}: {', '.join(missing)}"
        )


def _factor_panel_columns(
    definitions: Sequence[FactorDefinition],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(
                    column
                    for definition in definitions
                    for column in definition.input_columns
                ),
            )
        )
    )


__all__ = [
    "calculate_factors_for_panel",
    "factor_panel_input_columns",
    "factor_lookback_bars",
    "factor_read_window_for_panel",
]
