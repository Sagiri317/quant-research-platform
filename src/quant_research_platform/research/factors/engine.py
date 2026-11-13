"""Dispatch selected factor calculations and determine their required historical
windows.
"""

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
    history_keys: tuple[str, ...] = ("asset_id",),
    min_fresh_fraction: float = 0.75,
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
    if families == ("style_structure",):
        from .style_structure import calculate

        return _with_keys(
            output_source,
            calculate(
                panel_frame,
                factor_columns=tuple(d.output_column for d in requested),
                history_keys=history_keys,
            ),
            output_columns=tuple(d.output_column for d in requested),
        )
    if families == ("market_structure",):
        from .market_structure import calculate

        return _with_keys(
            output_source,
            calculate(
                panel_frame,
                factor_columns=tuple(d.output_column for d in requested),
                history_keys=history_keys,
            ),
            output_columns=tuple(d.output_column for d in requested),
        )
    if families == ("risk_residual",):
        from quant_research_platform.contracts.risk import RISK_STYLE_COLUMNS

        from .residual import calculate_residual_factors

        return _with_keys(
            output_source,
            calculate_residual_factors(
                panel_frame,
                style_columns=RISK_STYLE_COLUMNS,
                factor_columns=tuple(item.output_column for item in requested),
                history_keys=history_keys,
                min_fresh_fraction=min_fresh_fraction,
            ),
            output_columns=tuple(item.output_column for item in requested),
        )
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
        history_keys=history_keys,
        min_fresh_fraction=min_fresh_fraction,
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
    """Return preceding sessions consumed by the requested definitions."""
    lookback = 0
    for definition in definitions:
        current = max(
            (
                value
                for value in (definition.lookback_window, *definition.required_windows)
                if value is not None
            ),
            default=0,
        )
        if definition.family in {"market_structure", "style_structure"}:
            # Structure definitions already declare preceding sessions explicitly.
            current = definition.lookback_window or 0
        column = definition.output_column
        # Unshifted windows include today's sample; endpoint lags and shocks
        # consume the full number of preceding sessions.
        if (
            current
            and definition.family == PRICE_VOLUME_FACTOR_FAMILY
            and not column.startswith(("reversal_", "momentum_"))
            and not column.endswith("_change_20d")
            and column not in {"amount_shock_20d", "volume_shock_20d"}
        ):
            current -= 1
        lookback = max(lookback, current)
    return lookback


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
    keys = source.loc[:, ["market_id", "asset_id", "timestamp"]]
    values = calculated.loc[:, ["asset_id", "timestamp", *output_columns]]
    return keys.merge(
        values,
        on=["asset_id", "timestamp"],
        how="left",
    )
