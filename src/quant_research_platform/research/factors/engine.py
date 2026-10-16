"""Common factor dispatch and dependency-window calculations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

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


def _market_model_frames(
    panels: Iterable[pd.DataFrame],
    *,
    trading_calendar: pd.DataFrame | None,
    market_history: pd.DataFrame | None,
    min_fresh_fraction: float = 0.75,
) -> Iterator[pd.DataFrame]:
    from quant_research_platform.compute.market_model import MarketModelRiskState
    from quant_research_platform.contracts.validation import (
        strict_bool_series,
        strict_integer_series,
        strict_timestamp_series,
    )
    from quant_research_platform.research.filtering import a_share_eligibility

    open_markets = None
    if trading_calendar is not None:
        calendar = trading_calendar.assign(
            calendar_date=strict_timestamp_series(
                trading_calendar.calendar_date, error="invalid_calendar_date"
            ).dt.normalize(),
            source_market_type=strict_integer_series(
                trading_calendar.source_market_type, error="invalid_source_market_type"
            ),
            is_trading_day=strict_bool_series(
                trading_calendar.is_trading_day, error="invalid_is_trading_day"
            ),
        )
        if calendar.duplicated(["calendar_date", "source_market_type"]).any():
            raise ValueError("duplicate_trading_calendar_key")
        open_markets = {
            timestamp: frozenset(group.loc[group.is_trading_day, "source_market_type"])
            for timestamp, group in calendar.groupby("calendar_date", sort=False)
        }
    events = (
        iter(())
        if market_history is None
        else iter(
            market_history.sort_values("timestamp", kind="mergesort").itertuples(
                index=False
            )
        )
    )
    event = next(events, None)
    membership = {}
    state = MarketModelRiskState(min_fresh_fraction=min_fresh_fraction)
    for panel in panels:
        pieces = []
        for timestamp, cross in panel.sort_values(
            "timestamp", kind="mergesort"
        ).groupby("timestamp", sort=False):
            while event is not None and event.timestamp <= timestamp:
                membership[event.asset_id] = event.market_type
                event = next(events, None)
            cross = cross.assign(eligible=a_share_eligibility(cross, "a_share_all"))
            styles = state.observe(
                cross,
                open_market_types=None
                if open_markets is None
                else open_markets[timestamp.normalize()],
                asset_market_types=membership,
                market_input_complete=(
                    bool(cross["_panel_observed"].all())
                    if "_panel_observed" in cross
                    else True
                ),
            )
            piece = cross.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
            piece["beta_252d"] = piece.asset_id.map(styles.beta)
            piece["residual_volatility_60d"] = piece.asset_id.map(
                styles.residual_volatility
            )
            if "_panel_observed" in cross:
                piece = piece.loc[cross["_panel_observed"].to_numpy()]
            pieces.append(piece)
        if pieces:
            yield pd.concat(pieces, ignore_index=True)


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
