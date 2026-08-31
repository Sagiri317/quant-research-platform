"""A-share-specific factor calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_FACTOR_FAMILY,
)
from quant_research_platform.research.factors.price_volume.windows import (
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
    if families == ("fundamental",):
        return calculate_fundamental_factors_for_panel(panel_frame)
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_factor_family: {unsupported}")
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
    if families == ("fundamental",):
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
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_factor_family: {unsupported}")
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
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    definitions: Sequence[FactorDefinition],
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
) -> TimeWindow:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return read_window
    if families == ("fundamental",):
        return read_window
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_factor_family: {unsupported}")
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
            f"missing_factor_input_columns: {family}: {', '.join(missing)}"
        )


def calculate_fundamental_factors(
    pit_facts: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the fixed phase-1 raw fundamental factors from PIT facts."""

    required_facts = {
        "asset_id",
        "timestamp",
        "metric_id",
        "value",
        "report_period",
        "previous_value",
    }
    if not required_facts.issubset(pit_facts.columns) or not {
        "asset_id",
        "timestamp",
        "market_cap",
    }.issubset(market_cap.columns):
        raise ValueError("a_share_fundamental_factor_columns_invalid")
    values = (
        pit_facts.pivot_table(
            index=["asset_id", "timestamp"],
            columns="metric_id",
            values="value",
            aggfunc="first",
        )
        .reindex(
            columns=(
                "total_assets",
                "total_liabilities",
                "total_equity",
                "operating_revenue",
                "net_income",
                "operating_cash_flow",
            )
        )
        .reset_index()
    )
    wide = (
        pit_facts.loc[:, ("asset_id", "timestamp")]
        .drop_duplicates()
        .merge(
            values,
            on=("asset_id", "timestamp"),
            how="left",
            validate="one_to_one",
        )
        .merge(
            market_cap.loc[:, ("asset_id", "timestamp", "market_cap")],
            on=("asset_id", "timestamp"),
            how="left",
            validate="many_to_one",
        )
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
    )

    wide["market_id"] = "a_share"
    prior_assets = pit_facts.loc[
        pit_facts["metric_id"].eq("total_assets"),
        ("asset_id", "timestamp", "previous_value"),
    ].rename(columns={"previous_value": "previous_total_assets"})
    wide = wide.merge(
        prior_assets,
        on=("asset_id", "timestamp"),
        how="left",
        validate="one_to_one",
    )
    prior_assets = wide["previous_total_assets"].where(
        wide["previous_total_assets"] > 0.0
    )
    wide["previous_total_assets"] = prior_assets
    return calculate_fundamental_factors_for_panel(wide)


def calculate_fundamental_factors_for_panel(panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "market_id",
        "asset_id",
        "timestamp",
        "net_income",
        "market_cap",
        "total_equity",
        "operating_cash_flow",
        "total_assets",
        "previous_total_assets",
        "total_liabilities",
    }
    if not required.issubset(panel.columns):
        raise ValueError("a_share_fundamental_factor_columns_invalid")
    output = panel.loc[:, ("market_id", "asset_id", "timestamp")].copy()

    def ratio(numerator: str, denominator: str) -> pd.Series:
        denominator_values = panel[denominator].where(panel[denominator] > 0.0)
        return panel[numerator] / denominator_values

    output["fundamental_raw_earnings_yield"] = ratio("net_income", "market_cap")
    output["fundamental_raw_book_to_market"] = ratio("total_equity", "market_cap")
    output["fundamental_raw_roe"] = ratio("net_income", "total_equity")
    output["fundamental_raw_operating_cash_flow_to_assets"] = ratio(
        "operating_cash_flow", "total_assets"
    )
    output["fundamental_raw_asset_growth"] = (
        ratio("total_assets", "previous_total_assets") - 1.0
    )
    output["fundamental_raw_leverage"] = ratio("total_liabilities", "total_assets")
    return output.reset_index(drop=True)
