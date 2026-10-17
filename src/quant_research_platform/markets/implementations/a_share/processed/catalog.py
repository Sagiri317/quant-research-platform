"""A-share market-fact bindings for the common processed-data DAG."""

from __future__ import annotations

import json

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    EXECUTION_MASK_ID,
    FORWARD_RETURNS_DATA_ID,
    FUNDAMENTAL_DATA_ID,
    FUNDAMENTAL_PIT_ID,
    MARKET_ID,
    MARKET_PANEL_ID,
    ON_ID_DECOMPOSITION_DATA_ID,
    PRICE_VOLUME_DATA_ID,
)
from quant_research_platform.research.factors.expectations.definitions import (
    ALL_EXPECTATIONS_FACTORS,
)
from quant_research_platform.research.factors.fundamental.definitions import (
    FUNDAMENTAL_FACTORS,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PATH_FACTORS,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)
from quant_research_platform.research.protocols.a_share import (
    LABEL_HORIZONS,
    MARKET_MODEL_FACTORS,
)

PANEL_COLUMNS = (
    *PLATFORM_ASSET_TIME_KEY,
    "session_date",
    "available_at",
    "stock_name",
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "prev_close",
    "volume",
    "amount",
    "after_hours_volume",
    "after_hours_amount",
    "period_return",
    "turnover",
    "limit_status",
    "is_trading",
    "is_a_share_scope",
    "is_st",
    "is_pt",
    "listing_date",
    "delisting_date",
    "delisting_date_unresolved",
    "listing_age_trading_days",
    "has_252_trading_sessions",
    "industry_standard_id",
    "industry_level1_code",
    "industry_level1_name",
    "is_nonfinancial_industry",
    "share_change_date",
    "total_shares",
    "a_float_shares",
    "total_market_cap",
    "float_market_cap",
)
EXECUTION_COLUMNS = (*PLATFORM_ASSET_TIME_KEY, "can_enter", "can_exit")
FINANCIAL_PIT_COLUMNS = (
    *PLATFORM_ASSET_TIME_KEY,
    "total_assets",
    "total_liabilities",
    "parent_equity",
    "cash",
    "short_term_borrowings",
    "current_portion_noncurrent_liabilities",
    "long_term_borrowings",
    "bonds_payable",
    "operating_revenue",
    "operating_cost",
    "consolidated_net_income",
    "parent_net_income",
    "operating_cash_flow",
    "capex_cash_paid",
    "beginning_total_assets",
    "beginning_parent_equity",
    "prior_year_total_assets",
    "prior_year_parent_equity",
    "prior_year_parent_net_income_ttm",
    "prior_year_operating_revenue_ttm",
    "average_total_assets",
    "average_parent_equity",
    "operating_revenue_ttm",
    "operating_cost_ttm",
    "consolidated_net_income_ttm",
    "parent_net_income_ttm",
    "operating_cash_flow_ttm",
    "capex_cash_paid_ttm",
    "has_legal_fundamental_pit",
)
LABEL_COLUMNS = tuple(
    column
    for horizon in LABEL_HORIZONS
    for column in (
        forward_return_label_column(horizon),
        forward_return_available_at_column(horizon),
    )
)
FACTOR_COLUMNS = (*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS)
BENCHMARK_WEIGHTS_ID = "benchmark_weights"
BENCHMARK_WEIGHTS_COLUMNS = (
    "market_id",
    "benchmark_id",
    "asset_id",
    "timestamp",
    "weight",
)


def processed_catalog() -> ProcessedDataCatalog:
    return ProcessedDataCatalog(
        (
            _spec(
                MARKET_PANEL_ID,
                ProcessedDataKind.PANEL,
                PANEL_COLUMNS,
                optional_source_input_data_ids=(
                    "industry_classification",
                    "industry_classification_sets",
                    "industry_classification_versions",
                    "stock_info_changes",
                    "a_share_industry_snapshots",
                ),
                source_input_data_ids=(
                    "market_bars_1d",
                    "stock_master",
                    "market_cap",
                    "trading_calendar",
                    "industry_classification",
                    "industry_classification_sets",
                    "industry_classification_versions",
                    "stock_info_changes",
                    "a_share_industry_snapshots",
                    "trading_no_limit",
                    "special_treatment_changes",
                ),
            ),
            _spec(
                EXECUTION_MASK_ID,
                ProcessedDataKind.ANNOTATION,
                EXECUTION_COLUMNS,
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=("trading_calendar",),
            ),
            _spec(
                FUNDAMENTAL_PIT_ID,
                ProcessedDataKind.ANNOTATION,
                FINANCIAL_PIT_COLUMNS,
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=(
                    "trading_calendar",
                    "financial_balance",
                    "financial_income",
                    "financial_cashflow",
                    "financial_first_disclosure",
                ),
            ),
            _spec(
                PRICE_VOLUME_DATA_ID,
                ProcessedDataKind.FACTOR,
                (*PLATFORM_ASSET_TIME_KEY, *FACTOR_COLUMNS),
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=("trading_calendar",),
                factor_columns=FACTOR_COLUMNS,
            ),
            _spec(
                "market_model_factors",
                ProcessedDataKind.FACTOR,
                (*PLATFORM_ASSET_TIME_KEY, *MARKET_MODEL_FACTORS),
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=(
                    "trading_calendar",
                    "market_bars_1d",
                    "stock_master",
                ),
                factor_columns=MARKET_MODEL_FACTORS,
            ),
            _spec(
                FUNDAMENTAL_DATA_ID,
                ProcessedDataKind.FACTOR,
                (
                    *PLATFORM_ASSET_TIME_KEY,
                    *FUNDAMENTAL_FACTORS,
                    "has_legal_fundamental_pit",
                ),
                processed_input_data_ids=(FUNDAMENTAL_PIT_ID, MARKET_PANEL_ID),
                factor_columns=FUNDAMENTAL_FACTORS,
            ),
            _spec(
                "expectations_factors",
                ProcessedDataKind.FACTOR,
                (*PLATFORM_ASSET_TIME_KEY, *ALL_EXPECTATIONS_FACTORS),
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=("analyst_forecast", "trading_calendar"),
                factor_columns=ALL_EXPECTATIONS_FACTORS,
            ),
            _spec(
                FORWARD_RETURNS_DATA_ID,
                ProcessedDataKind.LABEL,
                (*PLATFORM_ASSET_TIME_KEY, *LABEL_COLUMNS),
                processed_input_data_ids=(MARKET_PANEL_ID,),
                source_input_data_ids=("trading_calendar",),
                label_columns=LABEL_COLUMNS,
            ),
            on_id_decomposition_spec(),
            _spec(
                BENCHMARK_WEIGHTS_ID,
                ProcessedDataKind.ANNOTATION,
                BENCHMARK_WEIGHTS_COLUMNS,
                source_input_data_ids=("index_info", "index_weights"),
                primary_key=(
                    "market_id",
                    "benchmark_id",
                    "asset_id",
                    "timestamp",
                ),
            ),
        )
    )


def _spec(
    data_id: str,
    kind: ProcessedDataKind,
    columns: tuple[str, ...],
    *,
    source_input_data_ids: tuple[str, ...] = (),
    optional_source_input_data_ids: tuple[str, ...] = (),
    processed_input_data_ids: tuple[str, ...] = (),
    factor_columns: tuple[str, ...] = (),
    label_columns: tuple[str, ...] = (),
    primary_key: tuple[str, ...] | None = None,
) -> ProcessedDataSpec:
    roles = {column: _role(column, factor_columns, label_columns) for column in columns}
    selected_primary_key = primary_key or PLATFORM_ASSET_TIME_KEY
    for column in selected_primary_key:
        if column != "timestamp":
            roles[column] = ColumnRole.KEY
    availability = {
        forward_return_label_column(horizon): forward_return_available_at_column(
            horizon
        )
        for horizon in LABEL_HORIZONS
        if forward_return_label_column(horizon) in label_columns
    }
    return ProcessedDataSpec(
        column_units={column: "decimal_return" for column in availability},
        label_availability_columns=availability,
        data_id=data_id,
        market_id=MARKET_ID,
        kind=kind,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=selected_primary_key,
        partition_columns=("year",),
        source_input_data_ids=source_input_data_ids,
        optional_source_input_data_ids=optional_source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        parameters_json=json.dumps({"min_fresh_fraction": 0.75})
        if data_id in {PRICE_VOLUME_DATA_ID, "market_model_factors"}
        else "{}",
    )


def _role(
    column: str,
    factor_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> ColumnRole:
    if column in factor_columns:
        return ColumnRole.FACTOR
    if column in label_columns:
        return ColumnRole.LABEL
    if column == "timestamp":
        return ColumnRole.TIMESTAMP
    if column in PLATFORM_ASSET_TIME_KEY:
        return ColumnRole.KEY
    if column in {
        "can_enter",
        "can_exit",
    }:
        return ColumnRole.MASK
    return ColumnRole.CONTEXT


def on_id_decomposition_spec(horizons=(20,)) -> ProcessedDataSpec:
    """Declare one reusable decomposition, with explicit component label clocks."""
    if (
        type(horizons) not in (tuple, list)
        or not horizons
        or any(
            type(value) is not int or value not in LABEL_HORIZONS for value in horizons
        )
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("on_id_horizons_invalid")
    roles = {
        column: ColumnRole.TIMESTAMP if column == "timestamp" else ColumnRole.KEY
        for column in PLATFORM_ASSET_TIME_KEY
    }
    roles["listing_date"] = ColumnRole.CONTEXT
    labels = {"on_id_pre_entry_log_return": "log_return"}
    for horizon in horizons:
        labels.update(
            {
                f"on_id_holding_intraday_log_return_{horizon}d": "log_return",
                f"on_id_holding_overnight_log_return_{horizon}d": "log_return",
                f"on_id_endpoint_return_{horizon}d": "decimal_return",
            }
        )
    for column in labels:
        roles[column] = ColumnRole.LABEL
        roles[f"{column}_available_at"] = ColumnRole.CONTEXT
        roles[f"{column}_reason"] = ColumnRole.CONTEXT
    for horizon in horizons:
        roles[f"on_id_decomposition_qualified_{horizon}d"] = ColumnRole.MASK
        roles[f"on_id_decomposition_reason_{horizon}d"] = ColumnRole.CONTEXT
    return ProcessedDataSpec(
        data_id=ON_ID_DECOMPOSITION_DATA_ID,
        market_id=MARKET_ID,
        kind=ProcessedDataKind.LABEL,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        column_units=labels,
        label_availability_columns={
            column: f"{column}_available_at" for column in labels
        },
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        processed_input_data_ids=(MARKET_PANEL_ID,),
        source_input_data_ids=("trading_calendar",),
        parameters_json=json.dumps({"horizons": list(horizons)}),
    )
