"""Small explicit native input schemas and numerical frames for portfolio tests."""

from __future__ import annotations

from math import log

import pandas as pd

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_catalog,
)
from quant_research_platform.sources.reference.interim import (
    interim_catalog as reference_catalog,
)
from tests.helpers.research_configuration import PV25_FACTORS


def _interim_specs() -> InterimDataCatalog:
    calendar = InterimDataSpec(
        data_id="trading_calendar",
        source_id="csmar",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
        column_roles={
            "timestamp": ColumnRole.TIMESTAMP,
            "calendar_date": ColumnRole.SOURCE,
            "source_market_type": ColumnRole.KEY,
            "is_trading_day": ColumnRole.CONTEXT,
        },
        primary_key=("timestamp", "source_market_type"),
        partition_columns=("year",),
    )
    fee = InterimDataSpec(
        data_id="a_share_fee_schedule",
        source_id="reference",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles={
            "fee_code": ColumnRole.KEY,
            "effective_date": ColumnRole.TIMESTAMP,
            "end_date": ColumnRole.KEY,
            "market": ColumnRole.KEY,
            "side": ColumnRole.KEY,
            "basis": ColumnRole.KEY,
            "rate": ColumnRole.CONTEXT,
            "included_in_ledger": ColumnRole.CONTEXT,
            "included_in": ColumnRole.CONTEXT,
        },
        primary_key=(
            "fee_code",
            "effective_date",
            "end_date",
            "market",
            "side",
            "basis",
        ),
        partition_columns=("year",),
    )
    return InterimDataCatalog(
        (
            calendar,
            fee,
            *(
                (
                    reference_catalog()
                    if target == "a_share_equity_events"
                    else csmar_catalog()
                ).get(target)
                for target in (
                    "index_market",
                    "market_bars_1d",
                    "stock_master",
                    "corporate_actions",
                    "a_share_equity_events",
                )
            ),
        )
    )


def _processed_specs() -> ProcessedDataCatalog:
    common = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    panel = ProcessedDataSpec(
        data_id="market_panel_1d",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **common,
            "listing_date": ColumnRole.CONTEXT,
            "open": ColumnRole.CONTEXT,
            "adj_open": ColumnRole.CONTEXT,
            "adj_close": ColumnRole.CONTEXT,
            "amount": ColumnRole.CONTEXT,
            "market_type": ColumnRole.CONTEXT,
            "period_return": ColumnRole.CONTEXT,
            "float_market_cap": ColumnRole.CONTEXT,
            "total_market_cap": ColumnRole.CONTEXT,
            "industry_standard_id": ColumnRole.KEY,
            "industry_level1_code": ColumnRole.KEY,
            "is_nonfinancial_industry": ColumnRole.CONTEXT,
            "is_a_share_scope": ColumnRole.CONTEXT,
            "is_trading": ColumnRole.CONTEXT,
            "is_st": ColumnRole.CONTEXT,
            "is_pt": ColumnRole.CONTEXT,
            "listing_age_trading_days": ColumnRole.CONTEXT,
            "has_252_trading_sessions": ColumnRole.CONTEXT,
            "close": ColumnRole.CONTEXT,
            "volume": ColumnRole.CONTEXT,
            "turnover": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.PANEL,
    )
    execution = ProcessedDataSpec(
        data_id="execution_mask",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **common,
            "can_enter": ColumnRole.MASK,
            "can_exit": ColumnRole.MASK,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.ANNOTATION,
    )
    benchmark = ProcessedDataSpec(
        data_id="benchmark_weights",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "benchmark_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "weight": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "benchmark_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.ANNOTATION,
    )
    risk_context = tuple(
        ProcessedDataSpec(
            data_id=data_id,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                **common,
                **{column: ColumnRole.FACTOR for column in columns},
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id="a_share",
            kind=ProcessedDataKind.FACTOR,
        )
        for data_id, columns in (
            (
                "price_volume_facts",
                ("log_total_market_cap", "momentum_252_20", "amihud_60d"),
            ),
            ("fundamental_factors", ("earnings_yield",)),
            ("market_model_factors", ("beta_252d", "residual_volatility_60d")),
        )
    )
    return ProcessedDataCatalog((panel, execution, benchmark, *risk_context))


def _dataset_spec() -> ResearchDatasetSpec:
    statistical_factors = tuple(factor for factor in PV25_FACTORS if factor != "x")
    return ResearchDatasetSpec(
        data_id="pv_all",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "x": ColumnRole.FACTOR,
            **{factor: ColumnRole.FACTOR for factor in statistical_factors},
            "label": ColumnRole.LABEL,
            "label_available_at": ColumnRole.CONTEXT,
            "forward_return_20d": ColumnRole.LABEL,
            "forward_return_20d_available_at": ColumnRole.CONTEXT,
            "eligible": ColumnRole.MASK,
            "float_market_cap": ColumnRole.CONTEXT,
            "total_market_cap": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        input_data_ids=("market_panel_1d",),
        base_input_data_id="market_panel_1d",
        factor_columns=("x", *statistical_factors),
        label_columns=(
            "label",
            "forward_return_20d",
        ),
        filter_columns=("eligible",),
        context_columns=(
            "float_market_cap",
            "total_market_cap",
            "label_available_at",
            "forward_return_20d_available_at",
        ),
    )


def _frames(dates: pd.DatetimeIndex | None = None) -> dict[str, pd.DataFrame]:
    if dates is None:
        dates = pd.bdate_range("2014-01-01", "2023-12-31")
    calendar_dates = dates.append(
        pd.bdate_range(dates[-1] + pd.offsets.BDay(1), periods=21)
    )
    assets = ("a", "b")
    panel = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "open": 100.0,
                "adj_open": 100.0,
                "adj_close": 100.0,
                "amount": 10_000.0,
                "market_type": 1,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    execution = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "can_enter": True,
                "can_exit": True,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    dataset = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "x": float(asset == "b"),
                **{
                    factor: float(asset == "b")
                    for factor in PV25_FACTORS
                    if factor != "x"
                },
                "label": float(asset == "b") * 0.01,
                "label_available_at": calendar_dates[date_index + 1],
                "forward_return_20d": float(asset == "b") * 0.02,
                "forward_return_20d_available_at": calendar_dates[date_index + 21],
                "eligible": True,
                "float_market_cap": 1_000.0 if asset == "a" else 2_000.0,
                "total_market_cap": 2_000.0 if asset == "a" else 5_000.0,
            }
            for asset in assets
            for date_index, timestamp in enumerate(dates)
        ]
    )
    panel["period_return"] = 0.001
    panel["float_market_cap"] = panel["asset_id"].map({"a": 1_000.0, "b": 2_000.0})
    panel["total_market_cap"] = panel["asset_id"].map({"a": 2_000.0, "b": 5_000.0})
    panel["industry_standard_id"] = "CSRC2012"
    panel["industry_level1_code"] = "A"
    panel["is_nonfinancial_industry"] = True
    panel = panel.assign(
        listing_date=dates[0],
        is_a_share_scope=True,
        is_trading=True,
        is_st=False,
        is_pt=False,
        listing_age_trading_days=300,
        has_252_trading_sessions=True,
        close=panel.adj_close,
        volume=1000.0,
        turnover=0.01,
    )
    from quant_research_platform.research.factors.engine import (
        _market_model_frames,
    )

    market_factors = pd.concat(
        _market_model_frames((panel,), trading_calendar=None, market_history=None),
        ignore_index=True,
    )
    calendar = pd.DataFrame(
        {
            "timestamp": calendar_dates,
            "calendar_date": calendar_dates,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    fees = pd.DataFrame(
        [
            {
                "fee_code": fee_code,
                "effective_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": side,
                "basis": "amount",
                "rate": rate,
                "included_in_ledger": True,
                "included_in": None,
            }
            for fee_code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.0001),
            )
            for side in ("buy", "sell")
        ]
        + [
            {
                "fee_code": "STAMP_DUTY",
                "effective_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": "sell",
                "basis": "amount",
                "rate": 0.001,
                "included_in_ledger": True,
                "included_in": None,
            }
        ]
    )
    fees["included_in"] = pd.Series([pd.NA] * len(fees), dtype="string")
    bars = (
        panel[["asset_id", "timestamp", "market_type"]]
        .rename(columns={"asset_id": "source_asset_id"})
        .assign(filling_flag=0)
    )
    first = panel.groupby("asset_id", sort=False).timestamp.min()
    stocks = pd.DataFrame(
        {
            "source_asset_id": first.index,
            "timestamp": first.to_numpy(),
            "listing_date": first.to_numpy(),
            "delisting_date": pd.NaT,
            "delisting_date_unresolved": False,
            "company_status": "active",
        }
    )
    return {
        "market_bars_1d": bars.reindex(
            columns=csmar_catalog().get("market_bars_1d").required_columns
        ),
        "corporate_actions": pd.DataFrame(
            {
                column: pd.Series(
                    dtype=(
                        "datetime64[ns]"
                        if column
                        in {"timestamp", "announcement_date", "settlement_date"}
                        else "float64"
                        if column in {"amount", "subscription_price"}
                        else "Int64"
                        if column == "market_type"
                        else "string"
                    )
                )
                for column in csmar_catalog().get("corporate_actions").required_columns
            }
        ),
        "stock_master": stocks.reindex(
            columns=csmar_catalog().get("stock_master").required_columns
        ),
        "market_panel_1d": panel,
        "market_model_factors": market_factors,
        "price_volume_facts": dataset.loc[
            :,
            [
                "market_id",
                "asset_id",
                "timestamp",
                "momentum_252_20",
                "amihud_60d",
            ],
        ].assign(log_total_market_cap=dataset.total_market_cap.map(log)),
        "fundamental_factors": dataset.loc[
            :, ["market_id", "asset_id", "timestamp"]
        ].assign(earnings_yield=0.01),
        "index_market": pd.DataFrame(
            [
                {
                    "source_asset_id": code,
                    "timestamp": timestamp,
                    "trade_date": timestamp,
                    "index_name": code,
                    "open": float("nan"),
                    "close": 100.0,
                    "period_return": 0.0,
                }
                for code in ("H00905", "H00852")
                for timestamp in dates
            ]
        ),
        "execution_mask": execution,
        "pv_all": dataset,
        "trading_calendar": calendar,
        "a_share_fee_schedule": fees,
        "benchmark_weights": pd.DataFrame(
            [
                {
                    "market_id": "a_share",
                    "benchmark_id": benchmark_id,
                    "asset_id": asset,
                    "timestamp": timestamp,
                    "weight": 0.5,
                }
                for benchmark_id in ("csi500", "csi1000")
                for timestamp in dates
                for asset in assets
            ]
        ),
    }
