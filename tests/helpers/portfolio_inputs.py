"""Small explicit native input schemas and numerical frames for portfolio tests."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from math import log

import numpy as np
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
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.risk import FactorRiskEstimate
from quant_research_platform.sources.csmar.interim import (
    interim_catalog as csmar_catalog,
)
from quant_research_platform.sources.reference.interim import (
    interim_catalog as reference_catalog,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.portfolio_ledger import write_market_evidence
from tests.helpers.research_configuration import PV25_FACTORS
from tests.helpers.research_specs import make_demo_research_dataset_spec


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


def portfolio_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, TimeWindow]:
    dates = pd.bdate_range("2022-11-14", "2023-01-10")
    assets = ("a", "b")
    panel = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "listing_date": dates[0],
                "open": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "adj_open": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "adj_close": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "amount": 1000.0,
                "total_market_cap": 1000.0 if asset == "a" else 2000.0,
                "market_type": 1,
                "is_a_share_scope": True,
                "is_trading": True,
                "is_st": False,
                "is_pt": False,
                "listing_age_trading_days": 300,
                "has_252_trading_sessions": True,
                "close": 100.0 + timestamp.day * (1 if asset == "a" else 2),
                "volume": 10.0,
                "turnover": 0.01,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    dataset = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "eligible": True,
                "total_market_cap": 1000.0 if asset == "a" else 2000.0,
                "float_market_cap": 1000.0 if asset == "a" else 2000.0,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    execution = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": timestamp,
                "can_enter": True,
                "can_exit": True,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    fee = pd.DataFrame(
        [
            {
                "fee_code": fee_code,
                "effective_date": pd.Timestamp("2023-01-01"),
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
                "effective_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": "sell",
                "basis": "amount",
                "rate": 0.0005,
                "included_in_ledger": True,
                "included_in": None,
            }
        ]
    )
    window = TimeWindow("2023-01-03", "2023-01-06")
    return (
        {
            "dataset": dataset,
            "panel": panel,
            "execution": execution,
            "fee": fee,
            "calendar": pd.DataFrame(
                {
                    "calendar_date": dates,
                    "source_market_type": 1,
                    "is_trading_day": True,
                }
            ),
        },
        fee,
        window,
    )


def closed_market_inputs(tmp_path, *, fault=None, open_assets=6):
    dates = pd.bdate_range("2022-12-01", "2023-01-06")
    closed = pd.DatetimeIndex(
        []
        if fault in {"normal_open", "confirmed_suspension", "adjacent_relisting"}
        else ["2023-01-03", "2023-01-04"]
    )
    assets = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(open_assets)]
    panel = pd.DataFrame(
        [
            {
                "listing_date": dates[0],
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": day,
                "market_type": 1 if asset.startswith("a") else 4,
                "adj_open": 121.0
                if asset.startswith("a") and day >= pd.Timestamp("2023-01-05")
                else 100.0,
                "adj_close": 110.0
                if asset.startswith("a") and day == pd.Timestamp("2023-01-02")
                else 121.0
                if asset.startswith("a") and day >= pd.Timestamp("2023-01-05")
                else 100.0,
                "amount": 1000.0,
                "period_return": 0.0,
                "total_market_cap": 1000.0,
                "float_market_cap": 1000.0,
                "industry_standard_id": "fixture",
                "industry_level1_code": "one",
                "is_nonfinancial_industry": True,
                "is_a_share_scope": True,
                "is_trading": True,
                "is_st": False,
                "is_pt": False,
                "listing_age_trading_days": 300,
                "has_252_trading_sessions": True,
                "close": 100.0,
                "volume": 1000.0,
                "turnover": 0.01,
            }
            for day in dates
            for asset in assets
        ]
    )
    bars = panel[["asset_id", "timestamp", "market_type"]].rename(
        columns={"asset_id": "source_asset_id"}
    )
    bars["filling_flag"] = 0
    holiday = bars.source_asset_id.str.startswith("a") & bars.timestamp.isin(closed)
    bars.loc[holiday, "filling_flag"] = 2
    bars["adj_close"] = 999.0
    panel = panel.loc[
        ~(panel.asset_id.str.startswith("a") & panel.timestamp.isin(closed))
    ].copy()
    calendar = pd.DataFrame(
        [
            {
                "timestamp": day,
                "calendar_date": day,
                "source_market_type": market,
                "is_trading_day": day in dates and not (market == 1 and day in closed),
            }
            for day in pd.date_range(dates[0], dates[-1])
            for market in (1, 4)
        ]
    )
    stocks = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": dates[0],
            "listing_date": dates[0],
            "delisting_date": pd.NaT,
            "company_status": "active",
        }
    )
    if fault == "actual_activity_on_closed":
        bars.loc[holiday, "filling_flag"] = 0
    elif fault == "unknown_barrier":
        bars.loc[
            bars.source_asset_id.eq("a0")
            & bars.timestamp.eq(pd.Timestamp("2023-01-02")),
            "filling_flag",
        ] = pd.NA
    elif fault == "missing_calendar":
        calendar = calendar.loc[
            ~(calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[0]))
        ]
    elif fault == "unknown_membership":
        bars.loc[
            bars.source_asset_id.eq("a0") & bars.timestamp.eq(closed[0]), "market_type"
        ] = pd.NA
    elif fault == "delisted":
        stocks.loc[stocks.source_asset_id.eq("a0"), "delisting_date"] = pd.Timestamp(
            "2023-01-02"
        )
    elif fault == "future_lifecycle":
        stocks.loc[stocks.source_asset_id.eq("a0"), ["timestamp", "listing_date"]] = (
            pd.Timestamp("2023-01-05")
        )
    elif fault == "missing_open_price":
        panel.loc[
            panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")),
            "adj_open",
        ] = np.nan
    elif fault == "missing_open_row":
        panel = panel.loc[
            ~(panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")))
        ]
    if fault == "closed_only_axis":
        panel = panel.loc[panel.asset_id.str.startswith("a")].copy()
        bars = bars.loc[bars.source_asset_id.str.startswith("a")].copy()
        stocks = stocks.loc[stocks.source_asset_id.str.startswith("a")].copy()
    if fault == "new_lifecycle":
        new = stocks.loc[stocks.source_asset_id.eq("a0")].assign(
            timestamp=closed[0], listing_date=closed[0]
        )
        stocks.loc[stocks.source_asset_id.eq("a0"), "delisting_date"] = closed[
            0
        ] - pd.Timedelta(days=1)
        stocks = pd.concat((stocks, new), ignore_index=True)
    if fault == "future_lifecycle_append":
        selected = stocks.source_asset_id.eq("a0")
        new = stocks.loc[selected].assign(
            timestamp=pd.Timestamp("2023-01-09"),
            listing_date=pd.Timestamp("2023-01-09"),
        )
        stocks.loc[selected, "delisting_date"] = pd.Timestamp("2023-01-08")
        stocks = pd.concat((stocks, new), ignore_index=True)
    if fault == "unknown_before_suspension":
        calendar = calendar.loc[
            ~(calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[0]))
        ].copy()
        calendar.loc[
            calendar.source_market_type.eq(4) & calendar.calendar_date.eq(closed[0]),
            "is_trading_day",
        ] = False
        calendar.loc[
            calendar.source_market_type.eq(1) & calendar.calendar_date.eq(closed[1]),
            "is_trading_day",
        ] = True
        bars = bars.loc[
            ~(bars.source_asset_id.str.startswith("b") & bars.timestamp.eq(closed[0]))
        ].copy()
        bars.loc[
            bars.source_asset_id.str.startswith("a") & bars.timestamp.eq(closed[1]),
            "filling_flag",
        ] = 1
        suspended = panel.loc[
            panel.asset_id.str.startswith("a")
            & panel.timestamp.eq(pd.Timestamp("2023-01-02"))
        ].assign(
            timestamp=closed[1],
            adj_open=110.0,
            adj_close=110.0,
            is_trading=False,
            amount=0.0,
            volume=0.0,
        )
        panel = pd.concat(
            (panel.loc[~panel.timestamp.eq(closed[0])], suspended), ignore_index=True
        )
    if fault == "confirmed_suspension":
        suspension_dates = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
        suspended = panel.asset_id.str.startswith("a") & panel.timestamp.isin(
            suspension_dates
        )
        bars.loc[
            bars.source_asset_id.str.startswith("a")
            & bars.timestamp.isin(suspension_dates),
            "filling_flag",
        ] = 1
        panel.loc[suspended, ["adj_open", "adj_close"]] = 110.0
        panel.loc[suspended, ["amount", "volume"]] = 0.0
        panel.loc[suspended, "is_trading"] = False
    if fault == "missing_decision_industry":
        panel.loc[
            panel.asset_id.eq("a0") & panel.timestamp.eq(pd.Timestamp("2023-01-02")),
            "industry_level1_code",
        ] = pd.NA
    if fault == "adjacent_relisting":
        boundary = dates[-1]
        changed = stocks.source_asset_id.str.startswith("a")
        new = stocks.loc[changed].assign(timestamp=boundary, listing_date=boundary)
        stocks.loc[changed, "delisting_date"] = dates[-2]
        stocks = pd.concat((stocks, new), ignore_index=True)
        rows = panel.asset_id.str.startswith("a") & panel.timestamp.eq(boundary)
        panel.loc[rows, "listing_date"] = boundary
        panel.loc[panel.asset_id.str.startswith("a"), ["adj_open", "adj_close"]] = 10.0
        panel.loc[rows, ["adj_open", "adj_close"]] = 100.0
        panel.loc[rows, "has_252_trading_sessions"] = False
        panel.loc[rows, "listing_age_trading_days"] = 1
    if fault in {"partial_open_quote", "partial_close_quote"}:
        selected = panel.asset_id.eq("a0")
        panel.loc[selected, ["adj_open", "adj_close"]] = 10.0
        panel.loc[selected & panel.timestamp.eq("2023-01-02"), "adj_close"] = 11.0
        partial = selected & panel.timestamp.eq("2023-01-05")
        panel.loc[partial, ["adj_open", "adj_close"]] = np.nan
        phase = "open" if fault == "partial_open_quote" else "close"
        panel.loc[partial, f"adj_{phase}"] = 12.0
        panel.loc[
            selected & panel.timestamp.eq("2023-01-06"), ["adj_open", "adj_close"]
        ] = np.nan
    execution = panel[["market_id", "asset_id", "timestamp"]].assign(
        can_enter=True, can_exit=True
    )
    initial = execution.timestamp.eq(
        pd.Timestamp("2022-12-30")
    ) & execution.asset_id.isin(["a3", "a4", "a5"])
    if fault != "adjacent_relisting":
        execution.loc[initial, "can_enter"] = False
    if fault == "unknown_reopen_mask":
        execution["can_exit"] = execution.can_exit.astype("boolean")
        execution.loc[
            execution.asset_id.eq("a0")
            & execution.timestamp.eq(pd.Timestamp("2023-01-02")),
            "can_exit",
        ] = pd.NA
    data = panel[["market_id", "asset_id", "timestamp", "float_market_cap"]].assign(
        eligible=True, signal=1.0, label=0.0
    )
    # This fixture uses identical raw and research price paths, with no actions.
    panel["open"] = panel.adj_open
    panel["close"] = panel.adj_close
    roots = write_market_evidence(tmp_path, panel, bars=bars, stocks=stocks)
    specs = _processed_specs()
    dataset_spec = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    for target, frame, spec in (
        ("market_panel_1d", panel, specs.get("market_panel_1d")),
        ("execution_mask", execution, specs.get("execution_mask")),
        ("trading_calendar", calendar, _interim_specs().get("trading_calendar")),
        (
            "a_share_fee_schedule",
            portfolio_inputs()[1].assign(
                included_in=lambda frame: frame.included_in.astype("string"),
                effective_date=pd.Timestamp("2022-12-01"),
            ),
            _interim_specs().get("a_share_fee_schedule"),
        ),
        ("fixture", data, dataset_spec),
    ):
        root = tmp_path / target
        write_dataset(root, frame, spec=spec)
        roots[target] = root
    inputs = {
        "dataset": roots["fixture"],
        "panel": roots["market_panel_1d"],
        "execution": roots["execution_mask"],
        "calendar": roots["trading_calendar"],
        "fees": roots["a_share_fee_schedule"],
        "market_bars": roots["market_bars_1d"],
        "stock_master": roots["stock_master"],
        "corporate_actions": roots["corporate_actions"],
        "risk_context": (),
    }
    return inputs, roots, calendar, panel


def decision_inputs(*, benchmark=(0.5, 0.5), holdings=None, fee_rate=0.0):
    assets = list("abc"[: len(benchmark)])
    day, entry, end = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    life = pd.Timestamp("2000-01-01")
    panel = pd.DataFrame(
        {
            "asset_id": assets,
            "listing_date": life,
            "timestamp": day,
            "open": 100.0,
            "close": 100.0,
            "amount": 1000.0,
            "market_type": 1,
            "is_trading": True,
            "is_nonfinancial_industry": True,
            "research_open_mark_at": pd.NaT,
        }
    )
    _, fees, _ = portfolio_inputs()
    fees["rate"] = fee_rate
    values = pd.Series(benchmark if holdings is None else holdings, index=assets) * 100
    alpha = pd.Series([0.1, *([0.0] * (len(assets) - 1))], index=assets)
    risk = FactorRiskEstimate(
        pd.DataFrame(),
        pd.DataFrame([[0.01]], index=["style"], columns=["style"]),
        pd.Series(0.01, index=assets),
        126.0,
        63.0,
        0.2,
        252.0,
    )
    enhanced = production._EnhancedDecision(
        pd.Series(benchmark, index=assets),
        alpha,
        pd.DataFrame({"asset_id": assets, "style": 1.0}),
        risk,
        production.AlphaCalibration(
            1.0, 0.0, 1.0, 1.0, 0.5, -1.0, 1.0, 0.0, 1.0, 6, 6, None
        ),
    )
    return dict(
        candidate=PortfolioCandidateSpec(
            "timing",
            "global",
            "benchmark_enhanced",
            "champion",
            "daily",
            None,
            "csi500",
            ("style",),
        ),
        decision_timestamp=day,
        next_timestamp=entry,
        panel=panel,
        next_panel=panel.assign(timestamp=entry),
        endpoint_panel=panel.assign(timestamp=end),
        closed_assets=frozenset(),
        dataset=pd.DataFrame(
            {
                "asset_id": assets,
                "timestamp": day,
                "eligible": True,
                "total_market_cap": 1000.0,
                "float_market_cap": 1000.0,
            }
        ),
        execution=pd.DataFrame(
            {
                "asset_id": assets,
                "timestamp": day,
                "execution_at": entry,
                "can_enter": True,
                "can_exit": True,
            }
        ),
        holdings=values,
        held_units=values.map(lambda value: Fraction(value) / 100),
        cash=0.0,
        recovery_cash=0.0,
        decision_receivable=0.0,
        receivable=0.0,
        locked_quantities={},
        nav=100.0,
        amount_history={(asset, life): deque([1000.0] * 20) for asset in assets},
        alpha=alpha,
        enhanced_decision=enhanced,
        target_override=None,
        decision_reason="enhanced_optimizer",
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        fee_schedule=fee_schedule_from_frame(fees, content_identity="synthetic"),
        fee_identity="synthetic",
        prepare_trade_cost=prepare_trade_cost,
        market_name_for_type=lambda _: "shanghai",
    )
