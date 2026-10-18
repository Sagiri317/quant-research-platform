"""Independent monetary-domain checks through the actual portfolio producer."""

from fractions import Fraction

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.accounting import EquityEntitlements
from quant_research_platform.research.portfolios.diagnostics import (
    _portfolio_statistics_tables,
)
from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs
from tests.helpers.statistics_artifacts import (
    portfolio_statistics_config,
    statistics_metadata,
)
from tests.research.test_equity_recovery_production import recovered_batches
from tests.research.test_portfolio_production import _build_producer
from tests.sources.reference.test_equity_events import native_equity_inputs


def test_finite_money_survives_exact_units_and_native_ledger_roundtrip(
    tmp_path, monkeypatch
):
    capital, price = 100.0, 3.0
    frames, fees, window = _inputs()
    fees["rate"] = 0.0
    frames["panel"].loc[:, ["open", "close"]] = price
    chunks, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        initial_nav=float(capital),
        typed_batches=True,
    )
    batches = list(chunks)
    assert batches
    expected_units = Fraction(float(capital)) / Fraction(float(price))
    assert len(batches[0].trades) == 1
    assert batches[0].trades[0].quantity == expected_units
    assert batches[0].trades[0].notional == capital
    assert not any(batch.trades for batch in batches[1:])
    for batch in batches:
        assert all(flow.kind == "principal" for flow in batch.cashflows)
        assert sum(flow.amount for flow in batch.cashflows) == (
            -capital if batch.sequence == 0 else 0.0
        )
        for row in batch.valuations:
            if row.asset_id == "__cash__":
                expected_cash = (
                    capital if batch.sequence == 0 and row.phase == "before" else 0.0
                )
                assert row.value == expected_cash
            elif row.phase != "before" or batch.sequence != 0:
                assert row.quantity == expected_units
                assert row.value == capital
                assert float(row.quantity * Fraction(row.mark_price)) == capital

    capital_spec = PortfolioCapital(
        **{
            "candidate_id": "independent_numeric_portfolio",
            "initial_capital": float(capital),
            "valuation_currency": "CNY",
            "valuation_method": "cash_equity",
        }
    )
    metadata = {
        "window": {
            key: value
            for key, value in {
                "start": "2023-01-03",
                "end": "2023-01-06",
                "access": "validation",
            }.items()
            if key != "access"
        },
        "sample_role": "validation",
    }

    def produce(emit):
        for batch in batches:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital_spec, metadata=metadata, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    assert list(artifact.iter_chunks()) == batches
    economics = _LedgerEconomics(
        PortfolioCapital(
            "independent_numeric_portfolio", float(capital), "CNY", "cash_equity"
        )
    )
    for batch in artifact.iter_chunks():
        economics.consume(batch)
    summary = economics.summary()
    assert summary["nav"] == capital
    assert summary["net_pnl"] == summary["capital_return"] == 0.0
    assert summary["fill_events"] == 1
    assert summary["fill_quantity"] == float(expected_units)


def test_active_share_recovery_requires_terminated_holder_entitlement(tmp_path):
    def events(base):
        return [dict(base, kind="cash_recovery", amount_per_share=3.0)]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    stocks = read_dataset(roots["stock_master"])
    stocks.loc[stocks.source_asset_id.eq("000001"), "delisting_date"] = pd.NaT
    stocks.loc[stocks.source_asset_id.eq("000001"), "company_status"] = "active"
    active_root = roots["stock_master"].parent / ("d" * 64)
    write_dataset(active_root, stocks, spec=interim_catalog().get("stock_master"))
    layout = {**layout, "stock_master": active_root}
    sessions = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=layout["market_bars"],
            stock_master_root=layout["stock_master"],
            equity_events_root=layout["equity_events"],
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2022-12-30", "2023-01-06"),
        )
    )
    events = pd.concat(view.rights_events for view in sessions)
    assert not events.entitlement_proved.all()
    with pytest.raises(PortfolioTargetBlocked, match="portfolio_held_rights_unproved"):
        EquityEntitlements().process(
            pd.DataFrame(),
            events,
            pd.Series({"000001": Fraction(1)}),
            pd.Series({"000001": pd.Timestamp("2022-12-01")}),
            pd.Timestamp("2023-01-04"),
        )


@pytest.mark.parametrize("endpoint", ("new_lifecycle", "absent"))
def test_fully_sold_rights_do_not_require_next_lifecycle_linkage(
    tmp_path, monkeypatch, endpoint
):
    frames, fees, window = _inputs()
    fees["rate"] = 0.0
    frames["panel"].loc[:, ["open", "close"]] = 10.0
    frames["dataset"].loc[
        frames["dataset"].asset_id.eq("b")
        & frames["dataset"].timestamp.ge("2023-01-04"),
        "eligible",
    ] = False
    frames["panel"].loc[
        frames["panel"].asset_id.eq("b") & frames["panel"].timestamp.ge("2023-01-06"),
        "listing_date",
    ] = pd.Timestamp("2023-01-06")
    if endpoint == "absent":
        for name in ("panel", "dataset", "execution"):
            frame = frames[name]
            frames[name] = frame.loc[
                ~(frame.asset_id.eq("b") & frame.timestamp.ge("2023-01-06"))
            ].copy()
    chunks, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        typed_batches=True,
    )
    batches = list(chunks)
    old_sale = [
        trade
        for batch in batches
        for trade in batch.trades
        if trade.asset_id == "b" and trade.notional < 0
    ]
    assert len(old_sale) == 1 and old_sale[0].notional == -100.0
    assert old_sale[0].timestamp_ns == pd.Timestamp("2023-01-05").value
    assert (
        sum(row.value for row in batches[-1].valuations if row.phase == "end") == 100.0
    )


@pytest.mark.parametrize("kind", ("cash_cancel", "conversion", "quote"))
def test_admitted_recovery_reaches_native_statistics_without_losing_money(
    tmp_path, kind
):
    batches = recovered_batches(tmp_path / "source", kind)
    final_nav = float(
        sum(
            (
                Fraction(row.value)
                for row in batches[-1].valuations
                if row.phase == "end"
            ),
            Fraction(),
        )
    )
    fees = float(
        -sum(
            (
                Fraction(flow.amount)
                for batch in batches
                for flow in batch.cashflows
                if flow.kind.endswith("_cost")
            ),
            Fraction(),
        )
    )
    candidates, manifest = portfolio_statistics_config(sample_role="validation")
    window = {"start": "2022-12-30", "end": "2023-01-06", "access": "validation"}
    ledgers = {}
    for candidate in candidates:
        capital_spec = PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        )
        metadata = {
            "window": {key: value for key, value in window.items() if key != "access"},
            "sample_role": "validation",
        }

        def produce(emit):
            for batch in batches:
                emit(batch)

        root = write_portfolio_ledger_artifact(
            tmp_path / candidate.candidate_id,
            capital=capital_spec,
            metadata=metadata,
            produce=produce,
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(root)
    dates = pd.bdate_range("2022-12-30", "2023-01-06")
    inputs = [
        HoldingFactorInputs(
            dates[i],
            dates[i + 1],
            dates[i + 2],
            None,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            "holding_factor_fit_unidentified",
        )
        for i in range(len(dates) - 2)
    ]
    tables = _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        config=manifest,
        time_window=TimeWindow(dates[0], dates[-1]),
        benchmark_opens=pd.DataFrame(columns=("benchmark_id", "timestamp", "open")),
        calendar=pd.DataFrame({"calendar_date": dates}),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        holding_inputs={
            candidate.risk_style_columns: iter(inputs) for candidate in candidates
        },
    )
    performance = tables["portfolio_performance"].set_index(
        ["candidate_id", "metric_id"]
    )
    for candidate in candidates:
        assert (
            performance.loc[(candidate.candidate_id, "terminal_nav"), "value"]
            == final_nav
        )
        assert performance.loc[
            (candidate.candidate_id, "cost"), "value"
        ] == pytest.approx(fees, abs=1e-13)
    metadata = statistics_metadata(
        TimeWindow(window["start"], window["end"]),
        sample_role="validation",
        candidate_ids=[c.candidate_id for c in candidates],
        comparison_portfolio_ids=manifest["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
    )
    root = write_statistics_artifact(
        tmp_path / "statistics",
        metadata=metadata,
        tables=tables,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(root)
    with artifact.table("portfolio_daily.parquet") as daily:
        assert daily.groupby("candidate_id").nav_end.last().eq(final_nav).all()


def test_native_writer_rejects_negative_zero_valued_long_only_rights(tmp_path):
    start, end = pd.Timestamp("2023-01-04").value, pd.Timestamp("2023-01-05").value
    values = []
    for phase in ("before", "after", "end"):
        stamp = end if phase == "end" else start
        values.extend(
            (
                PortfolioValuation(
                    stamp,
                    "asset",
                    phase,
                    Fraction(-1),
                    0.0,
                    0.0,
                    lifecycle_id="2000-01-01",
                ),
                PortfolioValuation(stamp, "__cash__", phase, 100.0, 100.0),
            )
        )
    batch = PortfolioLedgerBatch(
        0,
        (),
        (),
        tuple(values),
        (
            PortfolioDiagnostic("asset", None, "target_notional", 0.0),
            PortfolioDiagnostic("__cash__", None, "target_notional", 100.0),
            PortfolioDiagnostic(
                "__cash__", None, "decision_reason", text="simple_rule"
            ),
        ),
    )
    capital_spec = PortfolioCapital(
        **{
            "candidate_id": "invalid_negative_rights",
            "initial_capital": 100.0,
            "valuation_currency": "CNY",
            "valuation_method": "cash_equity",
        }
    )
    metadata = {
        "window": {
            key: value
            for key, value in {
                "start": "2023-01-03",
                "end": "2023-01-06",
                "access": "validation",
            }.items()
            if key != "access"
        },
        "sample_role": "validation",
    }
    with pytest.raises(ValueError, match="portfolio_ledger_notional_invalid"):
        write_portfolio_ledger_artifact(
            tmp_path / "ledger",
            capital=capital_spec,
            metadata=metadata,
            produce=lambda emit: emit(batch),
        )
