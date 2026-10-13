"""Independent monetary-domain checks through the actual portfolio producer."""

from fractions import Fraction

import numpy as np
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
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.diagnostics import (
    _portfolio_statistics_tables,
)
from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.research_configuration import research_configuration
from tests.helpers.statistics_artifacts import statistics_envelope
from tests.research.test_equity_recovery_production import recovered_batches
from tests.research.test_portfolio_production import _build_producer, _inputs
from tests.sources.reference.test_equity_events import native_equity_inputs


@pytest.mark.parametrize(
    "capital, price",
    [
        (np.nextafter(0.0, 1.0), 10.0),
        (100.0, np.nextafter(0.0, 1.0)),
        (1e-300, 1e300),
        (100.0, 3.0),
    ],
    ids=("unit-underflow", "unit-overflow", "extreme-finite-ratio", "ordinary"),
)
def test_finite_money_survives_exact_units_and_native_ledger_roundtrip(
    tmp_path, monkeypatch, capital, price
):
    frames, fees, window = _inputs()
    fees["rate"] = 0.0
    frames["panel"].loc[:, ["adj_open", "adj_close"]] = price
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

    envelope = build_business_manifest_envelope(
        logical_artifact_id="independent_numeric_portfolio",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-03",
            "end": "2023-01-06",
            "access": "development",
        },
        upstream_identities={"prediction": "b" * 64},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "independent_numeric_portfolio",
                "initial_capital": float(capital),
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )

    def produce(emit):
        for batch in batches:
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", envelope=envelope, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    assert list(artifact.iter_chunks()) == batches
    economics = _LedgerEconomics(
        PortfolioCapital(
            "independent_numeric_portfolio", float(capital), "CNY", "adjusted_notional"
        )
    )
    for batch in artifact.iter_chunks():
        economics.consume(batch)
    summary = economics.summary()
    assert summary["nav"] == capital
    assert summary["net_pnl"] == summary["capital_return"] == 0.0
    assert summary["fill_events"] == 1
    try:
        represented_units = float(expected_units)
    except OverflowError:
        represented_units = None
    if represented_units in (None, 0.0):
        assert summary["fill_quantity"] is None
        assert summary["fill_quantity_reason"]
    else:
        assert summary["fill_quantity"] == represented_units


def test_active_share_cash_event_cannot_duplicate_adjusted_distribution(tmp_path):
    def events(base):
        return [dict(base, kind="cash_recovery", amount_per_share=3.0)]

    layout, roots, calendar = native_equity_inputs(tmp_path, events=events)
    stocks = read_dataset(roots["stock_master"])
    stocks.loc[stocks.source_asset_id.eq("000001"), "delisting_date"] = pd.NaT
    active_root = roots["stock_master"].parent / ("d" * 64)
    write_dataset(
        active_root, stocks, spec=research_profile().interim.get("stock_master")
    )
    layout = layout.with_target_roots({"stock_master": active_root})
    with pytest.raises(ValueError, match="equity_event"):
        list(
            portfolio_sessions.iter_portfolio_sessions(
                layout=layout,
                panel_root=roots["market_panel_1d"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=TimeWindow("2022-12-30", "2023-01-06"),
            )
        )


@pytest.mark.parametrize("endpoint", ("new_lifecycle", "absent"))
def test_fully_sold_rights_do_not_require_next_lifecycle_linkage(
    tmp_path, monkeypatch, endpoint
):
    frames, fees, window = _inputs()
    fees["rate"] = 0.0
    frames["panel"].loc[:, ["adj_open", "adj_close"]] = 10.0
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


@pytest.mark.parametrize("kind", ("cash_cancel", "conversion", "huge_quote"))
def test_admitted_recovery_reaches_native_statistics_without_losing_money(
    tmp_path, kind
):
    def huge_price(panel):
        panel = panel.copy()
        panel.loc[panel.asset_id.eq("000001"), ["adj_open", "adj_close"]] = 1e308
        return panel

    def huge_event(base):
        return [dict(base, amount_per_share=1e308, shares_per_old_unit=10)]

    batches = recovered_batches(
        tmp_path / "source",
        "quote" if kind == "huge_quote" else kind,
        events_override=huge_event if kind == "huge_quote" else None,
        panel_transform=huge_price if kind == "huge_quote" else None,
    )
    if kind == "huge_quote":
        quoted = [
            row
            for batch in batches
            for row in batch.valuations
            if row.asset_id == "000001" and row.mark_price > Fraction(1e308)
        ]
        assert quoted
        for row in quoted:
            assert row.mark_price == Fraction(1e308) * 10
            assert row.value == float(Fraction(row.quantity) * Fraction(1e308) * 10)
            assert np.isfinite(row.value)
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
    protocol = bind_research_configuration(
        research_profile(), research_configuration(name="pv25")
    ).protocol("pv_all")
    candidates = protocol.portfolio_candidates()
    manifest = protocol.manifest()
    manifest["access_window"] = "development"
    window = {"start": "2022-12-30", "end": "2023-01-06", "access": "development"}
    ledgers = {}
    for candidate in candidates:
        envelope = build_business_manifest_envelope(
            logical_artifact_id=candidate.candidate_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=window,
            upstream_identities={},
            semantic_metadata={
                "portfolio_capital": {
                    "candidate_id": candidate.candidate_id,
                    "initial_capital": 100.0,
                    "valuation_currency": "CNY",
                    "valuation_method": "adjusted_notional",
                }
            },
        )

        def produce(emit):
            for batch in batches:
                emit(batch)

        root = write_portfolio_ledger_artifact(
            tmp_path / candidate.candidate_id, envelope=envelope, produce=produce
        )
        ledgers[candidate.candidate_id] = read_portfolio_ledger_artifact(
            root, expected_identity=root.name
        )
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
        protocol_manifest=manifest,
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
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_portfolio_statistics",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window=window,
        upstream_identities={},
        semantic_metadata={},
    )
    envelope = statistics_envelope(envelope, group="portfolio", protocol=manifest)
    root = write_statistics_artifact(
        tmp_path / "statistics",
        envelope=envelope,
        tables=tables,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(root, expected_identity=root.name)
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
    envelope = build_business_manifest_envelope(
        logical_artifact_id="invalid_negative_rights",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-03",
            "end": "2023-01-06",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "invalid_negative_rights",
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    with pytest.raises(ValueError, match="portfolio_ledger_notional_invalid"):
        write_portfolio_ledger_artifact(
            tmp_path / "ledger", envelope=envelope, produce=lambda emit: emit(batch)
        )
