"""Admitted immutable equity evidence through native sessions and money ledger."""

import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    validate_portfolio_transition,
)
from quant_research_platform.research.portfolios.accounting import (
    portfolio_interval_economics,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.portfolio_inputs import _processed_specs
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.test_closed_market_portfolios import _ledger
from tests.sources.reference.test_equity_events import native_equity_inputs


def recovered_batches(
    tmp_path, kind, *, events_override=None, panel_transform=None, hold_affected=True
):
    def events(base):
        if kind == "quote":
            return [base]
        if kind.startswith("cash"):
            return [
                dict(
                    base,
                    event_id="recovery",
                    kind="cash_recovery",
                    amount_per_share=3.0,
                    remaining_old_units_per_old_unit=0.0
                    if kind == "cash_cancel"
                    else 1.0,
                )
            ]
        if kind == "cancel":
            return [
                dict(
                    base,
                    event_id="cancel",
                    kind="cancel",
                    amount_per_share=None,
                    remaining_old_units_per_old_unit=0.0,
                )
            ]
        successor = "000001" if kind == "same_code_conversion" else "000002"
        return [
            dict(
                base,
                event_id="conversion",
                kind="conversion",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
                successor_asset_id=successor,
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=0.5,
                successor_shares_per_unit=4.0,
                successor_unit_basis_date="2023-01-03",
            ),
            dict(
                base,
                event_id="successor-quote",
                asset_id=successor,
                listing_date="2023-01-03",
                shares_per_old_unit=4.0,
                amount_per_share=8.0,
            ),
        ]

    layout, roots, _ = native_equity_inputs(
        tmp_path, events=events_override or events, panel_transform=panel_transform
    )
    dataset_spec = make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    changed = {}
    for target in ("fixture", "execution_mask"):
        frame = read_dataset(roots[target])
        frame["asset_id"] = frame.asset_id.replace({"a0": "000001"})
        frame = frame.loc[
            ~(frame.asset_id.eq("000001") & frame.timestamp.gt("2023-01-02"))
        ]
        if target == "fixture" and not hold_affected:
            frame.loc[frame.asset_id.eq("000001"), "eligible"] = False
        root = roots[target].with_name(roots[target].name + "-updated")
        write_dataset(
            root,
            frame,
            spec=dataset_spec
            if target == "fixture"
            else _processed_specs().get(target),
        )
        changed[target] = root
    roots.update(changed)
    layout = {
        **layout,
        "dataset": changed["fixture"],
        "execution": changed["execution_mask"],
    }
    batches = list(_ledger(layout, roots))
    return batches


@pytest.mark.parametrize(
    "kind",
    (
        "quote",
        "cash_keep",
        "cash_cancel",
        "cancel",
        "conversion",
        "same_code_conversion",
    ),
)
def test_actual_reference_evidence_reaches_retained_equity_ledger(tmp_path, kind):
    batches = recovered_batches(tmp_path, kind)
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    previous = None
    economics = _LedgerEconomics(capital)
    for batch in batches:
        previous = validate_portfolio_transition(batch, capital, previous)
        economics.consume(batch)
        portfolio_interval_economics(batch)
    initial = next(
        row.quantity
        for row in batches[0].valuations
        if row.asset_id == "000001" and row.phase == "after"
    )
    assert initial > 0
    # No post-delisting PANEL rows exist; admitted evidence alone values rights.
    affected = [event for batch in batches for event in batch.rights_adjustments]
    cash = [
        flow for batch in batches for flow in batch.cashflows if flow.kind == "recovery"
    ]
    assert len(affected) == (0 if kind == "quote" else 1)
    assert len(cash) == (1 if kind.startswith("cash") else 0)
    if cash:
        assert cash[0].amount == pytest.approx(float(initial * 6))
    if kind in {"conversion", "same_code_conversion"}:
        assert affected[0].target_quantity == initial / 4
        successor = "000001" if kind == "same_code_conversion" else "000002"
        assert any(
            row.asset_id == successor
            and row.quantity == initial / 4
            and row.mark_price == 32
            for batch in batches
            for row in batch.valuations
            if row.phase == "after"
        )
    elif kind in {"cancel", "cash_cancel"}:
        assert affected[0].quantity_after == 0
    if kind != "quote":
        assert all(
            trade.asset_id != "000001"
            for batch in batches[1:]
            for trade in batch.trades
        )
    metadata = {
        "window": {"start": "2022-12-30", "end": "2023-01-06"},
        "sample_role": "validation",
    }
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata=metadata,
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == batches


def test_unvalued_successor_blocks_without_synthetic_quote(tmp_path):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    def events(base):
        return [
            dict(
                base,
                event_id="conversion",
                kind="conversion",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
                successor_asset_id="000002",
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=0.5,
                successor_shares_per_unit=4.0,
                successor_unit_basis_date="2023-01-03",
            )
        ]

    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_successor_valuation_missing"
    ):
        recovered_batches(tmp_path, "conversion", events_override=events)


def _replacement_events(kind, *, cash_per_share=50.0, publication="2023-01-03"):
    def events(base):
        quoted = dict(
            base,
            event_id="old-quote",
            effective_date="2023-01-02",
            publication_date="2023-01-02",
            amount_per_share=50.0,
        )
        replacement = dict(
            base,
            event_id="replacement",
            effective_date="2023-01-03",
            publication_date=publication,
            kind="cash_recovery",
            amount_per_share=cash_per_share,
            remaining_old_units_per_old_unit=0.5 if kind == "partial" else 0.0,
        )
        if kind == "cancel":
            replacement.update(kind="cancel", amount_per_share=None)
        if kind in {"conversion", "same_code_conversion"}:
            successor = "000001" if kind == "same_code_conversion" else "000002"
            replacement.update(
                kind="conversion",
                amount_per_share=None,
                successor_asset_id=successor,
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=0.5,
                successor_shares_per_unit=4.0,
                successor_unit_basis_date="2023-01-03",
            )
            successor_quote = dict(
                base,
                event_id="successor-quote",
                asset_id=successor,
                listing_date="2023-01-03",
                shares_per_old_unit=4.0,
                effective_date="2023-01-03",
                publication_date=publication,
                amount_per_share=100.0,
            )
            return [quoted, replacement, successor_quote]
        return [quoted, replacement]

    return events


def _flat_prices(frame):
    return frame.assign(adj_open=100.0, adj_close=100.0)


@pytest.mark.parametrize(
    "kind,cash_per_share,replacement_value",
    [
        ("cash_cancel", 50.0, 100.0),
        ("partial", 25.0, 100.0),
        ("cash_cancel", 25.0, 50.0),
        ("cash_cancel", 75.0, 150.0),
        ("cash_cancel", 0.0, 0.0),
        ("cancel", 0.0, 0.0),
        ("conversion", 0.0, 100.0),
        ("same_code_conversion", 0.0, 100.0),
    ],
)
def test_available_replacement_is_atomic_in_every_wealth_snapshot(
    tmp_path, kind, cash_per_share, replacement_value
):
    batches = recovered_batches(
        tmp_path / "input",
        kind,
        events_override=_replacement_events(kind, cash_per_share=cash_per_share),
        panel_transform=_flat_prices,
    )
    boundary, settlement = batches[1:3]
    old = {row.phase: row for row in boundary.valuations if row.asset_id == "000001"}
    assert old["after"].mark_price == 100.0
    assert old["end"].value == pytest.approx(
        float(old["after"].quantity) * replacement_value
    )
    assert old["end"].quantity == old["after"].quantity
    assert any(
        row.name == "research_end_mark_kind"
        and row.text == "evidenced_rights_entitlement"
        for row in boundary.diagnostics
    )
    opening = next(
        row
        for row in settlement.valuations
        if row.asset_id == "000001" and row.phase == "before"
    )
    assert opening.quantity == old["end"].quantity
    assert opening.value == old["end"].value
    assert opening.lifecycle_id == old["end"].lifecycle_id
    economic = portfolio_interval_economics(boundary)
    assert economic["gross_return"] == pytest.approx(
        float(old["after"].quantity)
        * (replacement_value - 100.0)
        / economic["nav_before"],
        abs=1e-14,
    )
    assert portfolio_interval_economics(settlement)["gross_return"] == pytest.approx(
        0.0, abs=1e-14
    )
    adjustments = [event for batch in batches for event in batch.rights_adjustments]
    assert len(adjustments) == 1
    assert not boundary.rights_adjustments
    assert not any(
        trade.asset_id == "000001" for batch in batches[1:] for trade in batch.trades
    )
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    previous = None
    for batch in batches:
        previous = validate_portfolio_transition(batch, capital, previous)
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "2022-12-30", "end": "2023-01-06"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == batches


def test_equal_replacement_native_statistics_has_no_artificial_drawdown_or_volatility(
    tmp_path,
):
    from dataclasses import asdict

    import numpy as np

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    batches = recovered_batches(
        tmp_path / "input",
        "cash_cancel",
        events_override=_replacement_events("cash_cancel"),
        panel_transform=_flat_prices,
    )
    economics = [portfolio_interval_economics(batch) for batch in batches]
    assert [row["gross_return"] for row in economics] == pytest.approx(
        [0.0] * len(batches), abs=1e-14
    )
    window = {"start": "2022-12-30", "end": "2023-01-06"}
    ledger = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital(
            "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
        ),
        metadata={
            "window": window,
            "candidate": asdict(
                PortfolioCandidateSpec(
                    "closed_market_portfolio",
                    "global",
                    "top20_equal_weight",
                    "explicit",
                    "daily",
                    0.5,
                    None,
                )
            ),
        },
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    result = read_statistics_artifact(
        portfolio_statistics(
            {
                "ledgers": [str(ledger)],
                "window": window,
                "analyses": ["performance"],
                "output": str(tmp_path / "statistics"),
            }
        )
    )
    with result.table("portfolio_daily.parquet") as daily:
        np.testing.assert_allclose(
            daily.net_return, [row["net_return"] for row in economics], atol=1e-14
        )
        assert daily.net_return.abs().max() < 0.001
    with result.table("portfolio_performance.parquet") as performance:
        metrics = performance.set_index("metric_id")
        assert -0.001 < metrics.loc["max_drawdown", "value"] <= 0.0
        assert metrics.loc["annualized_volatility", "value"] < 0.01


def test_future_publication_cannot_change_earlier_held_wealth(tmp_path):
    baseline = recovered_batches(
        tmp_path / "base",
        "quote",
        events_override=lambda base: _replacement_events("cash_cancel")(base)[:1],
        panel_transform=_flat_prices,
    )
    changed = recovered_batches(
        tmp_path / "future",
        "cash_cancel",
        events_override=_replacement_events(
            "cash_cancel", cash_per_share=75.0, publication="2023-01-05"
        ),
        panel_transform=_flat_prices,
    )
    for left, right in zip(baseline[:3], changed[:3], strict=True):
        assert left.valuations == right.valuations
        assert left.trades == right.trades
        assert left.cashflows == right.cashflows
    assert not any(batch.rights_adjustments for batch in changed)
    assert portfolio_interval_economics(changed[-1])["gross_return"] > 0.0


def test_unheld_unknown_successor_does_not_block_defined_account(tmp_path):
    def events(base):
        return _replacement_events("conversion")(base)[:2]

    batches = recovered_batches(
        tmp_path, "conversion", events_override=events, hold_affected=False
    )
    assert batches
    assert not any(batch.rights_adjustments for batch in batches)
    assert not any(
        row.quantity
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "000001"
    )


def test_event_wealth_and_cash_count_are_chunk_and_reporting_window_invariant(
    tmp_path, monkeypatch
):
    from dataclasses import asdict, replace

    import pandas as pd

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.markets.implementations.a_share import (
        portfolio_sessions,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )

    baseline = recovered_batches(
        tmp_path / "wide",
        "cash_cancel",
        events_override=_replacement_events("cash_cancel"),
        panel_transform=_flat_prices,
    )
    monkeypatch.setattr(portfolio_sessions, "DEFAULT_BATCH_ROWS", 1)
    changed = recovered_batches(
        tmp_path / "small",
        "cash_cancel",
        events_override=_replacement_events("cash_cancel"),
        panel_transform=_flat_prices,
    )
    for left, right in zip(baseline, changed, strict=True):
        assert left.valuations == right.valuations
        assert left.trades == right.trades
        assert left.cashflows == right.cashflows
    candidate = PortfolioCandidateSpec(
        "closed_market_portfolio",
        "global",
        "top20_equal_weight",
        "explicit",
        "daily",
        0.5,
        None,
    )
    for first in (0, 1, 2):
        # Keep the held initial state while changing the reporting window.
        selected = [
            replace(batch, sequence=i) for i, batch in enumerate(baseline[first:])
        ]
        capital = sum(
            row.value for row in selected[0].valuations if row.phase == "before"
        )
        start = pd.Timestamp(min(row.timestamp_ns for row in selected[0].valuations))
        window = {"start": str(start.date()), "end": "2023-01-06"}
        ledger = write_portfolio_ledger_artifact(
            tmp_path / f"ledger-{first}",
            capital=PortfolioCapital(
                candidate.candidate_id, capital, "CNY", "adjusted_notional"
            ),
            metadata={"window": window, "candidate": asdict(candidate)},
            produce=lambda emit, selected=selected: [emit(batch) for batch in selected],
        )
        restored = list(read_portfolio_ledger_artifact(ledger).iter_chunks())
        assert (
            sum(
                flow.kind == "recovery"
                for batch in restored
                for flow in batch.cashflows
            )
            == 1
        )
        result = read_statistics_artifact(
            portfolio_statistics(
                {
                    "ledgers": [str(ledger)],
                    "window": window,
                    "analyses": ["performance"],
                    "output": str(tmp_path / f"statistics-{first}"),
                }
            )
        )
        with result.table("portfolio_daily.parquet") as daily:
            assert daily.gross_return.tolist() == pytest.approx(
                [0.0] * len(selected), abs=1e-14
            )
            assert daily.nav_end.tolist() == pytest.approx(
                [
                    portfolio_interval_economics(batch)["nav_end"]
                    for batch in baseline[first:]
                ]
            )


def test_rights_only_successor_can_settle_to_cash_without_disappearing(tmp_path):
    def events(base):
        successor = dict(
            base,
            event_id="successor-quote",
            asset_id="000002",
            listing_date="2023-01-03",
            unit_basis_date="2023-01-03",
            shares_per_old_unit=4.0,
            effective_date="2023-01-03",
            publication_date="2023-01-03",
            amount_per_share=100.0,
        )
        return [
            dict(
                base,
                event_id="initial-quote",
                effective_date="2023-01-02",
                publication_date="2023-01-02",
                amount_per_share=50.0,
            ),
            dict(
                base,
                event_id="convert",
                kind="conversion",
                effective_date="2023-01-03",
                publication_date="2023-01-03",
                amount_per_share=None,
                remaining_old_units_per_old_unit=0.0,
                successor_asset_id="000002",
                successor_listing_date="2023-01-03",
                successor_shares_per_old_share=0.5,
                successor_shares_per_unit=4.0,
                successor_unit_basis_date="2023-01-03",
            ),
            successor,
            dict(
                successor,
                event_id="successor-cash",
                kind="cash_recovery",
                effective_date="2023-01-04",
                publication_date="2023-01-04",
                remaining_old_units_per_old_unit=0.0,
            ),
        ]

    batches = recovered_batches(
        tmp_path / "inputs",
        "cash_cancel",
        events_override=events,
        panel_transform=lambda frame: frame.assign(adj_open=100.0, adj_close=100.0),
    )
    capital = PortfolioCapital(
        "closed_market_portfolio", 100.0, "CNY", "adjusted_notional"
    )
    previous = None
    for batch in batches:
        previous = validate_portfolio_transition(batch, capital, previous)
        assert portfolio_interval_economics(batch)["gross_return"] == pytest.approx(
            0.0, abs=1e-14
        )
    assert sum(len(batch.rights_adjustments) for batch in batches) == 2
    recovery = [
        flow for batch in batches for flow in batch.cashflows if flow.kind == "recovery"
    ]
    initial = next(
        row.quantity
        for row in batches[0].valuations
        if row.phase == "after" and row.asset_id == "000001"
    )
    assert len(recovery) == 1
    assert recovery[0].amount == pytest.approx(float(initial * 100), rel=1e-14, abs=0.0)
    assert not any(
        trade.asset_id in {"000001", "000002"}
        for batch in batches[1:]
        for trade in batch.trades
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "2022-12-30", "end": "2023-01-06"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == batches
