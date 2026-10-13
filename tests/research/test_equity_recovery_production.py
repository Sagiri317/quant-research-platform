"""Admitted immutable equity evidence through native sessions and money ledger."""

import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    validate_portfolio_transition,
)
from quant_research_platform.research.portfolios.accounting import (
    portfolio_interval_economics,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.test_closed_market_portfolios import _ledger
from tests.sources.reference.test_equity_events import native_equity_inputs
from tests.workflows.test_portfolio_workflow_integration import _processed_specs


def recovered_batches(tmp_path, kind, *, events_override=None, panel_transform=None):
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
        root = roots[target].parent / ("c" * 64)
        write_dataset(
            root,
            frame,
            spec=dataset_spec
            if target == "fixture"
            else _processed_specs().get(target),
        )
        changed[target] = root
    roots.update(changed)
    layout = layout.with_target_roots(changed)
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
    envelope = build_business_manifest_envelope(
        logical_artifact_id=capital.candidate_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2022-12-30",
            "end": "2023-01-06",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": capital.candidate_id,
                "initial_capital": 100.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        envelope=envelope,
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert (
        list(
            read_portfolio_ledger_artifact(
                root, expected_identity=root.name
            ).iter_chunks()
        )
        == batches
    )


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
            )
        ]

    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_successor_valuation_missing"
    ):
        recovered_batches(tmp_path, "conversion", events_override=events)
