"""Actual initial purchases start from cash and obey ordinary execution costs."""

from collections import deque
from dataclasses import replace

import pandas as pd
import pytest

import quant_research_platform.research.portfolios.daily as daily
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.research.portfolios.optimizer import OptimizerConfig
from tests.helpers.portfolio_inputs import decision_inputs
from tests.research.test_portfolio_production import _build_producer, _enhanced_inputs


def test_unequal_capacity_can_fund_a_proportional_risk_improving_stock_leg():
    args = decision_inputs(holdings=(0.0, 0.0))
    args.update(cash=100.0, initializing=True)
    args["enhanced_decision"].exposures["style"] = [-1.0, 1.0]
    args["amount_history"] = {
        (asset, pd.Timestamp("2000-01-01")): deque([capacity] * 20)
        for asset, capacity in (("a", 10.0), ("b", 30.0))
    }
    args["optimizer_config"] = OptimizerConfig(
        active_name_limit=0.1, style_active_limit=0.001, tracking_error_limit=0.1
    )
    account, _, optimized, _, ledger = daily._build_session(**args)
    assert optimized is None
    # Buying 10/30 breaches the style band; buying 10/10 preserves it and
    # reduces both name shortfalls from 0.5 to 0.4 within actual capacity.
    assert ledger.executed_quantity.tolist() == [1000, 1000]
    assert account.post_trade_cash == pytest.approx(80.0)
    facts = ledger.attrs["realized_constraints"]
    assert facts["execution_constraints_status"] == "IMPROVED"
    assert facts["realized_name_active_weight"] == pytest.approx(0.4)
    assert facts["realized_style_active_exposure"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "blocked_first, capacity", [(False, 1000.0), (True, 1000.0), (False, 10.0)]
)
def test_initial_actual_purchase_exempts_turnover_but_preserves_fees_capacity_and_retry(
    tmp_path, monkeypatch, blocked_first, capacity
):
    frames, fees, window, candidate = _enhanced_inputs(monkeypatch)
    frames["panel"].loc[:, ["open", "close", "adj_open", "adj_close"]] = 0.01
    frames["panel"]["amount"] = capacity
    fees["rate"] = 0.0
    fees.loc[fees.fee_code.eq("BROKER_COMMISSION"), "rate"] = 0.01
    if blocked_first:
        frames["execution"].loc[
            frames["execution"].timestamp.eq("2023-01-03"), "can_enter"
        ] = False
    attempts = []
    original = daily._build_session

    def observe(**kwargs):
        result = original(**kwargs)
        attempts.append(
            (
                kwargs["decision_timestamp"],
                kwargs["initializing"],
                kwargs["optimizer_config"],
                result[0],
            )
        )
        return result

    monkeypatch.setattr(daily, "_build_session", observe)
    calibration = daily.AlphaCalibration(
        1.0, 0.0, 1.0, 1.0, 0.5, -1.0, 1.0, 0.0, 1.0, 6, 6, None
    )
    if capacity == 10.0:
        calibration = replace(
            calibration, undefined_reason="alpha_history_insufficient"
        )
    chunks, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        frozen_calibration=calibration,
        typed_batches=True,
    )
    batches = list(chunks)
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital(candidate.candidate_id, 100.0, "CNY", "cash_equity"),
        metadata={"window": {"start": str(window.start), "end": str(window.end)}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    batches = list(read_portfolio_ledger_artifact(root).iter_chunks())
    first = batches[0]
    assert all(
        float(row.quantity) == 0.0
        for row in first.valuations
        if row.phase == "before" and row.asset_id != "__cash__"
    )
    assert (
        next(
            row.value
            for row in first.valuations
            if row.phase == "before" and row.asset_id == "__cash__"
        )
        == 100.0
    )
    first_fill = next(
        batch for batch in batches if any(trade.quantity > 0 for trade in batch.trades)
    )
    expected_date = "2023-01-05" if blocked_first else "2023-01-04"
    assert {trade.timestamp_ns for trade in first_fill.trades} == {
        pd.Timestamp(expected_date).value
    }
    bought = sum(trade.notional for trade in first_fill.trades if trade.quantity > 0)
    costs = -sum(
        flow.amount for flow in first_fill.cashflows if flow.kind.endswith("_cost")
    )
    assert costs == pytest.approx(bought * 0.01)
    assert sum(
        row.value for row in first_fill.valuations if row.phase == "after"
    ) == pytest.approx(100.0 - costs)
    if capacity == 1000.0:
        assert 90.0 < bought <= 100.0 / 1.01
    else:
        assert bought == pytest.approx(20.0)
    assert attempts[0][1] is True
    assert attempts[0][2].target_turnover_limit >= 1.0
    if blocked_first:
        assert not first.trades
        assert attempts[1][1] is True
        assert attempts[1][2].target_turnover_limit >= 1.0
    after_initial = attempts[2 if blocked_first else 1 :]
    assert after_initial
    assert all(not initializing for _, initializing, _, _ in after_initial)
    assert all(
        config is None or config.target_turnover_limit < 1.0
        for _, _, config, _ in after_initial
    )


def test_first_purchase_on_ex_date_does_not_inherit_prior_record_date_dividend(
    tmp_path, monkeypatch
):
    frames, fees, window, candidate = _enhanced_inputs(monkeypatch)
    fees["rate"] = 0.0
    panel = frames["panel"]
    panel.loc[:, ["open", "close", "adj_open", "adj_close"]] = 0.01
    panel.loc[panel.timestamp.ge("2023-01-04"), ["open", "close"]] = 0.009
    frames["corporate_actions"] = pd.DataFrame(
        [
            dict(
                source_asset_id=asset,
                timestamp=pd.Timestamp("2023-01-04"),
                kind="CA",
                market_type=1,
                announcement_date=pd.Timestamp("2023-01-02"),
                settlement_date=pd.Timestamp("2023-01-05"),
                amount=0.001,
            )
            for asset in ("a", "b")
        ]
    )
    chunks, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
    )
    batches = list(chunks)
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital(candidate.candidate_id, 100.0, "CNY", "cash_equity"),
        metadata={"window": {"start": str(window.start), "end": str(window.end)}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    restored = list(read_portfolio_ledger_artifact(root).iter_chunks())
    assert restored == batches
    batches = restored
    assert batches[0].trades
    assert all(
        trade.timestamp_ns == pd.Timestamp("2023-01-04").value
        for trade in batches[0].trades
    )
    assert not any(
        flow.kind == "dividend" for batch in batches for flow in batch.cashflows
    )
    assert all(row.receivable == 0 for batch in batches for row in batch.valuations)
    assert sum(
        row.value for row in batches[-1].valuations if row.phase == "end"
    ) == pytest.approx(100.0)
