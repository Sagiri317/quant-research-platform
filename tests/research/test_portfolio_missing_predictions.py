"""Unavailable forecasts preserve shares and do not excuse invalid economics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.accounting import EquityEntitlements
from tests.research.test_portfolio_decision_execution import _decision_inputs
from tests.research.test_portfolio_production import _build_producer, _inputs


class _Forecasts:
    def __init__(self, missing):
        self.missing = missing

    def values_for_frame(self, frame, *, mark_matched):
        return (
            frame.asset_id.map({"a": 0.0, "b": 1.0}).to_numpy(),
            bytearray(
                not self.missing(row.timestamp, row.asset_id)
                for row in frame.itertuples()
            ),
        )

    def coverage_counts(self):
        return 0, 0


def _persist(root, batches):
    saved = write_portfolio_ledger_artifact(
        root,
        capital=PortfolioCapital("missing-score", 100.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2023-01-03", "end": "2023-01-09"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    restored = list(read_portfolio_ledger_artifact(saved).iter_chunks())
    assert restored == batches
    return restored


def _native(root, monkeypatch, missing, *, dividend=False, independent_exit=False):
    frames, fees, _ = _inputs()
    fees["rate"] = 0.0
    panel = frames["panel"]
    panel.loc[:, ["open", "close", "adj_open", "adj_close"]] = 100.0
    panel.loc[
        panel.asset_id.eq("b") & panel.timestamp.ge("2023-01-05"), ["open", "close"]
    ] = 90.0 if dividend else 120.0
    if dividend:
        frames["corporate_actions"] = pd.DataFrame(
            [
                {
                    "source_asset_id": "b",
                    "timestamp": pd.Timestamp("2023-01-05"),
                    "kind": "CA",
                    "market_type": 1,
                    "announcement_date": pd.Timestamp("2023-01-02"),
                    "settlement_date": pd.Timestamp("2023-01-06"),
                    "amount": 10.0,
                }
            ]
        )
    if independent_exit:
        frames["dataset"].loc[
            frames["dataset"].asset_id.eq("b")
            & frames["dataset"].timestamp.ge("2023-01-04"),
            "eligible",
        ] = False
    batches, _ = _build_producer(
        root / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=TimeWindow("2023-01-03", "2023-01-09"),
        prediction_lookup=_Forecasts(missing),
        typed_batches=True,
    )
    return _persist(root / "ledger", list(batches))


def _diagnostic(batch, name):
    return next(
        row
        for row in batch.diagnostics
        if row.asset_id == "__cash__" and row.name == name
    )


def test_all_missing_keeps_initial_cash_then_resumes(tmp_path, monkeypatch):
    batches = _native(
        tmp_path, monkeypatch, lambda day, _: day == pd.Timestamp("2023-01-03")
    )
    assert not batches[0].trades
    assert (
        _diagnostic(batches[0], "decision_skip_reason").text
        == "all_predictions_missing"
    )
    assert (
        next(
            row.value
            for row in batches[0].valuations
            if row.phase == "after" and row.asset_id == "__cash__"
        )
        == 100.0
    )
    assert batches[1].trades and all(trade.quantity > 0 for trade in batches[1].trades)


@pytest.mark.parametrize("partial", [False, True])
def test_missing_score_holds_exact_shares_across_opening_price_drift(
    tmp_path, monkeypatch, partial
):
    batches = _native(
        tmp_path,
        monkeypatch,
        lambda day, asset: (
            day >= pd.Timestamp("2023-01-04") and (not partial or asset == "b")
        ),
    )
    bought = next(
        trade.quantity for trade in batches[0].trades if trade.asset_id == "b"
    )
    assert bought == 1
    assert all(not batch.trades for batch in batches[1:])
    assert all(
        row.quantity == bought
        for batch in batches[1:]
        for row in batch.valuations
        if row.asset_id == "b"
    )
    assert _diagnostic(batches[1], "decision_status").text == (
        "PARTIAL" if partial else "SKIPPED"
    )


def test_missing_new_asset_cannot_enter(tmp_path, monkeypatch):
    batches = _native(tmp_path, monkeypatch, lambda _, asset: asset == "b")
    assert not any(trade.asset_id == "b" for batch in batches for trade in batch.trades)
    assert any(
        trade.asset_id == "a" and trade.quantity > 0
        for batch in batches
        for trade in batch.trades
    )


def test_missing_predictions_still_process_dividend_receivable_and_cash(
    tmp_path, monkeypatch
):
    batches = _native(
        tmp_path,
        monkeypatch,
        lambda day, _: day >= pd.Timestamp("2023-01-04"),
        dividend=True,
    )
    assert all(not batch.trades for batch in batches[1:])
    assert (
        sum(
            flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "dividend"
        )
        == 10.0
    )
    assert (
        sum(row.value for row in batches[-1].valuations if row.phase == "end") == 100.0
    )


def test_missing_predictions_do_not_cancel_independent_eligibility_exit(
    tmp_path, monkeypatch
):
    batches = _native(
        tmp_path,
        monkeypatch,
        lambda day, _: day >= pd.Timestamp("2023-01-04"),
        independent_exit=True,
    )
    assert any(
        trade.asset_id == "b" and trade.quantity < 0 for trade in batches[1].trades
    )
    assert not any(
        trade.quantity > 0 for batch in batches[1:] for trade in batch.trades
    )


def test_enhanced_missing_hold_infeasibility_skips_without_relaxation(tmp_path):
    args = _decision_inputs(holdings=(0.6, 0.4))
    args["alpha"].loc["a"] = np.nan
    account, adv, optimized, delta, ledger = production._build_session(**args)
    assert optimized is None
    assert account.asset_ledger.executed_delta_notional.eq(0.0).all()
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "optimizer_frozen_constraints_infeasible"
    )
    life = args["next_panel"].set_index("asset_id").listing_date
    batch = production._ledger_batch(
        account=account,
        asset_ledger=ledger,
        sequence=0,
        end_timestamp=pd.Timestamp("2023-01-05"),
        decision_timestamp=args["decision_timestamp"],
        held_units=args["held_units"],
        execution_units=args["held_units"],
        execution_lives=life,
        previous_values={},
        rights_adjustments=(),
        prediction_rank_score=args["alpha"],
        adv20=adv,
        enhanced_decision=args["enhanced_decision"],
        benchmark_weights=args["enhanced_decision"].benchmark_weights,
        optimization_result=optimized,
        decision_reason=args["decision_reason"],
        decision_delta=delta,
        next_panel=args["next_panel"],
        endpoint_panel=args["endpoint_panel"],
        endpoint_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        endpoint_corporate_events=pd.DataFrame(),
        entitlements=EquityEntitlements(),
    )
    assert not _persist(tmp_path / "ledger", [batch])[0].trades


@pytest.mark.parametrize(
    "error",
    [
        "optimizer_solution_invalid",
        "optimizer_covariance_invalid",
        "optimizer_failed_closed",
    ],
)
def test_numerical_or_input_failure_is_never_a_skipped_decision(monkeypatch, error):
    def invalid(*args, **kwargs):
        raise ValueError(error)

    monkeypatch.setattr(production, "optimize_portfolio", invalid)
    with pytest.raises(ValueError, match=f"^{error}$"):
        production._build_session(**_decision_inputs())


def test_partial_missing_neutral_decision_holds_without_new_projection():
    args = _decision_inputs()
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id=None,
        risk_style_columns=(),
        neutrality="size",
    )
    args["enhanced_decision"] = None
    args["alpha"].loc["a"] = np.nan
    result = production._build_session(**args)
    assert result[0].asset_ledger.executed_delta_notional.eq(0.0).all()
    assert (
        result[4].attrs["decision_availability"]["decision_skip_reason"]
        == "partial_predictions_missing_neutrality"
    )


@pytest.mark.parametrize(
    "capacity,turnover,reason",
    [
        (0.01, 1.0, "optimizer_capacity_bounds_infeasible"),
        (1.0, 0.1, "optimizer_turnover_bounds_infeasible"),
    ],
)
def test_proven_economic_bounds_skip_without_solver_relaxation(
    capacity, turnover, reason
):
    args = _decision_inputs(holdings=(1.0, 0.0))
    args["assumptions"] = replace(args["assumptions"], adv_participation=capacity)
    args["optimizer_config"] = production.OptimizerConfig(
        target_turnover_limit=turnover
    )
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is None
    assert account.asset_ledger.executed_delta_notional.eq(0.0).all()
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] == reason


@pytest.mark.parametrize("unknown_weight", [0.0, 0.05])
def test_neutrality_diagnostics_use_stock_leg_and_disclose_unknown_reference(
    monkeypatch, unknown_weight
):
    args = _decision_inputs(holdings=(0.25, 0.25))
    args["cash"] = 50.0
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id=None,
        risk_style_columns=(),
        neutrality="size",
    )
    args["enhanced_decision"] = None
    args["alpha"].loc[:] = np.nan
    context = pd.DataFrame(
        {"benchmark_weight": (1.0 - unknown_weight) / 2.0, "size": 1.0},
        index=["a", "b"],
    )
    context.attrs["unknown_industry_weight"] = unknown_weight
    monkeypatch.setattr(production, "_neutrality_context", lambda *a, **k: context)
    account, _, _, _, ledger = production._build_session(**args)
    assert account.post_trade_cash == 50.0
    diagnostics = ledger.attrs["neutrality_exposures"]
    assert diagnostics["benchmark_unknown_industry_weight"] == unknown_weight
    assert diagnostics["benchmark_exposure:size"] == 1.0 - unknown_weight
    for phase in ("target", "realized"):
        if unknown_weight:
            assert diagnostics[f"{phase}_exposure_status"] == "UNDEFINED"
            assert (
                diagnostics[f"{phase}_exposure_reason"] == "benchmark_unknown_industry"
            )
        else:
            assert diagnostics[f"{phase}_exposure:size"] == 1.0


@pytest.mark.parametrize("technical_fill", [False, True])
def test_prediction_domain_accepts_partial_original_inputs_not_all_missing(
    technical_fill,
):
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.Timestamp("2023-01-04"),
            "eligible": True,
            "feature_x": [np.nan, 1.0, np.nan],
            "feature_y": [2.0, np.nan, np.nan],
        }
    )
    if technical_fill:
        for name in ("x", "y"):
            frame[f"feature_observed_{name}"] = (
                frame[f"feature_{name}"].notna().astype(int)
            )
            frame[f"feature_{name}"] = frame[f"feature_{name}"].fillna(0.0)
    result = production._prediction_alpha(
        frame,
        _Forecasts(lambda *_: False),
        ("feature_x", "feature_y"),
        declared_columns=frame.columns,
    )
    assert result.loc[["a", "b"]].tolist() == [0.0, 1.0]
    assert pd.isna(result.loc["c"])
