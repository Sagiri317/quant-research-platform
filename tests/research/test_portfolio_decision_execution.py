"""Keep decision feasibility separate from future execution constraints."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.portfolios.daily as daily
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioCapital,
)
from tests.helpers.portfolio_inputs import decision_inputs as _decision_inputs


def test_future_execution_changes_actual_constraints_without_changing_target(tmp_path):
    base = _decision_inputs()
    base["execution"].loc[base["execution"].asset_id.eq("a"), "can_exit"] = False
    baseline = daily._build_session(**base)
    changed = _decision_inputs()
    changed["next_panel"].loc[changed["next_panel"].asset_id.eq("a"), "open"] = 0.015
    changed["execution"].loc[changed["execution"].asset_id.eq("a"), "can_exit"] = False
    account, adv, optimized, delta, snapshot = daily._build_session(**changed)
    pd.testing.assert_series_equal(optimized.weights, baseline[2].weights)
    pd.testing.assert_series_equal(delta, baseline[3])
    assert optimized.weights.tolist() == pytest.approx([0.505, 0.495, 0.0])
    assert optimized.predicted_tracking_error == baseline[2].predicted_tracking_error
    assert (
        baseline[4].attrs["realized_constraints"]["realized_constraints_status"]
        == "SATISFIED"
    )
    facts = snapshot.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "BREACHED"
    assert facts["realized_constraints_reason"] == "name_active_weight"
    assert facts["realized_name_active_weight"] == pytest.approx(0.1)
    assert facts["realized_tracking_error"] == pytest.approx(np.sqrt(0.0002))
    assert snapshot.executed_delta_notional.eq(0.0).all()
    assert snapshot.post_trade_notional.tolist() == [75.0, 50.0]
    assert account.nav_after == 125.0 and account.post_trade_cash == 0.0
    changed["execution"]["can_exit"] = True
    tradable, _, tradable_target, tradable_delta, tradable_snapshot = (
        daily._build_session(**changed)
    )
    pd.testing.assert_series_equal(tradable_target.weights, optimized.weights)
    pd.testing.assert_series_equal(tradable_delta, delta)
    assert (
        tradable_snapshot.attrs["realized_constraints"]["realized_constraints_status"]
        == "SATISFIED"
    )
    assert tradable.asset_ledger.executed_delta_notional.tolist() == pytest.approx(
        [-12.0, 12.0]
    )
    batch = daily._ledger_batch(
        account=account,
        asset_ledger=snapshot,
        sequence=0,
        end_timestamp=changed["endpoint_panel"].timestamp.iloc[0],
        decision_timestamp=changed["decision_timestamp"],
        held_units=changed["held_units"],
        execution_units=changed["held_units"],
        previous_values={},
        rights_adjustments=(),
        prediction_rank_score=changed["alpha"],
        adv20=adv,
        enhanced_decision=changed["enhanced_decision"],
        benchmark_weights=changed["enhanced_decision"].benchmark_weights,
        optimization_result=optimized,
        decision_reason="enhanced_optimizer",
        decision_delta=delta,
        next_panel=changed["next_panel"],
        endpoint_panel=changed["endpoint_panel"],
        endpoint_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        endpoint_corporate_events=pd.DataFrame(),
        entitlements=daily.EquityEntitlements(),
        execution_lives=changed["next_panel"].set_index("asset_id").listing_date,
    )
    path = write_portfolio_ledger_artifact(
        tmp_path / "drift",
        capital=PortfolioCapital("timing", 125.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2023-01-03", "end": "2023-01-05"}},
        produce=lambda emit: emit(batch),
    )
    saved = read_portfolio_ledger_artifact(path)
    restored = list(saved.iter_chunks())
    diagnostics = {
        row.name: row.number if row.number is not None else row.text
        for row in restored[0].diagnostics
        if row.asset_id == "__cash__"
    }
    assert diagnostics["realized_constraints_status"] == "BREACHED"
    assert diagnostics["realized_tracking_error"] == facts["realized_tracking_error"]
    assert not restored[0].trades
    assert summarize_portfolio_ledger(saved.capital, iter(restored))["nav"] == 100.0


def test_known_fees_cannot_make_frozen_holding_exceed_decision_limit(monkeypatch):
    captured = []
    solve = daily.optimize_portfolio

    def observe(*args, **kwargs):
        result = solve(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(daily, "optimize_portfolio", observe)
    for rate in (0.0, 0.001):
        inputs = _decision_inputs(
            benchmark=(0.5, 0.25, 0.25), holdings=(0.505, 0.245, 0.25), fee_rate=rate
        )
        inputs["enhanced_decision"] = replace(
            inputs["enhanced_decision"],
            expected_alpha=pd.Series([0.0, 1.0, 0.0], index=list("abc")),
        )
        inputs["amount_history"].pop(("a", pd.Timestamp("2000-01-01")))
        if rate:
            account, _, optimized, _, ledger = daily._build_session(**inputs)
            # Frozen a is already at .505. Any strictly positive fee c would
            # make its net weight .505/(1-c) > .505, so only zero-cost no-trade
            # is feasible. The correct optimizer returns that optimum.
            assert optimized is not None
            assert account.asset_ledger.executed_delta_notional.to_numpy() == (
                pytest.approx(np.zeros(3), abs=1e-10)
            )
            assert account.nav_after == pytest.approx(account.nav_before, abs=1e-10)
            assert (
                ledger.attrs["realized_constraints"]["realized_constraints_status"]
                == "SATISFIED"
            )
        else:
            account, _, _, _, _ = daily._build_session(**inputs)
            assert (
                account.asset_ledger.set_index("asset_id").loc[
                    "a", "post_trade_notional"
                ]
                == 50.5
            )
    assert len(captured) == 2
    assert captured[0].weights.tolist() == pytest.approx([0.505, 0.25, 0.245, 0.0])
    assert captured[1].weights.tolist() == pytest.approx(
        [0.505, 0.245, 0.25, 0.0], abs=1e-12
    )
    assert captured[1].expected_transaction_cost == pytest.approx(0.0, abs=1e-12)
    assert captured[1].weights.loc["a"] * 100.0 == 50.5


@pytest.mark.parametrize("missing", ["specific", "exposure"])
def test_exact_zero_active_missing_risk_is_zero_but_future_active_is_undefined(missing):
    inputs = _decision_inputs()
    enhanced = inputs["enhanced_decision"]
    inputs["enhanced_decision"] = replace(
        enhanced, expected_alpha=enhanced.expected_alpha * 0.0
    )
    if missing == "specific":
        enhanced.risk.specific_variance.loc["a"] = np.nan
    else:
        enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.nan
    _, _, optimized, _, snapshot = daily._build_session(**inputs)
    assert optimized.predicted_tracking_error == 0.0
    assert snapshot.attrs["realized_constraints"]["realized_tracking_error"] == 0.0
    inputs["next_panel"].loc[inputs["next_panel"].asset_id.eq("a"), "open"] = 0.015
    inputs["execution"]["can_exit"] = False
    _, _, changed, _, snapshot = daily._build_session(**inputs)
    pd.testing.assert_series_equal(changed.weights, optimized.weights)
    facts = snapshot.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_constraints_reason"] == "active_risk_unavailable"
    assert facts["realized_tracking_error_status"] == "UNDEFINED"
    assert "realized_tracking_error" not in facts
    assert facts["realized_name_active_weight"] == pytest.approx(0.1)
    assert facts["realized_style_active_exposure_status"] == (
        "DEFINED" if missing == "specific" else "UNDEFINED"
    )
    if missing == "specific":
        assert facts["realized_style_active_exposure"] == pytest.approx(0.0, abs=1e-15)
    else:
        assert "realized_style_active_exposure" not in facts


@pytest.mark.parametrize("missing", ["specific", "exposure"])
def test_frozen_nonzero_missing_risk_skips_target_and_keeps_real_account(missing):
    inputs = _decision_inputs(holdings=(0.501, 0.499))
    inputs["amount_history"].pop(("a", pd.Timestamp("2000-01-01")))
    enhanced = inputs["enhanced_decision"]
    if missing == "specific":
        enhanced.risk.specific_variance.loc["a"] = np.nan
    else:
        enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.nan
    account, _, optimized, _, ledger = daily._build_session(**inputs)
    assert optimized is None
    assert account.nav_after == 100.0
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert ledger.post_trade_notional.tolist() == [50.1, 49.9]
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "optimizer_held_risk_unavailable"
    )
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_constraints_reason"] == "active_risk_unavailable"
    assert facts["realized_tracking_error_status"] == "UNDEFINED"
    assert "realized_tracking_error" not in facts
    assert facts["realized_style_active_exposure_status"] == (
        "DEFINED" if missing == "specific" else "UNDEFINED"
    )


@pytest.mark.parametrize("objective", ["rank_to_risk", "net_return_to_risk"])
def test_pool_exit_without_forecast_keeps_weekly_target_and_real_execution(objective):
    inputs = _decision_inputs(holdings=(0.2, 0.8))
    inputs["candidate"] = replace(
        inputs["candidate"], construction="robust_long_only", benchmark_id=None
    )
    inputs["optimizer_config"] = daily.AbsoluteRiskConfig(fully_invested=True)
    inputs["robust_config"] = daily.RobustPortfolioConfig(objective=objective)
    inputs["dataset"].loc[0, "eligible"] = False
    inputs["alpha"] = pd.Series({"a": np.nan, "b": 0.1})
    inputs["enhanced_decision"] = replace(
        inputs["enhanced_decision"],
        benchmark_weights=pd.Series(dtype=float),
        expected_alpha=inputs["alpha"],
        holding_sessions=5,
    )
    _, _, target, _, book = daily._build_session(**inputs)
    assert target.weights.loc["a"] == 0
    assert book.set_index("asset_id").at["a", "post_trade_notional"] == 0
    inputs["execution"].loc[0, "can_exit"] = False
    _, _, blocked, _, book = daily._build_session(**inputs)
    pd.testing.assert_series_equal(blocked.weights, target.weights)
    assert book.set_index("asset_id").at["a", "post_trade_notional"] == pytest.approx(
        20.0
    )
    inputs["locked_fractions"] = {"a": 0.05}
    _, _, partial, delta, _ = daily._build_session(**inputs)
    assert partial.weights["a"] == pytest.approx(0.01)
    assert partial.unscored_target_weight == pytest.approx(0.01)
    assert delta["a"] == pytest.approx(-0.19)
    inputs.pop("locked_fractions")
    inputs["amount_history"].pop(("a", pd.Timestamp("2000-01-01")))
    _, _, frozen, delta, book = daily._build_session(**inputs)
    assert frozen.weights["a"] == pytest.approx(0.2)
    assert delta["a"] == pytest.approx(0)
    assert book.attrs["decision_availability"]["decision_skip_reason"] is None


def test_cash_week_keeps_stock_when_actual_minimum_commission_exceeds_expected_loss():
    inputs = _decision_inputs(holdings=(0.5, 0.5), fee_rate=0.0001)
    inputs["candidate"] = replace(
        inputs["candidate"], construction="robust_long_only", benchmark_id=None
    )
    inputs["optimizer_config"] = daily.AbsoluteRiskConfig(fully_invested=True)
    inputs["robust_config"] = daily.RobustPortfolioConfig(
        objective="net_return_to_risk"
    )
    inputs["assumptions"] = replace(inputs["assumptions"], minimum_commission=5.0)
    alpha = pd.Series({"a": -0.02, "b": -0.02})
    inputs["alpha"] = alpha
    inputs["enhanced_decision"] = replace(
        inputs["enhanced_decision"],
        expected_alpha=alpha,
        benchmark_weights=pd.Series(dtype=float),
        holding_sessions=5,
    )
    _, _, target, _, book = daily._build_session(**inputs)
    assert target.cash_on_nonpositive_return
    assert target.weights.loc[["a", "b"]].sum() == pytest.approx(0, abs=2e-7)
    assert book.executed_delta_notional.eq(0).all()


@pytest.mark.parametrize("can_sell", [True, False])
def test_single_name_repair_only_trades_the_requested_book(can_sell, monkeypatch):
    inputs = _decision_inputs(
        benchmark=(0.45, 0.45, 0.1), holdings=(0.45, 0.30, 0.0), fee_rate=0.0002
    )
    inputs["cash"] = 25.0
    inputs["candidate"] = replace(
        inputs["candidate"], construction="robust_long_only", benchmark_id=None
    )
    inputs["optimizer_config"] = daily.AbsoluteRiskConfig(
        fully_invested=True, name_weight_limit=0.55
    )
    inputs["robust_config"] = daily.RobustPortfolioConfig(objective="rank_to_risk")
    inputs["alpha"] = pd.Series([0.9, 0.25, 0.4], index=list("abc"))
    enhanced = inputs["enhanced_decision"]
    enhanced.risk.specific_variance["c"] = np.nan
    inputs["enhanced_decision"] = replace(
        enhanced, benchmark_weights=pd.Series(dtype=float), holding_sessions=5
    )
    inputs["next_panel"].loc[0, "open"] = 0.023
    inputs["execution"].loc[0, "can_exit"] = can_sell
    execute = daily.execute_stock_target

    def execute_requested_book(orders, **kwargs):
        # C has no position and is excluded from the target by missing risk.
        assert orders.at["c", "target_quantity"] == 0
        return execute(orders, **kwargs)

    monkeypatch.setattr(daily, "execute_stock_target", execute_requested_book)
    account, _, target, _, book = daily._build_session(**inputs)
    assert target.weights["c"] == 0.0
    # A is worth 103.5 of 158.5. Selling 700 shares leaves 87.4, above 55%;
    # selling 800 brings it to 85.1. If A is frozen the 25 cash can still buy B.
    assert book.set_index("asset_id").at["a", "executed_quantity"] == (
        -800 if can_sell else 0
    )
    if not can_sell:
        assert book.set_index("asset_id").at["b", "executed_quantity"] == 2400
    facts = book.attrs["realized_constraints"]
    assert facts["realized_name_weight"] == pytest.approx(
        (85.1 if can_sell else 103.5) / account.nav_after
    )
    assert facts["realized_constraints_status"] == (
        "SATISFIED" if can_sell else "BREACHED"
    )


@pytest.mark.parametrize("sellable_lots", [0, 5])
def test_adjacent_lot_improves_other_positions_with_a_frozen_overweight(sellable_lots):
    inputs = _decision_inputs(benchmark=(0.5, 0.2, 0.3), holdings=(0.45, 0.15, 0.0))
    inputs["cash"] = 40.0
    inputs["candidate"] = replace(
        inputs["candidate"], construction="robust_long_only", benchmark_id=None
    )
    inputs["optimizer_config"] = daily.AbsoluteRiskConfig(
        fully_invested=True, name_weight_limit=0.55
    )
    inputs["robust_config"] = daily.RobustPortfolioConfig(objective="rank_to_risk")
    inputs["alpha"] = 0.1 * (1.0 + pd.Series([0.54, 0.1615, 0.2985], index=list("abc")))
    inputs["enhanced_decision"] = replace(
        inputs["enhanced_decision"],
        benchmark_weights=pd.Series(dtype=float),
        holding_sessions=5,
    )
    inputs["next_panel"].loc[0, "open"] = 0.03
    inputs["execution"].loc[0, "can_exit"] = bool(sellable_lots)
    if sellable_lots:
        inputs["amount_history"]["a", pd.Timestamp("2000-01-01")] = [15.0] * 20
    inputs["execution"].loc[2, "can_enter"] = False
    _, _, target, _, book = daily._build_session(**inputs)
    assert target.weights["b"] == pytest.approx(0.1615, abs=1e-6)
    # B's target is 3068.5 shares at the open. Buying 1600, from 1500,
    # invests the adjacent lot. Each 100-share lot sold reduces A by 3.
    assert book.executed_quantity.tolist() == [-100 * sellable_lots, 1600, 0]
    facts = book.attrs["realized_constraints"]
    assert facts["realized_name_weight"] == pytest.approx(
        (135.0 - 3.0 * sellable_lots) / 190.0
    )
    assert facts["realized_constraints_status"] == "BREACHED"


def test_zero_nav_neutrality_has_no_defined_weights_or_exposures():
    inputs = _decision_inputs()
    inputs.update(
        candidate=PortfolioCandidateSpec(
            "zero",
            "global",
            "quantile_equal_weight",
            "champion",
            "daily",
            1.0,
            None,
            neutrality="size",
        ),
        enhanced_decision=None,
        holdings=inputs["holdings"] * 0.0,
        held_units=inputs["held_units"] * 0,
        nav=0.0,
    )
    account, _, _, _, snapshot = daily._build_session(**inputs)
    assert account.nav_after == 0.0
    assert snapshot.realized_weight.isna().all()
    assert snapshot.target_weight.isna().all()
    for kind in ("target", "realized"):
        assert (
            snapshot.attrs["neutrality_exposures"][f"{kind}_exposure_status"]
            == "UNDEFINED"
        )
        assert (
            snapshot.attrs["neutrality_exposures"][f"{kind}_exposure_reason"]
            == "nonpositive_nav"
        )


def test_future_zero_nav_keeps_valid_target_and_reports_undefined_actual_risk():
    inputs = _decision_inputs()
    baseline = daily._build_session(**inputs)
    inputs["next_panel"]["open"] = 0.0
    account, _, optimized, _, snapshot = daily._build_session(**inputs)
    pd.testing.assert_series_equal(optimized.weights, baseline[2].weights)
    assert account.nav_after == 0.0
    assert snapshot.executed_delta_notional.eq(0.0).all()
    assert snapshot.attrs["realized_constraints"] == {
        "realized_constraints_status": "UNDEFINED",
        "realized_constraints_reason": "nonpositive_nav",
    }


@pytest.mark.parametrize("invalid", ["negative_variance", "infinite_exposure"])
def test_invalid_risk_is_not_downgraded_to_missing_target(invalid):
    inputs = _decision_inputs(holdings=(0.501, 0.499))
    inputs["amount_history"].pop(("a", pd.Timestamp("2000-01-01")))
    enhanced = inputs["enhanced_decision"]
    if invalid == "negative_variance":
        enhanced.risk.specific_variance.loc["a"] = -0.1
    else:
        enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.inf
    with pytest.raises(ValueError, match="optimizer_risk_invalid"):
        daily._build_session(**inputs)


@pytest.mark.parametrize("missing", ["specific", "exposure"])
@pytest.mark.parametrize("fee_rate", [0.0002, 0.005])
def test_integer_fees_do_not_turn_missing_active_risk_into_defined_zero(
    missing, fee_rate
):
    inputs = _decision_inputs(holdings=(0.4, 0.4), fee_rate=fee_rate)
    inputs["cash"] = 20.0
    enhanced = inputs["enhanced_decision"]
    inputs["enhanced_decision"] = replace(
        enhanced, expected_alpha=pd.Series([0.0, 0.2], index=["a", "b"])
    )
    if missing == "specific":
        enhanced.risk.specific_variance.loc["a"] = np.nan
    else:
        enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.nan
    account, _, optimized, _, ledger = daily._build_session(**inputs)
    assert optimized is not None
    # The continuous benchmark target is reachable; integer fills and fees
    # cannot establish exactly zero active exposure to the missing risk.
    assert account.nav_after == 100.0
    assert ledger.post_trade_notional.tolist() == [40.0, 40.0]
    assert account.post_trade_cash == 20.0
    assert ledger.executed_quantity.eq(0).all()
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_tracking_error_status"] == "UNDEFINED"
    assert "realized_tracking_error" not in facts
    fee_ratio = optimized.expected_transaction_cost / inputs["nav"]
    net_stock = optimized.weights.reindex(["a", "b"]) / (1.0 - fee_ratio)
    # Only normalization-sized unknown coefficients may be omitted. The
    # known stock's actual residual remains real factor and specific risk.
    assert abs(net_stock.loc["a"] - 0.5) <= (
        8 * np.finfo(float).eps * max(abs(net_stock.loc["a"]), 0.5)
    )
    expected_tracking_error = np.sqrt(0.01 + 0.01) * abs(net_stock.loc["b"] - 0.5)
    assert optimized.predicted_tracking_error == pytest.approx(
        expected_tracking_error, abs=1e-16, rel=1e-12
    )


@pytest.mark.parametrize("missing", ["specific", "exposure"])
def test_one_cent_future_price_move_requires_missing_active_risk(missing):
    benchmark_a = 0.01
    benchmark = (benchmark_a, 1.0 - benchmark_a)
    inputs = _decision_inputs(benchmark=benchmark)
    enhanced = inputs["enhanced_decision"]
    inputs["enhanced_decision"] = replace(
        enhanced, expected_alpha=pd.Series([0.0, 0.0], index=["a", "b"])
    )
    if missing == "specific":
        enhanced.risk.specific_variance.loc["a"] = np.nan
    else:
        enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.nan
    inputs["execution"][["can_enter", "can_exit"]] = False
    inputs["next_panel"].loc[inputs["next_panel"].asset_id.eq("a"), "open"] = 0.02
    account, _, optimized, _, ledger = daily._build_session(**inputs)
    assert optimized is not None
    assert optimized.predicted_tracking_error == 0.0
    active = (
        ledger.set_index("asset_id").loc["a", "post_trade_notional"] / account.nav_after
        - benchmark_a
    )
    assert active == pytest.approx(2.0 / 101.0 - benchmark_a)
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_constraints_reason"] == "active_risk_unavailable"
    assert facts["realized_tracking_error_status"] == "UNDEFINED"
    assert "realized_tracking_error" not in facts
