"""Keep decision feasibility separate from future execution constraints."""

from collections import deque
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    fee_schedule_from_frame,
    prepare_trade_cost,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.risk import FactorRiskEstimate
from tests.research.test_portfolio_production import _inputs


def _decision_inputs(*, benchmark=(0.5, 0.5), holdings=None, fee_rate=0.0):
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
    _, fees, _ = _inputs()
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


def test_future_execution_changes_actual_constraints_without_changing_target(tmp_path):
    base = _decision_inputs()
    base["execution"].loc[base["execution"].asset_id.eq("a"), "can_exit"] = False
    baseline = production._build_session(**base)
    changed = _decision_inputs()
    changed["next_panel"].loc[changed["next_panel"].asset_id.eq("a"), "open"] = 150.0
    changed["execution"].loc[changed["execution"].asset_id.eq("a"), "can_exit"] = False
    account, adv, optimized, delta, snapshot = production._build_session(**changed)
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
        production._build_session(**changed)
    )
    pd.testing.assert_series_equal(tradable_target.weights, optimized.weights)
    pd.testing.assert_series_equal(tradable_delta, delta)
    assert (
        tradable_snapshot.attrs["realized_constraints"]["realized_constraints_status"]
        == "SATISFIED"
    )
    assert tradable.asset_ledger.executed_delta_notional.tolist() == pytest.approx(
        [-11.875, 11.875]
    )
    batch = production._ledger_batch(
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
        entitlements=production.EquityEntitlements(),
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
    assert restored == [batch]
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
    solve = production.optimize_portfolio

    def observe(*args, **kwargs):
        result = solve(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(production, "optimize_portfolio", observe)
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
            account, _, optimized, _, ledger = production._build_session(**inputs)
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
            account, _, _, _, _ = production._build_session(**inputs)
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
    _, _, optimized, _, snapshot = production._build_session(**inputs)
    assert optimized.predicted_tracking_error == 0.0
    assert snapshot.attrs["realized_constraints"]["realized_tracking_error"] == 0.0
    inputs["next_panel"].loc[inputs["next_panel"].asset_id.eq("a"), "open"] = 150.0
    inputs["execution"]["can_exit"] = False
    _, _, changed, _, snapshot = production._build_session(**inputs)
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
    account, _, optimized, _, ledger = production._build_session(**inputs)
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
    account, _, _, _, snapshot = production._build_session(**inputs)
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
    baseline = production._build_session(**inputs)
    inputs["next_panel"]["open"] = 0.0
    account, _, optimized, _, snapshot = production._build_session(**inputs)
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
        production._build_session(**inputs)


@pytest.mark.parametrize("missing", ["specific", "exposure"])
@pytest.mark.parametrize("fee_rate", [0.0002, 0.005])
def test_fee_normalized_missing_risk_survives_actual_account_projection(
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
    account, _, optimized, _, ledger = production._build_session(**inputs)
    assert optimized is not None
    # This fee fixture charges commission and transfer fee at fee_rate each.
    rate = 2.0 * fee_rate
    fee = 0.2 * rate / (1.0 + rate)
    assert account.nav_after == pytest.approx(100.0 * (1.0 - fee), abs=1e-10)
    assert ledger.post_trade_notional.tolist() == pytest.approx(
        [50.0 * (1.0 - fee)] * 2, abs=1e-10
    )
    assert account.post_trade_cash == pytest.approx(0.0, abs=1e-10)
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "SATISFIED"
    assert facts["realized_tracking_error_status"] == "DEFINED"
    assert facts["realized_tracking_error"] == pytest.approx(0.0, abs=1e-14)
    assert optimized.predicted_tracking_error == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("missing", ["specific", "exposure"])
@pytest.mark.parametrize("benchmark_a, relative_move", [(0.5, 4e-12), (1e-12, 1e-8)])
def test_genuine_tiny_future_active_weight_still_requires_missing_risk(
    missing, benchmark_a, relative_move
):
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
    inputs["next_panel"].loc[inputs["next_panel"].asset_id.eq("a"), "open"] = 100.0 * (
        1.0 + relative_move
    )
    account, _, optimized, _, ledger = production._build_session(**inputs)
    assert optimized is not None
    assert optimized.predicted_tracking_error == 0.0
    active = (
        ledger.set_index("asset_id").loc["a", "post_trade_notional"] / account.nav_after
        - benchmark_a
    )
    assert 0.0 < active < 2e-12
    # A tiny benchmark position must use a relative rounding bound. Its true
    # 1e-20 active weight is far below a global epsilon or economic threshold.
    assert active > 100.0 * np.finfo(float).eps * benchmark_a
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_constraints_reason"] == "active_risk_unavailable"
    assert facts["realized_tracking_error_status"] == "UNDEFINED"
    assert "realized_tracking_error" not in facts


def test_post_projection_guard_rejects_corrupted_real_optimizer_target(monkeypatch):
    real = production.optimize_portfolio
    solved = []

    def corrupted_target(*args, **kwargs):
        result = real(*args, **kwargs)
        solved.append(result)
        # Deliberately corrupt a real solved target to exercise the defensive
        # projection guard, not a claim that correctly funded fees are infeasible.
        weights = pd.Series({"a": 0.6, "b": 0.4, "__cash__": 0.0})
        return replace(result, weights=weights.reindex(result.weights.index))

    monkeypatch.setattr(production, "optimize_portfolio", corrupted_target)
    inputs = _decision_inputs(fee_rate=0.001)
    account, _, optimized, _, ledger = production._build_session(**inputs)
    assert len(solved) == 1
    assert optimized is None
    availability = ledger.attrs["decision_availability"]
    assert availability["decision_status"] == "SKIPPED"
    assert (
        availability["decision_skip_reason"]
        == "portfolio_post_trade_constraints_infeasible"
    )
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert ledger.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].eq(0.0).all().all()
    pd.testing.assert_series_equal(
        ledger.set_index("asset_id").post_trade_notional,
        inputs["holdings"],
        check_names=False,
        check_exact=True,
    )
    assert account.nav_after == account.nav_before == 100.0
    assert account.post_trade_cash == 0.0
