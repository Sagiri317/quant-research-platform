"""Explicit exits survive forecast gaps and ordinary schedule boundaries."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_MAINTENANCE,
)
from quant_research_platform.research.portfolios import production
from tests.research.test_portfolio_decision_execution import _decision_inputs


@pytest.mark.parametrize(
    "construction",
    ["quantile_equal_weight", "quantile_float_cap", "quantile_equal_weight_half_step"],
)
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("can_exit", [True, False, None])
def test_off_schedule_research_exit_is_independent_of_unrelated_forecast(
    construction, missing, can_exit
):
    args = _decision_inputs()
    args["candidate"] = replace(
        args["candidate"],
        construction=construction,
        benchmark_id=None,
        top_quantile=0.5,
        rebalance_schedule="weekly-last-trading-day",
    )
    args["enhanced_decision"] = None
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    args["alpha"].loc["b"] = np.nan
    if missing:
        args["alpha"].loc["a"] = np.nan
    args["execution"]["can_exit"] = args["execution"]["can_exit"].astype("boolean")
    args["execution"].loc[args["execution"].asset_id.eq("b"), "can_exit"] = can_exit
    account, _, _, delta, ledger = production._build_session(**args)
    rows = ledger.set_index("asset_id")
    assert rows.loc["a", "executed_delta_notional"] == 0.0
    assert rows.loc["a", "post_trade_notional"] == 50.0
    assert rows.loc["b", "target_notional"] == 0.0
    assert delta.loc["b"] == -0.5
    assert rows.loc["b", "executed_delta_notional"] == (-50.0 if can_exit else 0.0)
    assert rows.loc["b", "post_trade_notional"] == (0.0 if can_exit else 50.0)
    assert account.post_trade_cash == (50.0 if can_exit else 0.0)
    if not can_exit:
        assert rows.loc["b", "reason"] == (
            "execution_fact_unknown" if can_exit is None else "market_blocked"
        )


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("can_exit", [True, False, None])
def test_proved_benchmark_removal_exits_despite_missing_forecasts(missing, can_exit):
    args = _decision_inputs()
    args["enhanced_decision"].benchmark_weights.drop("b", inplace=True)
    args["enhanced_decision"].benchmark_weights.loc["a"] = 1.0
    args["target_override"] = args["enhanced_decision"].benchmark_weights.copy()
    args["decision_reason"] = PORTFOLIO_DECISION_REASON_MAINTENANCE
    if missing:
        args["alpha"][:] = np.nan
    args["execution"]["can_exit"] = args["execution"]["can_exit"].astype("boolean")
    args["execution"].loc[args["execution"].asset_id.eq("b"), "can_exit"] = can_exit
    account, _, _, delta, ledger = production._build_session(**args)
    rows = ledger.set_index("asset_id")
    assert rows.loc["b", "target_notional"] == 0.0
    assert delta.loc["b"] == -0.5
    assert rows.loc["b", "executed_delta_notional"] == (-50.0 if can_exit else 0.0)
    assert rows.loc["b", "post_trade_notional"] == (0.0 if can_exit else 50.0)
    if missing:
        assert rows.loc["a", "executed_delta_notional"] == 0.0
        assert rows.loc["a", "post_trade_notional"] == 50.0
        assert account.post_trade_cash == (50.0 if can_exit else 0.0)
    if not can_exit:
        assert rows.loc["b", "reason"] == (
            "execution_fact_unknown" if can_exit is None else "market_blocked"
        )


def test_missing_benchmark_snapshot_is_not_removal_evidence():
    args = _decision_inputs()
    args["target_override"] = pd.Series(dtype="float64")
    args["decision_reason"] = PORTFOLIO_DECISION_REASON_MAINTENANCE
    args["alpha"][:] = np.nan
    account, _, _, delta, ledger = production._build_session(**args)
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert ledger.post_trade_notional.tolist() == [50.0, 50.0]
    assert delta.eq(0.0).all()


@pytest.mark.parametrize("missing", [False, True])
def test_off_schedule_independent_exit_respects_actual_capacity(missing):
    args = _decision_inputs()
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        benchmark_id=None,
        top_quantile=0.5,
        rebalance_schedule="weekly-last-trading-day",
    )
    args["enhanced_decision"] = None
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    args["alpha"].loc["b"] = np.nan
    if missing:
        args["alpha"].loc["a"] = np.nan
    args["assumptions"] = replace(
        args["assumptions"], adv_participation=0.01, adv_sensitivities=(0.01,)
    )
    account, _, _, delta, ledger = production._build_session(**args)
    rows = ledger.set_index("asset_id")
    assert rows.loc["b", "target_notional"] == 0.0
    assert rows.loc["b", "executed_delta_notional"] == -10.0
    assert rows.loc["b", "post_trade_notional"] == 40.0
    assert rows.loc["a", "executed_delta_notional"] == 0.0
    assert account.post_trade_cash == 10.0
    assert delta.loc["b"] == -0.5
