"""Independent economic oracles for the smooth transaction-cost objective."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    PreparedTradeCost,
)
from quant_research_platform.research.portfolios import optimizer, production
from tests.helpers.optimizer import with_cash_asset
from tests.helpers.portfolio_inputs import decision_inputs as _decision_inputs


def _owner(rate, *, impact=0.0):
    return PreparedTradeCost(
        (rate, 0.0, 0.0),
        (False, False, False),
        1.0,
        impact,
        "synthetic",
    )


def _cost(buy_owner, sell_owner, adv):
    def evaluate(buy, sell):
        buy_cost, buy_gradient = buy_owner.totals_and_gradients(buy[:-1], adv)
        sell_cost, sell_gradient = sell_owner.totals_and_gradients(sell[:-1], adv)
        return (
            float(buy_cost.sum() + sell_cost.sum()),
            np.r_[buy_gradient, 0.0],
            np.r_[sell_gradient, 0.0],
        )

    return evaluate


@pytest.mark.parametrize("amount", [-10.0, 0.0, 1e-10, 10.0, 1e6])
def test_prepared_cost_gradient_matches_its_economic_amount_formula(amount):
    owner = _owner(0.001, impact=10.0)
    adv = np.array([1e6])
    costs, gradient = owner.totals_and_gradients(np.array([amount]), adv)
    linear = 0.0011
    positive = max(amount, 0.0)
    expected = linear * amount + 0.001 * positive * np.sqrt(positive / adv[0])
    expected_gradient = linear + 0.0015 * np.sqrt(positive / adv[0])
    assert costs[0] == pytest.approx(expected, rel=1e-13, abs=1e-28)
    assert gradient[0] == pytest.approx(expected_gradient, rel=1e-13)
    if amount >= 0.0:
        assert costs == pytest.approx(owner.totals(np.array([amount]), adv), rel=1e-14)


@pytest.mark.parametrize(
    "seed, scale",
    [
        (0, "normal"),  # Both name bands, without binding cash.
        (1, "normal"),  # Both name bands and cash.
        (5, "normal"),  # Only the lower name band binds.
        (0, "capacity"),
        (2, "capacity"),  # Capacity and cash bind together.
        (1, "small_account"),  # Same active set under ordinary NAV rescaling.
    ],
)
def test_side_aware_cost_reaches_independent_lp_optimum(seed, scale):
    rng = np.random.default_rng(seed)
    n = 12
    benchmark = rng.dirichlet(np.full(n, 3.0))
    alpha = rng.normal(0.0, 0.002, n)
    style = rng.normal(size=n)
    style -= benchmark @ style
    nav = 1e4 if scale == "small_account" else 1e9
    capacity = (
        rng.uniform(0.0001, 0.001, n) if scale == "capacity" else np.full(n, 0.05)
    )
    adv = capacity * nav / 0.05
    buy, sell = _owner(0.0007), _owner(0.0011)
    evaluate = _cost(buy, sell, adv)
    assets = pd.DataFrame(
        dict(
            asset_id=[f"a{i}" for i in range(n)],
            expected_alpha=alpha,
            current_weight=benchmark,
            benchmark_weight=benchmark,
            specific_variance=0.0,
            adv=adv,
            style=style,
        )
    )
    # Independent LP in actual stock holdings, cash, buys and sells. Costs
    # shrink the name-band denominator and must be funded by pre-fee cash.
    buy_rate, sell_rate = 0.0008, 0.0012
    count = 3 * n + 1
    holding = np.eye(count)[:n]
    cash = np.eye(count)[n]
    buys = np.eye(count)[n + 1 : 2 * n + 1]
    sells = np.eye(count)[2 * n + 1 :]
    cost = buy_rate * buys.sum(axis=0) + sell_rate * sells.sum(axis=0)
    lower, upper = np.maximum(0.0, benchmark - 0.005), benchmark + 0.005
    inequalities = np.vstack(
        (
            -holding - lower[:, None] * cost,
            holding + upper[:, None] * cost,
            cost - cash,
            buys.sum(axis=0) + sells.sum(axis=0) + cash,
            (style - 0.2) @ holding,
            (-style - 0.2) @ holding,
            buys + sells,
        )
    )
    oracle = linprog(
        -alpha @ holding + cost,
        A_ub=inequalities,
        b_ub=np.r_[-lower, upper, 0.0, 0.4, 0.0, 0.0, capacity],
        A_eq=np.vstack((holding - buys + sells, holding.sum(axis=0) + cash)),
        b_eq=np.r_[benchmark, 1.0],
        bounds=[(0.0, 1.0)] * (n + 1) + [(0.0, value) for value in capacity] * 2,
        method="highs",
    )
    assert oracle.success
    # Certify the LP optimum is an actual signed-trade portfolio, rather than
    # a relaxation that manufactures fees from simultaneous buy/sell flow.
    assert np.minimum(buys @ oracle.x, sells @ oracle.x).max() <= 1e-10
    result = optimizer.optimize_portfolio(
        with_cash_asset(assets),
        cash_column="is_cash",
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=evaluate,
        nav=nav,
        assumptions=ExecutionAssumptions(nav, 1.0, 0.0, 0.05, (0.05,)),
    )
    weights = result.weights.to_numpy()
    stock = weights[:-1]
    fee = (
        buy_rate * np.maximum(stock - benchmark, 0).sum()
        + sell_rate * np.maximum(benchmark - stock, 0).sum()
    )
    economic = -alpha @ stock + fee
    assert economic == pytest.approx(oracle.fun, abs=2e-12, rel=0)
    assert result.expected_transaction_cost / nav == pytest.approx(fee, abs=2e-15)
    assert weights[-1] >= fee - 1e-10
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.abs(stock / (1.0 - fee) - benchmark).max() <= 0.005 + 1e-8
    assert np.max(np.abs(stock - benchmark) / capacity) <= 1 + 1e-7
    assert abs(style @ stock) <= 0.2 * stock.sum() + 1e-10
    assert np.abs(stock - benchmark).sum() + weights[-1] <= 0.4 + 1e-10


def test_seed18_real_cost_improves_over_the_proven_no_trade_kink():
    rng = np.random.default_rng(18)
    benchmark = rng.dirichlet(np.full(12, 3.0))
    alpha = rng.normal(0.0, 0.002, 12)
    style = rng.normal(size=12)
    style -= benchmark @ style
    owner = PreparedTradeCost(
        (0.001, 0.0, 0.0), (False, False, False), 0.0, 0.0, "synthetic"
    )
    adv = np.full(12, 1e12)
    assets = pd.DataFrame(
        dict(
            asset_id=[f"a{i}" for i in range(12)],
            expected_alpha=alpha,
            current_weight=benchmark,
            benchmark_weight=benchmark,
            specific_variance=0.01,
            adv=adv,
            style=style,
        )
    )
    result = optimizer.optimize_portfolio(
        with_cash_asset(assets),
        cash_column="is_cash",
        factor_covariance=pd.DataFrame([[0.01]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=_cost(owner, owner, adv),
        nav=1e9,
        assumptions=ExecutionAssumptions(1e9, 0.0, 0.0, 0.05, (0.05,)),
    )
    value = (
        alpha @ (result.weights.to_numpy()[:-1] - benchmark)
        - result.expected_transaction_cost / 1e9
    )
    sell = 0.005
    buy = sell * (1.0 - 0.001) / (1.0 + 0.001)
    fee = 0.001 * (buy + sell)
    hand_weights = benchmark.copy()
    hand_weights[9] += buy
    hand_weights[4] -= sell
    assert np.max(abs(hand_weights / (1.0 - fee) - benchmark)) <= 0.005
    hand_improvement = alpha[9] * buy - alpha[4] * sell - fee
    assert hand_improvement > 5.5e-6
    assert value >= hand_improvement - 1e-10


@pytest.mark.parametrize("cash", [0.0, 1e-13, 1e-6])
def test_zero_cash_has_the_same_legal_sell_to_cash_domain(cash):
    args = _decision_inputs(holdings=((1 - cash / 100) / 2,) * 2)
    enhanced = args["enhanced_decision"]
    args.update(
        cash=cash,
        enhanced_decision=replace(
            enhanced, expected_alpha=pd.Series([-0.1, -0.2], index=["a", "b"])
        ),
    )
    account, _, result, _, ledger = production._build_session(**args)
    assert result is not None
    assert result.weights.tolist() == pytest.approx([0.495, 0.495, 0.01], abs=1e-10)
    assert account.post_trade_cash == pytest.approx(1.0, abs=1e-8)
    assert ledger.executed_delta_notional.lt(0).all()


@pytest.mark.parametrize(
    "branch",
    [
        "optimized",
        "benchmark_zero",
        "benchmark_negative",
        "maintenance",
        "skipped",
        "risk_missing",
    ],
)
def test_actual_constraint_diagnostics_depend_on_account_and_available_facts(
    tmp_path, branch
):
    args = _decision_inputs()
    enhanced = args["enhanced_decision"]
    if branch.startswith("benchmark"):
        args["enhanced_decision"] = replace(
            enhanced,
            calibration=replace(
                enhanced.calibration, slope=0.0 if branch == "benchmark_zero" else -1.0
            ),
        )
    elif branch == "maintenance":
        args.update(
            target_override=enhanced.benchmark_weights,
            decision_reason="benchmark_maintenance",
        )
    elif branch == "skipped":
        args["decision_skip_reason"] = "all_predictions_missing"
    elif branch == "risk_missing":
        args["enhanced_decision"] = replace(enhanced, risk=None)
        args["panel"] = args["panel"].drop(columns="is_nonfinancial_industry")
    args["next_panel"].loc[args["next_panel"].asset_id.eq("a"), "open"] = 150.0
    args["execution"]["can_exit"] = False
    account, adv, result, delta, ledger = production._build_session(**args)
    assert account.nav_after == 125.0
    assert ledger.post_trade_notional.tolist() == [75.0, 50.0]
    facts = ledger.attrs["realized_constraints"]
    assert facts["realized_name_active_weight"] == pytest.approx(0.1)
    assert facts["realized_style_active_exposure"] == pytest.approx(0.0, abs=1e-15)
    if branch == "risk_missing":
        assert facts["realized_constraints_status"] == "UNDEFINED"
        assert facts["realized_tracking_error_status"] == "UNDEFINED"
        assert "realized_tracking_error" not in facts
    else:
        assert facts["realized_constraints_status"] == "BREACHED"
        assert facts["realized_constraints_reason"] == "name_active_weight"
        assert facts["realized_tracking_error"] == pytest.approx(np.sqrt(0.0002))
    if branch != "optimized":
        assert result is None
    batch = production._ledger_batch(
        account=account,
        asset_ledger=ledger,
        sequence=0,
        end_timestamp=args["endpoint_panel"].timestamp.iloc[0],
        decision_timestamp=args["decision_timestamp"],
        held_units=args["held_units"],
        execution_units=args["held_units"],
        previous_values={},
        rights_adjustments=(),
        prediction_rank_score=args["alpha"],
        adv20=adv,
        enhanced_decision=args["enhanced_decision"],
        benchmark_weights=enhanced.benchmark_weights,
        optimization_result=result,
        decision_reason=args["decision_reason"],
        decision_delta=delta,
        next_panel=args["next_panel"],
        endpoint_panel=args["endpoint_panel"],
        endpoint_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        endpoint_corporate_events=pd.DataFrame(),
        entitlements=production.EquityEntitlements(),
        execution_lives=args["next_panel"].set_index("asset_id").listing_date,
    )
    path = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("timing", 125.0, "CNY", "cash_equity"),
        metadata={"window": {"start": "2023-01-03", "end": "2023-01-05"}},
        produce=lambda emit: emit(batch),
    )
    restored = list(read_portfolio_ledger_artifact(path).iter_chunks())
    assert restored == [batch]
    persisted = {
        row.name: row.number if row.number is not None else row.text
        for row in restored[0].diagnostics
        if row.asset_id == "__cash__"
    }
    for name, value in facts.items():
        if value is not None:
            assert persisted[name] == value


def test_optimizer_iteration_limit_is_still_a_failure():
    assets = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            expected_alpha=[0.1, 0.0],
            current_weight=[0.5, 0.5],
            benchmark_weight=[0.5, 0.5],
            specific_variance=0.0,
            adv=1e6,
            style=0.0,
        )
    )
    with pytest.raises(ValueError, match="optimizer_failed_closed"):
        optimizer.optimize_portfolio(
            with_cash_asset(assets),
            cash_column="is_cash",
            factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
            style_columns=("style",),
            expected_transaction_cost=_cost(
                _owner(0.001), _owner(0.001), np.full(2, 1e6)
            ),
            nav=100.0,
            assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
            config=optimizer.OptimizerConfig(solver_maxiter=1),
        )


def test_impact_cost_reaches_independent_two_asset_stationary_optimum():
    fee, impact = 0.001, 0.001
    alpha_spread, nav, adv = 0.0025, 100.0, np.full(2, 10.0)
    owner = PreparedTradeCost(
        (fee, 0.0, 0.0), (False, False, False), 0.0, 10.0, "synthetic"
    )
    optimum = ((alpha_spread - 2 * fee) / (3 * impact)) ** 2 * adv[0] / nav
    assert 0.0 < optimum < 0.005
    assets = pd.DataFrame(
        dict(
            asset_id=["a", "b"],
            expected_alpha=[alpha_spread, 0.0],
            current_weight=[0.495, 0.495],
            benchmark_weight=[0.5, 0.5],
            specific_variance=0.0,
            adv=adv,
            style=0.0,
        )
    )
    assets = with_cash_asset(assets)
    assets.loc[assets.is_cash, "current_weight"] = 0.01
    assets["can_enter"] = ~assets.is_cash
    assets["can_exit"] = ~assets.is_cash
    result = optimizer.optimize_portfolio(
        assets,
        cash_column="is_cash",
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=_cost(owner, owner, adv),
        nav=nav,
        assumptions=ExecutionAssumptions(nav, 0.0, 10.0, 0.05, (0.05,)),
        config=optimizer.OptimizerConfig(active_name_limit=0.02),
    )
    delta = result.weights.iloc[0] - 0.495
    assert result.weights.iloc[-1] == pytest.approx(0.01)
    assert result.weights.iloc[-1] > result.expected_transaction_cost / nav
    curvature_floor = 1.5 * impact * np.sqrt(nav / adv[0] / 0.005)
    weight_error = np.sqrt(
        2 * optimizer.OptimizerConfig().solver_ftol / curvature_floor
    )
    assert delta == pytest.approx(optimum, abs=weight_error)
    expected_value = (
        alpha_spread - 2 * fee
    ) * optimum - 2 * impact * optimum * np.sqrt(optimum * nav / adv[0])
    assert (
        alpha_spread * delta - result.expected_transaction_cost / nav
        == pytest.approx(expected_value, abs=2e-12)
    )
