"""Independent rank-one robust objectives and native calibrated account flows."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from quant_research_platform.contracts.portfolios import RobustPortfolioConfig
from quant_research_platform.research.portfolios import calibration, optimizer
from tests.helpers.optimizer import ECONOMIC_UTILITY_ATOL


def _assets(alpha=(0.08, 0.0), current=(0.0, 1.0)):
    count = len(alpha)
    return pd.DataFrame(
        {
            "asset_id": [f"stock_{i}" for i in range(count - 1)] + ["cash"],
            "expected_alpha": alpha,
            "current_weight": current,
            "is_cash": [False] * (count - 1) + [True],
            "specific_variance": [0.25] * (count - 1) + [0.0],
        }
    )


def _solve(assets, loading, *, rate=0.0, **kwargs):
    rates = np.where(assets.is_cash, 0.0, rate)
    return optimizer.optimize_portfolio(
        assets,
        factor_covariance=pd.DataFrame(index=[], columns=[], dtype=float),
        expected_transaction_cost=lambda buy, sell: (
            float((buy + sell) @ rates),
            rates,
            rates,
        ),
        nav=100.0,
        cash_column="is_cash",
        config=kwargs.pop("config", optimizer.AbsoluteRiskConfig()),
        uncertainty_loading=pd.Series(loading, index=assets.asset_id),
        risk_aversion=kwargs.pop("risk_aversion", 2.0),
        holding_sessions=kwargs.pop("holding_sessions", 252),
        **kwargs,
    )


def test_hand_derived_return_robust_risk_fee_objective_and_nonrobust_baseline():
    assets = _assets()
    robust = _solve(assets, [0.02, 0.0], rate=0.01)
    # f(x) = (.08-.02-.01)*x - .5*2*.25*x^2.
    assert robust.weights.tolist() == pytest.approx([0.1, 0.9], abs=1e-7)
    assert robust.expected_return == pytest.approx(0.008, abs=1e-8)
    assert robust.robust_penalty == pytest.approx(0.002, abs=1e-8)
    assert robust.holding_variance == pytest.approx(0.0025, abs=1e-8)
    assert robust.risk_penalty == pytest.approx(0.0025, abs=1e-8)
    assert robust.expected_transaction_cost == pytest.approx(0.1, abs=1e-6)
    assert robust.objective_value == pytest.approx(0.0025, abs=1e-8)
    assert robust.solver_iterations > 0
    assert robust.solver_variables == 4
    baseline = _solve(assets, [0.0, 0.0], rate=0.01, risk_aversion=0.0)
    assert baseline.weights.iloc[0] == pytest.approx(1 / 1.01, abs=1e-8)
    assert baseline.weights.iloc[0] > robust.weights.iloc[0]


def test_rank_one_zero_norm_is_solved_exactly_without_a_smoothed_penalty():
    assets = _assets((0.04, 0.04, 0.0), (0.0, 0.0, 1.0))
    assets["specific_variance"] = [0.2, 0.2, 0.0]
    result = _solve(assets, [0.1, -0.1, 0.0], risk_aversion=1.0)
    # On x=y, f=.08*x-.2*x^2, whose unique maximum has x=y=.2.
    assert result.weights.tolist() == pytest.approx([0.2, 0.2, 0.6], abs=1e-6)
    assert result.robust_penalty == pytest.approx(0.0, abs=1e-10)


def test_fixed_and_one_direction_variables_preserve_the_economic_optimum():
    assets = _assets((0.08, 0.03, 0.0), (0.4, 0.0, 0.6))
    assets["trade_capacity_weight"] = [0.0, 1.0, 0.0]
    result = _solve(assets, [0.02, 0.01, 0.0])
    # a is frozen; b maximizes (.03-.01)*b-.25*b^2, hence b=.04.
    assert result.weights.tolist() == pytest.approx([0.4, 0.04, 0.56], abs=1e-7)
    assert result.solver_variables == 3


def test_holding_risk_uses_the_declared_covariance_annualization_and_horizon():
    assets = _assets()
    yearly = _solve(assets, [0.02, 0.0])
    monthly = _solve(assets, [0.02, 0.0], holding_sessions=21)
    alternate_scale = _solve(assets, [0.02, 0.0], risk_annualization=504.0)
    assert yearly.weights.iloc[0] == pytest.approx(0.12, abs=1e-7)
    assert monthly.objective_value == pytest.approx(0.06 - 0.25 / 12, abs=2e-8)
    assert monthly.weights.iloc[0] == pytest.approx(1.0, abs=2e-8 / (0.06 - 0.5 / 12))
    assert alternate_scale.weights.iloc[0] == pytest.approx(0.24, abs=1e-7)
    with pytest.raises(ValueError, match="optimizer_risk_horizon_invalid"):
        _solve(assets, [0.02, 0.0], holding_sessions=None)


def test_robust_solver_preserves_cash_capacity_turnover_and_relative_bands():
    assets = _assets((0.08, -0.08, 0.0), (0.4, 0.4, 0.2))
    assets["benchmark_weight"] = [0.5, 0.5, 0.0]
    assets["adv"] = [1000.0, 1000.0, np.inf]
    assets["style"] = [-1.0, 1.0, 0.0]
    from quant_research_platform.contracts.portfolios import ExecutionAssumptions

    result = optimizer.optimize_portfolio(
        assets,
        factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda buy, sell: (0.0, np.zeros(3), np.zeros(3)),
        nav=100.0,
        cash_column="is_cash",
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.01, (0.01,)),
        config=optimizer.OptimizerConfig(
            active_name_limit=0.2,
            tracking_error_limit=1.0,
            target_turnover_limit=0.1,
            style_active_limit=0.2,
        ),
        uncertainty_loading=pd.Series([0.02, -0.02, 0.0], index=assets.asset_id),
        risk_aversion=1.0,
        holding_sessions=21,
    )
    # Capacity caps the buy at .1; the stock-leg style band gives b >= 2*a/3.
    assert result.weights.tolist() == pytest.approx([0.5, 1 / 3, 1 / 6], abs=1e-7)
    assert result.target_turnover <= 0.1 + 1e-8
    assert result.maximum_adv_participation <= 0.01 + 1e-8
    assert result.predicted_tracking_error <= 1.0
    assert result.robust_penalty == pytest.approx(0.02 / 6, abs=1e-8)


def test_robust_objective_and_constraint_jacobians_match_central_differences(
    monkeypatch,
):
    assets = _assets((0.08, 0.03, 0.0), (0.2, 0.3, 0.5))

    def inspect(fun, initial, **kwargs):
        from scipy.optimize import LinearConstraint

        point = np.r_[0.22, 0.27, 0.51, 0.04, 0.05, 0.0005, 0.3]
        step = 1e-6
        numerical = np.array(
            [
                (fun(point + direction * step)[0] - fun(point - direction * step)[0])
                / (2 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert fun(point)[1] == pytest.approx(numerical, abs=1e-10, rel=1e-7)
        for constraint in kwargs["constraints"]:
            value = (
                (lambda x, c=constraint: c.A @ x)
                if isinstance(constraint, LinearConstraint)
                else constraint.fun
            )
            jacobian = (
                constraint.A
                if isinstance(constraint, LinearConstraint)
                else constraint.jac(point)
            )
            numerical = np.column_stack(
                [
                    (
                        np.atleast_1d(value(point + direction * step))
                        - np.atleast_1d(value(point - direction * step))
                    )
                    / (2 * step)
                    for direction in np.eye(len(point))
                ]
            )
            assert jacobian.toarray() == pytest.approx(numerical, abs=1e-9, rel=1e-7)
        direct = np.r_[np.full(3, 0.03), np.full(2, 0.01), 0.0, 0.02]
        assert kwargs["hess"](point) @ direct == pytest.approx(
            (fun(point + step * direct)[1] - fun(point - step * direct)[1])
            / (2 * step),
            abs=1e-9,
        )
        fee = next(
            item
            for item in kwargs["constraints"]
            if not isinstance(item, LinearConstraint)
            and np.array_equal(item.lb, item.ub)
        )
        inflated = point.copy()
        inflated[3:5] += 0.1
        assert fee.fun(point) == pytest.approx(0.0, abs=1e-15)
        assert fee.fun(inflated) == pytest.approx(0.0, abs=1e-15)
        assert fun(inflated)[0] > fun(point)[0]
        fake_fee = point.copy()
        fake_fee[-2] += 0.0005
        assert fee.fun(fake_fee) == pytest.approx(0.0005)
        raise RuntimeError("derivatives_checked")

    monkeypatch.setattr(optimizer, "minimize", inspect)
    with pytest.raises(RuntimeError, match="derivatives_checked"):
        _solve(assets, [0.02, -0.01, 0.0], rate=0.01)


def test_multiplier_projection_rechecks_an_invalid_one_sided_dual(monkeypatch):
    from scipy import sparse
    from scipy.optimize import LinearConstraint, OptimizeResult

    assets = _assets((0.07, 0.15, 0.0), (0.2, 0.3, 0.5))

    def inspect(fun, initial, **kwargs):
        point = np.r_[assets.current_weight.to_numpy(), 0.0, 0.0, 0.0]
        constraints = kwargs["constraints"]
        values, jacobians, multipliers = [], [], []
        for constraint in constraints:
            linear = isinstance(constraint, LinearConstraint)
            values.append(
                np.atleast_1d(constraint.A @ point if linear else constraint.fun(point))
            )
            jacobians.append(constraint.A if linear else constraint.jac(point))
            multiplier = np.zeros_like(values[-1])
            if linear and constraint.A.shape[0] == 4:
                multiplier[:] = [0.01, -0.005, -0.02, -0.005]
            multipliers.append(multiplier)
        values.append(point)
        jacobians.append(sparse.eye(len(point), format="csr"))
        multipliers.append(np.zeros(len(point)))
        gradient = fun(point)[1]
        raw_lagrangian = gradient + sum(
            jac.T @ dual for jac, dual in zip(jacobians, multipliers, strict=True)
        )
        # A positive lower-only multiplier cancels a genuine descent direction;
        # zero raw stationarity and complementarity cannot make it a valid dual.
        assert raw_lagrangian == pytest.approx(np.zeros(len(point)), abs=1e-14)
        state = OptimizeResult(
            grad=gradient,
            jac=jacobians,
            constr=values,
            v=multipliers,
            optimality=0.0,
            constr_violation=0.0,
        )
        assert kwargs["callback"](point, state) is False
        raise RuntimeError("invalid_dual_rejected")

    monkeypatch.setattr(optimizer, "minimize", inspect)
    with pytest.raises(RuntimeError, match="invalid_dual_rejected"):
        _solve(assets, [0.0, 0.0, 0.0], rate=0.01)


def test_bidirectional_real_impact_cost_matches_independent_stationary_optimum():
    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )
    from tests.research.test_portfolio_economic_optimality import _cost

    assets = _assets((0.0025, 0.0, 0.0), (0.495, 0.495, 0.01))
    assets["specific_variance"] = 0.0
    assets["can_enter"] = ~assets.is_cash
    assets["can_exit"] = ~assets.is_cash
    owner = PreparedTradeCost(
        (0.0003, 0.0, 0.0), (False, False, False), 0.0, 10.0, "synthetic"
    )
    result = optimizer.optimize_portfolio(
        assets,
        nav=100.0,
        cash_column="is_cash",
        factor_covariance=pd.DataFrame(index=[], columns=[], dtype=float),
        expected_transaction_cost=_cost(owner, owner, np.full(2, 10.0)),
        config=optimizer.AbsoluteRiskConfig(),
        uncertainty_loading=pd.Series([0.0001, -0.0001, 0.0], index=assets.asset_id),
    )
    delta = ((0.0025 - 0.0002 - 0.0006) / (3 * 0.001)) ** 2 / 10
    assert result.weights.iloc[0] - 0.495 == pytest.approx(delta, abs=2e-6)
    expected_cost = 100 * (2 * 0.0003 * delta + 2 * 0.001 * delta * np.sqrt(delta * 10))
    assert result.expected_transaction_cost == pytest.approx(expected_cost, abs=1e-6)
    assert result.weights.iloc[2] == 0.01
    assert result.solver_optimality <= optimizer.AbsoluteRiskConfig().solver_gtol
    assert (
        result.solver_constraint_violation <= optimizer.AbsoluteRiskConfig().solver_gtol
    )
    assert result.solver_complementarity_sum <= (
        result.solver_complementarity_count * optimizer.AbsoluteRiskConfig().solver_gtol
    )


def test_robust_iteration_exhaustion_is_a_failed_solve():
    with pytest.raises(ValueError, match="optimizer_failed_closed"):
        _solve(
            _assets(),
            [0.02, 0.0],
            config=optimizer.AbsoluteRiskConfig(solver_maxiter=1),
        )


def test_small_trade_funds_the_actual_fee_without_forcing_both_capacity_bounds():
    from quant_research_platform.contracts.portfolios import (
        portfolio_notional_tolerance,
    )
    from quant_research_platform.markets.implementations.a_share.execution import (
        PreparedTradeCost,
    )
    from tests.research.test_portfolio_economic_optimality import _cost

    assets = _assets((0.02, -0.02, 0.0), (0.5, 0.5, 0.0))
    capacity, rate, nav = 1e-6, 0.0004, 1e9
    assets["trade_capacity_weight"] = [capacity, capacity, 0.0]
    owner = PreparedTradeCost(
        (rate, 0.0, 0.0), (False, False, False), 0.0, 0.0, "synthetic"
    )
    result = optimizer.optimize_portfolio(
        assets,
        nav=nav,
        cash_column="is_cash",
        factor_covariance=None,
        expected_transaction_cost=_cost(owner, owner, np.full(2, 1e12)),
        config=optimizer.AbsoluteRiskConfig(),
    )
    bought = result.weights.iloc[0] - 0.5
    sold = 0.5 - result.weights.iloc[1]
    fee = rate * (bought + sold)
    # A full sale finances less than a full purchase because both sides pay.
    best_buy = capacity * (1.0 - rate) / (1.0 + rate)
    best_objective = (0.02 - rate) * (best_buy + capacity)
    assert 0.0 <= bought <= best_buy + 2e-12
    assert 0.0 <= sold <= capacity + 1e-12
    assert result.weights.iloc[-1] > 0.0
    assert result.weights.iloc[-1] >= fee - portfolio_notional_tolerance(nav) / nav
    assert result.expected_transaction_cost == pytest.approx(fee * nav, abs=1e-7)
    assert result.objective_value <= best_objective + 1e-15
    assert result.objective_value >= best_objective - ECONOMIC_UTILITY_ATOL


@pytest.mark.parametrize("blocked_sale", [False, True])
def test_defined_nonpositive_slope_exits_to_cash_without_consuming_covariance(
    blocked_sale,
):
    import quant_research_platform.research.portfolios.daily as daily
    from tests.helpers.portfolio_inputs import decision_inputs

    inputs = decision_inputs()
    decision = inputs["enhanced_decision"]
    inputs.update(
        candidate=replace(
            inputs["candidate"], construction="robust_long_only", benchmark_id=None
        ),
        enhanced_decision=replace(
            decision,
            benchmark_weights=pd.Series(dtype="float64"),
            expected_alpha=decision.expected_alpha * 0.0,
            risk=None,
            calibration=replace(
                decision.calibration,
                slope=0.0,
                slope_t_statistic=0.0,
                slope_pvalue=1.0,
            ),
        ),
        optimizer_config=optimizer.AbsoluteRiskConfig(total_volatility_limit=0.3),
        robust_config=RobustPortfolioConfig(),
        decision_reason="robust_optimizer",
    )
    if blocked_sale:
        inputs["execution"].loc[inputs["execution"].asset_id.eq("a"), "can_exit"] = (
            False
        )
    account, _, solved, _, ledger = daily._build_session(**inputs)
    assert solved is None
    assert ledger.attrs["decision_availability"]["decision_status"] == "AVAILABLE"
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] is None
    if blocked_sale:
        assert ledger.set_index("asset_id").loc["a", "post_trade_notional"] == 50.0
        assert (
            ledger.attrs["realized_constraints"]["realized_total_volatility_status"]
            == "UNDEFINED"
        )
        return
    assert account.post_trade_cash == 100.0
    assert ledger.executed_delta_notional.tolist() == [-50.0, -50.0]
    assert (
        ledger.attrs["realized_constraints"]["realized_total_volatility_status"]
        == "DEFINED"
    )
    assert ledger.attrs["realized_constraints"]["realized_total_volatility"] == 0.0


def _calibration_rows():
    sessions = pd.bdate_range("2022-01-03", periods=35)
    intervals = calibration.alpha_calibration_intervals(sessions, "daily")
    slopes = np.arange(8) * 0.002 + 0.02
    rows = pd.DataFrame(
        [
            {
                "timestamp": sessions[i],
                "prediction_rank_score": rank,
                "holding_return": 0.01 + slope * rank,
            }
            for i, slope in enumerate(slopes)
            for rank in (-0.25, 0.25)
        ]
    )
    return sessions, intervals, rows, slopes


def test_date_cluster_slope_uncertainty_is_parameter_error_and_excludes_immature_rows():
    sessions, intervals, rows, slopes = _calibration_rows()
    future = pd.DataFrame(
        {
            "timestamp": [sessions[8]] * 2,
            "prediction_rank_score": [-0.25, 0.25],
            "holding_return": [-10.0, 10.0],
        }
    )
    fit = calibration.calibrate_expected_alpha(
        pd.concat((rows, future), ignore_index=True),
        as_of=sessions[9],
        scheduled_intervals=intervals,
        trading_sessions=sessions,
    )
    # Equal date weights imply beta=mean(beta_t), SE=sqrt(sum((beta_t-mean)^2))/T.
    expected_se = np.linalg.norm(slopes - slopes.mean()) / len(slopes)
    assert fit.slope == pytest.approx(slopes.mean(), abs=1e-14)
    assert fit.slope_standard_error == pytest.approx(expected_se, abs=1e-14)
    ranks = pd.Series([-0.25, 0.25, 0.0])
    loading = calibration.alpha_uncertainty_loading(fit, ranks)
    expected = student_t.ppf(0.975, df=6) * expected_se * ranks
    assert loading.tolist() == pytest.approx(expected.tolist(), abs=1e-14)
    shifted = rows.assign(holding_return=rows.holding_return + 0.5)
    shifted_fit = calibration.calibrate_expected_alpha(
        shifted,
        as_of=sessions[9],
        scheduled_intervals=intervals,
        trading_sessions=sessions,
    )
    assert shifted_fit.intercept == pytest.approx(0.51)
    assert shifted_fit.slope == pytest.approx(fit.slope)
    assert calibration.alpha_uncertainty_loading(
        shifted_fit, ranks
    ).tolist() == pytest.approx(loading.tolist())
    zero = replace(fit, slope_standard_error=0.0)
    assert calibration.alpha_uncertainty_loading(zero, ranks).eq(0.0).all()


def test_robust_configuration_contains_only_confidence_and_risk_preference():
    assert asdict(RobustPortfolioConfig()) == {"confidence": 0.95, "risk_aversion": 1.0}
    for values in ({"confidence": 1.0}, {"risk_aversion": -1.0}):
        with pytest.raises(ValueError, match="robust_portfolio_config_invalid"):
            RobustPortfolioConfig(**values)


@pytest.fixture(scope="module")
def native_case(tmp_path_factory):
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.portfolios.production import calibrate
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.portfolio_inputs import (
        _dataset_spec,
        _frames,
        _interim_specs,
        _processed_specs,
    )
    from tests.helpers.portfolio_ledger import write_market_evidence
    from tests.research.direct_model_test_support import fit_frame

    root = tmp_path_factory.mktemp("robust-native")
    dates = pd.bdate_range("2022-11-01", "2023-01-13")
    frames = _frames(dates)
    for name, frame in frames.items():
        if "asset_id" in frame:
            frames[name] = pd.concat(
                [
                    frame.assign(
                        asset_id=frame.asset_id.map(
                            {"a": chr(97 + offset), "b": chr(98 + offset)}
                        )
                    )
                    for offset in range(0, 12, 2)
                ],
                ignore_index=True,
            )
    frames["benchmark_weights"]["weight"] = 1 / 12
    panel = frames["market_panel_1d"]
    for asset, drift in (
        (name, 0.0002 if name in "acegik" else 0.0015) for name in "abcdefghijkl"
    ):
        selected = panel.asset_id.eq(asset)
        changes = drift + 0.0002 * np.sin(np.arange(len(dates)))
        prices = 10.0 * np.cumprod(1.0 + changes)
        for column in ("open", "close", "adj_open", "adj_close"):
            panel.loc[selected, column] = prices
        panel.loc[selected, "period_return"] = changes
    panel["amount"] = 1e8
    sources = write_market_evidence(root / "inputs", panel)
    specifications = {
        "market_panel_1d": _processed_specs().get("market_panel_1d"),
        "execution_mask": _processed_specs().get("execution_mask"),
        "benchmark_weights": _processed_specs().get("benchmark_weights"),
        "price_volume_facts": _processed_specs().get("price_volume_facts"),
        "trading_calendar": _interim_specs().get("trading_calendar"),
        "a_share_fee_schedule": _interim_specs().get("a_share_fee_schedule"),
        "pv_all": _dataset_spec(),
    }
    for name, spec in specifications.items():
        sources[name] = write_dataset(
            root / "inputs" / name, frames[name], spec=spec
        ).root
    training = frames["pv_all"].loc[frames["pv_all"].timestamp.le(dates[7])].copy()
    fitted = fit_frame(
        root / "fit",
        training,
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            random_seed=42,
        ),
        market_id="a_share",
    )
    prediction_window = {"start": dates[8].date().isoformat(), "end": "2023-01-10"}
    predicted = predict(
        {
            "model": str(fitted.root),
            "inputs": str(sources["pv_all"]),
            "output": str(root / "predictions"),
            "prediction_window": prediction_window,
            "sample_role": "out_of_sample",
        }
    )
    prediction, _ = read_predictions(predicted)
    sessions = pd.DatetimeIndex(frames["trading_calendar"].calendar_date)
    intervals = calibration.alpha_calibration_intervals(
        sessions, "weekly-last-trading-day"
    )
    price = panel.set_index(["timestamp", "asset_id"]).adj_open
    observations = []
    for row in intervals.itertuples(index=False):
        if row.available_at > pd.Timestamp("2022-12-30"):
            continue
        for forecast in prediction.loc[
            prediction.timestamp.eq(row.timestamp)
        ].itertuples():
            observations.append(
                {
                    "timestamp": row.timestamp,
                    "asset_id": forecast.asset_id,
                    "prediction_rank_score": forecast.prediction_rank_score,
                    "holding_return": price.loc[(row.available_at, forecast.asset_id)]
                    / price.loc[(row.entry_at, forecast.asset_id)]
                    - 1.0,
                }
            )
    observation_path, intervals_path = (
        root / "observations.parquet",
        root / "intervals.parquet",
    )
    pd.DataFrame(observations).to_parquet(observation_path, index=False)
    intervals.to_parquet(intervals_path, index=False)
    frozen = calibrate(
        {
            "observations": str(observation_path),
            "scheduled_intervals": str(intervals_path),
            "calendar": str(sources["trading_calendar"]),
            "cutoff": "2022-12-30",
            "rebalance_schedule": "weekly-last-trading-day",
            "selection_group": "research",
            "output": str(root / "calibration.json"),
        }
    )
    inputs = {
        "dataset": str(sources["pv_all"]),
        "panel": str(sources["market_panel_1d"]),
        "execution": str(sources["execution_mask"]),
        "calendar": str(sources["trading_calendar"]),
        "fees": str(sources["a_share_fee_schedule"]),
        "market_bars": str(sources["market_bars_1d"]),
        "stock_master": str(sources["stock_master"]),
        "corporate_actions": str(sources["corporate_actions"]),
        "risk_context": [str(sources["price_volume_facts"])],
    }
    config = {
        "candidate": {
            "candidate_id": "robust",
            "selection_group": "research",
            "construction": "robust_long_only",
            "prediction_selector": "linear",
            "rebalance_schedule": "weekly-last-trading-day",
            "top_quantile": None,
            "benchmark_id": None,
            "risk_style_columns": ["momentum_252_20"],
        },
        "inputs": inputs,
        "predictions": [{"path": str(predicted), "window": prediction_window}],
        "output_window": {"start": "2023-01-02", "end": "2023-01-10"},
        "input_window": {"start": dates[0].date().isoformat(), "end": "2023-01-10"},
        "formal_dataset_window": {
            "start": dates[0].date().isoformat(),
            "end": "2023-01-10",
        },
        "prediction_window": prediction_window,
        "calibration": str(frozen),
        "risk_model": {
            "style_columns": ["momentum_252_20"],
            "config": {
                "factor_window": 20,
                "specific_window": 20,
                "minimum_factor_dates": 10,
                "minimum_specific_dates": 10,
                "minimum_cross_section": 2,
            },
        },
        "assumptions": {
            "initial_nav": 1e6,
            "slippage_bps": 0.0,
            "impact_bps": 0.0,
            "adv_participation": 1.0,
            "adv_sensitivities": [1.0],
            "minimum_commission": 0.0,
        },
    }
    return root, config, str(sources["benchmark_weights"]), pd.DataFrame(observations)


@pytest.mark.parametrize("construction", ["robust_long_only", "benchmark_enhanced"])
def test_native_prediction_calibration_robust_account_and_frozen_reuse(
    native_case, construction
):
    import json
    from copy import deepcopy
    from pathlib import Path

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
    )
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.storage.identity import native_filesystem_path

    root, config, benchmark_path, observations = native_case
    config = deepcopy(config)
    config["candidate"]["construction"] = construction
    config["output"] = str(root / construction)
    config["robust_config"] = {"confidence": 0.95, "risk_aversion": 1.0}
    if construction == "benchmark_enhanced":
        config["candidate"]["benchmark_id"] = "csi500"
        config["inputs"]["benchmark"] = benchmark_path
        config["optimizer_config"] = {
            "active_name_limit": 0.1,
            "tracking_error_limit": 1.0,
            "target_turnover_limit": 0.2,
            "industry_active_limit": 1.0,
            "style_active_limit": 0.5,
        }
    calibration_path = Path(native_filesystem_path(config["calibration"]))
    before = calibration_path.read_bytes()
    payload = json.loads(before)
    fit, _ = calibration.frozen_alpha_calibration(
        payload,
        cutoff="2022-12-30",
        selection_group="research",
        rebalance_schedule="weekly-last-trading-day",
    )
    assert fit.slope > 0.0 and fit.slope_standard_error > 0.0
    assert observations.timestamp.min() >= pd.Timestamp(
        config["prediction_window"]["start"]
    )
    with pytest.raises(ValueError, match="test_alpha_calibration_binding_invalid"):
        calibration.frozen_alpha_calibration(
            payload,
            cutoff="2022-12-30",
            selection_group="research",
            rebalance_schedule="daily",
        )
    with pytest.raises(ValueError, match="portfolio_calibration_not_frozen"):
        backtest(
            {**config, "output_window": {"start": "2022-12-30", "end": "2023-01-10"}}
        )
    output = backtest(config)
    artifact = read_portfolio_ledger_artifact(output)
    batches = list(artifact.iter_chunks())
    assert calibration_path.read_bytes() == before
    # The first decision is Monday; the weekly calibration first applies Friday.
    assert all(not batch.trades for batch in batches[:4])
    first = next(batch for batch in batches if batch.trades)
    facts = {
        row.name: row.number for row in first.diagnostics if row.asset_id == "__cash__"
    }
    assert facts["holding_sessions"] == 5.0
    assert facts["robust_penalty"] >= 0.0
    assert facts["risk_penalty"] >= 0.0
    assert facts["solver_iterations"] > 0.0
    assert facts["expected_transaction_cost"] > 0.0
    assert any(
        flow.kind == "commission_cost" and flow.amount < 0.0 for flow in first.cashflows
    )
    assert any(
        row.quantity > 0.0 for row in first.valuations if row.asset_id != "__cash__"
    )
    summary = summarize_portfolio_ledger(artifact.capital, iter(batches))
    assert summary["fees"] > 0.0
    assert np.isfinite(summary["capital_return"])
    assert artifact.final_alpha_calibration is None


def test_robust_index_membership_maintenance_is_independent_of_calibration_schedule(
    native_case,
):
    from copy import deepcopy

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.portfolio_inputs import _processed_specs

    root, config, benchmark_path, _ = native_case
    config = deepcopy(config)
    reference = read_dataset(benchmark_path)
    reference = reference.loc[reference.benchmark_id.eq("csi500")]
    reference = reference.loc[
        ~(reference.timestamp.ge("2023-01-04") & reference.asset_id.eq("l"))
    ].copy()
    reference.loc[reference.timestamp.ge("2023-01-04"), "weight"] = 1 / 11
    path = root / "membership"
    write_dataset(path, reference, spec=_processed_specs().get("benchmark_weights"))
    config["candidate"].update(construction="benchmark_enhanced", benchmark_id="csi500")
    config["inputs"]["benchmark"] = str(path)
    config["robust_config"] = {"confidence": 0.95, "risk_aversion": 1.0}
    config["output"] = str(root / "maintenance")
    output = backtest(config)
    artifact = read_portfolio_ledger_artifact(output)
    first = next(batch for batch in artifact.iter_chunks() if batch.trades)
    assert pd.Timestamp(first.trades[0].timestamp_ns) == pd.Timestamp("2023-01-05")
    reasons = {row.text for row in first.diagnostics if row.name == "decision_reason"}
    assert reasons == {"benchmark_maintenance"}
    assert not any(row.name == "solver_iterations" for row in first.diagnostics)


@pytest.mark.parametrize("risk_aversion", [0.0, 1.0])
def test_native_robust_risk_window_only_blocks_a_consumed_covariance(
    native_case, risk_aversion
):
    from copy import deepcopy

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.portfolios.production import backtest

    root, config, _, _ = native_case
    config = deepcopy(config)
    config["risk_model"]["config"].update(factor_window=80, minimum_factor_dates=60)
    config["robust_config"] = {"confidence": 0.95, "risk_aversion": risk_aversion}
    config["output"] = str(root / f"risk_unavailable_{risk_aversion}")
    artifact = read_portfolio_ledger_artifact(backtest(config))
    batches = list(artifact.iter_chunks())
    if risk_aversion == 0.0:
        assert any(batch.trades for batch in batches)
        optimized = next(
            batch
            for batch in batches
            if any(row.name == "solver_iterations" for row in batch.diagnostics)
        )
        assert any(
            row.name == "risk_penalty" and row.number == 0.0
            for row in optimized.diagnostics
        )
        assert any(
            row.name == "realized_total_volatility_status" and row.text == "UNDEFINED"
            for row in optimized.diagnostics
        )
        assert not any(
            row.name == "predicted_total_volatility" and row.number is not None
            for row in optimized.diagnostics
        )
        assert all(
            row.text != "factor_risk_factor_history_insufficient"
            for batch in batches
            for row in batch.diagnostics
            if row.name == "decision_skip_reason"
        )
        return
    assert all(not batch.trades for batch in batches)
    skipped = []
    for batch in batches:
        facts = {
            row.name: row.text
            for row in batch.diagnostics
            if row.asset_id == "__cash__" and row.text is not None
        }
        assert not any(row.name == "solver_iterations" for row in batch.diagnostics)
        if (
            facts.get("decision_skip_reason")
            == "factor_risk_factor_history_insufficient"
        ):
            assert facts["decision_status"] == "SKIPPED"
            assert facts["alpha_status"] == "DEFINED"
            skipped.append(batch.sequence)
    assert skipped


@pytest.mark.parametrize("mode", ["active", "cash", "benchmark"])
def test_native_unknown_unheld_industry_only_blocks_the_required_reference(
    native_case, mode
):
    from copy import deepcopy

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.portfolios.production import (
        backtest,
        calibrate,
    )
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.portfolio_inputs import _processed_specs

    root, config, benchmark_path, observations = native_case
    config = deepcopy(config)
    panel = read_dataset(config["inputs"]["panel"])
    panel.loc[panel.asset_id.eq("b"), "industry_level1_code"] = pd.NA
    path = root / f"unknown_industry_{mode}"
    config["inputs"]["panel"] = str(
        write_dataset(path, panel, spec=_processed_specs().get("market_panel_1d")).root
    )
    config["robust_config"] = {"confidence": 0.95, "risk_aversion": 1.0}
    config["output"] = str(root / f"account_unknown_industry_{mode}")
    if mode == "benchmark":
        config["candidate"].update(
            construction="benchmark_enhanced", benchmark_id="csi500"
        )
        config["inputs"]["benchmark"] = benchmark_path
    elif mode == "cash":
        observation_path = root / "nonpositive_observations.parquet"
        observations.assign(
            holding_return=-0.1 * observations.prediction_rank_score
        ).to_parquet(observation_path, index=False)
        config["calibration"] = str(
            calibrate(
                {
                    "observations": str(observation_path),
                    "scheduled_intervals": str(root / "intervals.parquet"),
                    "calendar": config["inputs"]["calendar"],
                    "cutoff": "2022-12-30",
                    "rebalance_schedule": "weekly-last-trading-day",
                    "selection_group": "research",
                    "output": str(root / "nonpositive_calibration.json"),
                }
            )
        )
        config["risk_model"]["config"].update(factor_window=80, minimum_factor_dates=60)
    batches = list(read_portfolio_ledger_artifact(backtest(config)).iter_chunks())
    assert batches
    facts = [
        {row.name: row.text for row in batch.diagnostics if row.asset_id == "__cash__"}
        for batch in batches
    ]
    if mode == "benchmark":
        assert all(not batch.trades for batch in batches)
        assert any(
            row.get("decision_status") == "SKIPPED"
            and row.get("decision_skip_reason") == "portfolio_industry_context_missing"
            for row in facts
        )
        return
    assert all(
        row.get("decision_skip_reason") != "portfolio_industry_context_missing"
        for row in facts
    )
    assert any(row.get("decision_status") == "AVAILABLE" for row in facts)
    if mode == "cash":
        assert all(not batch.trades for batch in batches)
        assert not any(
            row.name == "solver_iterations"
            for batch in batches
            for row in batch.diagnostics
        )
        assert any(
            row.name == "alpha_slope" and row.number < 0.0
            for batch in batches
            for row in batch.diagnostics
        )
        return
    optimized = [
        batch
        for batch in batches
        if any(row.name == "solver_iterations" for row in batch.diagnostics)
    ]
    assert optimized and any(batch.trades for batch in optimized)
    assert all(trade.asset_id != "b" for batch in batches for trade in batch.trades)
    assert all(
        row.quantity == 0.0
        for batch in batches
        for row in batch.valuations
        if row.asset_id == "b"
    )
