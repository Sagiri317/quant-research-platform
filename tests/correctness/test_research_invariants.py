from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.research.factors.fundamental.calculations import (
    calculate_fundamental_factors,
)
from quant_research_platform.research.factors.price_volume.columns import (
    FORMAL_CORE_PRICE_VOLUME_FACTORS,
    FORMAL_PRICE_VOLUME_FACTORS,
)
from quant_research_platform.research.factors.price_volume.formal import (
    calculate_formal_price_volume_factors,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.models.fitting import (
    _fit_model_result_owned_from_frames,
)
from quant_research_platform.research.models.scaler import (
    ModelMatrixScaler,
    equal_date_sample_weights,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.portfolios.accounting import (
    alpha_score,
    calibrate_expected_alpha,
    execute_continuous_target,
    one_way_turnover,
)
from quant_research_platform.research.portfolios.engine import (
    _construct_portfolio_positions,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from quant_research_platform.research.protocols.a_share import (
    PROTOCOL_ID,
    SCENARIO_IDS,
    protocol,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.choice.interim import readiness
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.runtime_resources import capture_test_policy


def test_research_protocol_identity_and_surfaces_are_explicit() -> None:
    assert len(FORMAL_PRICE_VOLUME_FACTORS) == 25
    assert len(FORMAL_CORE_PRICE_VOLUME_FACTORS) == 6
    assert SCENARIO_IDS == ("pv_all", "fusion_nf_core", "fusion_nf_expectations")
    research_protocol = protocol("pv_all")
    assert research_protocol.protocol_id == PROTOCOL_ID == "a_share_low_frequency"
    assert research_protocol.manifest()["protocol_id"] == PROTOCOL_ID
    assert "protocol_version" not in research_protocol.manifest()
    assert not hasattr(ModelSpec, "purge_gap_steps")
    assert not hasattr(PortfolioSpec, "execution_lag_steps")


def test_formal_price_volume_definitions_use_session_windows() -> None:
    sessions = pd.bdate_range("2020-01-01", periods=252)
    values = pd.Series(range(1, len(sessions) + 1), dtype="float64")
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": sessions,
            "adj_close": values,
            "period_return": 0.01,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "amount": 1_000.0,
            "volume": 100.0,
            "float_market_cap": 100_000.0,
        }
    )
    result = calculate_formal_price_volume_factors(
        panel,
        factor_columns=("amount_shock_20d", "amplitude_20d"),
    )
    assert pd.isna(result.loc[19, "amount_shock_20d"])
    assert result.loc[20, "amount_shock_20d"] == pytest.approx(0.0)
    assert result.loc[20, "amplitude_20d"] == pytest.approx(
        (values.iloc[1:21].rdiv(2.0)).mean()
    )


def test_session_label_uses_next_open_and_persists_availability() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=5)
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": sessions,
            "adj_open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "can_enter": [True, True, False, True, True],
        }
    )
    result = add_forward_return_labels(
        panel,
        horizons=(1,),
        sessions=sessions,
        can_enter_column="can_enter",
    )
    assert result.loc[0, "forward_return_1d"] == pytest.approx(12.0 / 11.0 - 1.0)
    assert pd.isna(result.loc[1, "forward_return_1d"])
    assert result.loc[0, "forward_return_1d_available_at"] == sessions[2]


def test_model_scaler_uses_equal_date_weights_and_zero_variance_zero() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 3.0, 5.0],
            "constant": [2.0, 2.0, 2.0],
        }
    )
    weights = equal_date_sample_weights(
        pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    )
    assert weights.sum() == pytest.approx(3.0)
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    transformed = scaler.transform(frame)
    assert transformed["constant"].tolist() == [0.0, 0.0, 0.0]


def test_training_eligibility_is_label_availability_not_a_fixed_gap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "label_available_at": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    result = legal_training_rows(frame, fitting_cutoff="2024-01-02")
    assert result.index.tolist() == [0]


def test_execution_mask_is_a_separate_future_fact() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "is_trading": [True, True, True],
            "scenario_universe": [True, True, True],
            "limit_status": [0, 1, 0],
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
        }
    )
    result = derive_execution_mask(panel)
    assert result.columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "can_enter",
        "can_exit",
    ]
    assert bool(result.loc[0, "can_enter"]) is False


def test_supervised_models_persist_a_training_only_scaler() -> None:
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_feature_columns=("feature",),
        label_column="model_target",
        hyperparameters={"alpha": 1.0},
    )
    preflight_model_specs((spec,))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "asset_id": ["a", "b", "a", "b"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "model_target": [0.1, 0.2, 0.3, 0.4],
            "eligible": True,
        }
    )
    with runtime_resource_policy_scope(None):
        result = _fit_model_result_owned_from_frames(
            [frame],
            spec,
            estimator=build_estimator_from_model_spec(spec),
        )
        fitted = result.value if hasattr(result, "value") else result
        assert fitted.metadata["scaler"]["columns"] == ["feature"]
        if hasattr(result, "release"):
            result.release(owner="test.scaler")


def test_rank_ensembles_learn_direction_and_keep_missing_neutral() -> None:
    spec = ModelSpec(
        model_id="rank",
        model_algorithm="equal_weight_rank_ensemble",
        input_factor_columns=("factor_a", "factor_b"),
        label_column="model_target",
    )
    preflight_model_specs((spec,))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "asset_id": ["a", "b", "c"] * 2,
            "factor_a": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "factor_b": [3.0, 2.0, 1.0, 3.0, 2.0, np.nan],
            "model_target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "eligible": True,
        }
    )
    with runtime_resource_policy_scope(None):
        result = _fit_model_result_owned_from_frames(
            [frame],
            spec,
            estimator=build_estimator_from_model_spec(spec),
        )
        fitted = result.value if hasattr(result, "value") else result
        predictions = fitted.prediction_model.predict(
            frame.loc[:, ["timestamp", "factor_a", "factor_b"]]
        )
        assert np.isfinite(predictions).all()
        assert fitted.metadata["scaler"] is None
        if hasattr(result, "release"):
            result.release(owner="test.rank")


def test_weekly_portfolio_rebalances_then_naturally_drifts() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame(
        {
            "asset_id": [asset for _date in sessions for asset in ("a", "b")],
            "timestamp": [date for date in sessions for _asset in ("a", "b")],
            "signal": [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            "forward_return_1d": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "scenario_universe": True,
            "can_enter": True,
            "can_exit": True,
            "amount": 1_000_000.0,
            "float_market_cap": 10_000_000.0,
            "industry_level1_code": "A",
        }
    )
    spec = PortfolioSpec(
        portfolio_id="weekly",
        market_id="a_share",
        research_dataset_id="pv_all",
        signals=(
            SignalSpec(
                signal_id="signal", source_kind="factor", source_column="signal"
            ),
        ),
        rebalance_schedule="weekly-last-trading-day",
        execution_policy_id="a_share_next_open",
        cost_policy_id="a_share_historical_costs",
        weight_config=WeightConfig(
            method="equal_weight_quantile",
            portfolio_mode="long_only",
            top_quantile=1.0,
        ),
        diagnostics_config=PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
            categorical_exposure_columns=("industry_level1_code",),
        ),
        tradability_config=TradabilityConfig(
            eligibility_column="scenario_universe",
            can_enter_column="can_enter",
            can_exit_column="can_exit",
        ),
    )
    result = _construct_portfolio_positions(
        frame,
        spec,
        source_columns_by_kind={
            "factor": ("signal",),
            "feature": (),
            "prediction": (),
        },
    )
    turnover = result.turnover.loc[result.turnover["signal"].eq("signal")]
    assert turnover["turnover"].tolist()[1:3] == [0.0, 0.0]
    first = result.positions.loc[result.positions["timestamp"].eq(sessions[0])]
    middle = result.positions.loc[result.positions["timestamp"].eq(sessions[1])]
    assert first.loc[first["asset"].eq("a"), "position"].iloc[0] == pytest.approx(0.5)
    assert middle.loc[middle["asset"].eq("a"), "position"].iloc[0] != pytest.approx(0.5)


def test_continuous_accounting_is_sell_first_and_adv_bounded() -> None:
    current = pd.Series({"a": 1.0, "b": 0.0})
    target = pd.Series({"a": 0.0, "b": 1.0})
    result = execute_continuous_target(
        current,
        target,
        nav=1_000_000.0,
        average_daily_volume=pd.Series({"a": 1_000_000.0, "b": 1_000_000.0}),
        can_enter=pd.Series({"a": True, "b": True}),
        can_exit=pd.Series({"a": True, "b": True}),
        participation=0.05,
    )
    assert result.executions.loc[0, "side"] == "sell"
    assert result.realized_weights["a"] == pytest.approx(0.95)
    assert 0.049 < result.realized_weights["b"] <= 0.05
    assert result.realized_turnover == pytest.approx(0.05)
    assert one_way_turnover(target, current) == pytest.approx(1.0)
    assert alpha_score(pd.Series([10.0, 20.0, 30.0])).tolist() == pytest.approx(
        [-1 / 3, 0.0, 1 / 3]
    )


def test_future_price_does_not_change_a_past_price_volume_factor() -> None:
    sessions = pd.bdate_range("2020-01-01", periods=300)
    values = pd.Series(np.arange(10.0, 310.0), index=sessions)
    base = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": sessions,
            "adj_close": values,
            "period_return": values.pct_change().fillna(0.0),
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "amount": 1_000.0,
            "volume": 100.0,
            "float_market_cap": 100_000.0,
        }
    )
    changed = base.copy()
    changed.loc[changed["timestamp"] >= sessions[220], "adj_close"] *= 3.0
    changed.loc[changed["timestamp"] >= sessions[220], "period_return"] = 0.2
    first = calculate_formal_price_volume_factors(
        base, factor_columns=("reversal_20d", "volatility_20d")
    )
    second = calculate_formal_price_volume_factors(
        changed, factor_columns=("reversal_20d", "volatility_20d")
    )
    assert second.loc[
        219, ["reversal_20d", "volatility_20d"]
    ].tolist() == pytest.approx(
        first.loc[219, ["reversal_20d", "volatility_20d"]].tolist()
    )


def test_alpha_calibration_is_schedule_matched_and_non_flipping() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "timestamp": [date for date in dates for _asset in ("a", "b")],
            "alpha_score": [-0.25, 0.25] * 6,
            "forward_return_1d": [0.01, 0.03] * 6,
        }
    )
    calibrated = calibrate_expected_alpha(
        frame,
        schedule="daily",
        minimum_valid_dates=6,
    )
    assert calibrated.benchmark_only is False
    assert calibrated.valid_rebalance_dates == 6
    assert calibrated.expected_alpha.iloc[1] > calibrated.expected_alpha.iloc[0]

    inverse = calibrate_expected_alpha(
        frame.assign(forward_return_1d=-frame["forward_return_1d"]),
        schedule="daily",
        minimum_valid_dates=6,
    )
    assert inverse.benchmark_only is True
    assert inverse.expected_alpha.eq(0.0).all()


def test_future_execution_and_disclosure_facts_do_not_rewrite_past_facts() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "adj_open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "can_enter": True,
        }
    )
    baseline = add_forward_return_labels(
        panel, horizons=(1,), sessions=sessions, can_enter_column="can_enter"
    )
    changed = panel.copy()
    changed.loc[5, "can_enter"] = False
    future_changed = add_forward_return_labels(
        changed, horizons=(1,), sessions=sessions, can_enter_column="can_enter"
    )
    assert future_changed.loc[0, "forward_return_1d"] == pytest.approx(
        baseline.loc[0, "forward_return_1d"]
    )

    observations = pd.DataFrame({"asset_id": ["asset"], "timestamp": [sessions[1]]})
    events = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "metric_id": ["total_assets"],
            "report_period": [sessions[0]],
            "available_at": [sessions[2]],
            "is_corrected": [False],
            "value": [100.0],
        }
    )
    from quant_research_platform.markets.implementations.a_share.fundamentals import (
        materialize_pit_fundamentals,
    )

    assert materialize_pit_fundamentals(observations, events)["value"].isna().all()
    assert readiness().status == "BLOCKED_EXTERNAL_DATA"


def test_enhanced_optimizer_is_deterministic_and_hard_constraint_only() -> None:
    exposure_columns = (
        "industry_a",
        "size",
        "beta",
        "momentum",
        "residual_volatility",
        "liquidity",
        "value",
    )
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "expected_alpha": [0.01, 0.02, 0.0, -0.01],
            "current_weight": [0.25] * 4,
            "benchmark_weight": [0.25] * 4,
            "specific_variance": [0.01] * 4,
            "adv": [1_000_000_000.0] * 4,
            **{column: [0.25, 0.25, 0.25, 0.25] for column in exposure_columns},
        }
    )
    covariance = pd.DataFrame(
        np.zeros((len(exposure_columns), len(exposure_columns))),
        index=exposure_columns,
        columns=exposure_columns,
    )
    config = OptimizerConfig(
        initial_nav=1_000_000_000.0,
        active_name_limit=0.005,
        tracking_error_limit=0.05,
        target_turnover_limit=0.20,
        industry_active_limit=0.02,
        style_active_limit=0.20,
        adv_participation_limit=0.05,
        commission_bps=3.0,
        slippage_bps=5.0,
        impact_bps=25.0,
        solver_ftol=1e-12,
        solver_maxiter=1000,
    )
    result = optimize_portfolio(
        assets,
        factor_covariance=covariance,
        industry_columns=("industry_a",),
        style_columns=exposure_columns[1:],
        config=config,
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.target_turnover <= 0.20 + 1e-8
    assert result.maximum_adv_participation <= 0.05 + 1e-8


def test_fundamental_value_factors_use_panel_total_cap() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset", "asset"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "total_assets": [100.0, 120.0],
            "total_liabilities": [40.0, 48.0],
            "parent_equity": [60.0, 72.0],
            "cash": [10.0, 12.0],
            "short_term_borrowings": [5.0, 6.0],
            "current_portion_noncurrent_liabilities": [2.0, 2.0],
            "long_term_borrowings": [8.0, 9.0],
            "bonds_payable": [1.0, 1.0],
            "operating_revenue_ttm": [30.0, 36.0],
            "operating_cost_ttm": [20.0, 24.0],
            "consolidated_net_income_ttm": [8.0, 9.0],
            "parent_net_income_ttm": [7.0, 8.0],
            "operating_cash_flow_ttm": [9.0, 10.0],
            "capex_cash_paid_ttm": [4.0, 5.0],
            "previous_total_assets": [80.0, 100.0],
            "previous_parent_equity": [50.0, 60.0],
            "previous_parent_net_income": [6.0, 7.0],
            "previous_operating_revenue": [25.0, 30.0],
            "average_total_assets": [90.0, 110.0],
            "average_parent_equity": [55.0, 66.0],
            "total_market_cap": [200.0, 220.0],
        }
    )
    result = calculate_fundamental_factors(
        frame, factor_columns=("earnings_yield", "book_to_market")
    )
    assert result["earnings_yield"].tolist() == pytest.approx([0.035, 8 / 220])
    assert result["book_to_market"].tolist() == pytest.approx([0.3, 72 / 220])


def test_existing_interim_outputs_block_dataset_without_identity(tmp_path) -> None:
    sessions = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.Series(np.arange(10.0, 270.0), index=sessions)
    bars = pd.DataFrame(
        {
            "source_asset_id": "000001",
            "timestamp": sessions,
            "trade_date": sessions,
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 100.0,
            "amount": 1_000.0,
            "after_hours_volume": 0.0,
            "after_hours_amount": 0.0,
            "prev_close": prices - 1.0,
            "limit_status": 0,
            "period_return": prices.pct_change().fillna(0.0),
            "adj_close": prices,
            "adj_open": prices,
            "float_market_cap": 1_000_000.0,
            "total_market_cap": 1_200_000.0,
            "turnover": np.nan,
        }
    )
    stock = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0]],
            "stock_name": ["demo"],
            "listing_date": [sessions[0] - pd.Timedelta(days=500)],
            "delisting_date": [pd.NaT],
            "status_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "company_status": ["normal"],
            "industry_level1_code": ["A"],
            "industry_level1_name": ["A"],
        }
    )
    cap = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0] - pd.Timedelta(days=500)],
            "share_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "total_shares": [12_000.0],
            "a_float_shares": [10_000.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "timestamp": sessions,
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
            "source_day_of_week": sessions.dayofweek,
            "source_state": "O",
        }
    )
    layout = StorageLayout(project_root=tmp_path)
    policy = capture_test_policy(
        available_memory_bytes=512 * 1024 * 1024,
        explicit_cap_bytes=256 * 1024 * 1024,
    )
    source_frames = {
        "market_bars_1d": bars,
        "stock_master": stock,
        "market_cap": cap,
        "trading_calendar": calendar,
    }
    with runtime_resource_policy_scope(policy):
        for data_id, frame in source_frames.items():
            spec = interim_catalog().get(data_id)
            write_dataset(
                layout.interim_data_root(spec.source_id, spec.data_id),
                frame.loc[:, list(spec.required_columns)],
                spec=spec,
                resource_policy=policy,
                borrowed_input=True,
            )
        from quant_research_platform.profiles.research import research_profile

        result = research.dataset_build(
            profile=research_profile(),
            layout=layout,
            dataset_id="pv_all",
            start_date=sessions[0].date().isoformat(),
            end_date=sessions[-1].date().isoformat(),
        )
        assert result["status"] == "BLOCKED"
        assert any(
            item["reason"] == "semantic_identity_not_available"
            for item in result["blockers"]
        )
