from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.research.factors.fundamental.calculations import (
    calculate_fundamental_factors,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    FORMAL_CORE_PRICE_VOLUME_FACTORS,
    FORMAL_PRICE_VOLUME_FACTORS,
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
    calibrate_expected_alpha,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskState,
    MarketModelRiskState,
)
from quant_research_platform.research.protocols.a_share import (
    PROTOCOL_ID,
    SCENARIO_IDS,
    protocol,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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
    result = calculate_price_volume_factors(
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
            "can_enter": [False, True, True, True, True],
        }
    )
    result = add_forward_return_labels(
        panel,
        horizons=(1,),
        sessions=sessions,
        can_enter_column="can_enter",
    )
    assert pd.isna(result.loc[0, "forward_return_1d"])
    assert result.loc[1, "forward_return_1d"] == pytest.approx(13.0 / 12.0 - 1.0)
    assert result.loc[1, "forward_return_1d_available_at"] == sessions[3]


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
            "eligible": [True, True],
        }
    )
    result = legal_training_rows(frame, fitting_cutoff="2024-01-02")
    assert result.index.tolist() == [0]


def test_training_selector_requires_eligible_finite_features_and_target() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "eligible": [True, False, True],
            "feature": [1.0, 2.0, np.nan],
            "model_target": [0.1, 0.2, 0.3],
            "label_available_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02"]
            ),
        }
    )

    result = legal_training_rows(
        frame,
        fitting_cutoff="2024-01-02",
        feature_columns=("feature",),
        target_column="model_target",
    )

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
            "forward_return_20d_available_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "eligible": True,
        }
    )
    with runtime_resource_policy_scope(None):
        result = _fit_model_result_owned_from_frames(
            [frame],
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2024-01-02",
        )
        fitted = result.value if hasattr(result, "value") else result
        assert fitted.metadata["scaler"]["columns"] == ["feature"]
        if hasattr(result, "release"):
            result.release(owner="test.scaler")


def test_model_fitting_excludes_labels_unavailable_at_fitting_cutoff() -> None:
    spec = ModelSpec(
        model_id="cutoff",
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
            "forward_return_20d_available_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "eligible": True,
        }
    )

    with runtime_resource_policy_scope(None):
        result = _fit_model_result_owned_from_frames(
            [frame],
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2024-01-02",
        )
        fitted = result.value if hasattr(result, "value") else result
        assert fitted.metadata["rows_after_input_label_dropna"] == 2
        assert fitted.metadata["train_end"] == "2024-01-01"
        if hasattr(result, "release"):
            result.release(owner="test.cutoff")


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
            "forward_return_20d_available_at": pd.to_datetime(
                ["2024-01-02"] * 3 + ["2024-01-03"] * 3
            ),
            "eligible": True,
        }
    )
    with runtime_resource_policy_scope(None):
        result = _fit_model_result_owned_from_frames(
            [frame],
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2024-01-02",
        )
        fitted = result.value if hasattr(result, "value") else result
        predictions = fitted.prediction_model.predict(
            frame.loc[:, ["timestamp", "factor_a", "factor_b"]]
        )
        assert np.isfinite(predictions).all()
        assert fitted.metadata["scaler"] is None
        if hasattr(result, "release"):
            result.release(owner="test.rank")


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
    first = calculate_price_volume_factors(base)
    second = calculate_price_volume_factors(changed)
    assert second.loc[
        219, ["reversal_20d", "volatility_20d"]
    ].tolist() == pytest.approx(
        first.loc[219, ["reversal_20d", "volatility_20d"]].tolist()
    )
    from quant_research_platform.contracts.time import TimeWindow

    window = TimeWindow(sessions[200].date(), sessions[219].date())
    selected = calculate_price_volume_factors(base, output_window=window)
    pd.testing.assert_frame_equal(
        selected, first.iloc[200:220].reset_index(drop=True), check_exact=True
    )
    pd.testing.assert_frame_equal(
        selected,
        calculate_price_volume_factors(changed, output_window=window),
        check_exact=True,
    )


def test_alpha_calibration_is_schedule_matched_and_non_flipping() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "timestamp": [date for date in dates for _asset in ("a", "b")],
            "alpha_score": [-0.25, 0.25] * 6,
            "holding_return": [0.01, 0.03] * 6,
        }
    )
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

    calendar = pd.bdate_range("2024-01-01", periods=8)
    intervals = alpha_calibration_intervals(calendar, "daily")
    calibrated = calibrate_expected_alpha(
        frame,
        as_of=calendar[-1],
        scheduled_intervals=intervals,
        minimum_valid_dates=6,
    )
    assert calibrated.benchmark_only is False
    assert calibrated.valid_rebalance_dates == 6
    assert calibrated.slope > 0.0
    assert calibrated.slope_standard_error == pytest.approx(0.0, abs=1e-14)
    assert calibrated.slope_ci_low == pytest.approx(calibrated.slope)
    assert calibrated.slope_ci_high == pytest.approx(calibrated.slope)
    assert 0.0 <= calibrated.slope_pvalue <= 1.0
    inverse = calibrate_expected_alpha(
        frame.assign(holding_return=-frame["holding_return"]),
        as_of=calendar[-1],
        scheduled_intervals=intervals,
        minimum_valid_dates=6,
    )
    assert inverse.benchmark_only is True
    assert inverse.slope < 0.0


def test_calibration_weighted_cluster_reference_and_true_mature_coverage():
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster

    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

    sessions = pd.bdate_range("2023-01-02", "2023-02-20")
    intervals = alpha_calibration_intervals(sessions, "weekly-last-trading-day")
    dates = intervals.timestamp.iloc[:6]
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "alpha_score": x,
                "holding_return": 0.02
                + (0.05 + 0.003 * g) * x
                + 0.002 * g
                + 0.001 * ((g + i) % 3 - 1),
            }
            for g, timestamp in enumerate(dates)
            for i, x in enumerate((-1.0, 0.0, 1.0))
        ]
    )
    result = calibrate_expected_alpha(
        frame, as_of=sessions[-1], scheduled_intervals=intervals
    )
    x = sm.add_constant(frame.alpha_score)
    weights = np.full(len(frame), 1 / 18)
    fit = sm.WLS(frame.holding_return, x, weights=weights).fit()
    # Same uncorrected date-cluster sandwich as production; no df policy change.
    covariance = cov_cluster(fit, frame.timestamp, use_correction=False)
    assert result.slope_standard_error == pytest.approx(
        np.sqrt(covariance[1, 1]), rel=1e-12
    )
    assert result.weighted_r2 == pytest.approx(fit.rsquared, rel=1e-12)
    assert result.scheduled_rebalance_dates == 6
    missing = calibrate_expected_alpha(
        frame.loc[frame.timestamp.ne(dates.iloc[1])],
        as_of=sessions[-1],
        scheduled_intervals=intervals,
    )
    assert missing.scheduled_rebalance_dates == 6 and missing.valid_rebalance_dates == 5
    assert missing.coverage == pytest.approx(5 / 6) and missing.benchmark_only
    stale = calibrate_expected_alpha(
        frame, as_of=pd.Timestamp("2024-03-01"), scheduled_intervals=intervals
    )
    assert stale.valid_rebalance_dates == 0 and stale.scheduled_rebalance_dates == 0
    assert stale.benchmark_only


def test_calibration_intervals_use_adjacent_execution_opens_across_holidays():
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

    sessions = pd.bdate_range("2024-01-02", "2024-01-23").difference(
        pd.to_datetime(["2024-01-12", "2024-01-15"])
    )
    weekly = alpha_calibration_intervals(sessions, "weekly-last-trading-day")
    first = weekly.iloc[0]
    assert first.timestamp == pd.Timestamp("2024-01-05")
    assert first.entry_at == pd.Timestamp("2024-01-08")
    assert first.available_at == pd.Timestamp("2024-01-16")
    frame = pd.DataFrame(
        {
            "timestamp": [first.timestamp] * 2,
            "alpha_score": [-1.0, 1.0],
            "holding_return": [0.01, 0.03],
        }
    )
    before = calibrate_expected_alpha(
        frame,
        as_of=pd.Timestamp("2024-01-15"),
        scheduled_intervals=weekly,
        minimum_valid_dates=1,
    )
    after = calibrate_expected_alpha(
        frame,
        as_of=pd.Timestamp("2024-01-16"),
        scheduled_intervals=weekly,
        minimum_valid_dates=1,
    )
    assert before.valid_rebalance_dates == 0
    assert after.valid_rebalance_dates == 1
    monthly = alpha_calibration_intervals(
        pd.bdate_range("2024-01-02", "2024-03-05").difference(
            pd.to_datetime(["2024-03-01"])
        ),
        "monthly-last-trading-day",
    )
    assert monthly.iloc[0].timestamp == pd.Timestamp("2024-01-31")
    assert monthly.iloc[0].entry_at == pd.Timestamp("2024-02-01")
    assert monthly.iloc[0].available_at == pd.Timestamp("2024-03-04")


@pytest.mark.parametrize("changed_eligibility", (False, True))
def test_market_risk_uses_daily_intercept_residuals_and_previous_session_caps(
    changed_eligibility,
):
    state = MarketModelRiskState(window=4, residual_window=3)
    histories = {asset: [] for asset in ("a", "b")}
    residuals = {asset: [] for asset in histories}
    previous_caps = {}
    for position, market in enumerate(
        (0.01, -0.02, 0.03, 0.02, -0.01, 0.04, 0.01, -0.03)
    ):
        suspended = changed_eligibility and position == 4
        panel = pd.DataFrame(
            {
                "asset_id": ["a", "b"],
                "period_return": [
                    0.0 if suspended else 0.02 + 1.5 * market,
                    -0.02 + 0.5 * market,
                ],
                "float_market_cap": [300.0 if suspended else 100.0, 100.0],
                "eligible": [not suspended, True],
            }
        )
        basket = panel.loc[panel.eligible]
        denominator = sum(previous_caps.get(asset, 0.0) for asset in basket.asset_id)
        benchmark = (
            sum(
                previous_caps[row.asset_id] * row.period_return
                for row in basket.itertuples()
            )
            / denominator
            if denominator
            else np.nan
        )
        styles = state.observe(panel)
        for row in panel.itertuples():
            history = histories[row.asset_id]
            if np.isfinite(benchmark):
                history.append((benchmark, row.period_return))
            expected_beta = expected_residual = np.nan
            if len(history) >= 4:
                values = np.array(history[-4:])
                alpha, expected_beta = np.linalg.lstsq(
                    np.column_stack((np.ones(4), values[:, 0])),
                    values[:, 1],
                    rcond=None,
                )[0]
                expected_residual = (
                    row.period_return - alpha - expected_beta * benchmark
                )
            residuals[row.asset_id].append(expected_residual)
            expected_volatility = np.std(residuals[row.asset_id][-3:], ddof=0)
            assert styles.beta.get(row.asset_id, np.nan) == pytest.approx(
                expected_beta, nan_ok=True
            )
            assert styles.residual_volatility.get(
                row.asset_id, np.nan
            ) == pytest.approx(expected_volatility, abs=1e-12, nan_ok=True)
        previous_caps = dict(zip(panel.asset_id, panel.float_market_cap, strict=True))
    if not changed_eligibility:
        assert styles.residual_volatility.to_numpy() == pytest.approx(
            [0.0, 0.0], abs=1e-12
        )


def test_market_risk_does_not_reuse_mature_residuals_after_missing_or_constant_days():
    state = MarketModelRiskState(window=3, residual_window=2)

    def observe(value, assets=("a", "b"), eligible=True):
        return state.observe(
            pd.DataFrame(
                {
                    "asset_id": assets,
                    "period_return": [value] * len(assets),
                    "float_market_cap": [100.0] * len(assets),
                    "eligible": [eligible] * len(assets),
                }
            )
        )

    for value in (0.0, 0.01, 0.03, -0.02, 0.02):
        mature = observe(value)
    assert mature.residual_volatility.notna().all()
    for _ in range(3):
        invalid = observe(0.02)
    assert invalid.beta.isna().all()
    assert invalid.residual_volatility.isna().all()
    observe(0.03, eligible=False)
    assert observe(0.04).beta.isna().all()
    for value in (0.01, -0.01, 0.02, 0.03):
        observe(value)
    observe(0.01, assets=("b",))
    returned = observe(-0.01)
    assert np.isnan(returned.beta["a"])
    assert np.isnan(returned.residual_volatility["a"])


@pytest.mark.parametrize("missing_column", ("period_return", "float_market_cap"))
def test_canonical_listing_null_does_not_invalidate_complete_market_history(
    missing_column,
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        panels,
    )

    state = MarketModelRiskState(window=3, residual_window=2)
    assets = ["a", "new_listing"] if missing_column == "float_market_cap" else ["a"]
    for value in (0.0, 0.01, 0.03, -0.02, 0.02):
        state.observe(
            pd.DataFrame(
                {
                    "asset_id": assets,
                    "period_return": value,
                    "float_market_cap": 100.0,
                    "eligible": True,
                }
            )
        )
    raw = pd.DataFrame(
        {
            "source_asset_id": ["a", "new_listing"],
            "timestamp": pd.Timestamp("2024-01-08"),
            "market_type": 1,
            "trading_status_code": 1,
            "filling_flag": 0,
            **{
                name: 100.0
                for name in (
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_open",
                    "adj_close",
                    "float_market_cap",
                    "total_market_cap",
                )
            },
            "volume": 100.0,
            "amount": 10000.0,
            "period_return": [0.04, 0.2],
        }
    )
    raw.loc[1, missing_column] = np.nan
    panel = panels._apply_suspension_semantics(panels._canonical_bars(raw))
    panel["eligible"] = panel.asset_id.eq("a")
    assert panel.loc[panel.asset_id.eq("new_listing"), missing_column].isna().all()
    result = state.observe(panel)
    assert result.beta["a"] == pytest.approx(1.0)
    assert result.residual_volatility["a"] == pytest.approx(0.0, abs=1e-12)
    if missing_column == "period_return":
        assert np.isnan(result.beta["new_listing"])
    panel["eligible"] = True
    panel["period_return"] = [0.01, -0.03]
    panel["float_market_cap"] = 100.0
    result = state.observe(panel)
    if missing_column == "float_market_cap":
        assert result.beta["a"] == pytest.approx(1.0)
    else:
        assert np.isnan(result.beta["new_listing"])
    panel["eligible"] = [False, True]
    panel["period_return"] = [np.nan, 0.03]
    interrupted = state.observe(panel)
    assert np.isnan(interrupted.beta["a"])
    assert np.isnan(interrupted.residual_volatility["a"])
    panel["eligible"] = True
    panel["period_return"] = [0.02, -0.02]
    assert np.isnan(state.observe(panel).beta["a"])


@pytest.mark.parametrize(
    "column,value",
    (
        ("period_return", np.inf),
        ("period_return", True),
        ("period_return", "invalid"),
        ("float_market_cap", np.inf),
        ("float_market_cap", 0.0),
        ("float_market_cap", -1.0),
    ),
)
def test_ineligible_market_risk_still_rejects_invalid_nonnull_inputs(column, value):
    panel = pd.DataFrame(
        {
            "asset_id": ["a"],
            "period_return": [0.01],
            "float_market_cap": [100.0],
            "eligible": [False],
        }
    )
    panel[column] = value
    with pytest.raises(ValueError, match="market_model_risk_(numeric|key)_invalid"):
        MarketModelRiskState().observe(panel)


def test_enhanced_risk_state_is_causal_and_produces_psd_risk() -> None:
    dates = pd.bdate_range("2022-01-03", periods=564)
    style_columns = (
        "log_float_market_cap",
        "beta_252d",
        "momentum_252_20",
        "residual_volatility_60d",
        "amihud_60d",
        "earnings_yield",
    )
    assets = tuple(f"a{index:02d}" for index in range(40))
    market_model = MarketModelRiskState()
    factor_risk = FactorRiskState(
        style_columns=style_columns,
        minimum_cross_section=2,
    )
    for position, timestamp in enumerate(dates):
        returns = pd.Series(
            {
                asset: 0.001 + ((index % 7) - 3) * 0.0002 + 0.0001 * (position % 5)
                for index, asset in enumerate(assets)
            },
            dtype="float64",
        )
        panel = pd.DataFrame(
            {
                "asset_id": assets,
                "period_return": returns,
                "float_market_cap": [1_000.0 + index for index in range(len(assets))],
                "eligible": [True] * len(assets),
            }
        )
        styles = market_model.observe(panel)
        if position == 0:
            continue
        cross = pd.DataFrame(
            {
                "timestamp": timestamp,
                "asset_id": assets,
                "asset_return": returns,
                "industry_level1_code": ["A01"] * len(assets),
                "float_market_cap": [1_000.0 + index for index in range(len(assets))],
                "eligible": [True] * len(assets),
                **{
                    column: (
                        styles.beta
                        if column == "beta_252d"
                        else styles.residual_volatility
                        if column == "residual_volatility_60d"
                        else [
                            ((index * (style_index + 2)) % 9 - 4) * 0.1
                            for index in range(len(assets))
                        ]
                    )
                    for style_index, column in enumerate(style_columns)
                },
            }
        )
        factor_risk.observe_return_session(cross)
    assert styles.beta.notna().all()
    assert styles.residual_volatility.notna().all()
    estimate = factor_risk.estimate()
    assert (
        np.linalg.eigvalsh(estimate.factor_covariance.to_numpy(dtype="float64")).min()
        >= -1e-12
    )
    assert estimate.specific_variance.notna().all()
    assert estimate.specific_variance.gt(0.0).all()


def test_factor_risk_projects_history_to_current_industry_domain() -> None:
    state = FactorRiskState(
        style_columns=("style",),
        minimum_cross_section=10,
        minimum_factor_dates=3,
        minimum_specific_dates=3,
    )
    x = np.arange(20, dtype="float64")
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{index}" for index in range(20)],
            "industry_level1_code": ["A"] * 10 + ["B"] * 10,
            "float_market_cap": 100 + x,
            "style": x,
            "asset_return": np.sin(x) / 100,
        }
    )
    for position in range(6):
        current = cross.assign(
            timestamp=pd.Timestamp("2020-01-01") + pd.Timedelta(days=position)
        )
        if position >= 3:
            current["industry_level1_code"] = "A"
        state.observe_return_session(current)
    exposures = state.current_exposures(current)
    assert "industry_B" not in exposures
    estimate = state.estimate()
    assert tuple(estimate.factor_covariance.columns) == ("industry_A", "style")
    assert len(estimate.factor_returns) == 6
    held = current.assign(eligible=True)
    held.loc[held.index[-1], ["industry_level1_code", "eligible"]] = ["B", False]
    assert "industry_B" in state.current_exposures(held)
    assert "industry_B" in state.estimate().factor_covariance
    current["industry_level1_code"] = "C"
    state.current_exposures(current)
    with pytest.raises(ValueError, match="factor_risk_factor_history_insufficient"):
        state.estimate()


@pytest.mark.parametrize(
    "entry,exit_price", [(-1.0, 2.0), (1.0, 0.0), (1.0, -1.0), (1e-308, 1e308)]
)
def test_forward_labels_reject_invalid_economic_prices_and_overflow(
    entry, exit_price
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": dates,
            "adj_open": [1.0, entry, exit_price],
            "can_enter": True,
        }
    )
    result = add_forward_return_labels(
        frame,
        horizons=(1,),
        sessions=dates,
        can_enter_column="can_enter",
    )
    assert pd.isna(result.loc[0, "forward_return_1d"])
    assert pd.isna(result.loc[0, "forward_return_1d_available_at"])


def test_factor_risk_windows_and_decay_follow_observed_sessions() -> None:
    dates = pd.bdate_range("2020-01-01", periods=509)
    specific_state = FactorRiskState(
        style_columns=("style",),
        minimum_cross_section=10,
        minimum_factor_dates=3,
        minimum_specific_dates=3,
    )
    factor_state = FactorRiskState(
        style_columns=("style",),
        minimum_cross_section=10,
        minimum_factor_dates=3,
        minimum_specific_dates=3,
    )
    x = np.arange(12, dtype="float64")

    def cross(position: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": dates[position],
                "asset_id": [f"a{index}" for index in range(12)],
                "industry_level1_code": "A",
                "float_market_cap": 100 + x,
                "style": x,
                "asset_return": np.sin(x + position) / 100,
            }
        )

    fits = []
    for position in range(4):
        current = cross(position)
        if position == 2:
            current["style"] = np.nan
        else:
            fits.append(specific_state.fit_return_session(current))
        specific_state.observe_return_session(current)
        factor_state.observe_return_session(current)
    estimate = specific_state.estimate()
    ages = np.array([3.0, 2.0, 0.0])
    weights = 0.5 ** (ages / 126.0)
    weights /= weights.sum()
    returns = np.vstack([fit.factor_returns.to_numpy() for fit in fits])
    centered = returns - np.average(returns, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered
    expected_covariance = (0.8 * covariance + 0.2 * np.diag(np.diag(covariance))) * 252
    np.testing.assert_allclose(
        estimate.factor_covariance, expected_covariance, rtol=1e-12, atol=1e-15
    )
    specific_weights = 0.5 ** (ages / 63.0)
    specific_weights /= specific_weights.sum()
    expected_specific = (
        np.dot(specific_weights, [fit.residuals.loc["a0"] ** 2 for fit in fits]) * 252
    )
    assert estimate.specific_variance.loc["a0"] == pytest.approx(
        expected_specific, rel=1e-12
    )
    assert estimate.factor_returns.timestamp.tolist() == dates[[0, 1, 3]].tolist()

    for position in range(4, len(dates)):
        current = cross(position)
        current.loc[current.asset_id.eq("a0"), "style"] = np.nan
        if position < 258:
            specific_state.observe_return_session(current)
        current["style"] = np.nan
        factor_state.observe_return_session(current)
    assert "a0" not in specific_state.estimate().specific_variance
    with pytest.raises(ValueError, match="factor_risk_factor_history_insufficient"):
        factor_state.estimate()


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
            "version_identity": ["original"],
            "value": [100.0],
        }
    )
    from quant_research_platform.markets.implementations.a_share.fundamentals import (
        materialize_pit_fundamentals,
    )

    assert materialize_pit_fundamentals(observations, events)["value"].isna().all()


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
        solver_ftol=1e-12,
        solver_maxiter=1000,
    )
    result = optimize_portfolio(
        assets,
        factor_covariance=covariance,
        industry_columns=("industry_a",),
        style_columns=exposure_columns[1:],
        expected_transaction_cost=lambda _weights: 0.0,
        config=config,
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.target_turnover <= 0.20 + 1e-8
    assert result.maximum_adv_participation <= 0.05 + 1e-8
    cash_assets = assets.copy()
    cash_assets["current_weight"] = 0.225
    cash_assets["is_cash"] = False
    cash_assets = pd.concat(
        [
            cash_assets,
            pd.DataFrame(
                [
                    {
                        "asset_id": "__cash__",
                        "expected_alpha": 0.0,
                        "current_weight": 0.1,
                        "benchmark_weight": 0.0,
                        "specific_variance": 0.0,
                        "adv": np.inf,
                        **{column: 0.0 for column in exposure_columns},
                        "is_cash": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    cash_result = optimize_portfolio(
        cash_assets,
        factor_covariance=covariance,
        industry_columns=("industry_a",),
        style_columns=exposure_columns[1:],
        expected_transaction_cost=lambda _weights: 0.0,
        cash_column="is_cash",
        config=config,
    )
    assert cash_result.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("adv", "requires_trade", "error"),
    [
        (1e10, True, None),
        (1e8, True, "optimizer_failed_closed"),
        (np.nan, True, "optimizer_trade_adv_unavailable"),
        (0.0, True, "optimizer_trade_adv_unavailable"),
        (np.nan, False, None),
    ],
)
def test_optimizer_fixed_missing_exposure_retains_real_trade_capacity(
    adv: float, requires_trade: bool, error: str | None
) -> None:
    assets = pd.DataFrame(
        {
            "asset_id": list("abcd"),
            "expected_alpha": [0.0, 0.2, -0.2, 0.1],
            "current_weight": [0.3, 0.2, 0.25, 0.25] if requires_trade else [0.25] * 4,
            "benchmark_weight": 0.25,
            "specific_variance": 0.01,
            "adv": [adv, 1e10, 1e10, 1e10],
            "industry": 1.0,
            "style": [np.nan, 1.0, -1.0, 0.0],
        }
    )
    covariance = pd.DataFrame(
        np.eye(2) * 0.01, index=("industry", "style"), columns=("industry", "style")
    )

    def optimize():
        return optimize_portfolio(
            assets,
            factor_covariance=covariance,
            industry_columns=("industry",),
            style_columns=("style",),
            expected_transaction_cost=lambda _weights: 0.0,
        )

    if error is not None:
        with pytest.raises(ValueError, match=error):
            optimize()
    else:
        result = optimize()
        assert result.weights.loc["a"] == pytest.approx(0.25)
        assert result.weights.loc["b"] > 0.25
        assert np.isfinite(result.predicted_tracking_error)
        assert result.maximum_adv_participation <= 0.05 + 1e-8


def test_fundamental_value_factors_use_panel_total_cap() -> None:
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.research.factors.engine import (
        calculate_factors_for_panel,
        factor_panel_input_columns,
    )
    from quant_research_platform.research.factors.fundamental.definitions import (
        FUNDAMENTAL_FACTORS,
    )

    frame = pd.DataFrame(
        {
            "market_id": "a_share",
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
            "prior_year_total_assets": [80.0, 100.0],
            "prior_year_parent_equity": [50.0, 60.0],
            "prior_year_parent_net_income_ttm": [6.0, 7.0],
            "prior_year_operating_revenue_ttm": [25.0, 30.0],
            "average_total_assets": [90.0, 110.0],
            "average_parent_equity": [55.0, 66.0],
            "total_market_cap": [200.0, 220.0],
        }
    )
    result = calculate_fundamental_factors(frame)
    assert result["earnings_yield"].tolist() == pytest.approx([0.035, 8 / 220])
    assert result["book_to_market"].tolist() == pytest.approx([0.3, 72 / 220])
    expected = [
        0.035,
        0.3,
        0.15,
        0.045,
        7 / 55,
        8 / 90,
        10 / 90,
        1 / 3,
        0.1,
        -1 / 90,
        0.25,
        0.2,
        1 / 90,
        0.2,
        4 / 90,
        0.4,
        0.16,
        0.1,
        0.06,
    ]
    assert result.loc[0, list(FUNDAMENTAL_FACTORS)].tolist() == pytest.approx(expected)
    definitions = tuple(
        definition
        for definition in research_profile().factors.list()
        if definition.family == "fundamental"
    )
    projected = calculate_factors_for_panel(
        frame.loc[:, factor_panel_input_columns(definitions)],
        definitions=definitions,
    )
    assert projected.loc[0, list(FUNDAMENTAL_FACTORS)].tolist() == pytest.approx(
        expected
    )


def test_missing_raw_identity_blocks_dataset_build(tmp_path) -> None:
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
            item["reason"] == "raw_content_identity_unavailable"
            for item in result["blockers"]
        )


def test_factor_return_wls_uses_sqrt_cap_statistical_weight():
    import statsmodels.api as sm

    from quant_research_platform.research.portfolios.risk import FactorRiskState
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS

    rng = np.random.default_rng(913)
    count = 100
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-03"),
            "asset_id": [f"asset_{i:03d}" for i in range(count)],
            "industry_level1_code": ["A01"] * 50 + ["B01"] * 50,
            "float_market_cap": np.geomspace(100.0, 1e8, count),
            "asset_return": rng.normal(scale=0.03, size=count),
            "eligible": True,
            **{column: rng.normal(size=count) for column in RISK_STYLE_COLUMNS},
        }
    )
    fit = FactorRiskState(style_columns=RISK_STYLE_COLUMNS).fit_return_session(
        cross, require_complete=True
    )
    assert fit is not None
    expected = sm.WLS(
        fit.asset_returns,
        fit.exposures,
        weights=np.sqrt(
            cross.set_index("asset_id").float_market_cap.reindex(fit.exposures.index)
        ),
    ).fit()
    np.testing.assert_allclose(
        fit.factor_returns, expected.params, rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(fit.residuals, expected.resid, rtol=1e-11, atol=1e-13)
