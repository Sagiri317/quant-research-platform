from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.market_model import MarketModelRiskState
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
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.models.fitting import (
    _collect_training_frame,
    _fit_model_result,
)
from quant_research_platform.research.models.scaler import (
    ModelMatrixScaler,
    equal_timestamp_sample_weights,
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
    FactorRiskConfig,
    FactorRiskState,
)
from quant_research_platform.research.protocols.a_share import (
    EXECUTION_ASSUMPTIONS,
    PROTOCOL_ID,
    SCENARIO_IDS,
    protocol,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.research_configuration import CORE6_FACTORS, PV25_FACTORS


def test_research_protocol_identity_and_surfaces_are_explicit() -> None:
    assert len(PV25_FACTORS) == 25
    assert len(CORE6_FACTORS) == 6
    assert SCENARIO_IDS == (
        "pv_all",
        "pv_expanded",
        "pv_nf_expectations",
        "fusion_nf_core",
        "fusion_nf_expectations",
    )
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


def test_model_scaler_uses_equal_timestamp_weights_and_zero_variance_zero() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 3.0, 5.0],
            "constant": [2.0, 2.0, 2.0],
        }
    )
    weights = equal_timestamp_sample_weights(
        pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    )
    assert weights.tolist() == pytest.approx([0.75, 0.75, 1.5])
    assert weights.sum() == pytest.approx(3.0)
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    transformed = scaler.transform(frame)
    assert transformed["constant"].tolist() == [0.0, 0.0, 0.0]
    assert scaler.mean_[0] == pytest.approx(3.5)
    assert scaler.scale_[0] == pytest.approx(np.sqrt(11.0) / 2.0)


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


def test_supervised_fit_uses_only_mature_training_rows_and_scaler() -> None:
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
            "model_target_available_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "eligible": True,
        }
    )
    result = _fit_model_result(
        _collect_training_frame([frame], spec, fitting_cutoff="2024-01-02"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    fitted = result
    assert fitted.metadata["scaler"]["columns"] == ["feature"]
    assert fitted.prediction_model.scaler.mean_ == (1.5,)
    assert fitted.prediction_model.scaler.scale_ == (0.5,)
    assert fitted.metadata["rows_after_input_label_dropna"] == 2
    assert fitted.metadata["train_end"] == "2024-01-01"
    frame.loc[2:, ["feature", "model_target"]] = [[1e9, -1e9], [-1e9, 1e9]]
    changed = _fit_model_result(
        _collect_training_frame([frame], spec, fitting_cutoff="2024-01-02"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    assert changed.metadata == fitted.metadata
    assert changed.prediction_model.scaler == fitted.prediction_model.scaler


@pytest.mark.parametrize(
    "algorithm",
    ("linear_regression", "equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"),
)
def test_fixed_model_parameters_agree_with_fitting_and_metadata(algorithm):
    from dataclasses import replace

    from quant_research_platform.contracts.models import validate_model_metadata_core

    spec = ModelSpec(
        model_id="fixed",
        model_algorithm=algorithm,
        input_factor_columns=("x",),
        label_column="y",
    )
    with pytest.raises(ValueError, match="^model_hyperparameters_not_supported$"):
        replace(spec, hyperparameters={"fit_intercept": False})
    preflight_model_specs((spec,))
    timestamp = pd.Timestamp("2020-01-02")
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "asset_id": ["a", "b", "c", "d"],
            "eligible": True,
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 7.0, 9.0, 11.0],
            "label_available_at": timestamp,
        }
    )
    result = _fit_model_result(
        _collect_training_frame((frame,), spec, fitting_cutoff=timestamp),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    metadata = validate_model_metadata_core(result.metadata)
    assert metadata["hyperparameters"] == metadata["resolved_hyperparameters"] == {}
    if algorithm == "linear_regression":
        np.testing.assert_allclose(
            result.prediction_model.predict(frame[["x"]]), frame.y
        )
    metadata["hyperparameters"] = {"fit_intercept": False}
    metadata["resolved_hyperparameters"] = {"fit_intercept": False}
    with pytest.raises(ValueError, match="^invalid_model_metadata$"):
        validate_model_metadata_core(metadata)


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
            "model_target_available_at": pd.to_datetime(
                ["2024-01-02"] * 3 + ["2024-01-03"] * 3
            ),
            "eligible": True,
        }
    )
    result = _fit_model_result(
        _collect_training_frame([frame], spec, fitting_cutoff="2024-01-02"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    fitted = result
    predictions = fitted.prediction_model.predict(
        frame.loc[:, ["timestamp", "factor_a", "factor_b"]]
    )
    assert fitted.prediction_model.directions == (1.0, -1.0)
    assert predictions[-1] == pytest.approx(1.0 / 6.0)
    assert np.isfinite(predictions).all()
    assert fitted.metadata["scaler"] is None


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
            "turnover": 0.01,
            "float_market_cap": 100_000.0,
            "total_market_cap": 150_000.0,
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
    assert calibrated.slope_ci_low is calibrated.slope_ci_high is None
    assert calibrated.slope_pvalue is None
    assert calibrated.inference_status == "UNDEFINED"
    assert calibrated.inference_reason == "alpha_calibration_zero_standard_error"
    inverse = calibrate_expected_alpha(
        frame.assign(holding_return=-frame["holding_return"]),
        as_of=calendar[-1],
        scheduled_intervals=intervals,
        minimum_valid_dates=6,
    )
    assert inverse.benchmark_only is True
    assert inverse.slope < 0.0


@pytest.mark.parametrize(
    "alpha_exponent,return_exponent",
    [(0, 0), (600, 0), (-600, 0), (0, 600), (0, -600), (600, 600), (-600, -600)],
)
def test_calibration_weighted_cluster_reference_and_unit_scaling(
    alpha_exponent, return_exponent
):
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
    alpha_scale = float(np.ldexp(1.0, alpha_exponent))
    return_scale = float(np.ldexp(1.0, return_exponent))
    coefficient_scale = float(np.ldexp(1.0, return_exponent - alpha_exponent))
    result = calibrate_expected_alpha(
        frame.assign(
            alpha_score=frame.alpha_score * alpha_scale,
            holding_return=frame.holding_return * return_scale,
        ),
        as_of=sessions[-1],
        scheduled_intervals=intervals,
    )
    x = sm.add_constant(frame.alpha_score)
    weights = np.full(len(frame), 1 / 18)
    fit = sm.WLS(frame.holding_return, x, weights=weights).fit()
    # Same uncorrected date-cluster sandwich as production; no df policy change.
    covariance = cov_cluster(fit, frame.timestamp, use_correction=False)
    assert result.slope_standard_error == pytest.approx(
        np.sqrt(covariance[1, 1]) * coefficient_scale, rel=1e-12, abs=0
    )
    assert result.slope == pytest.approx(
        fit.params.iloc[1] * coefficient_scale, rel=1e-12, abs=0
    )
    assert result.intercept == pytest.approx(
        fit.params.iloc[0] * return_scale, rel=1e-12, abs=0
    )
    assert result.slope_t_statistic == pytest.approx(
        fit.params.iloc[1] / np.sqrt(covariance[1, 1]), rel=1e-12, abs=0
    )
    assert result.weighted_r2 == pytest.approx(fit.rsquared, rel=1e-12)
    assert result.scheduled_rebalance_dates == 6


def test_alpha_calibration_missing_and_stale_date_coverage():
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

    sessions = pd.bdate_range("2023-01-02", "2023-02-20")
    intervals = alpha_calibration_intervals(sessions, "weekly-last-trading-day")
    dates = intervals.timestamp.iloc[:6]
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(2),
            "alpha_score": [0.0, 1.0] * 6,
            "holding_return": [0.0, 0.01] * 6,
        }
    )
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
    raw["listing_date"] = raw["timestamp"].min()
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


def test_enhanced_risk_state_uses_mature_estimates_and_produces_psd_risk() -> None:
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
        config=FactorRiskConfig(
            minimum_cross_section=2,
        ),
    )
    rng = np.random.default_rng(20260926)
    beta = np.linspace(0.5, 1.5, len(assets))
    residual_scale = np.linspace(0.001, 0.004, len(assets))
    market = rng.normal(0.0005, 0.01, len(dates))
    residuals = rng.normal(size=(len(dates), len(assets))) * residual_scale
    for position, timestamp in enumerate(dates):
        returns = pd.Series(
            beta * market[position] + residuals[position],
            index=assets,
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
                "industry_standard_id": "CSRC2012",
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
    assert styles.beta.std() > 0.1
    assert styles.residual_volatility.gt(0.0).all()
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
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
        ),
    )
    x = np.arange(20, dtype="float64")
    cross = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{index}" for index in range(20)],
            "industry_standard_id": "CSRC2012",
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
    assert "industry_CSRC2012:B" not in exposures
    estimate = state.estimate()
    assert tuple(estimate.factor_covariance.columns) == ("industry_CSRC2012:A", "style")
    assert len(estimate.factor_returns) == 6
    held = current.assign(eligible=True)
    held.loc[held.index[-1], ["industry_level1_code", "eligible"]] = ["B", False]
    assert "industry_CSRC2012:B" in state.current_exposures(held)
    assert "industry_CSRC2012:B" in state.estimate().factor_covariance
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
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
        ),
    )
    factor_state = FactorRiskState(
        style_columns=("style",),
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=3,
            minimum_specific_dates=3,
        ),
    )
    x = np.arange(12, dtype="float64")

    def cross(position: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": dates[position],
                "asset_id": [f"a{index}" for index in range(12)],
                "industry_standard_id": "CSRC2012",
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
        active_name_limit=0.005,
        tracking_error_limit=0.05,
        target_turnover_limit=0.20,
        industry_active_limit=0.02,
        style_active_limit=0.20,
        solver_ftol=1e-12,
        solver_maxiter=1000,
    )
    result = optimize_portfolio(
        assets,
        factor_covariance=covariance,
        industry_columns=("industry_a",),
        style_columns=exposure_columns[1:],
        expected_transaction_cost=lambda _weights: 0.0,
        nav=1_000_000_000.0,
        assumptions=EXECUTION_ASSUMPTIONS,
        config=config,
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.target_turnover <= 0.20 + 1e-8
    assert result.maximum_adv_participation <= 0.05 + 1e-8
    np.testing.assert_allclose(
        result.weights.reindex(list("abcd")),
        [0.255, 0.255, 0.245, 0.245],
        atol=1e-9,
        rtol=0,
    )
    active_alpha = (
        result.weights.reindex(list("abcd")).to_numpy() - 0.25
    ) @ assets.expected_alpha.to_numpy()
    assert active_alpha == pytest.approx(0.0002, abs=1e-11)
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
        nav=1_000_000_000.0,
        assumptions=EXECUTION_ASSUMPTIONS,
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
            nav=1_000_000_000.0,
            assumptions=EXECUTION_ASSUMPTIONS,
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
    from quant_research_platform.profiles.research import research_profile

    layout = StorageLayout(project_root=tmp_path)
    result = research.dataset_build(
        profile=research_profile(),
        layout=layout,
        dataset_id="pv_all",
        start_date="2020-01-01",
        end_date="2020-12-31",
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
            "industry_standard_id": "CSRC2012",
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


@pytest.mark.parametrize("constant", [0.0, 2.0])
@pytest.mark.parametrize("observed_dates", [1, 8])
def test_alpha_unidentified_statistics_are_not_defined_zero(constant, observed_dates):
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

    sessions = pd.bdate_range("2023-01-02", periods=10)
    intervals = alpha_calibration_intervals(sessions, "daily")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(intervals.timestamp.iloc[:observed_dates], 10),
            "alpha_score": constant,
            "holding_return": np.arange(observed_dates * 10) / 10000,
        }
    )
    design = np.column_stack([np.ones(len(frame)), frame.alpha_score])
    assert np.linalg.matrix_rank(design) == 1
    result = calibrate_expected_alpha(
        frame, as_of=sessions[-1], scheduled_intervals=intervals
    )
    assert result.status == "UNDEFINED"
    assert result.undefined_reason == (
        "alpha_calibration_rank_deficient"
        if observed_dates == 8
        else "alpha_calibration_insufficient_coverage"
    )
    assert result.benchmark_only
    assert result.coverage == observed_dates / 8
    for field in (
        "slope",
        "intercept",
        "slope_standard_error",
        "slope_t_statistic",
        "slope_pvalue",
        "slope_ci_low",
        "slope_ci_high",
        "weighted_r2",
    ):
        assert getattr(result, field) is None


@pytest.mark.parametrize("count", (3, 10, 100))
def test_decimal_constant_scaler_freezes_exact_training_value(count):
    training = pd.DataFrame(
        {"constant": np.full(count, 0.1), "varying": np.arange(count, dtype=float)}
    )
    scaler = ModelMatrixScaler.fit(training, sample_weight=np.linspace(1.0, 2.0, count))
    assert scaler.mean_[0] == 0.1
    assert scaler.scale_[0] == 1.0
    assert scaler.transform(training).constant.eq(0.0).all()
    assert (
        scaler.transform(
            pd.DataFrame({"constant": [0.2], "varying": [10.0]})
        ).constant.iloc[0]
        == 0.1
    )
    training.loc[0, "constant"] = np.nextafter(0.1, 1.0)
    assert ModelMatrixScaler.fit(training).scale_[0] != 1.0


@pytest.mark.parametrize("alpha_exponent", (0, 600, -600))
@pytest.mark.parametrize("case", ("exact_zero", "small_positive"))
def test_alpha_cluster_variance_preserves_zero_and_binary64_signal(
    case, alpha_exponent
):
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    counts = (3,) * 6 if case == "exact_zero" else (2, 3, 4, 5, 2, 7)
    rows = [
        {
            "timestamp": day,
            "alpha_score": x,
            "holding_return": 0.125 + 0.25 * x + (group - 2) * 0.0078125 * (x * x + 1),
        }
        for group, (day, count) in enumerate(zip(dates, counts, strict=True))
        for x in np.linspace(-1, 1, count)
    ]
    frame = pd.DataFrame(rows)
    scale = float(np.ldexp(1.0, alpha_exponent))
    frame["alpha_score"] *= scale
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=1),
            "available_at": dates + pd.Timedelta(days=2),
        }
    )
    result = calibrate_expected_alpha(
        frame, as_of=pd.Timestamp("2024-07-01"), scheduled_intervals=intervals
    )
    assert result.status == "DEFINED" and not result.benchmark_only
    assert result.slope == pytest.approx(0.25 / scale, rel=1e-12, abs=0)
    if case == "exact_zero":
        # Symmetric binary inputs give exactly zero date-cluster slope scores.
        assert result.slope_standard_error == 0.0
        assert result.slope_t_statistic is result.slope_pvalue is None
        assert result.slope_ci_low is result.slope_ci_high is None
    else:
        # Exact rational WLS and uncorrected six-date sandwich on binary64 inputs.
        standard_error = 1.3003283717694013e-18
        assert result.slope_standard_error == pytest.approx(
            standard_error / scale, rel=1e-12, abs=0
        )
        assert result.slope_t_statistic == pytest.approx(
            0.25 / standard_error, rel=1e-12, abs=0
        )


@pytest.mark.parametrize("quantity", ("slope", "standard_error"))
def test_alpha_calibration_rejects_nonzero_unrepresentable_coefficients(quantity):
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    if quantity == "slope":
        alpha = float(np.ldexp(1.0, 600))
        targets = [float(np.ldexp(1.0, -600))] * 6
    else:
        alpha = float(np.ldexp(1.0, 1020))
        # Residuals exceed row resolution, while the alpha scale underflows the SE.
        targets = [
            float(np.ldexp(1.0, -20) + group * np.ldexp(1.0, -60)) for group in range(6)
        ]
    frame = pd.DataFrame(
        [
            {
                "timestamp": day,
                "alpha_score": sign * alpha,
                "holding_return": sign * target,
            }
            for day, target in zip(dates, targets, strict=True)
            for sign in (-1.0, 1.0)
        ]
    )
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=1),
            "available_at": dates + pd.Timedelta(days=2),
        }
    )
    with pytest.raises(
        ValueError, match=f"^alpha_calibration_{quantity}_unrepresentable$"
    ):
        calibrate_expected_alpha(
            frame, as_of=pd.Timestamp("2024-07-01"), scheduled_intervals=intervals
        )


@pytest.mark.parametrize("scale", [1.0, 1e308, 5e-324])
def test_compound_fundamental_ratios_round_only_the_final_result(scale) -> None:
    from quant_research_platform.research.factors.fundamental.definitions import (
        FUNDAMENTAL_INPUT_COLUMNS,
    )

    values = dict.fromkeys(FUNDAMENTAL_INPUT_COLUMNS, scale)
    values.update(
        operating_cost_ttm=-scale,
        operating_cash_flow_ttm=-scale,
        prior_year_parent_net_income_ttm=-scale,
    )
    frame = pd.DataFrame(
        [dict(asset_id="asset", timestamp=pd.Timestamp("2024-05-02"), **values)]
    )
    expected = dict(
        gross_profitability=2.0,
        gross_margin=2.0,
        accruals=2.0,
        earnings_change_to_assets=2.0,
        debt_to_assets=4.0,
        net_debt_to_assets=3.0,
    )
    actual = calculate_fundamental_factors(frame)
    for column, value in expected.items():
        assert actual[column].iloc[0] == value

    # Missing constituents, nonpositive denominators and final overflow stay undefined.
    frame["short_term_borrowings"] = float("nan")
    frame["average_total_assets"] = 0.0
    frame["operating_revenue_ttm"] = -scale
    actual = calculate_fundamental_factors(frame)
    assert actual[list(expected)].isna().all().all()
    frame["average_total_assets"] = 5e-324
    frame["consolidated_net_income_ttm"] = 1e308
    assert pd.isna(calculate_fundamental_factors(frame)["accruals"].iloc[0])


def test_compound_fundamental_ratio_preserves_cancellation() -> None:
    from quant_research_platform.research.factors.fundamental.definitions import (
        FUNDAMENTAL_INPUT_COLUMNS,
    )

    values = dict.fromkeys(FUNDAMENTAL_INPUT_COLUMNS, 1.0)
    values.update(
        short_term_borrowings=1e308,
        current_portion_noncurrent_liabilities=1.0,
        long_term_borrowings=-1e308,
        bonds_payable=0.0,
        cash=0.0,
    )
    frame = pd.DataFrame(
        [dict(asset_id="asset", timestamp=pd.Timestamp("2024-05-02"), **values)]
    )
    actual = calculate_fundamental_factors(frame)
    assert actual["debt_to_assets"].iloc[0] == 1.0
    assert actual["net_debt_to_assets"].iloc[0] == 1.0


def test_fundamental_subset_requires_only_its_formula_inputs():
    from quant_research_platform.research.factors.fundamental.calculations import (
        calculate_fundamental_factors,
    )

    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": [pd.Timestamp("2024-01-02")],
            "parent_equity": [20.0],
            "total_market_cap": [100.0],
        }
    )
    result = calculate_fundamental_factors(frame, factor_columns=("book_to_market",))
    assert result.book_to_market.iloc[0] == 0.2
    with pytest.raises(ValueError, match="fundamental_factor_input_columns_missing"):
        calculate_fundamental_factors(frame, factor_columns=("roe",))


def test_label_maturity_preserves_ineligible_observations_for_unfiltered_evaluation():
    from quant_research_platform.contracts.experiments import label_maturity_mask

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3),
            "label_available_at": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "eligible": [True, False, True],
            "label": [0.1, 0.2, 0.3],
        }
    )
    mature = label_maturity_mask(
        frame, fitting_cutoff="2024-01-02", target_column="label"
    )
    assert frame.loc[mature].index.tolist() == [0, 1]
    assert label_maturity_mask(
        frame.drop(columns="eligible"),
        fitting_cutoff="2024-01-02",
        target_column="label",
    ).equals(mature)
    assert legal_training_rows(
        frame, fitting_cutoff="2024-01-02", target_column="label"
    ).index.tolist() == [0]


def test_flag_only_model_rejects_structural_invalidity_but_keeps_ordinary_missing():
    from quant_research_platform.research.models.fitting import (
        _collect_training_frame,
        model_training_read_columns,
    )

    inputs = ("feature_missing_x", "feature_observed_x")
    columns = model_training_read_columns(
        ("timestamp", "asset_id"),
        inputs,
        label_column="label",
        sample_filter_mode="unfiltered",
    )
    assert "feature_x" in columns
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3),
            "asset_id": ["a", "b", "c"],
            "eligible": [True] * 3,
            "label_available_at": pd.to_datetime(["2024-01-02"] * 3),
            "label": [0.1, 0.2, 0.3],
            "feature_x": [2.0, 0.0, np.nan],
            "feature_missing_x": [0, 1, 1],
            "feature_observed_x": [1, 0, 0],
        }
    )
    prepared = _collect_training_frame(
        [frame],
        ModelSpec(
            model_id="flags",
            model_algorithm="linear_regression",
            input_feature_columns=inputs,
            label_column="label",
            sample_filter_mode="unfiltered",
        ),
        fitting_cutoff=pd.Timestamp("2024-01-02"),
    )
    training = prepared.training
    assert training.asset_id.tolist() == ["a", "b"]
