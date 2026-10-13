"""Representable numerical boundaries retain their mathematical meaning."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.rank_ensemble import (
    _label_calibration,
    _training_rank_correlations,
)
from quant_research_platform.research.models.scaler import ModelMatrixScaler
from quant_research_platform.research.statistics import fama_macbeth_period


def test_scaler_preserves_tiny_weighted_population_scale():
    frame = pd.DataFrame({"x": [0.0, 1.0]})
    scaler = ModelMatrixScaler.fit(frame, sample_weight=[1e308, 1e-300])
    assert scaler.scale_[0] == pytest.approx(1e-304, rel=1e-14, abs=0)
    assert np.isfinite(scaler.transform(frame).to_numpy()).all()


def test_rank_direction_uses_mass_of_defined_cross_sections():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 3),
            "x": [1.0, 1.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 1.0, 3.0, 2.0],
        }
    )
    tiny = np.nextafter(0.0, 1.0)
    result = dict(
        _training_rank_correlations(frame, ("x",), "y", [1e308, 1.0, tiny, tiny, tiny])
    )
    assert result == {"x": 0.5}


def test_rank_direction_weights_may_have_unrepresentable_total_mass():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 3.0, 2.0],
        }
    )
    assert dict(_training_rank_correlations(frame, ("x",), "y", [1e308] * 3)) == {
        "x": 0.5
    }


@pytest.mark.parametrize("weights", (None, [1.0, 2.0, 3.0, 4.0]))
def test_rank_calibration_retains_subnormal_slope(weights):
    tiny = np.nextafter(0.0, 1.0)
    scores = np.array([-0.375, -0.125, 0.125, 0.375])
    labels = np.array([-3.0, -1.0, 1.0, 3.0]) * tiny
    intercept, slope = _label_calibration(scores, labels, weights)
    assert intercept == 0.0
    assert slope == 8 * tiny


def test_rank_calibration_retains_large_offset_slope():
    scores = np.array([-0.375, -0.125, 0.125, 0.375])
    labels = 1e308 + np.arange(4) * np.spacing(1e308)
    intercept, slope = _label_calibration(scores, labels, None)
    assert slope == 4 * np.spacing(1e308)
    assert intercept == float(sum(map(Fraction, labels)) / 4)


@pytest.mark.parametrize("weighted", (False, True))
def test_fama_macbeth_large_offset_exact_identity(weighted):
    values = 1e308 + np.arange(8) * np.spacing(1e308)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [str(i) for i in range(8)],
            "x": values,
            "y": values,
            "w": np.arange(1, 9),
        }
    )
    result = fama_macbeth_period(
        frame,
        return_column="y",
        factor_columns=("x",),
        weight_column="w" if weighted else None,
    )
    assert result.coefficients.to_dict() == {"const": 0.0, "x": 1.0}
    assert result.white_p_value is None
    assert result.white_reason == "white_test_undefined"


def test_fama_macbeth_white_is_invariant_to_large_target_scale():
    x = np.arange(20, dtype=float)
    y = 0.1 * x + np.sin(x) * (x + 1)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [str(i) for i in range(20)],
            "x": x,
            "y": y,
        }
    )
    expected = fama_macbeth_period(frame, return_column="y", factor_columns=("x",))
    frame["y"] *= 1e150
    actual = fama_macbeth_period(frame, return_column="y", factor_columns=("x",))
    assert actual.white_p_value == pytest.approx(expected.white_p_value, rel=1e-12)
    assert actual.coefficients["x"] / 1e150 == pytest.approx(expected.coefficients["x"])


def test_fama_macbeth_unrepresentable_slope_stays_explicit():
    tiny = np.nextafter(0.0, 1.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c", "d"],
            "x": np.arange(4) * tiny,
            "y": np.arange(4, dtype=float),
        }
    )
    assert Fraction(1) / Fraction(tiny) > Fraction(np.finfo(float).max)
    with pytest.raises(ValueError, match="fama_macbeth_coefficients_unrepresentable"):
        fama_macbeth_period(frame, return_column="y", factor_columns=("x",))


def test_linear_training_large_target_retains_representable_coefficients():
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.fitting import (
        _collect_training_frame,
        _fit_linear_model_result,
        validate_model_state,
    )

    stamps = pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"])
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "a"],
            "label_available_at": stamps,
            "eligible": True,
            "x": [-1.0, 0.0, 1.0],
            "y": [1.7e308] * 3,
        }
    )
    spec = ModelSpec(
        model_id="probe",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="y",
    )
    result = _fit_linear_model_result(
        _collect_training_frame((frame,), spec, fitting_cutoff="2020-01-03"), spec
    )
    validate_model_state(result.prediction_model, result.metadata)
    assert np.isfinite(result.prediction_model.coefficients).all()
    np.testing.assert_allclose(
        result.prediction_model.predict(frame[["x"]]) / 1.7e308, 1.0, rtol=1e-14
    )


def test_icir_divides_before_rounding_subnormal_mean_and_risk():
    from quant_research_platform.compute.dataframe.cross_section import (
        _summarize_correlation_aggregates,
    )

    tiny = Fraction(np.nextafter(0.0, 1.0))
    # The two observations tiny and 4*tiny have mean 2.5*tiny and std 1.5*tiny.
    result = _summarize_correlation_aggregates(
        count=2, mean=5 * tiny / 2, m2=9 * tiny * tiny / 2, positive_count=2
    )
    assert result["ratio"] == float(Fraction(5, 3))
    assert result["ratio_reason"] is None


def test_bootstrap_and_hac_do_not_overflow_a_representable_mean():
    from statistics import mean

    from quant_research_platform.research.statistics import (
        hac_mean,
        stationary_bootstrap_mean_ci,
    )

    values = np.tile([8e307, 9e307, 1e308, 1.1e308], 12)
    actual = stationary_bootstrap_mean_ci(values, block_size=4, repetitions=32, seed=17)
    scaled = stationary_bootstrap_mean_ci(
        values / 1e308, block_size=4, repetitions=32, seed=17
    )
    assert actual.mean == mean(values)
    assert actual.lower / 1e308 == pytest.approx(scaled.lower)
    assert actual.upper / 1e308 == pytest.approx(scaled.upper)
    hac = hac_mean(values, max_lags=2)
    assert hac.mean / 1e308 == pytest.approx(0.95)
    assert np.isfinite(hac.standard_error)


def test_rank_training_keeps_nullable_raw_domains_and_first_undefined_factor():
    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
        ICWeightedRankEnsembleFactory,
    )

    frame = pd.DataFrame(
        {
            "x0": pd.array([1, None, 2, 3], dtype="Int64"),
            "x1": pd.array([None, 2, 3, 4], dtype="Float64"),
            "timestamp": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2),
            "label": [1.0, 2.0, 3.0, 4.0],
        }
    ).iloc[[2, 0, 3, 1]]
    for factory in (EqualWeightRankEnsembleFactory, ICWeightedRankEnsembleFactory):
        model = factory(("x0", "x1"), "label").fit(frame, frame.label.to_numpy())
        np.testing.assert_array_equal(model.predict(frame), [-0.25, 0.0, 0.25, 0.0])
        assert _label_calibration(model.predict(frame), frame.label, None) == (2.5, 2.0)
    invalid = frame.assign(x0=np.nan, x1="invalid")
    with pytest.raises(
        ValueError, match="^ensemble_direction_undefined:x0:no_valid_training_rankic$"
    ):
        EqualWeightRankEnsembleFactory(("x0", "x1"), "label").fit(
            invalid, invalid.label.to_numpy()
        )


@pytest.mark.parametrize("direction", (-1, 1))
def test_rank_fit_preserves_underflowed_correlation_and_reload(direction):
    import pickle

    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
        ICWeightedRankEnsemble,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-01"] * 2 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3
            ),
            "x": [1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0],
        }
    )
    frame.loc[:1, "y"] *= direction
    frame["opposite"] = -frame["x"]
    tiny = np.nextafter(0.0, 1.0)
    weights = [tiny, tiny, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    correlation = dict(_training_rank_correlations(frame, ("x",), "y", weights))["x"]
    assert correlation * direction > 0
    assert float(correlation) == 0
    ew = EqualWeightRankEnsembleFactory(("x",), "y").fit(
        frame, frame["y"].to_numpy(), sample_weight=weights
    )
    assert ew.directions == (float(direction),)
    for columns, expected in (
        (("x",), (float(direction),)),
        (("x", "opposite"), (0.5 * direction, -0.5 * direction)),
    ):
        model = ICWeightedRankEnsemble.fit(
            frame, input_columns=columns, label_column="y", sample_weight=weights
        )
        assert model.weights == expected
        np.testing.assert_array_equal(
            pickle.loads(pickle.dumps(model)).predict(frame), model.predict(frame)
        )


def test_exact_zero_rank_correlation_stays_zero():
    from quant_research_platform.research.models.rank_ensemble import (
        ICWeightedRankEnsemble,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2),
            "x": [1.0, 2.0, 1.0, 2.0],
            "y": [1.0, 2.0, 2.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="all_ensemble_correlations_zero"):
        ICWeightedRankEnsemble.fit(frame, input_columns=("x",), label_column="y")


@pytest.mark.parametrize("coefficient", (0.0, 1e-308))
def test_linear_prediction_keeps_finite_affine_result_after_scaler_overflow(
    coefficient,
):
    import pickle

    from quant_research_platform.research.models.scaler import LinearScaledPredictor

    scaler = ModelMatrixScaler.fit(pd.DataFrame({"x": [1e-308, 2e-308]}))
    model = pickle.loads(
        pickle.dumps(LinearScaledPredictor((0.0, coefficient), scaler))
    )
    value = 10.0
    expected = float(
        Fraction(coefficient)
        * (Fraction(value) - Fraction(scaler.origin_[0]) - scaler.offset_[0])
        / Fraction(scaler.scale_[0])
    )
    assert model.predict(pd.DataFrame({"x": [value]}))[0] == expected


def test_joint_loss_tests_preserve_legal_internal_looking_model_name():
    from quant_research_platform.research.statistics import (
        spa_reality_check,
        stepm_superior_models,
    )

    axis = pd.date_range("2020-01-01", periods=32)
    benchmark = pd.Series(1.0 + np.sin(np.arange(32)), index=axis)
    models = pd.DataFrame({"challenger": np.cos(np.arange(32))}, index=axis)
    renamed = models.rename(columns={"challenger": "__benchmark_loss__"})
    options = dict(block_size=3, repetitions=99, seed=17)
    assert spa_reality_check(benchmark, models, **options) == spa_reality_check(
        benchmark, renamed, **options
    )
    expected = stepm_superior_models(benchmark, models, **options)
    actual = stepm_superior_models(benchmark, renamed, **options)
    assert actual.superior_models == tuple(
        "__benchmark_loss__" for _ in expected.superior_models
    )
    assert (actual.observation_count, actual.status, actual.reason) == (
        expected.observation_count,
        expected.status,
        expected.reason,
    )


@pytest.mark.parametrize("mode", ("ordinary", "tiny", "large", "constant"))
def test_rank_label_calibration_matches_full_fraction_accumulation(mode):
    rng = np.random.default_rng(193)
    scores = rng.normal(size=37)
    labels = rng.normal(size=37)
    weights = rng.uniform(0.01, 2, size=37)
    if mode == "tiny":
        weights[:18] = np.nextafter(0.0, 1.0)
        labels *= np.nextafter(0.0, 1.0)
    elif mode == "large":
        labels = 1e308 + np.arange(37) * np.spacing(1e308)
        weights *= 1e307
    elif mode == "constant":
        scores[:] = 0.125
    triples = [
        (Fraction(x), Fraction(y), Fraction(w))
        for x, y, w in zip(scores, labels, weights, strict=True)
    ]
    mass = sum(w for x, y, w in triples)
    xbar = sum(x * w for x, y, w in triples) / mass
    ybar = sum(y * w for x, y, w in triples) / mass
    variance = sum(w * (x - xbar) ** 2 for x, y, w in triples)
    slope = (
        sum(w * (x - xbar) * (y - ybar) for x, y, w in triples) / variance
        if variance
        else Fraction()
    )
    assert _label_calibration(scores, labels, weights) == (
        float(ybar - slope * xbar),
        float(slope),
    )


@pytest.mark.parametrize("identity", (False, True))
@pytest.mark.parametrize("scale", (1e-308, 1.0, 1e308))
def test_native_linear_fit_and_reload_preserves_finite_extrapolation(
    tmp_path, identity, scale
):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_prediction_output_bundle,
    )
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.fitting import prediction_callable
    from tests.research.direct_model_test_support import fit_frame, predict_frame

    stamps = pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2)
    frame = pd.DataFrame(
        dict(
            timestamp=stamps,
            asset_id=["a", "b", "a", "b"],
            eligible=True,
            label_available_at=stamps,
            x=[-scale, scale] * 2,
            y=[-scale, scale] * 2 if identity else [0.0] * 4,
        )
    )
    spec = ModelSpec(
        model_id="probe",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="y",
    )
    case = fit_frame(tmp_path / "fit", frame, spec, cutoff=pd.Timestamp("2020-01-03"))
    assert (case.root / "predictor.pkl").is_file()
    predictor = case.result.prediction_model
    inputs = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-04"),
            "asset_id": ["a", "b", "c"],
            "eligible": True,
            "x": [10.0, 1e-100, 0.0],
        }
    )
    expected = np.array(
        [
            float(
                Fraction(predictor.coefficients[0])
                + Fraction(predictor.coefficients[1])
                * (
                    Fraction(value)
                    - Fraction(predictor.scaler.origin_[0])
                    - predictor.scaler.offset_[0]
                )
                / Fraction(predictor.scaler.scale_[0])
            )
            for value in inputs.x
        ]
    )
    scores = prediction_callable(("x",), predictor)(inputs[["x"]])
    np.testing.assert_allclose(scores, expected, rtol=1e-14, atol=0)
    prediction, root, _metadata = predict_frame(
        tmp_path / "future",
        inputs,
        case,
        window={"start": "2020-01-04", "end": "2020-01-04"},
    )
    np.testing.assert_array_equal(prediction.prediction_score, scores)
    _, lookup = read_prediction_output_bundle(root)
    values, found = lookup.values_for_frame(
        inputs[["timestamp", "asset_id"]], mark_matched=True
    )
    assert all(found)
    np.testing.assert_array_equal(values, scores)


@pytest.mark.parametrize("value", (1e-100, 1.0, 10.0, -1e-100))
def test_linear_affine_underflow_preserves_representable_result(value):
    import pickle

    from quant_research_platform.research.models.scaler import LinearScaledPredictor

    scaler = ModelMatrixScaler.fit(pd.DataFrame({"x": [-1e308, 1e308] * 2}))
    predictor = pickle.loads(pickle.dumps(LinearScaledPredictor((0.0, 1e308), scaler)))
    frame = pd.DataFrame({"x": [value]})
    expected = float(Fraction(1e308) * Fraction(value) / Fraction(scaler.scale_[0]))
    assert predictor.predict(frame)[0] == expected == value


def test_fama_macbeth_white_scales_residuals_independently_of_small_target():
    from scipy.stats import chi2
    from statsmodels.tools.sm_exceptions import SingularMatrixWarning

    from quant_research_platform.research.statistics_production import (
        _fama_macbeth_period_records,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": list("abcde"),
            "x": [0.0, 1e-110, 2e-110, 3e-110, 1e200],
            "y": [0.0, 1e-110, 2e-110, 3e-110, 0.0],
            "total_market_cap": [1e308] * 4 + [np.nextafter(0.0, 1.0)],
        }
    )
    x, y, w = (
        [Fraction(float(v)) for v in frame[column]]
        for column in ("x", "y", "total_market_cap")
    )
    total = sum(w)
    mx = sum(a * b for a, b in zip(x, w, strict=True)) / total
    my = sum(a * b for a, b in zip(y, w, strict=True)) / total
    slope = sum(c * (a - mx) * (b - my) for a, b, c in zip(x, y, w, strict=True)) / sum(
        b * (a - mx) ** 2 for a, b in zip(x, w, strict=True)
    )
    # Redundant auxiliary columns leave White's fitted projection and LM unique.
    with pytest.warns(SingularMatrixWarning, match="rank-deficient"):
        actual = fama_macbeth_period(
            frame,
            return_column="y",
            factor_columns=("x",),
            weight_column="total_market_cap",
        )
    assert actual.coefficients.to_dict() == {
        "const": float(my - slope * mx),
        "x": float(slope),
    }
    # Binary64 White regressors identify the single dominant residual exactly.
    assert actual.white_p_value == pytest.approx(chi2.sf(5, 1), rel=1e-12)
    assert actual.white_reason is None
    with pytest.warns(SingularMatrixWarning, match="rank-deficient"):
        records = _fama_macbeth_period_records(
            frame, return_column="y", factors=("x",), core_factors=()
        )
    assert records and all(row["status"] == "DEFINED" for row in records)
