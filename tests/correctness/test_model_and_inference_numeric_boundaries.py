"""Ordinary-scale inference and native model contracts."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.rank_ensemble import (
    _label_calibration,
    _training_rank_correlations,
)
from quant_research_platform.research.statistics import fama_macbeth_period


def test_rank_direction_uses_mass_of_defined_cross_sections():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 3),
            "x": [1.0, 1.0, 1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 1.0, 3.0, 2.0],
        }
    )
    result = dict(
        _training_rank_correlations(frame, ("x",), "y", [2.0, 1.0, 0.5, 0.5, 0.5])
    )
    assert result == {"x": 0.5}


@pytest.mark.parametrize("weighted", (False, True))
def test_fama_macbeth_identity_has_undefined_white_variance(weighted):
    values = 0.01 + np.arange(8) * 0.002
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
    np.testing.assert_allclose(result.coefficients, [0.0, 1.0], rtol=1e-13, atol=1e-15)
    assert result.white_p_value is None
    assert result.white_reason == "white_test_undefined"


def test_fama_macbeth_white_is_invariant_to_target_units():
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
    frame["y"] *= 100
    actual = fama_macbeth_period(frame, return_column="y", factor_columns=("x",))
    assert actual.white_p_value == pytest.approx(expected.white_p_value, rel=1e-12)
    assert actual.coefficients["x"] / 100 == pytest.approx(expected.coefficients["x"])


def test_bootstrap_mean_preserves_ordinary_unit_scaling():
    from quant_research_platform.research.statistics import stationary_bootstrap_mean_ci

    values = np.tile([0.08, 0.09, 0.10, 0.11], 12)
    actual = stationary_bootstrap_mean_ci(values, block_size=4, repetitions=32, seed=17)
    scaled = stationary_bootstrap_mean_ci(
        values * 100, block_size=4, repetitions=32, seed=17
    )
    assert actual.mean == pytest.approx(0.095, rel=1e-12, abs=1e-14)
    assert actual.lower * 100 == pytest.approx(scaled.lower, rel=1e-12, abs=1e-14)
    assert actual.upper * 100 == pytest.approx(scaled.upper, rel=1e-12, abs=1e-14)


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


@pytest.mark.parametrize("mode", ("ordinary", "constant"))
def test_rank_label_calibration_matches_full_fraction_accumulation(mode):
    rng = np.random.default_rng(193)
    scores = rng.normal(size=37)
    labels = rng.normal(size=37)
    weights = rng.uniform(0.01, 2, size=37)
    if mode == "constant":
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
    np.testing.assert_allclose(
        _label_calibration(scores, labels, weights),
        (float(ybar - slope * xbar), float(slope)),
        rtol=2e-14,
        atol=0,
    )


@pytest.mark.parametrize("identity", (False, True))
@pytest.mark.parametrize("scale", (0.01, 1.0, 100.0))
def test_native_linear_fit_and_reload_preserves_finite_extrapolation(
    tmp_path, identity, scale
):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
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
            "x": [10.0, 0.001, 0.0],
        }
    )
    expected = np.array(
        [
            float(
                Fraction(predictor.coefficients[0])
                + Fraction(predictor.coefficients[1])
                * (Fraction(value) - Fraction(predictor.scaler.mean_[0]))
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
    np.testing.assert_allclose(prediction.prediction_score, scores, rtol=1e-14, atol=0)
    persisted = read_predictions(root)[0]
    actual = inputs[["timestamp", "asset_id"]].merge(
        persisted, on=["timestamp", "asset_id"], validate="one_to_one"
    )
    np.testing.assert_array_equal(actual.prediction_score, prediction.prediction_score)
