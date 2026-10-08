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
        label_column="y",
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
        assert model.label_calibration == (2.5, 2.0)
        np.testing.assert_array_equal(model.predict(frame), [-0.25, 0.0, 0.25, 0.0])
    invalid = frame.assign(x0=np.nan, x1="invalid")
    with pytest.raises(
        ValueError, match="^ensemble_direction_undefined:x0:no_valid_training_rankic$"
    ):
        EqualWeightRankEnsembleFactory(("x0", "x1"), "label").fit(
            invalid, invalid.label.to_numpy()
        )
