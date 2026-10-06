import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.scaler import ModelMatrixScaler
from quant_research_platform.research.portfolios.risk import preprocess_risk_styles
from quant_research_platform.research.statistics import hac_mean


@pytest.mark.parametrize("weights", ([1.0, 1.0], [1.0, 3.0]))
def test_scaler_preserves_split_center_across_chunks_and_pickle(weights):
    frame = pd.DataFrame({"x": [1.0, np.nextafter(1.0, 2.0)]})
    exact = [Fraction.from_float(value) for value in frame.x]
    exact_weights = [Fraction.from_float(weight) for weight in weights]
    center = sum(
        value * weight for value, weight in zip(exact, exact_weights, strict=True)
    ) / sum(exact_weights)
    variance = sum(
        (value - center) ** 2 * weight
        for value, weight in zip(exact, exact_weights, strict=True)
    ) / sum(exact_weights)
    with localcontext() as context:
        context.prec = 100
        scale = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        expected = [
            float(
                Decimal((value - center).numerator)
                / Decimal((value - center).denominator)
                / scale
            )
            for value in exact
        ]
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    actual = scaler.transform(frame).x.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)
    assert np.dot(actual, weights) == pytest.approx(0, abs=1e-15)
    restored = pickle.loads(pickle.dumps(scaler))
    chunks = pd.concat(
        [restored.transform(frame.iloc[:1]), restored.transform(frame.iloc[1:])]
    )
    pd.testing.assert_frame_equal(chunks, scaler.transform(frame))
    assert restored.to_payload() == scaler.to_payload()
    assert set(restored.to_payload()) == {
        "columns",
        "weighted_mean_origin",
        "weighted_mean_offset",
        "weighted_population_scale",
    }


def test_adjacent_risk_styles_remain_zero_centered():
    values = np.tile([1.0, np.nextafter(1.0, 2.0)], 50)
    frame = pd.DataFrame(
        {"timestamp": pd.Timestamp("2020-01-01"), "eligible": True, "style": values}
    )
    result = preprocess_risk_styles(frame, style_columns=("style",))
    np.testing.assert_array_equal(result["style"].to_numpy(), np.tile([-1.0, 1.0], 50))
    assert result["style"].mean() == 0


def test_constant_eligible_risk_style_projects_zero_to_complete_benchmark_domain():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["fit_a", "fit_b", "benchmark_only"],
            "eligible": [True, True, False],
            "style": [1.0, 1.0, 2.0],
        }
    )
    result = preprocess_risk_styles(frame, style_columns=("style",))
    pd.testing.assert_frame_equal(
        result.drop(columns="style"), frame.drop(columns="style")
    )
    pd.testing.assert_series_equal(result["style"], pd.Series([0.0] * 3, name="style"))


@pytest.mark.parametrize("unit", (1.0, 1e200, 1e-200, -1e200))
def test_hac_representable_extreme_units_match_fraction_bartlett_oracle(unit):
    values = np.array([(0.1 + 0.01 * i) * unit for i in range(24)])
    exact = [Fraction.from_float(float(value)) for value in values]
    count = len(exact)
    center = sum(exact) / count
    residuals = [value - center for value in exact]
    meat = sum(value * value for value in residuals)
    for lag in range(1, 20):
        meat += (
            2
            * Fraction(20 - lag, 20)
            * sum(residuals[i] * residuals[i - lag] for i in range(lag, count))
        )
    variance = meat / count**2
    with localcontext() as context:
        context.prec = 100
        standard_error = (
            Decimal(variance.numerator) / Decimal(variance.denominator)
        ).sqrt()
        mean = Decimal(center.numerator) / Decimal(center.denominator)
        t_statistic = mean / standard_error
    actual = hac_mean(values, max_lags=19)
    assert actual.mean == pytest.approx(float(mean), rel=1e-14, abs=0)
    assert actual.standard_error == pytest.approx(
        float(standard_error), rel=1e-14, abs=0
    )
    assert actual.t_statistic == pytest.approx(float(t_statistic), rel=1e-14, abs=0)
    assert np.isfinite(actual.confidence_interval).all()
