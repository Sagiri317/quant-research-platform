import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.scaler import ModelMatrixScaler


@pytest.mark.parametrize("magnitude", [1.0, 1e308])
@pytest.mark.parametrize("reverse", [False, True])
def test_tiny_weight_preserves_representable_center_and_reloaded_coordinates(
    reverse, magnitude
):
    x = [magnitude, 0.0]
    weights = [np.nextafter(0.0, 1.0), 2.0]
    if reverse:
        x.reverse()
        weights.reverse()
    frame = pd.DataFrame({"x": x})
    mass = sum(map(Fraction, weights))
    mean = (
        sum(Fraction(w) * Fraction(v) for w, v in zip(weights, x, strict=True)) / mass
    )
    variance = (
        sum(
            Fraction(w) * (Fraction(v) - mean) ** 2
            for w, v in zip(weights, x, strict=True)
        )
        / mass
    )
    with localcontext() as ctx:
        ctx.prec = 100
        scale = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        expected = np.array(
            [
                float(
                    (
                        Decimal.from_float(v)
                        - Decimal(mean.numerator) / Decimal(mean.denominator)
                    )
                    / scale
                )
                for v in x
            ]
        )
    fitted = ModelMatrixScaler.fit(frame, sample_weight=weights)
    assert fitted.mean_[0] == float(mean)
    assert fitted.scale_[0] == pytest.approx(float(scale), rel=2e-15)
    restored = pickle.loads(pickle.dumps(fitted))
    full = restored.transform(frame)
    pieces = pd.concat(
        [restored.transform(frame.iloc[:1]), restored.transform(frame.iloc[1:])]
    )
    pd.testing.assert_frame_equal(full, pieces, check_exact=True)
    np.testing.assert_allclose(full.x, expected, rtol=2e-15, atol=0)
    assert full.x.iloc[x.index(0.0)] != 0
    payload = restored.to_payload()
    assert set(payload) == {
        "columns",
        "weighted_mean_origin",
        "weighted_mean_offset",
        "weighted_population_scale",
    }
    assert Fraction(restored.origin_[0]) + restored.offset_[0] == mean
    assert Fraction(*payload["weighted_mean_offset"][0]) == restored.offset_[0]


def test_center_survives_cancellation_of_large_anchor():
    frame = pd.DataFrame({"x": [1e200, 1.0, -1e200]})
    fitted = ModelMatrixScaler.fit(frame)
    assert fitted.mean_[0] == float(Fraction(1, 3))
    assert fitted.transform(pd.DataFrame({"x": [0.0]})).x.iloc[0] < 0


def test_center_with_native_equal_timestamp_extreme_weights():
    from quant_research_platform.contracts.models import TrainingWeighting
    from quant_research_platform.research.models.scaler import (
        equal_timestamp_sample_weights,
    )

    weights = equal_timestamp_sample_weights(
        pd.date_range("2020-01-01", periods=2),
        TrainingWeighting(half_life_periods=1 / 1075),
        trading_sessions=pd.date_range("2020-01-01", periods=2),
    )
    np.testing.assert_array_equal(weights, [np.nextafter(0.0, 1.0), 2.0])
    fitted = ModelMatrixScaler.fit(
        pd.DataFrame({"x": [1e308, 0.0]}), sample_weight=weights
    )
    assert fitted.mean_[0] == 2.4703282292062327e-16
    assert fitted.transform(pd.DataFrame({"x": [0.0]})).x.iloc[0] != 0


def test_unrepresentable_scores_keep_existing_trial_consumer_rejection():
    from quant_research_platform.research.models.scaler import LinearScaledPredictor
    from quant_research_platform.research.models.trial_selection import (
        evaluate_model_trial_metrics_from_chunks,
    )

    frame = pd.DataFrame({"x": [1e308, -1e308]})
    for values, weights in [
        ([1.0, np.nextafter(1.0, 2.0)], None),
        ([1.0, 0.0], [np.nextafter(0.0, 1.0), 2.0]),
    ]:
        scaler = ModelMatrixScaler.fit(
            pd.DataFrame({"x": values}), sample_weight=weights
        )
        predictor = LinearScaledPredictor((0.0, 1.0), scaler)
        scores = predictor.predict(frame)
        np.testing.assert_array_equal(scores, [np.inf, -np.inf])
        validation = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01")] * 2,
                "asset_id": ["a", "b"],
                "label": [1.0, 2.0],
                "prediction": scores,
            }
        )
        with pytest.raises(ValueError, match="^invalid_trial_metric_values$"):
            evaluate_model_trial_metrics_from_chunks(
                [validation],
                model_id="linear",
                label_column="label",
                prediction_column="prediction",
            )
