"""Weighted population moments and frozen transformations at research scales."""

import pickle
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.scaler import ModelMatrixScaler


@pytest.mark.parametrize("weighted", [False, True])
def test_weighted_population_moments_and_frozen_transform(weighted):
    frame = pd.DataFrame(
        [[-3.0, 0.2, 8.0], [1.0, -0.7, 2.0], [2.0, 1.1, -1.0], [0.0, 0.4, 3.0]],
        columns=list("xyz"),
    )
    weights = [0.5, 1.0, 2.0, 3.0] if weighted else None
    exact_weights = [Fraction(value) for value in weights or [1] * len(frame)]
    means, scales, expected = [], [], {}
    with localcontext() as context:
        context.prec = 100
        for column in frame:
            values = [Fraction.from_float(value) for value in frame[column]]
            mean = sum(
                value * weight
                for value, weight in zip(values, exact_weights, strict=True)
            ) / sum(exact_weights)
            variance = sum(
                (value - mean) ** 2 * weight
                for value, weight in zip(values, exact_weights, strict=True)
            ) / sum(exact_weights)
            scale = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
            means.append(float(mean))
            scales.append(float(scale))
            expected[column] = [
                float(
                    Decimal((value - mean).numerator)
                    / Decimal((value - mean).denominator)
                    / scale
                )
                for value in values
            ]
    fitted = ModelMatrixScaler.fit(frame, sample_weight=weights)
    np.testing.assert_allclose(fitted.mean_, means, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(fitted.scale_, scales, rtol=1e-15, atol=1e-15)
    full = fitted.transform(frame)
    pd.testing.assert_frame_equal(
        full, pd.DataFrame(expected), check_exact=False, rtol=1e-15, atol=1e-15
    )
    np.testing.assert_allclose(
        np.average(full, axis=0, weights=weights), 0, rtol=0, atol=1e-15
    )
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.to_payload() == fitted.to_payload()
    assert restored.to_payload()["scaling"] is True
    assert set(restored.to_payload()) == {
        "columns",
        "scaling",
        "weighted_mean",
        "weighted_population_scale",
    }
    pieces = pd.concat(
        [restored.transform(frame.iloc[:1]), restored.transform(frame.iloc[1:])]
    )
    pd.testing.assert_frame_equal(full, pieces, check_exact=True)


def test_true_constant_keeps_unit_scale_and_future_variation():
    frame = pd.DataFrame({"x": [2.3] * 4})
    fitted = ModelMatrixScaler.fit(
        frame, sample_weight=np.linspace(0.5, 3.0, len(frame))
    )
    assert fitted.mean_ == (2.3,)
    assert fitted.scale_ == (1.0,)
    assert fitted.transform(frame).x.eq(0).all()
    np.testing.assert_allclose(
        fitted.transform(pd.DataFrame({"x": [2.0, 3.0]})).x, [-0.3, 0.7]
    )
