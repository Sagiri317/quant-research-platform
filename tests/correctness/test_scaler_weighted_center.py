"""Weighted population moments and frozen transformations at research scales."""

import pickle

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.scaler import ModelMatrixScaler


@pytest.mark.parametrize("weighted", [False, True])
def test_centered_columns_match_weighted_moments_and_reload(weighted):
    frame = pd.DataFrame(
        [[-3.0, 0.2, 8.0], [1.0, -0.7, 2.0], [2.0, 1.1, -1.0], [0.0, 0.4, 3.0]],
        columns=list("xyz"),
    )
    weights = np.array([0.5, 1.0, 2.0, 3.0]) if weighted else None
    mean = np.average(frame, axis=0, weights=weights)
    scale = np.sqrt(np.average((frame - mean) ** 2, axis=0, weights=weights))
    fitted = ModelMatrixScaler.fit(frame, sample_weight=weights)
    np.testing.assert_allclose(fitted.mean_, mean, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(fitted.scale_, scale, rtol=1e-13, atol=1e-14)
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.to_payload() == fitted.to_payload()
    assert set(restored.to_payload()) == {
        "columns",
        "scaling",
        "weighted_mean",
        "weighted_population_scale",
    }
    full = restored.transform(frame)
    pieces = pd.concat(
        [restored.transform(frame.iloc[:1]), restored.transform(frame.iloc[1:])]
    )
    pd.testing.assert_frame_equal(full.isna(), pieces.isna(), check_exact=True)
    pd.testing.assert_frame_equal(
        full, pieces, check_exact=False, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(full, (frame - mean) / scale, rtol=1e-13, atol=1e-14)


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
