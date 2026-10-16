"""Subnormal period decay survives actual fit, native publication and reload."""

from fractions import Fraction
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from tests.research.direct_model_test_support import fit_frame


def _permutation(count, correlation):
    return next(
        values
        for values in permutations(range(1, count + 1))
        if 1
        - Fraction(
            6 * sum((i - v) ** 2 for i, v in enumerate(values, 1)),
            count * (count * count - 1),
        )
        == correlation
    )


@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize(
    "algorithm", ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"]
)
def test_sparse_decay_preserves_direction_weight_and_calibration(
    tmp_path, negative, ordinary, algorithm
):
    count, assets = (1082, 6) if negative else (1083, 4)
    if ordinary:
        count = 32
    factors = ("x",) if negative else ("x", "z")
    correlations = (Fraction(-1, 35),) if negative else (Fraction(1, 5), Fraction(4, 5))
    patterns = [_permutation(assets, c) for c in correlations]
    dates = pd.date_range("2020-01-01", periods=count)
    rows = [
        {
            "timestamp": stamp,
            "asset_id": str(asset),
            "eligible": True,
            "label": float(asset + 1) if day == 0 else 0.0,
            "label_available_at": stamp,
            **{
                column: float(patterns[j][asset]) if day == 0 else 1.0
                for j, column in enumerate(factors)
            },
        }
        for day, stamp in enumerate(dates)
        for asset in range(assets if day == 0 else 1)
    ]
    frame = pd.DataFrame(rows)
    spec = ModelSpec(
        model_id="rank",
        model_algorithm=algorithm,
        input_factor_columns=factors,
        target_column="label",
        random_seed=17,
        training_weighting=TrainingWeighting(half_life_periods=1),
    )
    weights = equal_timestamp_sample_weights(
        frame.timestamp, spec.training_weighting, trading_sessions=dates
    )
    assert (weights > 0).all() and np.isfinite(weights).all()
    if not ordinary:
        assert weights[:assets].max() <= np.nextafter(0.0, 1.0) * 4
    calendar = pd.DataFrame(
        dict(calendar_date=dates, source_market_type=1, is_trading_day=True)
    )
    case = fit_frame(
        tmp_path, frame.assign(market_type=1), spec, trading_calendar=calendar
    )
    predictor = case.result.prediction_model
    expected = np.array(
        [-1.0]
        if negative
        else [1.0, 1.0]
        if algorithm == "equal_weight_rank_ensemble"
        else [0.2, 0.8]
    )
    actual = (
        predictor.directions
        if algorithm == "equal_weight_rank_ensemble"
        else predictor.weights
    )
    np.testing.assert_array_equal(actual, expected)
    rank = np.column_stack(
        [
            (np.asarray(pattern, dtype=float) - (assets + 1) / 2) / assets
            for pattern in patterns
        ]
    )
    scores = (
        rank
        @ expected
        / (len(factors) if algorithm == "equal_weight_rank_ensemble" else 1)
    )
    np.testing.assert_array_equal(
        fitting.prediction_callable(factors, predictor)(frame.iloc[:assets]), scores
    )
    assert (
        predictor.label_calibration is not None
        and np.isfinite(predictor.label_calibration).all()
    )
    fitting.validate_model_state(predictor, case.result.metadata)
