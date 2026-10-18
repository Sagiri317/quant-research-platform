"""Training RankIC keeps each factor's finite pair domain."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.rank_ensemble import (
    _training_rank_correlations,
)


@pytest.mark.parametrize("with_timestamp", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
def test_training_ranks_label_on_each_factors_pair_domain(with_timestamp, weighted):
    frame = pd.DataFrame(
        {
            "a": [5.0, np.nan, 3.0, 2.0, 1.0, 7.0],
            "b": [np.nan, 4.0, 3.0, 2.0, 1.0, 0.0],
            "label": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
        },
        index=[8, 2, 6, 4, 9, 1],
    )
    if with_timestamp:
        frame["timestamp"] = pd.Timestamp("2020-01-01")
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] if weighted else None
    assert dict(_training_rank_correlations(frame, ("a", "b"), "label", weights)) == {
        "a": -1.0,
        "b": -1.0,
    }


def test_rank_calibration_true_constant_scores_and_targets():
    from quant_research_platform.research.models.rank_ensemble import _label_calibration

    scores = np.full(40, 0.1)
    labels = np.linspace(-0.01, 0.03, 40)
    weights = np.linspace(1, 3, 40)
    intercept, slope = _label_calibration(scores, labels, weights)
    assert slope == 0.0
    assert intercept == pytest.approx(np.average(labels, weights=weights), abs=1e-16)
    assert _label_calibration(labels, scores, weights) == (0.1, 0.0)


def test_training_rank_correlations_keep_declared_period_weight_mass():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 4),
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, np.nan],
            "label": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    result = dict(_training_rank_correlations(frame, ("x",), "label", weights))
    assert result["x"] == pytest.approx((3 - 8) / 11, abs=1e-14)
