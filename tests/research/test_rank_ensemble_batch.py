from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.validation import strict_finite_numeric_series
from quant_research_platform.research.models import rank_ensemble as rank


def _legacy_ranks(frame, columns):
    result = pd.DataFrame(0.0, index=frame.index, columns=columns)
    groups = (
        frame.groupby("timestamp", sort=False)
        if "timestamp" in frame
        else [(None, frame)]
    )
    for _, group in groups:
        for column in columns:
            values = strict_finite_numeric_series(
                group[column],
                error=f"invalid_rank_ensemble_column: {column}",
                allow_null=True,
            )
            valid = values.notna()
            count = int(valid.sum())
            if count:
                ranks = values.loc[valid].rank(method="average")
                result.loc[ranks.index, column] = (ranks - (count + 1.0) / 2.0) / count
    return result.to_numpy(dtype="float64", copy=False)


@pytest.mark.parametrize("timestamp", (True, False))
@pytest.mark.parametrize("nullable", (True, False))
@pytest.mark.parametrize("missing", ("mixed", "all", "none"))
def test_batched_rank_preserves_rank_and_prediction_values(
    timestamp, nullable, missing
):
    rng = np.random.default_rng(71)
    columns = ("a", "b", "c", "d")
    frame = pd.DataFrame(
        rng.integers(-3, 4, size=(120, 4)).astype(float), columns=columns
    )
    frame["timestamp"] = np.repeat(pd.date_range("2020-01-01", periods=20), 6)
    frame["label"] = rng.normal(size=len(frame))
    predictor = rank.ICWeightedRankEnsemble.fit(
        frame,
        input_columns=columns,
        label_column="label",
    )
    assert len(set(predictor.weights)) > 1
    if missing == "mixed":
        frame.loc[::3, "a"] = np.nan
        frame.loc[::7, "b"] = np.nan
        frame.loc[30:35, "c"] = np.nan
    elif missing == "all":
        frame.loc[:, list(columns)] = np.nan
    if missing != "all":
        frame.loc[0, "a"] = np.finfo(float).max
        frame.loc[1, "a"] = -np.finfo(float).max
    if nullable:
        frame = frame.astype(dict.fromkeys(columns, "Float64"))
    frame = frame.sample(frac=1, random_state=9)
    if not timestamp:
        frame = frame.drop(columns="timestamp")
    expected = _legacy_ranks(frame, columns)
    actual = rank._rank_frame_columns(frame, columns)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=np.finfo(float).eps)
    np.testing.assert_allclose(
        predictor.predict(frame),
        expected @ np.asarray(predictor.weights, dtype="float64"),
        rtol=8 * np.finfo(float).eps,
        atol=8 * np.finfo(float).eps,
    )
    equal = rank.RankEnsemblePredictor(columns, (1.0, -1.0, 1.0, -1.0))
    np.testing.assert_allclose(
        equal.predict(frame),
        expected @ np.asarray(equal.directions) / len(columns),
        rtol=8 * np.finfo(float).eps,
        atol=8 * np.finfo(float).eps,
    )


@pytest.mark.parametrize("invalid", ("bad", np.inf, -np.inf))
def test_batched_rank_rejects_invalid_values(invalid):
    frame = pd.DataFrame({"a": [1.0, invalid]})
    with pytest.raises(ValueError, match="invalid_rank_ensemble_column"):
        rank.RankEnsemblePredictor(("a",)).predict(frame)
