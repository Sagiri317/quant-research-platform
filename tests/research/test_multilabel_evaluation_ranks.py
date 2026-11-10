"""Paired ranks and prediction groups keep distinct sample domains."""

import math

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from tests.research.evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _ranks(values):
    # Count strict predecessors and half the tie class, without sorting or float
    # conversion. This also protects nullable integers above float64 precision.
    return [
        1
        + sum(other < value for other in values)
        + (sum(other == value for other in values) - 1) / 2
        for value in values
    ]


def _correlation(left, right):
    x = [value - sum(left) / len(left) for value in left]
    y = [value - sum(right) / len(right) for value in right]
    return sum(a * b for a, b in zip(x, y, strict=True)) / math.sqrt(
        sum(a * a for a in x) * sum(b * b for b in y)
    )


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
def test_native_multilabel_ranks_preserve_pairwise_ties_and_prediction_groups(
    tmp_path, direction
):
    offset = 2**53
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("abcdefg"),
            "x": pd.Series(
                [
                    offset,
                    offset + 1,
                    offset + 1,
                    offset + 3,
                    offset + 4,
                    None,
                    offset + 6,
                ],
                dtype="Int64",
            ),
            "z": [3.0, np.nan, 2.0, 2.0, 5.0, 7.0, 1.0],
            "y1": [2.0, 2.0, 1.0, np.nan, 4.0, 3.0, 0.0],
            "y2": [np.nan, 4.0, 1.0, 2.0, np.nan, 5.0, 3.0],
            "available_at": pd.Timestamp("2024-01-02"),
            "eligible": True,
        }
    )
    signals = tuple(
        SignalSpec(
            signal_id=f"s_{name}",
            source_kind="factor",
            source_column=name,
            direction=direction,
        )
        for name in ("x", "z")
    )
    spec = EvaluationSpec(
        evaluation_id="paired_ranks",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=signals,
        label_columns=("y1", "y2"),
        label_availability_columns={"y1": "available_at", "y2": "available_at"},
        n_groups=3,
    )
    inputs = write_evaluation_input(
        tmp_path / "inputs", frame.iloc[::-1], factors=("x", "z"), labels=("y1", "y2")
    )
    output = evaluate(
        evaluation_config(
            inputs, tmp_path / "evaluation", spec, start="2024-01-02", end="2024-01-03"
        )
    )
    _, tables = read_evaluation_outputs(output)
    metrics = tables["metrics"].set_index(["signal", "label"])
    grouped = tables["grouped_label_values"].set_index(["signal", "label"]).sort_index()
    sign = 1 if direction == "higher_is_better" else -1
    for signal in signals:
        predicted = frame[signal.source_column].notna()
        prediction_ranks = _ranks(frame.loc[predicted, signal.source_column].tolist())
        if sign < 0:
            prediction_ranks = [
                len(prediction_ranks) + 1 - rank for rank in prediction_ranks
            ]
        groups = pd.Series(
            [
                math.floor((rank - 1) * 3 / len(prediction_ranks)) + 1
                for rank in prediction_ranks
            ],
            index=frame.index[predicted],
        )
        for label in spec.label_columns:
            paired = predicted & frame[label].notna()
            expected = sign * _correlation(
                _ranks(frame.loc[paired, signal.source_column].tolist()),
                _ranks(frame.loc[paired, label].tolist()),
            )
            metric = metrics.loc[(signal.signal_id, label)]
            assert metric.n_obs == paired.sum()
            assert metric.rankic == pytest.approx(expected, abs=1e-14)
            observed = grouped.loc[(signal.signal_id, label)]
            assert observed.group.tolist() == sorted(groups.unique())
            for row in observed.itertuples():
                labels = frame.loc[groups.index[groups.eq(row.group)], label].dropna()
                assert row.n_obs == len(labels)
                assert (
                    row.mean_label_value
                    == pytest.approx(math.fsum(labels) / len(labels), abs=1e-14)
                    if len(labels)
                    else pd.isna(row.mean_label_value)
                )
