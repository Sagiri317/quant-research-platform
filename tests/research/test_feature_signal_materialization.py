from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.signals import (
    SignalSpec,
    _cross_signal_id_column_overlaps,
)
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)


def test_signal_overlap_sets_preserve_unique_contract_and_error_order():
    names = ("a", "b", "c", "prediction_score")
    count = 0
    for length in range(1, len(names) + 1):
        for ids in permutations(names, length):
            for columns in permutations(names, length):
                signals = tuple(
                    SignalSpec(
                        signal_id=signal_id,
                        source_kind="prediction",
                        source_column=column,
                    )
                    for signal_id, column in zip(ids, columns, strict=True)
                )
                expected = sorted(
                    {
                        signal_id
                        for position, signal_id in enumerate(ids)
                        for other, column in enumerate(columns)
                        if position != other and signal_id == column
                    }
                )
                assert _cross_signal_id_column_overlaps(signals) == expected
                arguments = dict(
                    evaluation_id="case",
                    market_id="crypto",
                    research_dataset_id="case",
                    signals=signals,
                    label_columns=("label",),
                )
                if expected:
                    with pytest.raises(
                        ValueError,
                        match="^evaluation_signal_id_column_overlap: "
                        + ", ".join(expected)
                        + "$",
                    ):
                        EvaluationSpec(**arguments)
                else:
                    assert EvaluationSpec(**arguments).signals == signals
                count += 1
    assert count == 1312


def test_winsorize_clips_by_timestamp_quantiles_instead_of_zscore() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "momentum_20d": [0.0, 10.0, 100.0],
        }
    )
    spec = FeatureTransformSpec(
        transform_id="momentum_20d_winsorized",
        source_columns=("momentum_20d",),
        transform="winsorize",
        output_columns=("momentum_20d_winsorized",),
        winsorize_quantiles=(0.2, 0.8),
    )
    result = apply_feature_transforms(frame, (spec,))
    assert result["momentum_20d_winsorized"].tolist() == pytest.approx(
        [4.0, 10.0, 64.0]
    )


def test_winsorization_preserves_positional_groups_and_duplicate_indexes():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-03",
                    None,
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "factor": [0.0, 7.0, -1e308, 10.0, np.nan, 100.0, 1e308, 10.0, np.nan, 5.0],
        },
        index=[3, 3, 1, 1, 3, 7, 3, 7, 1, 1],
    )
    before = frame.copy(deep=True)
    spec = FeatureTransformSpec(
        "positional", ("factor",), "winsorize", ("feature",), (0.25, 0.75)
    )
    actual = apply_feature_transforms(frame, (spec,))
    expected = pd.Series(
        [7.5, 7.0, -5e307, 10.0, np.nan, 32.5, 5e307, 10.0, np.nan, 5.0],
        index=frame.index,
        name="feature",
    )
    pd.testing.assert_series_equal(actual["feature"], expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    empty = apply_feature_transforms(frame.iloc[:0], (spec,))
    assert empty.empty and empty["feature"].dtype == "float64"


def test_fillna_transform_names_apply_their_fill_policy() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "factor": [1.0, None, 3.0],
        }
    )
    zero_spec = FeatureTransformSpec(
        transform_id="factor_fillna_zero",
        source_columns=("factor",),
        transform="fillna_zero",
        output_columns=("factor_zero",),
    )
    median_spec = FeatureTransformSpec(
        transform_id="factor_fillna_median",
        source_columns=("factor",),
        transform="fillna_median",
        output_columns=("factor_median",),
    )
    zero_result = apply_feature_transforms(frame, (zero_spec,))
    median_result = apply_feature_transforms(frame, (median_spec,))
    assert zero_result["factor_zero"].tolist() == pytest.approx([1.0, 0.0, 3.0])
    assert median_result["factor_median"].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_zero_variance_zscore_keeps_numeric_output() -> None:
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-02")] * 3, "factor": [1.0, 1.0, 1.0]}
    )
    result = apply_feature_transforms(
        frame,
        (
            FeatureTransformSpec(
                transform_id="factor_zscore",
                source_columns=("factor",),
                transform="cross_sectional_zscore",
                output_columns=("factor_zscore",),
            ),
        ),
    )
    assert result["factor_zscore"].dtype == "float64"
    assert result["factor_zscore"].tolist() == [0.0, 0.0, 0.0]


def test_tiny_nonconstant_zscores_remain_valid() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 3,
            "factor": np.array([1e-320, 2e-320, 3e-320]),
        }
    )
    feature = FeatureTransformSpec(
        transform_id="factor_zscore",
        source_columns=("factor",),
        transform="cross_sectional_zscore",
        output_columns=("factor_zscore",),
    )
    signal = SignalSpec(
        signal_id="factor_signal",
        source_kind="factor",
        source_column="factor",
        transform="zscore",
    )
    feature_result = apply_feature_transforms(frame, (feature,))
    signal_result = normalize_signal_frame(frame, (signal,))
    expected = [-1.224744871391589, 0.0, 1.224744871391589]
    assert feature_result["factor_zscore"].tolist() == pytest.approx(expected)
    assert signal_result["factor_signal"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf")])
def test_feature_transforms_reject_invalid_numeric_sources(value: object) -> None:
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")], "factor": [value]})
    spec = FeatureTransformSpec(
        transform_id="factor_rank",
        source_columns=("factor",),
        transform="cross_sectional_rank",
        output_columns=("factor_rank",),
    )
    with pytest.raises(ValueError, match="invalid_feature_transform_source"):
        apply_feature_transforms(frame, (spec,))


def test_signal_materialization_normalizes_direction_and_transform() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "risk": [1.0, 2.0, 3.0],
        }
    )
    signal = SignalSpec(
        signal_id="risk_signal",
        source_kind="factor",
        source_column="risk",
        direction="lower_is_better",
        transform="rank",
    )
    result = normalize_signal_frame(frame, (signal,))
    assert result["risk_signal"].tolist() == pytest.approx([-1.0, -2.0, -3.0])
    assert "risk" not in result.columns


def test_signal_materialization_rejects_invalid_numeric_source() -> None:
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-02")], "risk": ["not-a-number"]}
    )
    signal = SignalSpec(
        signal_id="risk_signal", source_kind="factor", source_column="risk"
    )
    with pytest.raises(ValueError, match="invalid_signal_values"):
        normalize_signal_frame(frame, (signal,))
    with pytest.raises(ValueError, match="invalid_signal_values"):
        normalize_signal_frame(frame.assign(risk=["1.0"]), (signal,))


@pytest.mark.parametrize(
    "timestamps",
    [[pd.NaT, pd.Timestamp("2024-01-02")], ["not-a-timestamp", "2024-01-02"]],
)
def test_signal_materialization_rejects_invalid_group_timestamps(timestamps) -> None:
    frame = pd.DataFrame({"timestamp": timestamps, "risk": [1.0, 2.0]})
    signal = SignalSpec(
        signal_id="risk_signal",
        source_kind="factor",
        source_column="risk",
        transform="rank",
    )
    with pytest.raises(ValueError, match="invalid_signal_timestamps"):
        normalize_signal_frame(frame, (signal,))


def test_signal_materialization_enforces_source_kind_column_ownership() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "raw_factor": [1.0],
            "model_feature": [2.0],
        }
    )
    signal = SignalSpec(
        signal_id="bad_feature_signal",
        source_kind="feature",
        source_column="raw_factor",
    )
    with pytest.raises(ValueError, match="invalid_signal_source_kind_column"):
        normalize_signal_frame(
            frame,
            (signal,),
            source_columns_by_kind={
                "factor": ("raw_factor",),
                "feature": ("model_feature",),
                "prediction": (),
            },
        )
