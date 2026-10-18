from __future__ import annotations

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
from quant_research_platform.research.signals.normalization import (
    normalize_signal_frame,
)


@pytest.mark.parametrize(
    "ids,columns,literal_overlaps",
    [
        (("a",), ("a",), []),
        (("a", "b"), ("score", "prediction_score"), []),
        (("a", "b"), ("b", "a"), ["a", "b"]),
        (("a", "b"), ("b", "c"), ["b"]),
        (("c", "b", "a"), ("b", "a", "c"), ["a", "b", "c"]),
        (("a", "b", "c"), ("a", "c", "b"), ["b", "c"]),
    ],
)
def test_signal_overlap_sets_preserve_unique_contract_and_error_order(
    ids, columns, literal_overlaps
):
    signals = tuple(
        SignalSpec(signal_id=signal_id, source_kind="prediction", source_column=column)
        for signal_id, column in zip(ids, columns, strict=True)
    )
    assert _cross_signal_id_column_overlaps(signals) == literal_overlaps
    arguments = dict(
        evaluation_id="case",
        market_id="crypto",
        research_dataset_id="case",
        signals=signals,
        label_columns=("label",),
    )
    if literal_overlaps:
        with pytest.raises(
            ValueError,
            match="^evaluation_signal_id_column_overlap: "
            + ", ".join(literal_overlaps)
            + "$",
        ):
            EvaluationSpec(**arguments)
    else:
        assert EvaluationSpec(**arguments).signals == signals


@pytest.mark.parametrize(
    "ids,columns,error",
    [
        (("a", "a"), ("x", "y"), "evaluation_signal_ids_must_not_contain_duplicates"),
        (
            ("a", "b"),
            ("x", "x"),
            "evaluation_signal_columns_must_not_contain_duplicates",
        ),
    ],
)
def test_signal_overlap_requires_unique_declarations(ids, columns, error):
    signals = tuple(
        SignalSpec(signal_id=key, source_kind="prediction", source_column=column)
        for key, column in zip(ids, columns, strict=True)
    )
    with pytest.raises(ValueError, match=f"^{error}$"):
        EvaluationSpec(
            evaluation_id="case",
            market_id="crypto",
            research_dataset_id="case",
            signals=signals,
            label_columns=("label",),
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
            "factor": [0.0, 7.0, -100.0, 10.0, np.nan, 100.0, 100.0, 10.0, np.nan, 5.0],
        },
        index=[3, 3, 1, 1, 3, 7, 3, 7, 1, 1],
    )
    before = frame.copy(deep=True)
    from quant_research_platform.research.features.transforms import _winsorize

    values = frame["factor"]
    actual = _winsorize(values, values.groupby(frame["timestamp"], sort=False))
    expected = pd.Series(
        [0.3, 7.0, -98.0, 10.0, np.nan, 97.3, 98.0, 10.0, np.nan, 5.0],
        index=frame.index,
        name="factor",
    )
    pd.testing.assert_series_equal(actual, expected, rtol=1e-15, atol=1e-15)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    empty = _winsorize(
        values.iloc[:0], values.iloc[:0].groupby(frame["timestamp"].iloc[:0])
    )
    assert empty.empty and empty.dtype == "float64"


def test_zero_variance_zscore_keeps_numeric_output() -> None:
    frame = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2024-01-02")] * 3, "factor": [1.0, 1.0, 1.0]}
    )
    from quant_research_platform.compute.dataframe.cross_section import _stable_zscore

    result = _stable_zscore(frame["factor"])
    assert result.dtype == "float64"
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_small_nonconstant_zscores_remain_valid() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 3,
            "factor": np.array([1e-8, 2e-8, 3e-8]),
        }
    )
    signal = SignalSpec(
        signal_id="factor_signal",
        source_kind="factor",
        source_column="factor",
        transform="zscore",
    )
    from quant_research_platform.compute.dataframe.cross_section import _stable_zscore

    feature_result = _stable_zscore(frame["factor"])
    signal_result = normalize_signal_frame(frame, (signal,))
    expected = [-1.224744871391589, 0.0, 1.224744871391589]
    assert feature_result.tolist() == pytest.approx(expected)
    assert signal_result["factor_signal"].tolist() == pytest.approx(expected)


@pytest.mark.parametrize("value", ["not-a-number", float("inf"), float("-inf")])
def test_feature_transforms_reject_invalid_numeric_sources(value: object) -> None:
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")], "factor": [value]})
    frame = frame.assign(
        eligible=True,
        industry_standard_id="fixture",
        industry_level1_code="A",
        log_total_market_cap=0.0,
    )
    spec = FeatureTransformSpec("a_share_canonical", ("factor",))
    with pytest.raises(ValueError, match="invalid_feature_factor"):
        apply_feature_transforms(frame, (spec,))


def test_signal_materialization_normalizes_direction_and_transform() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "risk": [8.0, 2.0, 2.0, 4.0, 9.0, 1.0],
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
    assert result["risk_signal"].tolist() == pytest.approx(
        [-3.0, -1.5, -1.5, -2.0, -3.0, -1.0]
    )
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
