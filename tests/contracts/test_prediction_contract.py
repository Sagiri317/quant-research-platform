"""Prediction windows describe extent; test eligibility comes from frozen records."""

from datetime import date

import pandas as pd
import pytest

from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    prediction_window_mask,
)


def test_prediction_window_has_no_oos_flags_or_test_aliases():
    value = PredictionWindow("2020-01-02", "2020-01-03")
    assert value.end.isoformat() == "2020-01-03"
    with pytest.raises(ValueError):
        TimeWindow("2020-01-02", None)
    with pytest.raises(TypeError):
        PredictionWindow("2020-01-02", "2020-01-03", is_oos=True)
    with pytest.raises(TypeError):
        PredictionWindow("2020-01-02", "2020-01-03", test_start="2020-01-02")


def test_only_prediction_window_allows_an_open_end():
    value = PredictionWindow("2020-01-02", None)
    assert value.end is None
    timestamps = pd.Series(pd.date_range("2020-01-01", periods=3))
    assert prediction_window_mask(timestamps, value).tolist() == [
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "window",
    [
        {"start": None, "end": None},
        {"start": "NaT", "end": None},
        {"start": pd.NaT, "end": None},
        {"start": True, "end": None},
        {"start": "2020-01-02", "end": "2020-01-01"},
        {"start": "2020-01-02", "end": "2020-01-03T00:00:00"},
        {"start": "2020-01-02", "end": None, "extra": True},
        {"start": "2020-01-02"},
    ],
)
def test_prediction_window_rejects_invalid_or_unknown_bounds(window):
    with pytest.raises((TypeError, ValueError)):
        PredictionWindow(**window)


def test_open_prediction_normalizes_date_and_timezone_timestamp_bounds():
    class ObservationDate(date):
        pass

    assert PredictionWindow(ObservationDate(2020, 1, 2), None).start == date(2020, 1, 2)
    value = PredictionWindow(pd.Timestamp("2020-01-02T08:00:00.000000001+08:00"), None)
    assert value.start == pd.Timestamp("2020-01-02T00:00:00.000000001")
    assert value.start.tzinfo is None and value.start.nanosecond == 1


def test_intraday_window_preserves_midnight_and_nanoseconds():
    value = PredictionWindow(
        "2020-01-02T00:00:00.000000001", "2020-01-02T00:00:00.000000002"
    )
    assert str(value.start).endswith("000000001")
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-01-02T00:00:00",
                "2020-01-02T00:00:00.000000001",
                "2020-01-02T00:00:00.000000002",
                "2020-01-02T00:00:00.000000003",
            ],
            format="mixed",
        )
    )
    assert prediction_window_mask(timestamps, value).tolist() == [
        False,
        True,
        True,
        False,
    ]
    with pytest.raises(ValueError):
        PredictionWindow("2020-01-02", "2020-01-02T01:00:00")


@pytest.mark.parametrize(
    "key",
    [
        "prediction_mode",
        "prediction_column",
        "primary_key",
        "train_end",
        "is_oos",
        "test_start",
        "training_asset_ids",
        "evaluation_asset_ids",
        "sample_filter_mode",
    ],
)
def test_native_prediction_rejects_removed_overrides(key):
    from quant_research_platform.research.predictions.runner import predict

    with pytest.raises(ValueError, match="operation_config_fields_invalid"):
        predict(
            {
                "model": "unused",
                "inputs": "unused",
                "output": "unused",
                "prediction_window": {"start": "2020-01-02", "end": None},
                "sample_role": "validation",
                key: "unused",
            }
        )


@pytest.mark.parametrize("assets", [[], ["a", "a"], [""], [True], "a"])
def test_native_prediction_asset_scope_requires_distinct_names(assets):
    from quant_research_platform.research.predictions.runner import predict

    with pytest.raises(ValueError, match="prediction_asset_ids_invalid"):
        predict(
            {
                "model": "unused",
                "inputs": "unused",
                "output": "unused",
                "prediction_window": {"start": "2020-01-02", "end": None},
                "sample_role": "validation",
                "asset_ids": assets,
            }
        )
