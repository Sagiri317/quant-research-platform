from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.evaluations.diagnostics import (
    summarize_group_spread,
    summarize_period_ic,
    summarize_signal_label_coverage,
)


def test_summarize_signal_label_coverage_handles_empty_metrics() -> None:
    summary = summarize_signal_label_coverage(pd.DataFrame())

    assert summary == {
        "metric_rows": 0,
        "rows_after_signal_label_dropna": 0,
        "timestamp_count": 0,
        "min_timestamp": None,
        "max_timestamp": None,
        "skipped_timestamp_count": 0,
    }


def test_summarize_signal_label_coverage_counts_minimum_observation_skips() -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
            "signal": ["s1", "s1", "s2"],
            "label": ["l1", "l1", "l1"],
            "ic": [0.1, float("nan"), 0.2],
            "rankic": [0.2, float("nan"), 0.3],
            "n_obs": [3, 1, 2],
        }
    )

    summary = summarize_signal_label_coverage(metrics)

    assert summary["metric_rows"] == 3
    assert summary["rows_after_signal_label_dropna"] == 6
    assert summary["timestamp_count"] == 2
    assert summary["min_timestamp"] == "2024-01-02"
    assert summary["max_timestamp"] == "2024-01-03"
    assert summary["skipped_timestamp_count"] == 1


def test_summarize_period_ic_makes_all_nan_behavior_explicit() -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "signal": ["s1", "s1"],
            "label": ["l1", "l1"],
            "ic": [float("nan"), float("nan")],
            "rankic": [float("nan"), float("nan")],
            "n_obs": [1, 1],
        }
    )

    summary = summarize_period_ic(metrics)

    assert summary.loc[0, "valid_ic_count"] == 0
    assert pd.isna(summary.loc[0, "mean_ic"])


def test_summarize_group_spread_is_deterministically_sorted() -> None:
    grouped = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "signal": ["s1", "s1", "s1", "s1"],
            "label": ["l1", "l1", "l1", "l1"],
            "group": [2, 2, 1, 1],
            "mean_label_value": [0.4, 0.3, 0.1, 0.2],
            "n_obs": [2, 2, 2, 2],
        }
    )

    summary = summarize_group_spread(grouped)

    assert summary.loc[0, "signal"] == "s1"
    assert summary.loc[0, "label"] == "l1"
    assert summary.loc[0, "top_bottom_label_value_spread_mean"] == pytest.approx(0.2)


def test_evaluation_diagnostics_reports_fail_closed_on_bad_inputs() -> None:
    with pytest.raises(ValueError, match="missing_evaluation_metric_columns"):
        summarize_signal_label_coverage(pd.DataFrame({"timestamp": ["2024-01-02"]}))

    with pytest.raises(ValueError, match="invalid_evaluation_timestamp_values"):
        summarize_signal_label_coverage(
            pd.DataFrame({"timestamp": ["bad"], "n_obs": [1]})
        )

    with pytest.raises(ValueError, match="invalid_evaluation_observation_counts"):
        summarize_signal_label_coverage(
            pd.DataFrame({"timestamp": ["2024-01-02"], "n_obs": [1.5]})
        )
