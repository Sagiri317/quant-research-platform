from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.evaluations.panel_validation import PanelSpec
from quant_research_platform.kernels.dataframe.cross_section import (
    analyze_grouped_returns,
    evaluate_ic_by_timestamp,
    summarize_group_spread,
    summarize_ic,
)


def _perfect_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )


def test_evaluate_ic_by_timestamp_computes_pearson_and_rankic() -> None:
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    result = evaluate_ic_by_timestamp(_perfect_panel(), spec)

    assert result["timestamp"].nunique() == 2
    assert result["feature"].tolist() == ["factor", "factor"]
    assert result["ic"].round(6).tolist() == [1.0, 1.0]
    assert result["rankic"].round(6).tolist() == [1.0, 1.0]
    assert result["n_obs"].tolist() == [3, 3]


def test_evaluate_ic_by_timestamp_does_not_call_pandas_series_corr(
    monkeypatch,
) -> None:
    def fail_corr(*args, **kwargs):
        raise AssertionError("IC should use vectorized correlation math")

    monkeypatch.setattr(pd.Series, "corr", fail_corr)

    result = evaluate_ic_by_timestamp(
        _perfect_panel(),
        PanelSpec(feature_columns=("factor",), label_column="forward_return"),
    )

    assert result["ic"].round(6).tolist() == [1.0, 1.0]
    assert result["rankic"].round(6).tolist() == [1.0, 1.0]


def test_analyze_grouped_returns_assigns_timestamp_level_groups() -> None:
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    result = analyze_grouped_returns(_perfect_panel(), spec, n_groups=3)

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result["mean_forward_return"].round(6).tolist() == [
        0.01,
        0.02,
        0.03,
        0.01,
        0.02,
        0.03,
    ]


def test_analyze_grouped_returns_does_not_call_pandas_qcut(monkeypatch) -> None:
    def fail_qcut(*args, **kwargs):
        raise AssertionError("group assignment should use deterministic rank buckets")

    monkeypatch.setattr(pd, "qcut", fail_qcut)

    result = analyze_grouped_returns(
        _perfect_panel(),
        PanelSpec(feature_columns=("factor",), label_column="forward_return"),
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]


def test_analyze_grouped_returns_does_not_group_assigned_pairs(
    monkeypatch,
) -> None:
    def fail_assign(*args, **kwargs):
        raise AssertionError("grouped returns should aggregate rank buckets directly")

    monkeypatch.setattr(pd.DataFrame, "assign", fail_assign)

    result = analyze_grouped_returns(
        _perfect_panel(),
        PanelSpec(feature_columns=("factor",), label_column="forward_return"),
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]


def test_summarize_ic_reports_icir_and_positive_rates() -> None:
    ic = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "feature": ["factor", "factor", "factor"],
            "label": ["forward_return", "forward_return", "forward_return"],
            "ic": [0.2, -0.1, 0.4],
            "rankic": [0.3, 0.1, -0.2],
            "n_obs": [10, 10, 10],
        }
    )

    summary = summarize_ic(ic)

    row = summary.iloc[0]
    assert row["mean_ic"] == pytest.approx((0.2 - 0.1 + 0.4) / 3)
    assert row["std_ic"] > 0
    assert row["icir"] != 0
    assert row["positive_ic_rate"] == pytest.approx(2 / 3)
    assert row["mean_rankic"] == pytest.approx((0.3 + 0.1 - 0.2) / 3)
    assert row["positive_rankic_rate"] == pytest.approx(2 / 3)


def test_summarize_group_spread_reports_top_bottom_spread() -> None:
    grouped_returns = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "feature": ["factor", "factor", "factor", "factor"],
            "label": ["forward_return"] * 4,
            "group": [1, 2, 1, 2],
            "mean_forward_return": [-0.01, 0.03, 0.02, 0.01],
            "n_obs": [5, 5, 5, 5],
        }
    )

    summary = summarize_group_spread(grouped_returns)

    row = summary.iloc[0]
    assert row["top_group_mean_return"] == pytest.approx(0.02)
    assert row["bottom_group_mean_return"] == pytest.approx(0.005)
    assert row["top_bottom_spread_mean"] == pytest.approx(0.015)
    assert row["top_bottom_spread_hit_rate"] == pytest.approx(0.5)
    assert row["group_monotonicity_score"] == pytest.approx(0.0)


def test_summarize_group_spread_uses_dataframe_aggregation(monkeypatch) -> None:
    original_from_records = pd.DataFrame.from_records

    def fail_from_records(*args, **kwargs):
        raise AssertionError("group spread summary should avoid row-record loops")

    grouped_returns = original_from_records(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "feature": ["factor", "factor", "factor", "factor"],
            "label": ["forward_return"] * 4,
            "group": [1, 2, 1, 2],
            "mean_forward_return": [-0.01, 0.03, 0.02, 0.01],
            "n_obs": [5, 5, 5, 5],
        }
    )
    monkeypatch.setattr(pd.DataFrame, "from_records", fail_from_records)

    summary = summarize_group_spread(grouped_returns)

    assert summary.iloc[0]["top_bottom_spread_mean"] == pytest.approx(0.015)
