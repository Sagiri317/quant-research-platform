from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _assign_groups_array,
    _average_ranks,
    _pearson_correlation_values,
    analyze_grouped_returns,
    evaluate_ic_and_grouped_returns,
    evaluate_ic_and_grouped_returns_from_validated_panel,
    evaluate_ic_by_timestamp,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.compute.panels import ComputePanelSpec


def test_dataframe_cross_section_metrics_and_summaries() -> None:
    spec = ComputePanelSpec(
        feature_columns=("factor", "factor_alt"),
        label_column="forward_return",
    )
    frame = _panel()

    ic = evaluate_ic_by_timestamp(frame, spec)
    combined_ic, grouped = evaluate_ic_and_grouped_returns(frame, spec, n_groups=3)
    grouped_only = analyze_grouped_returns(frame, spec, n_groups=3)
    ic_summary = summarize_ic(ic)
    spread_summary = summarize_group_spread(grouped)

    assert len(ic) == 4
    assert combined_ic.equals(ic)
    assert grouped_only.equals(grouped)
    assert ic_summary["feature"].tolist() == ["factor", "factor_alt"]
    assert set(spread_summary["feature"]) == {"factor", "factor_alt"}


def test_dataframe_cross_section_validated_panel_and_empty_summaries() -> None:
    spec = ComputePanelSpec(feature_columns=("factor",), label_column="forward_return")
    frame = _panel().assign(segment=["a", "a", "a", "b", "b", "b"])

    ic, grouped = evaluate_ic_and_grouped_returns_from_validated_panel(
        frame,
        spec,
        n_groups=2,
    )
    ic["segment"] = ["a", "b"]
    grouped["segment"] = ["a", "a", "b", "b"]

    assert summarize_ic(ic, segment_columns=("segment",))["segment"].tolist() == [
        "a",
        "b",
    ]
    assert not summarize_group_spread(
        grouped,
        segment_columns=("segment",),
    ).empty
    assert summarize_ic(ic.iloc[0:0]).empty
    assert summarize_group_spread(grouped.iloc[0:0]).empty


def test_dataframe_cross_section_rejects_invalid_group_counts_and_columns() -> None:
    spec = ComputePanelSpec(feature_columns=("factor",), label_column="forward_return")

    with pytest.raises(TypeError, match="n_groups must be an integer"):
        evaluate_ic_and_grouped_returns(_panel(), spec, n_groups=True)
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        analyze_grouped_returns(_panel(), spec, n_groups=1)
    with pytest.raises(ValueError, match="ic is missing columns"):
        summarize_ic(pd.DataFrame({"feature": ["factor"]}))
    with pytest.raises(ValueError, match="grouped_returns is missing columns"):
        summarize_group_spread(pd.DataFrame({"feature": ["factor"]}))


def test_dataframe_cross_section_numeric_helpers() -> None:
    assert _assign_groups_array(np.array([3.0, 1.0, 2.0]), 5).tolist() == [3, 1, 2]
    assert _average_ranks(np.array([2.0, 2.0, 1.0])).tolist() == [2.5, 2.5, 1.0]
    assert np.isnan(
        _pearson_correlation_values(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
    )


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "factor_alt": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )
