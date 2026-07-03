from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _assign_groups_array,
    _average_ranks,
    _pearson_correlation_values,
    compute_grouped_label_values,
    compute_ic_and_grouped_label_values,
    compute_ic_and_grouped_label_values_from_validated_panel,
    compute_ic_by_timestamp,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from tests.helpers.compute_inputs import make_cross_section_compute_panel


def test_dataframe_cross_section_metrics_and_summaries() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor", "factor_alt"),
        label_column="forward_return",
    )
    frame = make_cross_section_compute_panel()

    ic = compute_ic_by_timestamp(frame, spec)
    combined_ic, grouped = compute_ic_and_grouped_label_values(
        frame,
        spec,
        n_groups=3,
    )
    grouped_only = compute_grouped_label_values(frame, spec, n_groups=3)
    ic_summary = summarize_ic(ic)
    spread_summary = summarize_group_spread(grouped)

    assert len(ic) == 4
    assert combined_ic.equals(ic)
    assert grouped_only.equals(grouped)
    assert ic_summary["signal"].tolist() == ["factor", "factor_alt"]
    assert set(spread_summary["signal"]) == {"factor", "factor_alt"}


def test_dataframe_cross_section_validated_panel_and_empty_summaries() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    frame = make_cross_section_compute_panel().assign(
        segment=["a", "a", "a", "b", "b", "b"]
    )

    ic, grouped = compute_ic_and_grouped_label_values_from_validated_panel(
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
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    with pytest.raises(TypeError, match="n_groups must be an integer"):
        compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=True,
        )
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        compute_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=1,
        )
    with pytest.raises(ValueError, match="ic is missing columns"):
        summarize_ic(pd.DataFrame({"signal": ["factor"]}))
    with pytest.raises(ValueError, match="grouped_label_values is missing columns"):
        summarize_group_spread(pd.DataFrame({"signal": ["factor"]}))


def test_dataframe_cross_section_numeric_helpers() -> None:
    assert _assign_groups_array(np.array([3.0, 1.0, 2.0]), 5).tolist() == [3, 1, 2]
    assert _average_ranks(np.array([2.0, 2.0, 1.0])).tolist() == [2.5, 2.5, 1.0]
    assert np.isnan(
        _pearson_correlation_values(np.array([1.0, 1.0]), np.array([2.0, 3.0]))
    )
