from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _assign_groups_from_ranks,
    _average_ranks,
    _compute_ic_and_grouped_label_values,
    _pearson_correlation_result,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from tests.helpers.compute_inputs import make_cross_section_compute_panel


def test_dataframe_cross_section_accepts_ignored_extra_columns() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor", "factor_alt"), label_columns=("forward_return",)
    )
    frame = make_cross_section_compute_panel().assign(
        segment=["a", "a", "a", "b", "b", "b"]
    )

    ic, grouped = _compute_ic_and_grouped_label_values(
        frame,
        spec,
        n_groups=3,
    )
    expected_ic, expected_grouped = _compute_ic_and_grouped_label_values(
        frame.drop(columns="segment"), spec, n_groups=3
    )
    assert len(ic) == 4
    assert set(ic["signal"]) == {"factor", "factor_alt"}
    assert not grouped.empty
    pd.testing.assert_frame_equal(ic, expected_ic)
    pd.testing.assert_frame_equal(grouped, expected_grouped)


def test_dataframe_cross_section_rejects_invalid_group_counts_and_columns() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor",), label_columns=("forward_return",)
    )

    with pytest.raises(ValueError, match="panel is missing columns"):
        _compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel().drop(columns=["asset"]),
            spec,
            n_groups=True,
        )
    with pytest.raises(TypeError, match="n_groups must be an integer"):
        _compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=True,
        )
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        _compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=1,
        )


def test_dataframe_cross_section_numeric_helpers() -> None:
    assert _assign_groups_from_ranks(np.array([3.0, 1.0, 2.0]), 5).tolist() == [3, 1, 2]
    assert _average_ranks(np.array([2.0, 2.0, 1.0])).tolist() == [2.5, 2.5, 1.0]


def test_pearson_preserves_small_nonconstant_variation() -> None:
    values = np.array([1e-4, 2e-4, 3e-4])

    correlation, reason = _pearson_correlation_result(values, values)

    assert correlation == pytest.approx(1.0)
    assert reason is None


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_pearson_preserves_linear_relations(sign):
    values = np.array([-2.0, 0.0, 1.0, 4.0])
    actual, reason = _pearson_correlation_result(values, 3 + sign * values)
    assert actual == pytest.approx(sign, abs=1e-15)
    assert reason is None


def test_pearson_preserves_true_zero_and_undefined_domains():
    assert _pearson_correlation_result(
        np.array([-1.0, 0.0, 1.0]), np.array([1.0, -2.0, 1.0])
    ) == (0.0, None)
    assert _pearson_correlation_result(np.ones(3), np.arange(3.0)) == (
        None,
        "zero_variance",
    )
    assert _pearson_correlation_result(np.array([1.0]), np.array([2.0])) == (
        None,
        "insufficient_observations",
    )
    assert _pearson_correlation_result(
        np.array([np.nan, 1.0]), np.array([2.0, 3.0])
    ) == (None, "nonfinite_result")
