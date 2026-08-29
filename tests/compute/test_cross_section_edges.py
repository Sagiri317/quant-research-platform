from __future__ import annotations

import numpy as np
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _assign_groups_array,
    _average_ranks,
    _pearson_correlation_from_moments,
    _pearson_correlation_result,
    compute_ic_and_grouped_label_values,
    compute_ic_and_grouped_label_values_from_validated_panel,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from tests.helpers.compute_inputs import make_cross_section_compute_panel


def test_dataframe_cross_section_metrics() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor", "factor_alt"),
        label_column="forward_return",
    )
    frame = make_cross_section_compute_panel()

    ic, grouped = compute_ic_and_grouped_label_values(
        frame,
        spec,
        n_groups=3,
    )
    assert len(ic) == 4
    assert set(ic["signal"]) == {"factor", "factor_alt"}
    assert not grouped.empty


def test_dataframe_cross_section_validated_panel() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    frame = make_cross_section_compute_panel().assign(
        segment=["a", "a", "a", "b", "b", "b"]
    )

    ic, grouped = compute_ic_and_grouped_label_values_from_validated_panel(
        frame,
        spec,
        n_groups=2,
    )
    assert len(ic) == 2
    assert len(grouped) == 4


def test_dataframe_cross_section_rejects_invalid_group_counts_and_columns() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    with pytest.raises(TypeError, match="n_groups must be an integer"):
        compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=True,
        )
    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        compute_ic_and_grouped_label_values(
            make_cross_section_compute_panel(),
            spec,
            n_groups=1,
        )
def test_dataframe_cross_section_numeric_helpers() -> None:
    assert _assign_groups_array(np.array([3.0, 1.0, 2.0]), 5).tolist() == [3, 1, 2]
    assert _average_ranks(np.array([2.0, 2.0, 1.0])).tolist() == [2.5, 2.5, 1.0]


def test_pearson_preserves_tiny_nonconstant_variation() -> None:
    values = np.array([1e-320, 2e-320, 3e-320])

    correlation, reason = _pearson_correlation_result(values, values)

    assert correlation == pytest.approx(1.0)
    assert reason is None


@pytest.mark.parametrize("covariance, expected", ((4.0, 1.0), (-4.0, -1.0)))
def test_pearson_moments_preserve_exact_boundaries(
    covariance: float,
    expected: float,
) -> None:
    assert _pearson_correlation_from_moments(
        count=17,
        left_m2=2.0,
        right_m2=8.0,
        covariance=covariance,
    ) == (expected, None)


@pytest.mark.parametrize("expected", (1.0, -1.0))
def test_pearson_moments_normalize_only_one_ulp_overflow(expected: float) -> None:
    overflow = float(np.copysign(np.nextafter(1.0, np.inf), expected))

    assert _pearson_correlation_from_moments(
        count=17,
        left_m2=1.0,
        right_m2=1.0,
        covariance=overflow,
    ) == (expected, None)


def test_pearson_moments_preserve_materially_invalid_result() -> None:
    invalid = float(np.nextafter(np.nextafter(1.0, np.inf), np.inf))

    assert _pearson_correlation_from_moments(
        count=17,
        left_m2=1.0,
        right_m2=1.0,
        covariance=invalid,
    ) == (invalid, None)
