from __future__ import annotations

import numpy as np
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _assign_groups_array,
    _average_ranks,
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
