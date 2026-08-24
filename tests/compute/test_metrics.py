from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from tests.helpers.compute_inputs import make_perfect_cross_section_compute_panel


def test_validate_compute_panel_normalizes_and_sorts_valid_panel() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result = validate_compute_panel(
        make_perfect_cross_section_compute_panel().iloc[::-1],
        spec,
    )

    assert result["timestamp"].tolist()[0] == pd.Timestamp("2026-01-01")
    assert result["asset"].tolist()[:3] == ["A", "B", "C"]


def test_validate_compute_panel_rejects_missing_signal_column() -> None:
    spec = ComputePanelSpec(signal_columns=("missing",), label_column="forward_return")

    with pytest.raises(ValueError, match="panel is missing columns"):
        validate_compute_panel(
            make_perfect_cross_section_compute_panel(),
            spec,
        )


def test_validate_compute_panel_rejects_duplicate_timestamp_asset_keys() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    panel = pd.concat(
        [
            make_perfect_cross_section_compute_panel(),
            make_perfect_cross_section_compute_panel().head(1),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(panel, spec)


def test_compute_ic_by_timestamp_computes_pearson_and_rankic() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result, _ = compute_ic_and_grouped_label_values(
        make_perfect_cross_section_compute_panel(),
        spec,
    )

    assert result["timestamp"].nunique() == 2
    assert result["signal"].tolist() == ["factor", "factor"]
    assert result["ic"].round(6).tolist() == [1.0, 1.0]
    assert result["rankic"].round(6).tolist() == [1.0, 1.0]
    assert result["n_obs"].tolist() == [3, 3]


def test_compute_ic_by_timestamp_does_not_call_pandas_series_corr(
    monkeypatch,
) -> None:
    def fail_corr(*args, **kwargs):
        raise AssertionError("IC should use vectorized correlation math")

    monkeypatch.setattr(pd.Series, "corr", fail_corr)

    result, _ = compute_ic_and_grouped_label_values(
        make_perfect_cross_section_compute_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
    )

    assert result["ic"].round(6).tolist() == [1.0, 1.0]
    assert result["rankic"].round(6).tolist() == [1.0, 1.0]


def test_compute_grouped_label_values_assigns_timestamp_level_groups() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    _, result = compute_ic_and_grouped_label_values(
        make_perfect_cross_section_compute_panel(),
        spec,
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result["mean_label_value"].round(6).tolist() == [
        0.01,
        0.02,
        0.03,
        0.01,
        0.02,
        0.03,
    ]


def test_compute_grouped_label_values_does_not_call_pandas_qcut(monkeypatch) -> None:
    def fail_qcut(*args, **kwargs):
        raise AssertionError("group assignment should use deterministic rank buckets")

    monkeypatch.setattr(pd, "qcut", fail_qcut)

    _, result = compute_ic_and_grouped_label_values(
        make_perfect_cross_section_compute_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]


def test_compute_grouped_label_values_does_not_group_assigned_pairs(
    monkeypatch,
) -> None:
    def fail_assign(*args, **kwargs):
        raise AssertionError(
            "grouped label values should aggregate rank buckets directly"
        )

    monkeypatch.setattr(pd.DataFrame, "assign", fail_assign)

    _, result = compute_ic_and_grouped_label_values(
        make_perfect_cross_section_compute_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]
