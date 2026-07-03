from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.backends import (
    DATAFRAME_BACKEND,
    SUPPORTED_BACKENDS,
    TENSOR_BACKEND,
)
from quant_research_platform.compute.dataframe.cross_section import (
    compute_grouped_label_values,
    compute_ic_by_timestamp,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from tests.helpers.compute_inputs import make_cross_section_compute_panel


def test_backend_constants_import_from_backend_owner() -> None:
    assert DATAFRAME_BACKEND == "dataframe"
    assert TENSOR_BACKEND == "tensor"
    assert SUPPORTED_BACKENDS == {DATAFRAME_BACKEND, TENSOR_BACKEND}


def test_validate_compute_panel_normalizes_and_sorts_valid_panel() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result = validate_compute_panel(
        make_cross_section_compute_panel(scenario="perfect").iloc[::-1],
        spec,
    )

    assert result["timestamp"].tolist()[0] == pd.Timestamp("2026-01-01")
    assert result["asset"].tolist()[:3] == ["A", "B", "C"]


def test_validate_compute_panel_rejects_missing_signal_column() -> None:
    spec = ComputePanelSpec(signal_columns=("missing",), label_column="forward_return")

    with pytest.raises(ValueError, match="panel is missing columns"):
        validate_compute_panel(
            make_cross_section_compute_panel(scenario="perfect"),
            spec,
        )


def test_validate_compute_panel_rejects_duplicate_timestamp_asset_keys() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    panel = pd.concat(
        [
            make_cross_section_compute_panel(scenario="perfect"),
            make_cross_section_compute_panel(scenario="perfect").head(1),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(panel, spec)


def test_compute_ic_by_timestamp_computes_pearson_and_rankic() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result = compute_ic_by_timestamp(
        make_cross_section_compute_panel(scenario="perfect"),
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

    result = compute_ic_by_timestamp(
        make_cross_section_compute_panel(scenario="perfect"),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
    )

    assert result["ic"].round(6).tolist() == [1.0, 1.0]
    assert result["rankic"].round(6).tolist() == [1.0, 1.0]


def test_compute_grouped_label_values_assigns_timestamp_level_groups() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result = compute_grouped_label_values(
        make_cross_section_compute_panel(scenario="perfect"),
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

    result = compute_grouped_label_values(
        make_cross_section_compute_panel(scenario="perfect"),
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

    result = compute_grouped_label_values(
        make_cross_section_compute_panel(scenario="perfect"),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        n_groups=3,
    )

    assert result["group"].tolist() == [1, 2, 3, 1, 2, 3]


def test_summarize_ic_reports_icir_and_positive_rates() -> None:
    ic = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "signal": ["factor", "factor", "factor"],
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
    assert row["valid_ic_count"] == 3
    assert row["mean_rankic"] == pytest.approx((0.3 + 0.1 - 0.2) / 3)
    assert row["positive_rankic_rate"] == pytest.approx(2 / 3)
    assert row["valid_rankic_count"] == 3


def test_summarize_ic_uses_nan_when_no_valid_ic_or_rankic() -> None:
    ic = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "signal": ["factor", "factor"],
            "label": ["forward_return", "forward_return"],
            "ic": [np.nan, np.nan],
            "rankic": [np.nan, np.nan],
            "n_obs": [1, 1],
        }
    )

    summary = summarize_ic(ic)

    row = summary.iloc[0]
    assert row["valid_ic_count"] == 0
    assert pd.isna(row["mean_ic"])
    assert pd.isna(row["std_ic"])
    assert pd.isna(row["icir"])
    assert pd.isna(row["positive_ic_rate"])
    assert row["valid_rankic_count"] == 0
    assert pd.isna(row["mean_rankic"])
    assert pd.isna(row["std_rankic"])
    assert pd.isna(row["rankicir"])
    assert pd.isna(row["positive_rankic_rate"])


def test_summarize_ic_uses_nan_icir_when_valid_count_or_std_is_insufficient() -> None:
    one_valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "signal": ["factor"],
            "label": ["forward_return"],
            "ic": [0.4],
            "rankic": [-0.3],
            "n_obs": [10],
        }
    )
    constant_valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "signal": ["factor", "factor"],
            "label": ["forward_return", "forward_return"],
            "ic": [0.4, 0.4],
            "rankic": [-0.3, -0.3],
            "n_obs": [10, 10],
        }
    )

    one_row = summarize_ic(one_valid).iloc[0]
    constant_row = summarize_ic(constant_valid).iloc[0]

    assert one_row["valid_ic_count"] == 1
    assert pd.isna(one_row["icir"])
    assert one_row["valid_rankic_count"] == 1
    assert pd.isna(one_row["rankicir"])
    assert constant_row["valid_ic_count"] == 2
    assert constant_row["std_ic"] == pytest.approx(0.0)
    assert pd.isna(constant_row["icir"])
    assert constant_row["valid_rankic_count"] == 2
    assert constant_row["std_rankic"] == pytest.approx(0.0)
    assert pd.isna(constant_row["rankicir"])


def test_summarize_group_spread_reports_top_bottom_spread() -> None:
    grouped_label_values = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "signal": ["factor", "factor", "factor", "factor"],
            "label": ["forward_return"] * 4,
            "group": [1, 2, 1, 2],
            "mean_label_value": [-0.01, 0.03, 0.02, 0.01],
            "n_obs": [5, 5, 5, 5],
        }
    )

    summary = summarize_group_spread(grouped_label_values)

    row = summary.iloc[0]
    assert row["top_group_mean_label_value"] == pytest.approx(0.02)
    assert row["bottom_group_mean_label_value"] == pytest.approx(0.005)
    assert row["top_bottom_label_value_spread_mean"] == pytest.approx(0.015)
    assert row["top_bottom_label_value_spread_hit_rate"] == pytest.approx(0.5)
    assert row["group_monotonicity_score"] == pytest.approx(0.0)


def test_summarize_group_spread_uses_dataframe_aggregation(monkeypatch) -> None:
    original_from_records = pd.DataFrame.from_records

    def fail_from_records(*args, **kwargs):
        raise AssertionError("group spread summary should avoid row-record loops")

    grouped_label_values = original_from_records(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "signal": ["factor", "factor", "factor", "factor"],
            "label": ["forward_return"] * 4,
            "group": [1, 2, 1, 2],
            "mean_label_value": [-0.01, 0.03, 0.02, 0.01],
            "n_obs": [5, 5, 5, 5],
        }
    )
    monkeypatch.setattr(pd.DataFrame, "from_records", fail_from_records)

    summary = summarize_group_spread(grouped_label_values)

    assert summary.iloc[0]["top_bottom_label_value_spread_mean"] == pytest.approx(
        0.015
    )
