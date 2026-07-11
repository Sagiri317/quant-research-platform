from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.portfolio_returns import (
    _build_portfolio_return_record_from_positions,
    _timestamp_label_lookup,
)
from quant_research_platform.compute.panels import ComputePanelSpec


def test_dataframe_portfolio_return_record_handles_empty_or_missing_legs() -> None:
    group_values = {"timestamp": pd.Timestamp("2024-01-02"), "signal": "factor"}
    label_lookup = {"A": 0.01, "B": -0.02}

    assert _build_portfolio_return_record_from_positions(
        group_values=group_values,
        portfolio_mode="long_short",
        positions={"A": 1.0},
        label_lookup=label_lookup,
        return_source_column="label",
        turnover=1.0,
        cost=0.0,
        segment_columns=(),
    ) is None

    long_only = _build_portfolio_return_record_from_positions(
        group_values=group_values,
        portfolio_mode="long_only",
        positions={"A": 1.0, "C": -1.0},
        label_lookup=label_lookup,
        return_source_column="label",
        turnover=1.0,
        cost=0.001,
        segment_columns=(),
    )

    assert long_only is not None
    assert long_only["long_count"] == 1
    assert long_only["short_count"] == 0
    assert long_only["net_portfolio_return"] == pytest.approx(0.009)


def test_dataframe_portfolio_return_timestamp_label_lookup_uses_asset_strings() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="label")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": [1, 2],
            "label": [0.1, 0.2],
        }
    )

    assert _timestamp_label_lookup(frame, spec, "label") == {"1": 0.1, "2": 0.2}
