from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.portfolio_returns import (
    _portfolio_return_components,
    _timestamp_label_lookup,
)
from quant_research_platform.compute.panels import ComputePanelSpec


def test_dataframe_portfolio_return_record_handles_empty_legs() -> None:
    label_lookup = {"A": 0.01, "B": -0.02}

    long_short = _portfolio_return_components(
        portfolio_mode="long_short",
        positions={"A": 1.0},
        label_lookup=label_lookup,
    )

    assert long_short == pytest.approx((0.01, 0.0, 0.01, 1, 0))

    long_only = _portfolio_return_components(
        portfolio_mode="long_only",
        positions={"A": 1.0},
        label_lookup=label_lookup,
    )

    assert long_only == pytest.approx((0.01, 0.0, 0.01, 1, 0))


def test_dataframe_portfolio_return_components_keep_cash_row() -> None:
    components = _portfolio_return_components(
        portfolio_mode="long_short",
        positions={},
        label_lookup={},
    )

    assert components == (0.0, 0.0, 0.0, 0, 0)


@pytest.mark.parametrize(
    "label_lookup",
    [{}, {"A": float("nan")}, {"A": float("inf")}],
)
def test_dataframe_portfolio_return_rejects_nonfinite_active_return(
    label_lookup: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="nonfinite_active_position_return"):
        _portfolio_return_components(
            portfolio_mode="long_only",
            positions={"A": 1.0},
            label_lookup=label_lookup,
        )


def test_timestamp_label_lookup_keeps_validated_asset_ids() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="label")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["1", "2"],
            "label": [0.1, 0.2],
        }
    )

    assert _timestamp_label_lookup(frame, spec, "label") == {"1": 0.1, "2": 0.2}
