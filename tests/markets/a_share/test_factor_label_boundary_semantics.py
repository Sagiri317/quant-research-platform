from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    a_share_label_catalog,
)


def test_factor_boundary_nan_is_allowed_without_pre_window_history() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=280))

    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=factor_definitions.a_share_factor_catalog().list(),
    )

    first = factors.sort_values("timestamp").iloc[0]
    middle = factors.loc[factors["timestamp"].eq(pd.Timestamp("2014-02-12"))].iloc[0]
    assert pd.isna(first["mom_20d"])
    assert pd.notna(middle["mom_20d"])


def test_factor_middle_window_keeps_values_when_history_exists() -> None:
    panel = _factor_label_panel(pd.bdate_range("2013-11-01", periods=90))

    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=factor_definitions.a_share_factor_catalog().list(),
    )
    requested = factors.loc[
        factors["timestamp"].between(
            pd.Timestamp("2014-01-02"),
            pd.Timestamp("2014-01-31"),
        )
    ]

    assert not requested.empty
    assert requested["mom_20d"].notna().all()


def test_label_tail_nan_is_allowed_without_post_window_future() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=40))

    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=a_share_label_catalog().list(),
    )

    middle = labels.loc[labels["timestamp"].eq(pd.Timestamp("2014-01-15"))].iloc[0]
    tail = labels.sort_values("timestamp").iloc[-1]
    assert pd.notna(middle["forward_return_5d"])
    assert pd.isna(tail["forward_return_5d"])


def test_label_middle_window_is_not_artificially_truncated_when_future_exists() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=80))

    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=a_share_label_catalog().list(),
    )
    requested = labels.loc[
        labels["timestamp"].between(
            pd.Timestamp("2014-01-15"),
            pd.Timestamp("2014-02-14"),
        )
    ]

    assert not requested.empty
    assert requested["forward_return_20d"].notna().all()


def test_factor_calculation_fails_closed_for_missing_required_input_columns() -> None:
    panel = _factor_label_panel(pd.bdate_range("2014-01-01", periods=30)).drop(
        columns=["amount"]
    )

    with pytest.raises(ValueError, match="missing_price_volume_input_columns: amount"):
        factor_calculations.calculate_factors_for_panel(
            panel,
            definitions=factor_definitions.a_share_factor_catalog().list(),
        )


def _factor_label_panel(dates) -> pd.DataFrame:
    rows = []
    for offset, date in enumerate(pd.to_datetime(dates), start=1):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": pd.Timestamp(date),
                "adj_open": float(100 + offset),
                "adj_close": float(100 + offset),
                "return_1d": 0.001 * offset,
                "amount": 10000.0 + offset,
                "turnover": 0.01 + offset / 100000.0,
                "high": float(101 + offset),
                "low": float(99 + offset),
                "prev_close": float(99 + offset),
                "total_market_cap": float(1_000_000 + offset),
                "float_market_cap": float(500_000 + offset),
            }
        )
    return pd.DataFrame(rows)
