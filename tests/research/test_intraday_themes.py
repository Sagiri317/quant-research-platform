"""Independent intraday path arithmetic and segment-local missingness."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.microstructure import compute_factors
from quant_research_platform.research.protocols.a_share_hft import (
    INTRADAY_THEME_FACTORS,
)


def theme_panel(prices):
    prices = np.asarray(prices, dtype=float)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "segment_id": "morning",
            "timestamp": pd.date_range(
                "2024-01-02T01:30:00", periods=len(prices), freq="3s"
            ),
            "mid": prices,
        }
    )
    for level in range(1, 11):
        frame[f"bid_px{level}"] = prices - 0.01 * level
        frame[f"ask_px{level}"] = prices + 0.01 * level
        frame[f"bid_vol{level}"] = 100
        frame[f"ask_vol{level}"] = 100
    for kind in ("trade", "order", "cancel"):
        frame[f"observed_{kind}_volume"] = 10.0
        frame[f"observed_signed_{kind}_volume"] = -5.0
    return frame


def test_intraday_reference_extrema_volume_and_sell_pressure_by_hand():
    frame = theme_panel(np.r_[np.linspace(10, 20, 100), 15])
    actual = compute_factors(frame, columns=tuple(INTRADAY_THEME_FACTORS))
    last = actual.iloc[-1]
    assert last.price_position_300s == pytest.approx(0.5)
    assert last.peak_drawdown_300s == pytest.approx(0.25)
    assert last.support_distance_300s == pytest.approx(0.5)
    assert last.resistance_distance_300s == pytest.approx(-0.25)
    assert last.trade_activity_ratio_30_300s == pytest.approx(1)
    assert last.drawdown_sell_pressure_300s == pytest.approx(0.125)
    frame.loc[50, "observed_signed_trade_volume"] = np.nan
    changed = compute_factors(frame, columns=tuple(INTRADAY_THEME_FACTORS))
    pd.testing.assert_series_equal(
        actual.trade_activity_ratio_30_300s, changed.trade_activity_ratio_30_300s
    )
    pd.testing.assert_series_equal(
        actual.peak_drawdown_300s, changed.peak_drawdown_300s
    )


def test_failed_breach_uses_frozen_pre_event_level_and_no_future_confirmation():
    prices = np.r_[np.full(101, 10.0), 11.0, np.full(10, 9.5)]
    frame = theme_panel(prices)
    columns = ("failed_high_break_300s", "failed_low_break_300s")
    actual = compute_factors(frame, columns=columns)
    assert actual.failed_high_break_300s.iloc[110] == pytest.approx(0.1)
    assert actual.failed_low_break_300s.iloc[110] == 0
    frame.loc[111, "mid"] = 100
    changed = compute_factors(frame, columns=columns)
    pd.testing.assert_frame_equal(actual.iloc[:111], changed.iloc[:111])
    missing = theme_panel(prices)
    missing.loc[105, "mid"] = np.nan
    unknown = compute_factors(missing, columns=columns)
    assert unknown.loc[110, list(columns)].isna().all()
    frame.loc[102:, "segment_id"] = "new_segment"
    reset = compute_factors(frame, columns=columns)
    assert reset.loc[102:, list(columns)].isna().all(axis=None)
