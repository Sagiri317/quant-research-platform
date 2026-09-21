"""Causal state alignment, full windows, and observed-only liquidation quantities."""

import copy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_derivative_factor_frames,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_CHANNELS,
)


def test_derivative_oracle_windows_age_chunks_and_causal_observations():
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-08-01", periods=3003, freq="100ms"),
            "segment_id": ["a"] * 3002 + ["b"],
            "mid": 100.0,
        }
    )
    origin = panel.timestamp.iloc[0].value
    events = [
        (
            origin,
            "premiumIndex",
            dict(
                markPrice="102",
                indexPrice="101",
                lastFundingRate="0.0001",
                nextFundingTime=origin // 1000000 + 3600000,
                time=origin // 1000000 - 10,
            ),
            0,
        ),
        (origin, "openInterest", dict(openInterest="10", time=origin // 1000000), 0),
        (
            origin + 4500000000,
            "forceOrder",
            {"E": origin // 1000000 + 4400, "o": {"S": "SELL", "z": "2", "q": "100"}},
            0,
        ),
        (
            origin + 4600000000,
            "forceOrder",
            {"E": origin // 1000000 + 4500, "o": {"S": "BUY", "z": "6", "q": "1"}},
            0,
        ),
        (
            origin + 30000000000,
            "openInterest",
            dict(openInterest="20", time=origin // 1000000 + 29900),
            0,
        ),
    ]
    ages = dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000)
    ages["premiumIndex"] = 5000

    def compute(frames, observations=events):
        return pd.concat(
            iter_derivative_factor_frames(
                frames, observations=iter(observations), max_age_ms=ages
            ),
            ignore_index=True,
        )

    whole = compute([panel])
    split = compute([panel.iloc[i : i + 17] for i in range(0, len(panel), 17)])
    pd.testing.assert_frame_equal(whole, split, check_exact=True)
    assert whole.mark_mid_basis.iloc[0] == pytest.approx(0.02)
    assert whole.mark_index_basis.iloc[0] == pytest.approx(102 / 101 - 1)
    assert whole.perp_index_basis.iloc[0] == pytest.approx(100 / 101 - 1)
    assert whole.funding_rate.iloc[0] == 0.0001
    assert whole.time_to_next_funding.iloc[50] == 3595
    assert whole.mark_mid_basis.iloc[51:].isna().all()
    assert whole.premiumIndex_age_ms.iloc[51] == 5100
    assert whole.premiumIndex_observed_at.iloc[0] == panel.timestamp.iloc[0]
    assert whole.premiumIndex_event_at.iloc[0] == panel.timestamp.iloc[
        0
    ] - pd.Timedelta(milliseconds=10)
    assert whole.log_open_interest.iloc[300] == pytest.approx(np.log(20))
    for column, first in (
        ("open_interest_change_30s", 300),
        ("open_interest_change_1m", 600),
        ("open_interest_change_5m", 3000),
    ):
        assert whole[column].iloc[:first].isna().all()
        assert whole[column].iloc[first] == 1
        assert pd.isna(whole[column].iloc[-1])
    assert whole.observed_force_order_imbalance_5s.iloc[:49].isna().all()
    assert whole.observed_force_order_imbalance_5s.iloc[49] == 0.5
    assert whole.time_since_observed_force_order.iloc[49] == 0.3
    assert pd.isna(whole.observed_force_order_imbalance_5s.iloc[100])
    changed = copy.deepcopy(events)
    changed[3][2]["o"]["z"] = "600"
    pd.testing.assert_frame_equal(
        whole.iloc[:46], compute([panel], changed).iloc[:46], check_exact=True
    )
    gap = panel.iloc[:700].copy()
    gap.loc[200:, "segment_id"] = "after_gap"
    reset = compute([gap])
    assert reset.open_interest_change_30s.iloc[200:600].isna().all()
