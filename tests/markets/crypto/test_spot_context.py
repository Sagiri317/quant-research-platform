"""Spot information is causal and uses the existing microstructure mathematics."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_spot_perp_factor_frames,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_SPOT_FACTOR_COLUMNS,
)


def panel(*, spot=False):
    time = pd.date_range("2026-08-01T00:00:01", periods=20, freq="100ms")
    return pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": time,
            "observed_utc": time.tz_localize("UTC") - pd.Timedelta(milliseconds=1),
            "segment_id": "a",
            "mid": 200 + np.arange(20, dtype=float) if spot else 100.0,
            "book_spread": 2.0,
            "bid_depth_1": 6.0 if spot else 3.0,
            "ask_depth_1": 2.0 if spot else 1.0,
            "bid_depth_5": 12.0,
            "ask_depth_5": 8.0,
            "bid_depth_10": 24.0,
            "ask_depth_10": 16.0,
            "book_ofi_100ms": 4.0 if spot else 2.0,
            "observed_signed_trade_volume_100ms": 2.0,
            "observed_taker_buy_volume_100ms": 5.0 if spot else 3.0,
            "observed_taker_sell_volume_100ms": 3.0 if spot else 1.0,
        }
    )


def compute(perpetual, spot):
    return pd.concat(
        iter_spot_perp_factor_frames(
            perpetual, spot_frames=spot, max_age_ms=200, batch_rows=7
        ),
        ignore_index=True,
    )


def test_spot_oracle_chunks_gaps_missing_source_and_future_invariance():
    perpetual, spot = (panel(), panel(spot=True))
    whole = compute([perpetual], [spot])
    split = compute(
        [perpetual.iloc[i : i + 3] for i in range(0, 20, 3)],
        [spot.iloc[i : i + 4] for i in range(0, 20, 4)],
    )
    pd.testing.assert_frame_equal(whole, split, check_exact=True)
    at = whole.iloc[5]
    assert at.spot_perp_basis == pytest.approx(205 / 100 - 1)
    assert at.spot_return_100ms == pytest.approx(205 / 204 - 1)
    assert at.spot_return_500ms == pytest.approx(0.025)
    assert at.spot_ofi_500ms == 20
    assert at.spot_tfi_500ms == 0.25
    assert at.spot_perp_return_divergence_500ms == pytest.approx(0.025)
    assert at.spot_perp_normalized_ofi_divergence_500ms == 0
    assert at.spot_perp_tfi_divergence_500ms == -0.25
    assert at.lagged_spot_return_100ms == pytest.approx(204 / 203 - 1)
    assert at.lagged_spot_ofi_500ms == 20
    assert at.lagged_spot_tfi_500ms == 0.25
    assert at.spot_book_age_ms == 1
    assert whole.spot_return_500ms.iloc[:5].isna().all()
    changed = spot.copy()
    changed.loc[8:, "mid"] *= 2
    pd.testing.assert_frame_equal(
        whole.iloc[:8], compute([perpetual], [changed]).iloc[:8], check_exact=True
    )
    gap = spot.drop(index=range(8, 14)).copy()
    gap.loc[14:, "segment_id"] = "after_gap"
    limited = compute([perpetual], [gap])
    assert limited.timestamp.equals(whole.timestamp)
    assert limited.spot_grid_age_ms.iloc[10] == 300
    assert limited.loc[10:13, list(HFT_SPOT_FACTOR_COLUMNS)].isna().all().all()
    assert pd.isna(limited.lagged_spot_ofi_500ms.iloc[14])
    assert limited.spot_return_500ms.iloc[14:19].isna().all()
    missing = compute([perpetual], [])
    assert missing.timestamp.equals(whole.timestamp)
    assert missing.loc[:, list(HFT_SPOT_FACTOR_COLUMNS)].isna().all().all()
    invalid = spot.copy()
    invalid["observed_utc"] += pd.Timedelta(milliseconds=2)
    with pytest.raises(ValueError, match="observed_after_grid"):
        compute([perpetual], [invalid])
