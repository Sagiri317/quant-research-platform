"""External observations preserve units, causal clocks, and complete windows."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_external_context_frames,
)


def test_external_units_clocks_windows_and_chunk_invariance():
    time = pd.date_range("2026-08-01T00:00:01", periods=10, freq="100ms")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": time,
            "segment_id": "a",
            "mid": 100.0,
        }
    )
    quotes = [
        {
            "local_timestamp": stamp.value // 1000,
            "timestamp": stamp.value // 1000 - 7000,
            "bid_price": 199.0 + i,
            "ask_price": 201.0 + i,
        }
        for i, stamp in enumerate(time)
    ]
    trades = [
        {
            "local_timestamp": time[1].value // 1000,
            "timestamp": time[0].value // 1000,
            "side": "buy",
            "price": 200.0,
            "amount": 3.0,
        },
        {
            "local_timestamp": time[3].value // 1000,
            "timestamp": time[0].value // 1000,
            "side": "sell",
            "price": 210.0,
            "amount": 1.0,
        },
    ]

    def compute(frames, quote_rows=quotes, trade_rows=trades, multiplier=1.0):
        return pd.concat(
            iter_external_context_frames(
                frames,
                quotes=quote_rows,
                trades=trade_rows,
                tickers=[],
                venue="bybit",
                base_units_per_amount=multiplier,
                max_age_ms=100,
            ),
            ignore_index=True,
        )

    actual = compute([frame])
    split = compute([frame.iloc[:3], frame.iloc[3:7], frame.iloc[7:]])
    pd.testing.assert_frame_equal(actual, split, check_exact=True)
    mature = actual.iloc[5]
    assert mature.bybit_return_500ms == pytest.approx(205 / 200 - 1)
    assert mature.bybit_observed_signed_base_flow_500ms == 2
    assert mature.bybit_observed_signed_notional_flow_500ms == 390
    assert mature.bybit_observed_tfi_500ms == 0.5
    assert actual.bybit_observed_tfi_500ms.iloc[:5].isna().all()
    assert pd.isna(actual.bybit_observed_tfi_500ms.iloc[9])
    assert mature.bybit_price_difference == pytest.approx(1.05)
    assert mature.bybit_relative_mid_dispersion == pytest.approx(105 / 305)
    contracts = [{**row, "amount": row["amount"] * 100} for row in trades]
    converted = compute([frame], trade_rows=contracts, multiplier=0.01)
    pd.testing.assert_frame_equal(actual, converted, check_exact=True)
    future = [
        {**row, "bid_price": 1000.0, "ask_price": 1002.0} if i >= 7 else row
        for i, row in enumerate(quotes)
    ]
    pd.testing.assert_frame_equal(
        actual.iloc[:7],
        compute([frame], quote_rows=future).iloc[:7],
        check_exact=True,
    )
    stale = compute([frame], quote_rows=quotes[:2])
    assert pd.isna(stale.bybit_mid_usdt.iloc[3])
    assert stale.bybit_quote_age_ms.iloc[3] == 200
    gap = frame.copy()
    gap.loc[5:, "segment_id"] = "b"
    reset = compute([gap])
    assert reset.bybit_return_500ms.iloc[5:].isna().all()
    assert reset.bybit_observed_tfi_500ms.iloc[5:].isna().all()
