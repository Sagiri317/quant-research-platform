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


@pytest.mark.parametrize(
    "mode", ["no_gap", "future_gap", "other_asset_reset", "same_asset_reset"]
)
def test_native_epoch_invalidates_all_assets_only_after_known_clock(tmp_path, mode):
    import json

    from quant_research_platform.markets.implementations.crypto.processed import catalog
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.sources import open_market_source
    from quant_research_platform.sources.tardis.raw import STATE_CHANNELS, import_slices
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.sources.tardis.test_raw import _slices

    origin = pd.Timestamp("2026-09-01").value

    def line(milliseconds, symbol, channel, payload):
        stamp = (
            pd.Timestamp(origin + milliseconds * 1_000_000, tz="UTC")
            .isoformat()
            .replace("+00:00", "Z")
        )
        return (
            stamp
            + " "
            + json.dumps({"stream": symbol.lower() + "@" + channel, "data": payload})
            + "\n"
        ).encode()

    native = line(
        4500,
        "BTCUSDT",
        "forceOrder",
        {
            "e": "forceOrder",
            "E": origin // 1_000_000 + 4500,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "q": "2",
                "l": "2",
                "z": "2",
                "T": origin // 1_000_000 + 4500,
            },
        },
    )
    if mode != "no_gap":
        native += b"\n"
    next_ms = 5100 if mode == "future_gap" else 4600
    symbol = "BTCUSDT" if mode == "same_asset_reset" else "ETHUSDT"
    native += line(
        next_ms,
        symbol,
        "openInterest",
        {
            "symbol": symbol,
            "time": origin // 1_000_000 + next_ms,
            "openInterest": "10",
        },
    )
    native += line(
        6000,
        "BTCUSDT",
        "openInterest",
        {
            "symbol": "BTCUSDT",
            "time": origin // 1_000_000 + 6000,
            "openInterest": "10",
        },
    )
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-09-01", periods=51, freq="100ms"),
            "segment_id": "independent_continuous_source",
            "mid": 100.0,
        }
    )
    with runtime_resource_policy_scope() as policy:
        root = tmp_path / "archive"
        import_slices(
            root,
            date="2026-09-01",
            slices=_slices(tmp_path / "native", native, count=1),
            byte_budget=1024**2,
            resource_policy=policy,
            channels=STATE_CHANNELS,
        )
        with open_market_source(root, resource_policy=policy) as archive:
            frames = iter_derivative_factor_frames(
                [panel],
                observations=archive.iter_state_observations(symbol="BTCUSDT"),
                max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
            )
            spec = catalog.hft_processed_catalog().get(
                "binance_futures_derivative_factors_100ms"
            )
            layout = StorageLayout(tmp_path)
            output = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
            write_dataset(
                output, frames, spec=spec, resource_policy=policy, borrowed_input=True
            )
            result = read_dataset(output, resource_policy=policy)
            value = result.observed_force_order_imbalance_5s.iloc[-1]
            if mode in ("no_gap", "future_gap"):
                assert value == -1
            else:
                assert pd.isna(value)


@pytest.mark.parametrize("count", [200, 400])
def test_derivative_catchup_retains_no_expired_force_orders(monkeypatch, count):
    from collections import deque

    import quant_research_platform.markets.implementations.crypto.hft as hft

    peak = 0

    class TrackedDeque(deque):
        def append(self, value):
            nonlocal peak
            super().append(value)
            peak = max(peak, len(self))

    monkeypatch.setattr(hft, "deque", TrackedDeque)
    origin = pd.Timestamp("2026-09-01").value
    panel = pd.DataFrame(
        {
            "market_id": ["crypto"],
            "asset_id": ["btc_usdt"],
            "timestamp": [pd.Timestamp(origin)],
            "segment_id": ["s"],
            "mid": [100.0],
        }
    )
    events = [
        (
            origin - 31_000_000_000 - (count - 1 - i) * 1_000_000,
            "forceOrder",
            {
                "E": (origin - 31_000_000_000 - (count - 1 - i) * 1_000_000)
                // 1_000_000,
                "o": {"S": "SELL", "z": "2", "q": "100"},
            },
            0,
        )
        for i in range(count)
    ]
    result = next(
        iter_derivative_factor_frames(
            [panel],
            observations=iter(events),
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
        )
    )
    assert peak == 0
    assert result.time_since_observed_force_order.iloc[0] == 31
    assert result.observed_force_order_imbalance_30s.isna().all()
