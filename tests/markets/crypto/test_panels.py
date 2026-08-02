from __future__ import annotations

import importlib
import importlib.util
import math
import warnings
import weakref

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.processed import (
    panels,
)


def compute_market_panel(
    bars: pd.DataFrame,
    *,
    trades: pd.DataFrame,
    depth: pd.DataFrame,
    max_quote_age: pd.Timedelta = panels.DEFAULT_MAX_QUOTE_AGE,
) -> pd.DataFrame:
    prepared = panels.prepare_panel_bars(bars)
    try:
        ordered_depth = (
            depth
            if depth.empty
            else depth.sort_values(
                ["source_asset_id", "timestamp", "side", "level"],
                kind="mergesort",
            ).reset_index(drop=True)
        )
    except TypeError:
        ordered_depth = depth
    test_budget = 512 * 1024 * 1024
    return panels.calculate_market_panel_from_chunks(
        prepared,
        trade_chunks=(trades,),
        depth_chunks=(ordered_depth,),
        payload_target_bytes=test_budget,
        max_quote_age=max_quote_age,
    )


def test_panel_releases_trade_reducer_state_before_depth_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_reference: weakref.ReferenceType[object] | None = None

    class TradeReducer:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal trade_reference
            trade_reference = weakref.ref(self)

        def add_chunk(self, _chunk: pd.DataFrame) -> None:
            pass

        def finish(self) -> pd.DataFrame:
            return pd.DataFrame()

    class DepthReducer:
        def __init__(self, **_kwargs: object) -> None:
            assert trade_reference is not None
            assert trade_reference() is None

        def add_chunk(self, _chunk: pd.DataFrame) -> None:
            pass

        def finish(self) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        panels.panel_input_reducers,
        "AggregateTradeReducer",
        TradeReducer,
    )
    monkeypatch.setattr(
        panels.panel_input_reducers,
        "DepthSnapshotReducer",
        DepthReducer,
    )
    monkeypatch.setattr(
        panels,
        "_calculate_market_panel",
        lambda panel, **_kwargs: panel,
    )

    output = panels.calculate_market_panel_from_chunks(
        pd.DataFrame({"market_id": ["crypto"]}),
        trade_chunks=(),
        depth_chunks=(),
        payload_target_bytes=1,
    )

    assert output["market_id"].tolist() == ["crypto"]


class _StringLikeCryptoSymbol:
    def __str__(self) -> str:
        return "BTCUSDT"


def test_compute_market_panel_canonicalizes_source_asset_and_asof_quote() -> None:
    bars = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "open_time": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "close_time": pd.to_datetime(
                ["2026-07-06 00:00:59.999", "2026-07-06 00:01:59.999"]
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "quote_volume": [1005.0, 1116.5],
            "trade_count": [1, 1],
            "taker_buy_volume": [4.0, 5.0],
            "taker_buy_quote_volume": [402.0, 507.5],
            "source_market_type": ["spot", "spot"],
            "interval": ["1m", "1m"],
        }
    )
    trades = _aggregate_trades_frame(
        quantities=(10.0,),
        signed_quantities=(10.0,),
        aggregate_trade_ids=(1,),
        first_trade_ids=(1,),
        last_trade_ids=(1,),
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00:58",
                    "2026-07-06 00:00:58",
                    "2026-07-06 00:02:01",
                    "2026-07-06 00:02:01",
                ]
            ),
            "side": ["bid", "ask", "bid", "ask"],
            "level": [1, 1, 1, 1],
            "price": [100.0, 100.2, 101.0, 101.2],
            "quantity": [1.5, 1.0, 2.0, 2.0],
            "source_market_type": ["spot", "spot", "spot", "spot"],
        }
    )

    panel = compute_market_panel(
        bars,
        trades=trades,
        depth=depth,
        max_quote_age=pd.Timedelta(seconds=30),
    )

    assert panel["asset_id"].tolist() == ["btc_usdt", "btc_usdt"]
    assert "source_asset_id" not in panel.columns
    assert bool(panel.loc[0, "quote_observed"]) is True
    assert panel.loc[0, "quote_age_seconds"] == 1.999
    assert panel.loc[0, "best_bid"] == 100.0
    assert bool(panel.loc[0, "trade_flow_observed"]) is True
    assert bool(panel.loc[1, "quote_observed"]) is False
    assert bool(panel.loc[1, "trade_flow_observed"]) is False
    assert math.isnan(panel.loc[1, "best_bid"])


def test_compute_market_panel_distinguishes_missing_trade_feed_from_zero_flow() -> None:
    bars = pd.DataFrame(
        {
            "source_asset_id": ["ETHUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "open_time": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "close_time": pd.to_datetime(
                ["2026-07-06 00:00:59.999", "2026-07-06 00:01:59.999"]
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [0.0, 10.0],
            "quote_volume": [0.0, 1015.0],
            "trade_count": [0, 2],
            "taker_buy_volume": [0.0, 5.0],
            "taker_buy_quote_volume": [0.0, 507.5],
            "source_market_type": ["spot", "spot"],
            "interval": ["1m", "1m"],
        }
    )

    panel = compute_market_panel(
        bars,
        trades=pd.DataFrame(
            columns=["source_asset_id", "timestamp", "signed_quantity"],
        ),
        depth=pd.DataFrame(),
    )

    assert bool(panel.loc[0, "trade_flow_observed"]) is True
    assert panel.loc[0, "signed_volume"] == 0.0
    assert bool(panel.loc[1, "trade_flow_observed"]) is False
    assert math.isnan(panel.loc[1, "signed_volume"])


def test_complete_trade_feed_is_observed_only_after_summary_validation() -> None:
    bars = _bars_frame()
    bars.loc[0, "trade_count"] = 2

    panel = compute_market_panel(
        bars,
        trades=_aggregate_trades_frame(),
        depth=pd.DataFrame(),
    )

    assert panel.loc[0, "signed_volume"] == 2.0
    assert bool(panel.loc[0, "trade_flow_observed"]) is True


@pytest.mark.parametrize(
    ("quantities", "first_trade_ids", "last_trade_ids"),
    [
        ((5.0, 4.0), (10, 11), (10, 11)),
        ((6.0, 4.0), (10, 11), (10, 12)),
    ],
)
def test_partial_trade_feed_fails_closed(
    quantities: tuple[float, ...],
    first_trade_ids: tuple[int, ...],
    last_trade_ids: tuple[int, ...],
) -> None:
    bars = _bars_frame()
    bars.loc[0, "trade_count"] = 2

    with pytest.raises(ValueError, match="invalid_crypto_trade_flow_coverage"):
        compute_market_panel(
            bars,
            trades=_aggregate_trades_frame(
                quantities=quantities,
                signed_quantities=quantities,
                first_trade_ids=first_trade_ids,
                last_trade_ids=last_trade_ids,
            ),
            depth=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    ("aggregate_trade_ids", "first_trade_ids", "last_trade_ids", "error"),
    [
        ((1, 2), (10, 12), (10, 12), "invalid_crypto_trade_flow_coverage"),
        ((1, 2), (10, 11), (11, 11), "invalid_crypto_trade_flow_coverage"),
        ((1, 2), (11, 11), (10, 11), "invalid_crypto_trade_flow_coverage"),
        ((1, 1), (10, 11), (10, 11), "out_of_order_crypto_trade_dependency"),
        ((2, 1), (10, 11), (10, 11), "out_of_order_crypto_trade_dependency"),
    ],
)
def test_trade_id_contract_fails_closed(
    aggregate_trade_ids: tuple[int, ...],
    first_trade_ids: tuple[int, ...],
    last_trade_ids: tuple[int, ...],
    error: str,
) -> None:
    bars = _bars_frame()
    bars.loc[0, "trade_count"] = 2

    with pytest.raises(ValueError, match=error):
        compute_market_panel(
            bars,
            trades=_aggregate_trades_frame(
                aggregate_trade_ids=aggregate_trade_ids,
                first_trade_ids=first_trade_ids,
                last_trade_ids=last_trade_ids,
            ),
            depth=pd.DataFrame(),
        )


def test_zero_trade_bar_rejects_unexpected_trade_feed() -> None:
    bars = _bars_frame()
    bars.loc[0, ["volume", "quote_volume", "trade_count"]] = 0
    bars.loc[0, ["taker_buy_volume", "taker_buy_quote_volume"]] = 0

    with pytest.raises(ValueError, match="invalid_crypto_trade_flow_coverage"):
        compute_market_panel(
            bars,
            trades=_aggregate_trades_frame(
                quantities=(1.0,),
                signed_quantities=(1.0,),
                aggregate_trade_ids=(1,),
                first_trade_ids=(1,),
                last_trade_ids=(1,),
            ),
            depth=pd.DataFrame(),
        )


@pytest.mark.parametrize(
    ("volume", "trade_count"),
    [(0.0, 1), (1.0, 0)],
)
def test_inconsistent_bar_volume_and_count_fails_closed(
    volume: float,
    trade_count: int,
) -> None:
    bars = _bars_frame()
    bars.loc[0, "volume"] = volume
    bars.loc[0, "trade_count"] = trade_count
    bars.loc[0, "taker_buy_volume"] = 0.0

    with pytest.raises(ValueError, match="invalid_crypto_bar_trade_summary"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_one_sided_depth_maps_to_quote_observed_false() -> None:
    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"],
                "timestamp": pd.to_datetime(["2026-07-06 00:00:58"]),
                "side": ["bid"],
                "level": [1],
                "price": [100.0],
                "quantity": [1.5],
                "source_market_type": ["spot"],
            }
        ),
    )

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_one_sided_quote_fields_do_not_set_quote_observed_true() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58"]),
            "side": ["bid"],
            "level": [1],
            "price": [100.0],
            "quantity": [1.5],
            "source_market_type": ["spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_crossed_or_invalid_quote_sets_quote_observed_false() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.2, 100.0],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "mid_price"])


def test_quote_after_close_not_used() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:01:00", "2026-07-06 00:01:00"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_quote_matching_allows_exact_close_time() -> None:
    close_time = _bars_frame().loc[0, "close_time"]
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": [close_time, close_time],
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is True
    assert panel.loc[0, "quote_age_seconds"] == 0.0


def test_latest_invalid_quote_does_not_fall_back_to_older_valid_quote() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00:50",
                    "2026-07-06 00:00:50",
                    "2026-07-06 00:00:58",
                    "2026-07-06 00:00:58",
                ]
            ),
            "side": ["bid", "ask", "bid", "ask"],
            "level": [1, 1, 1, 1],
            "price": [100.0, 100.2, 100.3, 100.1],
            "quantity": [1.5, 1.0, 1.5, 1.0],
            "source_market_type": ["spot"] * 4,
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_quote_matching_isolates_assets() -> None:
    bars = pd.concat(
        (
            _bars_frame(),
            _bars_frame().assign(source_asset_id="ETHUSDT"),
        ),
        ignore_index=True,
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58"] * 4),
            "side": ["bid", "ask", "bid", "ask"],
            "level": [1, 1, 1, 1],
            "price": [100.0, 100.2, 200.0, 200.4],
            "quantity": [1.5, 1.0, 2.5, 2.0],
            "source_market_type": ["spot"] * 4,
        }
    )

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=depth)

    assert panel.set_index("asset_id")["best_bid"].to_dict() == {
        "btc_usdt": 100.0,
        "eth_usdt": 200.0,
    }


def test_depth_reducer_merges_snapshot_across_chunks() -> None:
    module = _panel_input_reducers_module()
    depth = _ordered_depth_levels_frame()
    reducer = module.DepthSnapshotReducer(
        bars=panels.prepare_panel_bars(_bars_frame()),
        payload_target_bytes=16 * 1024,
        normalize_chunk=panels.normalize_depth_chunk,
    )

    reducer.add_chunk(depth.iloc[:2].copy())
    reducer.add_chunk(depth.iloc[2:].copy())
    quotes = reducer.finish()

    assert quotes.loc[0, "best_bid"] == 100.0
    assert quotes.loc[0, "best_ask"] == 100.2
    assert quotes.loc[0, "bid_depth"] == 2.5
    assert quotes.loc[0, "ask_depth"] == 2.0

    assert len(reducer._candidates) == 1
    assert sum(len(levels) for levels in reducer._current_levels.values()) == 0


def test_depth_reducer_rejects_out_of_order_across_chunks() -> None:
    module = _panel_input_reducers_module()
    depth = _ordered_depth_levels_frame()
    reducer = module.DepthSnapshotReducer(
        bars=panels.prepare_panel_bars(_bars_frame()),
        payload_target_bytes=16 * 1024,
        normalize_chunk=panels.normalize_depth_chunk,
    )

    reducer.add_chunk(depth.iloc[2:].copy())

    with pytest.raises(ValueError, match="out_of_order_crypto_depth_dependency"):
        reducer.add_chunk(depth.iloc[:2].copy())


@pytest.mark.parametrize("split_chunks", [False, True])
def test_market_panel_rejects_out_of_order_depth_independent_of_chunking(
    split_chunks: bool,
) -> None:
    reversed_depth = _ordered_depth_levels_frame().iloc[::-1].reset_index(drop=True)
    depth_chunks = (
        (reversed_depth.iloc[:2].copy(), reversed_depth.iloc[2:].copy())
        if split_chunks
        else (reversed_depth,)
    )

    with pytest.raises(ValueError, match="out_of_order_crypto_depth_dependency"):
        panels.calculate_market_panel_from_chunks(
            panels.prepare_panel_bars(_bars_frame()),
            trade_chunks=(),
            depth_chunks=depth_chunks,
            payload_target_bytes=16 * 1024,
        )


def test_depth_reducer_high_density_state_is_bounded_by_bars() -> None:
    module = _panel_input_reducers_module()
    snapshot_count = 200
    timestamps = pd.date_range(
        "2026-07-06 00:00:00",
        periods=snapshot_count,
        freq="100ms",
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * snapshot_count * 2,
            "timestamp": [value for value in timestamps for _ in range(2)],
            "side": [side for _ in timestamps for side in ("ask", "bid")],
            "level": [1] * snapshot_count * 2,
            "price": [price for _ in timestamps for price in (100.2, 100.0)],
            "quantity": [1.0] * snapshot_count * 2,
            "source_market_type": ["spot"] * snapshot_count * 2,
        }
    )
    reducer = module.DepthSnapshotReducer(
        bars=panels.prepare_panel_bars(_bars_frame()),
        payload_target_bytes=64 * 1024,
        normalize_chunk=panels.normalize_depth_chunk,
    )

    for start in range(0, len(depth), 17):
        reducer.add_chunk(depth.iloc[start : start + 17].copy())
    reducer.finish()

    assert len(reducer._candidates) <= len(_bars_frame())
    assert sum(len(levels) for levels in reducer._current_levels.values()) == 0


def test_trade_aggregate_state_reduces_dense_chunks_incrementally() -> None:
    module = _panel_input_reducers_module()
    trade_count = 2_000
    bars = _bars_frame()
    bars.loc[0, "volume"] = float(trade_count)
    bars.loc[0, "trade_count"] = trade_count
    bars.loc[0, "taker_buy_volume"] = float(trade_count // 2)
    trades = _aggregate_trades_frame(
        quantities=(1.0,) * trade_count,
        signed_quantities=tuple(
            -1.0 if index % 2 else 1.0 for index in range(trade_count)
        ),
        aggregate_trade_ids=tuple(range(1, trade_count + 1)),
        first_trade_ids=tuple(range(10_000, 10_000 + trade_count)),
        last_trade_ids=tuple(range(10_000, 10_000 + trade_count)),
        timestamps=tuple(
            pd.date_range(
                "2026-07-06 00:00:01",
                periods=trade_count,
                freq="ms",
            )
        ),
    )
    reducer = module.AggregateTradeReducer(
        bars=panels.prepare_panel_bars(bars),
        payload_target_bytes=64 * 1024,
        normalize_chunk=panels.normalize_trade_chunk,
    )

    reducer.add_chunk(trades.iloc[:1_000].copy())
    reducer.add_chunk(trades.iloc[1_000:].copy())
    aggregates = reducer.finish()

    assert aggregates["signed_volume"].tolist() == [0.0]
    assert aggregates["quantity_sum"].tolist() == [2_000.0]
    assert aggregates["covered_trade_count"].tolist() == [2_000]
    assert aggregates["trade_flow_observed"].tolist() == [True]


def test_trade_reducer_checks_range_continuity_across_chunks() -> None:
    module = _panel_input_reducers_module()
    bars = _bars_frame()
    bars.loc[0, "trade_count"] = 2
    trades = _aggregate_trades_frame(first_trade_ids=(10, 12), last_trade_ids=(10, 12))
    reducer = module.AggregateTradeReducer(
        bars=panels.prepare_panel_bars(bars),
        payload_target_bytes=16 * 1024,
        normalize_chunk=panels.normalize_trade_chunk,
    )

    reducer.add_chunk(trades.iloc[:1].copy())

    with pytest.raises(ValueError, match="invalid_crypto_trade_flow_coverage"):
        reducer.add_chunk(trades.iloc[1:].copy())


def test_trade_reducer_checks_range_continuity_across_bar_reducers() -> None:
    module = _panel_input_reducers_module()
    continuity = module.AggregateTradeContinuityState()
    bars = _bars_frame()
    trades = _aggregate_trades_frame(
        first_trade_ids=(10, 12),
        last_trade_ids=(10, 12),
        timestamps=(
            pd.Timestamp("2026-07-06 00:00:01"),
            pd.Timestamp("2026-07-06 00:01:01"),
        ),
    )
    first = module.AggregateTradeReducer(
        bars=panels.prepare_panel_bars(bars.iloc[:1].copy()),
        payload_target_bytes=16 * 1024,
        normalize_chunk=panels.normalize_trade_chunk,
        continuity_state=continuity,
    )
    first.add_chunk(trades.iloc[:1].copy())
    first.finish()
    second_bars = bars.iloc[:1].copy()
    second_bars["timestamp"] = pd.Timestamp("2026-07-06 00:01:00")
    second_bars["open_time"] = pd.Timestamp("2026-07-06 00:01:00")
    second_bars["close_time"] = pd.Timestamp("2026-07-06 00:01:59.999")
    second = module.AggregateTradeReducer(
        bars=panels.prepare_panel_bars(second_bars),
        payload_target_bytes=16 * 1024,
        normalize_chunk=panels.normalize_trade_chunk,
        continuity_state=continuity,
    )

    with pytest.raises(ValueError, match="invalid_crypto_trade_flow_coverage"):
        second.add_chunk(trades.iloc[1:].copy())


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        (
            "timestamp",
            pd.Timestamp("2026-07-06 00:01:00"),
            "invalid_crypto_bar_open_time",
        ),
        (
            "open_time",
            pd.Timestamp("2026-07-06 00:00:01"),
            "invalid_crypto_bar_open_time",
        ),
        (
            "close_time",
            pd.Timestamp("2026-07-06 00:01:00"),
            "invalid_crypto_bar_close_time",
        ),
        ("interval", "5m", "invalid_crypto_bar_interval"),
        ("source_market_type", "futures", "invalid_crypto_bar_source_market_type"),
    ],
)
def test_invalid_one_minute_bar_contract_fails_closed(
    column: str,
    value: object,
    error: str,
) -> None:
    bars = _bars_frame()
    bars[column] = bars[column].astype(object)
    bars.at[0, column] = value

    with pytest.raises(ValueError, match=error):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_bar_close_crossing_future_minute_fails_before_quote_matching() -> None:
    bars = _bars_frame()
    bars.loc[0, "close_time"] = pd.Timestamp("2026-07-06 00:01:59")
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:01:30", "2026-07-06 00:01:30"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_bar_close_time"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=depth)


def test_bar_timestamp_and_open_time_must_be_minute_aligned() -> None:
    bars = _bars_frame()
    bars.loc[0, "timestamp"] = pd.Timestamp("2026-07-06 00:00:01")
    bars.loc[0, "open_time"] = pd.Timestamp("2026-07-06 00:00:01")
    bars.loc[0, "close_time"] = pd.Timestamp("2026-07-06 00:01:00.999")

    with pytest.raises(ValueError, match="invalid_crypto_bar_open_time"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_stale_quote_sets_quote_observed_false() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:10", "2026-07-06 00:00:10"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=depth,
        max_quote_age=pd.Timedelta(seconds=10),
    )

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "quote_age_seconds"])


def test_invalid_ohlc_sets_eligible_false() -> None:
    bars = _bars_frame()
    bars.loc[0, "high"] = 99.0

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False


def test_invalid_bar_quantity_sets_eligible_false_and_masks_taker_sell_volume() -> None:
    bars = _bars_frame()
    bars.loc[0, "taker_buy_volume"] = 11.0

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False
    assert math.isnan(panel.loc[0, "taker_sell_volume"])


def test_invalid_signed_trade_quantity_fails_closed() -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": ["bad"],
            "source_market_type": ["spot"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_signed_quantity"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


def test_zero_quantity_quote_is_not_observed() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [0.0, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_invalid_depth_inputs_fail_closed() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "bid"],
            "level": [1, 1],
            "price": [100.0, 99.9],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    with pytest.raises(ValueError, match="duplicate_crypto_depth_level"):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


def test_invalid_bar_timestamp_fails_closed() -> None:
    bars = pd.DataFrame(
        {
            **_bars_frame().to_dict(orient="list"),
            "timestamp": ["bad"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match="invalid_crypto_panel_timestamp"):
            compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", "bad"),
        ("high", True),
        ("open", float("inf")),
        ("low", float("-inf")),
        ("close", np.nan),
    ],
)
def test_invalid_bar_ohlc_fails_closed(column: str, value: object) -> None:
    bars = _bars_frame()
    bars[column] = bars[column].astype(object)
    bars.loc[0, column] = value

    with pytest.raises(ValueError, match="invalid_crypto_ohlc"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("volume", "bad"),
        ("taker_buy_volume", True),
        ("volume", float("inf")),
        ("quote_volume", float("-inf")),
        ("taker_buy_quote_volume", np.nan),
    ],
)
def test_invalid_bar_quantity_inputs_fail_closed(
    column: str,
    value: object,
) -> None:
    bars = _bars_frame()
    bars[column] = bars[column].astype(object)
    bars.loc[0, column] = value

    with pytest.raises(ValueError, match="invalid_crypto_bar_quantity"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_bool_trade_count_fails_closed() -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = True

    with pytest.raises(ValueError, match="invalid_crypto_trade_count"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_infinite_trade_count_fails_closed() -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = float("inf")

    with pytest.raises(ValueError, match="invalid_crypto_trade_count"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize("trade_count", [1.5, -1])
def test_non_integer_or_negative_trade_count_sets_eligible_false(
    trade_count: object,
) -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = trade_count

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("price", "bad", "invalid_crypto_depth_price"),
        ("price", pd.NA, "invalid_crypto_depth_price"),
        ("price", np.nan, "invalid_crypto_depth_price"),
        ("price", float("inf"), "invalid_crypto_depth_price"),
        ("price", True, "invalid_crypto_depth_price"),
        ("quantity", "bad", "invalid_crypto_depth_quantity"),
        ("quantity", pd.NA, "invalid_crypto_depth_quantity"),
        ("quantity", np.nan, "invalid_crypto_depth_quantity"),
        ("quantity", float("inf"), "invalid_crypto_depth_quantity"),
        ("quantity", True, "invalid_crypto_depth_quantity"),
    ],
)
def test_invalid_depth_price_quantity_fail_closed(
    column: str,
    value: object,
    error: str,
) -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )
    depth[column] = depth[column].astype(object)
    depth.loc[0, column] = value

    with pytest.raises(ValueError, match=error):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


@pytest.mark.parametrize("signed_quantity", [True, np.nan, "bad", float("inf")])
def test_invalid_signed_trade_quantity_edge_cases_fail_closed(
    signed_quantity: object,
) -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [signed_quantity],
            "source_market_type": ["spot"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_signed_quantity"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


@pytest.mark.parametrize("side", [["bid"], {}, None, "buy"])
def test_invalid_depth_side_values_fail_closed(side: object) -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": [side, "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_depth_side"):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


def test_valid_depth_side_values_still_pass() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": ["spot", "spot"],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is True


@pytest.mark.parametrize(
    ("side", "levels", "prices", "error"),
    [
        ("bid", [1, 3], [100.0, 99.9], "invalid_crypto_depth_level_sequence"),
        ("ask", [2], [100.2], "invalid_crypto_depth_level_sequence"),
        ("bid", [1, 2], [100.0, 100.1], "invalid_crypto_bid_price_order"),
        ("ask", [1, 2], [100.2, 100.1], "invalid_crypto_ask_price_order"),
    ],
)
def test_depth_level_sequence_and_price_order_fail_closed(
    side: str,
    levels: list[int],
    prices: list[float],
    error: str,
) -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * len(levels),
            "timestamp": [pd.Timestamp("2026-07-06 00:00:58")] * len(levels),
            "side": [side] * len(levels),
            "level": levels,
            "price": prices,
            "quantity": [0.0] * len(levels),
            "source_market_type": ["spot"] * len(levels),
        }
    )

    with pytest.raises(ValueError, match=error):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


@pytest.mark.parametrize("frame_name", ["trades", "depth"])
def test_auxiliary_source_market_type_must_be_spot(frame_name: str) -> None:
    trades = _aggregate_trades_frame(
        quantities=(10.0,),
        signed_quantities=(10.0,),
        aggregate_trade_ids=(1,),
        first_trade_ids=(1,),
        last_trade_ids=(1,),
    )
    trades["source_market_type"] = "futures" if frame_name == "trades" else "spot"
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
            "source_market_type": [
                "futures" if frame_name == "depth" else "spot",
                "futures" if frame_name == "depth" else "spot",
            ],
        }
    )
    error = {
        "trades": "invalid_crypto_trade_source_market_type",
        "depth": "invalid_crypto_depth_source_market_type",
    }[frame_name]

    with pytest.raises(ValueError, match=error):
        compute_market_panel(_bars_frame(), trades=trades, depth=depth)


def test_invalid_trade_source_asset_fails_closed() -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["unknown"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [1.0],
            "source_market_type": ["spot"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_source_asset_id"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


@pytest.mark.parametrize(
    "source_asset_id",
    [1, 1.0, True, [], {}, set(), None, "", _StringLikeCryptoSymbol()],
)
def test_invalid_panel_source_asset_id_values_fail_closed(
    source_asset_id: object,
) -> None:
    bars = _bars_frame()
    bars["source_asset_id"] = bars["source_asset_id"].astype(object)
    bars.at[0, "source_asset_id"] = source_asset_id

    with pytest.raises(ValueError, match="invalid_crypto_source_asset_id"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_missing_observation_flags_are_bool_without_downcast_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        panel = compute_market_panel(
            _bars_frame(),
            trades=pd.DataFrame(),
            depth=pd.DataFrame(),
        )

    assert panel["quote_observed"].dtype == "bool"
    assert panel["trade_flow_observed"].dtype == "bool"
    assert panel["quote_observed"].tolist() == [False]
    assert panel["trade_flow_observed"].tolist() == [False]


def test_quote_observed_and_trade_flow_observed_do_not_control_eligible() -> None:
    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=pd.DataFrame(),
    )

    assert bool(panel.loc[0, "eligible"]) is True
    assert bool(panel.loc[0, "quote_observed"]) is False
    assert bool(panel.loc[0, "trade_flow_observed"]) is False


def _bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00"]),
            "open_time": pd.to_datetime(["2026-07-06 00:00"]),
            "close_time": pd.to_datetime(["2026-07-06 00:00:59.999"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
            "quote_volume": [1005.0],
            "trade_count": [1],
            "taker_buy_volume": [4.0],
            "taker_buy_quote_volume": [402.0],
            "source_market_type": ["spot"],
            "interval": ["1m"],
        }
    )


def _aggregate_trades_frame(
    *,
    quantities: tuple[float, ...] = (6.0, 4.0),
    signed_quantities: tuple[float, ...] = (6.0, -4.0),
    aggregate_trade_ids: tuple[int, ...] = (1, 2),
    first_trade_ids: tuple[int, ...] = (10, 11),
    last_trade_ids: tuple[int, ...] = (10, 11),
    timestamps: tuple[pd.Timestamp, ...] | None = None,
) -> pd.DataFrame:
    size = len(quantities)
    if timestamps is None:
        timestamps = tuple(
            pd.Timestamp("2026-07-06 00:00:01") + pd.Timedelta(milliseconds=index)
            for index in range(size)
        )
    return pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * size,
            "timestamp": list(timestamps),
            "aggregate_trade_id": list(aggregate_trade_ids),
            "quantity": list(quantities),
            "first_trade_id": list(first_trade_ids),
            "last_trade_id": list(last_trade_ids),
            "signed_quantity": list(signed_quantities),
            "source_market_type": ["spot"] * size,
        }
    )


def _depth_levels_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * 4,
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58"] * 4),
            "side": ["bid", "ask", "bid", "ask"],
            "level": [1, 1, 2, 2],
            "price": [100.0, 100.2, 99.9, 100.3],
            "quantity": [1.5, 1.0, 1.0, 1.0],
            "source_market_type": ["spot"] * 4,
        }
    )


def _ordered_depth_levels_frame() -> pd.DataFrame:
    return (
        _depth_levels_frame()
        .sort_values(
            ["source_asset_id", "timestamp", "side", "level"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _panel_input_reducers_module():
    module_name = (
        "quant_research_platform.markets.implementations.crypto.processed."
        "panel_input_reducers"
    )
    module_spec = importlib.util.find_spec(module_name)
    assert module_spec is not None, "crypto panel input reducer module is missing"
    return importlib.import_module(module_name)
