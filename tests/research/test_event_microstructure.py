"""Hand-computed depth, event-volume, and observed lifecycle examples."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.research.factors.event_microstructure import (
    compute_event_factors,
    parameters,
    trade_statistics,
)
from quant_research_platform.research.factors.microstructure import compute_factors
from quant_research_platform.research.protocols.a_share_hft import EVENT_FACTORS
from quant_research_platform.sources.venvoo import normalize
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)


def test_multilevel_ofi_and_actual_price_depth_shape():
    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    columns = (
        "book_ofi_5",
        "book_ofi_10",
        "depth_normalized_ofi_10",
        "bid_depth_concentration",
        "ask_near_depth_fraction",
    )
    result = compute_factors(panel, columns=columns)
    np.testing.assert_allclose(result.book_ofi_5.iloc[1:], 2000)
    np.testing.assert_allclose(result.book_ofi_10.iloc[1:], 4000)
    np.testing.assert_allclose(result.depth_normalized_ofi_10.iloc[1:], 1)
    np.testing.assert_allclose(result.bid_depth_concentration, 0.1)
    np.testing.assert_allclose(result.ask_near_depth_fraction, 0.3)


def test_partial_first_trade_in_volume_window_and_same_millisecond_intervals():
    trades = pd.DataFrame(
        {
            "event_kind": "trade",
            "side": ["buy", "sell", "buy", "buy"],
            "volume": [200.0, 100.0, 100.0, 100.0],
            "price": [10.0, 10.1, 10.1, 10.2],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T01:30:00",
                    "2024-01-02T01:30:01",
                    "2024-01-02T01:30:01",
                    "2024-01-02T01:30:03",
                ]
            ),
            "source_row": np.arange(4),
        }
    )
    config = parameters({"event_window": 2, "history_window": 2, "volume_window": 150})
    _, result = trade_statistics(trades, config)
    np.testing.assert_allclose(
        result.trade_volume_flow_imbalance.iloc[1:], [-1 / 3, 1 / 3, 1]
    )
    assert result.trade_interval_seconds.iloc[2] == 0
    assert np.isnan(result.trade_intensity.iloc[2])
    assert result.trade_signed_run.tolist() == [1, -1, 1, 2]
    _, prefix = trade_statistics(trades.iloc[:3], config)
    pd.testing.assert_frame_equal(prefix, result.iloc[:3])


def test_lifecycle_links_only_observed_original_orders_and_resets_at_gap():
    quotes = normalize(
        raw_quotes((93000000, 93003000, 93006000, 93030000)), "hft_quotes", INSTRUMENTS
    )
    raw_orders = raw_events("hft_orders").iloc[:2].copy()
    raw_orders["time"], raw_orders["volume"] = [93001000, 93002000], [100.0, 200.0]
    raw_orders["order_id"], raw_orders["ex_order_id"] = [11, 12], [111, 112]
    raw_orders["order_code"] = ["B", "S"]
    raw_trades = raw_events("hft_trades").iloc[:2].copy()
    raw_trades["time"], raw_trades["volume"] = [93004000, 93005000], [40.0, 60.0]
    raw_trades["trade_code"] = ["T", "C"]
    raw_trades["bid_order_id"], raw_trades["ask_order_id"] = [111, 111], [112, 0]
    orders, trades = (
        normalize(raw_orders, "hft_orders", INSTRUMENTS),
        normalize(raw_trades, "hft_trades", INSTRUMENTS),
    )
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    result = compute_event_factors(
        panel,
        orders,
        trades,
        columns=tuple(EVENT_FACTORS),
        config={
            "event_window": 2,
            "history_window": 2,
            "volume_window": 100,
            "order_link_field": "ex_order_id",
        },
    )
    assert result.observed_order_age_seconds.iloc[1] == 4 / 3
    assert result.observed_cancel_age_seconds.iloc[2] == 4
    assert result.observed_completed_lifetime_seconds.iloc[2] == 4
    assert result.observed_lifecycle_match_fraction.iloc[2] == 1
    assert result.iloc[-1][list(EVENT_FACTORS)].isna().all()
    assert panel.observed_trade_volume.iloc[-1] != panel.observed_trade_volume.iloc[-1]


@pytest.mark.parametrize("exchange", ["SSE", "SZSE"])
def test_cancel_windows_keep_same_stream_order_and_cross_stream_tie_convention(
    exchange,
):
    code = "600000.SH" if exchange == "SSE" else "000001.SZ"
    instruments = [{"wind_code": code, "asset_id": code, "exchange": exchange}]
    raw_orders = raw_events("hft_orders").iloc[: 3 if exchange == "SSE" else 2].copy()
    raw_orders["wind_code"] = code
    raw_orders["time"], raw_orders["volume"] = 93001000, 100.0
    raw_orders["order_type"] = ["D", "2", "2"] if exchange == "SSE" else "2"
    raw_trades = raw_events("hft_trades").iloc[: 0 if exchange == "SSE" else 1].copy()
    raw_trades["wind_code"] = code
    raw_trades["time"], raw_trades["volume"], raw_trades["trade_code"] = (
        93001000,
        100.0,
        "C",
    )
    quotes = normalize(
        raw_quotes((93000000, 93003000)).assign(wind_code=code),
        "hft_quotes",
        instruments,
    )
    orders = normalize(raw_orders, "hft_orders", instruments)
    trades = normalize(raw_trades, "hft_trades", instruments)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    columns = ("observed_cancel_order_ratio", "observed_cancel_count_fraction")
    result = compute_event_factors(
        panel, orders, trades, columns=columns, config={"event_window": 2}
    )
    # Both conventions put the cancellation before the two new orders.
    assert result.loc[result.index[-1], list(columns)].tolist() == [0.0, 0.0]


def test_event_factor_prepare_and_native_assembly(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.protocols.a_share_hft import processed_specs
    from quant_research_platform.sources.venvoo import interim_spec
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    paths = {
        name: str(tmp_path / name) for name in ("hft_orders", "hft_trades", "hft_panel")
    }
    for name, frame in (
        ("hft_orders", orders),
        ("hft_trades", trades),
        ("hft_panel", panel),
    ):
        spec = (
            interim_spec(name)
            if name != "hft_panel"
            else next(item for item in processed_specs() if item.data_id == name)
        )
        write_dataset(paths[name], frame, spec=spec)
    window = {"start": "2018-02-14", "end": "2018-02-14"}
    columns = ["trade_signed_run", "trade_volume_flow_imbalance"]
    factors = prepare(
        {
            "market_id": "a_share",
            "data_id": "hft_event_factors",
            "inputs": paths,
            "time_window": window,
            "output_root": str(tmp_path / "event_factors"),
            "factor_columns": columns,
            "parameters": {"event_window": 2, "volume_window": 150},
        }
    )
    dataset = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "event_test",
            "inputs": {
                "hft_panel": paths["hft_panel"],
                "hft_event_factors": str(factors),
            },
            "base_input_data_id": "hft_panel",
            "factor_columns": columns,
            "feature_recipe": None,
            "filter_columns": ["eligible"],
            "time_window": window,
            "output_root": str(tmp_path / "dataset"),
        }
    )
    result = read_dataset(dataset)
    assert result.trade_signed_run.tolist() == [1, 1, 2, 2]
    assert result.trade_volume_flow_imbalance.iloc[2] == 1
