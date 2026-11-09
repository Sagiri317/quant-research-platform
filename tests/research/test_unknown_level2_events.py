"""Unknown event types preserve causal gaps without damaging independent quotes."""

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
from quant_research_platform.research.factors.hawkes import compute_hawkes_factors
from quant_research_platform.research.factors.microstructure import compute_factors
from quant_research_platform.research.labels.quote_returns import compute_labels
from quant_research_platform.research.models.hawkes import hawkes_parameters
from quant_research_platform.research.protocols.a_share_hft import label_catalog
from quant_research_platform.sources.venvoo import normalize
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)


@pytest.mark.parametrize("exchange", ("SSE", "SZSE"))
@pytest.mark.parametrize("code", (None, "", "\x00"))
def test_not_applicable_sse_trade_code_is_not_unknown_szse_code(exchange, code):
    instruments = [{**INSTRUMENTS[0], "exchange": exchange}]
    raw = raw_events("hft_trades").iloc[:1].copy()
    raw["trade_code"] = pd.Series([code], dtype="str")
    result = normalize(raw, "hft_trades", instruments)
    assert result.event_kind.tolist() == ["trade" if exchange == "SSE" else "unknown"]
    assert result.volume.tolist() == [100]
    assert result.price.tolist() == [10]


@pytest.mark.parametrize("code", (None, "?", "2"))
def test_sse_unknown_order_type_does_not_default_to_addition(code):
    raw = raw_events("hft_orders").iloc[:1].copy()
    raw["order_type"] = pd.Series([code], dtype="str")
    frame = normalize(raw, "hft_orders", [{**INSTRUMENTS[0], "exchange": "SSE"}])
    assert frame.event_kind.tolist() == ["unknown"]


def test_cancel_windows_recover_after_unknown_message_leaves_window():
    instruments = [{**INSTRUMENTS[0], "exchange": "SSE"}]
    raw = pd.concat([raw_events("hft_orders").iloc[[0]]] * 5, ignore_index=True)
    raw["time"] = [93001000, 93002000, 93003000, 93004000, 93005000]
    raw["order_type"] = ["A", "?", "D", "A", "A"]
    orders = normalize(raw, "hft_orders", instruments)
    trades = normalize(raw_events("hft_trades").iloc[:0], "hft_trades", instruments)
    quotes = normalize(
        raw_quotes((93000000, 93001000, 93002000, 93003000, 93004000, 93005000)),
        "hft_quotes",
        instruments,
    )
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    columns = ("observed_cancel_count_fraction", "observed_cancel_order_ratio")
    result = compute_event_factors(
        panel, orders, trades, columns=columns, config={"event_window": 2}
    )
    assert result.loc[2:3, list(columns)].isna().all(axis=None)
    np.testing.assert_allclose(result.loc[4, list(columns)].to_numpy(float), [0.5, 1])
    np.testing.assert_allclose(result.loc[5, list(columns)].to_numpy(float), [0, 0])


def _case():
    quotes = normalize(
        raw_quotes((93000000, 93003000, 93006000, 93009000, 93012000)),
        "hft_quotes",
        INSTRUMENTS,
    )
    raw = raw_events("hft_trades").copy()
    raw["time"], raw["volume"], raw["trade_code"] = (
        [93000000, 93000500, 93004000, 93007000],
        [1.0, 2.0, 7.0, 10.0],
        ["T", "T", "?", "T"],
    )
    raw["bs_flag"] = "B"
    trades = normalize(raw, "hft_trades", INSTRUMENTS)
    raw_order = raw_events("hft_orders").iloc[:1].copy()
    raw_order["time"], raw_order["volume"], raw_order["ex_order_id"] = (
        93001000,
        10.0,
        11,
    )
    orders = normalize(raw_order, "hft_orders", INSTRUMENTS)
    return quotes, orders, trades


def test_unknown_interval_is_not_zero_and_later_known_intervals_recover():
    quotes, orders, trades = _case()
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    np.testing.assert_allclose(
        panel.observed_trade_volume, [np.nan, 2, np.nan, 10, 0], equal_nan=True
    )
    np.testing.assert_allclose(
        panel.observed_signed_trade_volume, [np.nan, 2, np.nan, 10, 0], equal_nan=True
    )
    factors = compute_factors(panel, columns=("relative_spread", "imbalance_1"))
    assert factors.iloc[2].notna().all()
    labels = compute_labels(panel, [label_catalog().get("future_mid_return_3s")])
    assert labels.future_mid_return_3s.iloc[2] == pytest.approx(
        quotes.price.iloc[3] / quotes.price.iloc[2] - 1
    )
    healthy = build_panel(
        quotes, orders, trades.loc[trades.event_kind.eq("trade")], max_gap_seconds=6
    )
    pd.testing.assert_series_equal(panel.mid, healthy.mid)
    pd.testing.assert_series_equal(
        panel.observed_order_volume, healthy.observed_order_volume
    )


def test_unknown_trade_marker_does_not_compress_event_or_volume_windows():
    start = pd.Timestamp("2024-01-02T01:30")
    trades = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(np.arange(1, 7), unit="s"),
            "event_kind": ["trade", "unknown", "trade", "trade", "trade", "trade"],
            "side": "buy",
            "price": 10.0,
            "volume": 10.0,
            "source_row": np.arange(6),
        }
    )
    times, values = trade_statistics(
        trades,
        parameters({"event_window": 2, "history_window": 2, "volume_window": 15}),
    )
    assert len(times) == 6
    assert values.iloc[1].isna().all()
    assert np.isnan(values.trade_interval_seconds.iloc[2])
    assert values.trade_signed_run.iloc[2] == 1
    assert np.isnan(values.trade_event_flow_imbalance.iloc[2])
    assert np.isnan(values.trade_volume_flow_imbalance.iloc[2])
    assert values.trade_event_flow_imbalance.iloc[3] == 1
    assert values.trade_volume_flow_imbalance.iloc[3] == 1
    assert values.trade_interval_seconds.iloc[3] == 1


def test_recursive_lifecycle_and_hawkes_keep_known_prefix_only():
    quotes, orders, trades = _case()
    trades.loc[trades.event_kind.eq("unknown"), "bid_order_id"] = 11
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    life = compute_event_factors(
        panel,
        orders,
        trades,
        columns=("observed_order_age_seconds",),
        config={"order_link_field": "ex_order_id"},
    )
    assert life.observed_order_age_seconds.iloc[1] == 2
    assert life.observed_order_age_seconds.iloc[2:].isna().all()
    model = (
        hawkes_parameters(
            {
                "baseline": [1.0, 2.0],
                "branching": [[0.1, 0.1], [0.1, 0.1]],
                "decay": 1.0,
            }
        ),
        pd.Timestamp("2018-02-13"),
    )
    values = compute_hawkes_factors(panel, trades, model)
    assert values.hawkes_buy_intensity.iloc[:2].notna().all()
    assert values.iloc[2:].filter(like="hawkes_").isna().all(axis=None)
    tied = trades.copy()
    tied.loc[tied.event_kind.eq("unknown"), "timestamp"] = quotes.timestamp.iloc[2]
    values = compute_hawkes_factors(panel, tied, model)
    assert values.hawkes_buy_intensity.iloc[2] > 0
    assert values.iloc[3:].filter(like="hawkes_").isna().all(axis=None)
