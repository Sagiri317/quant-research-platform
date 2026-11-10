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
from quant_research_platform.research.protocols.a_share_hft import (
    EVENT_FACTORS,
    FACTOR_INPUTS,
)
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
    raw_orders["order_type"] = ["D", "A", "A"] if exchange == "SSE" else "2"
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


@pytest.mark.parametrize("column", tuple(FACTOR_INPUTS))
def test_selected_snapshot_factor_uses_only_declared_inputs(column):
    from quant_research_platform.research.protocols import a_share_hft as protocol

    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    minimal = panel.loc[
        :,
        list(
            dict.fromkeys(
                (*protocol.KEY, "segment_id", *protocol.FACTOR_INPUTS[column])
            )
        ),
    ]
    actual = compute_factors(minimal, columns=[column])
    expected = compute_factors(panel, columns=tuple(protocol.FACTOR_INPUTS))
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.loc[:, [*protocol.KEY, column]].reset_index(drop=True),
    )


def test_selected_trade_interval_needs_no_order_stream_or_price_volume():
    from quant_research_platform.research.protocols import a_share_hft as protocol

    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": pd.to_datetime(["2024-01-02T01:30:00", "2024-01-02T01:30:03"]),
            "segment_id": "am",
            "session_date": pd.Timestamp("2024-01-02"),
            "exchange": "SSE",
        }
    )
    trades = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": panel.timestamp,
            "source_row": [0, 1],
            "event_kind": "trade",
        }
    )
    result = compute_event_factors(
        panel,
        None,
        trades,
        columns=["trade_interval_seconds"],
        config={},
        missing_streams={"hft_orders"},
    )
    assert np.isnan(result.trade_interval_seconds.iloc[0])
    assert result.trade_interval_seconds.iloc[1] == 3
    assert list(result) == [*protocol.KEY, "trade_interval_seconds"]


@pytest.mark.parametrize("data_id", ("hft_factors", "hft_event_factors"))
def test_local_undefined_values_survive_native_factor_publication(tmp_path, data_id):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.protocols.a_share_hft import processed_specs
    from quant_research_platform.sources.venvoo import interim_spec
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    instruments = [{"wind_code": "600000.SH", "asset_id": "sse", "exchange": "SSE"}]
    raw = raw_quotes((93000000, 93003000, 93006000, 93030000)).assign(
        wind_code="600000.SH"
    )
    raw["bid_px1"], raw["ask_px1"] = 99900, 100100
    raw["bid_vol1"], raw["ask_vol1"] = [100, 0, 100, 100], [50, 0, 50, 50]
    quotes = normalize(raw, "hft_quotes", instruments)
    raw_orders = raw_events("hft_orders").copy()
    raw_orders["wind_code"] = "600000.SH"
    raw_orders["time"] = [93001000, 93001000, 93004000, 93005000]
    raw_orders["volume"] = [100.0, 40.0, 20.0, 60.0]
    raw_orders["order_type"], raw_orders["order_code"] = ["A", "D", "D", "A"], "B"
    raw_orders["ex_order_id"] = [11, 11, 11, 12]
    orders = normalize(raw_orders, "hft_orders", instruments)
    raw_trades = raw_events("hft_trades").iloc[:2].copy()
    raw_trades["wind_code"], raw_trades["time"] = "600000.SH", [93000000, 93000500]
    raw_trades["bid_order_id"], raw_trades["ask_order_id"] = 0, 0
    raw_trades["bs_flag"] = "B"
    trades = normalize(raw_trades, "hft_trades", instruments)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    inputs = {}
    for name, frame in (
        ("hft_panel", panel),
        ("hft_orders", orders),
        ("hft_trades", trades),
    ):
        root = tmp_path / name
        spec = (
            next(item for item in processed_specs() if item.data_id == name)
            if name == "hft_panel"
            else interim_spec(name)
        )
        write_dataset(root, frame, spec=spec)
        inputs[name] = str(root)
    columns = (
        ["spread_cny", "book_ofi", "depth_normalized_ofi"]
        if data_id == "hft_factors"
        else [
            "trade_signed_run",
            "trade_volume_flow_imbalance",
            "observed_order_age_seconds",
            "observed_cancel_age_seconds",
            "observed_lifecycle_match_fraction",
        ]
    )
    output = prepare(
        {
            "market_id": "a_share",
            "data_id": data_id,
            "inputs": {"hft_panel": inputs["hft_panel"]}
            if data_id == "hft_factors"
            else inputs,
            "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
            "output_root": str(tmp_path / "factors"),
            "factor_columns": columns,
            **(
                {
                    "parameters": {
                        "order_link_field": "ex_order_id",
                        "event_window": 2,
                        "volume_window": 100,
                    }
                }
                if data_id == "hft_event_factors"
                else {}
            ),
        }
    )
    result = read_dataset(output)
    if data_id == "hft_factors":
        np.testing.assert_allclose(result.spread_cny, 0.02)
        assert result.book_ofi.iloc[1] == -50
        assert np.isnan(result.depth_normalized_ofi.iloc[1])
        assert result.depth_normalized_ofi.iloc[2] == pytest.approx(1 / 3)
    else:
        # Same-stream source rows prove 100 - 40 - 20 = 40 shares remain.
        # Sub-millisecond cancel age stays unknown; later observed age is usable.
        np.testing.assert_allclose(
            result.observed_order_age_seconds.iloc[1:3], [2, (40 * 5 + 60 * 1) / 100]
        )
        assert result.observed_cancel_age_seconds.iloc[1:3].isna().all()
        assert result.observed_lifecycle_match_fraction.iloc[2] == 1
        assert result.trade_volume_flow_imbalance.iloc[1] == 1
        assert result.iloc[-1][columns].isna().all()
