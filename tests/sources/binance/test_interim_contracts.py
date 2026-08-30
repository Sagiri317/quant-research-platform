from __future__ import annotations

import pandas as pd
import pytest

import quant_research_platform.sources.binance.raw_exports as raw_exports
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.sources.binance.interim import (
    _normalize_depth,
    interim_catalog,
    materialize_interim_data,
)
from quant_research_platform.storage.layout import StorageLayout


def test_binance_interim_catalog_registers_approved_source_specs() -> None:
    catalog = interim_catalog()

    assert tuple(spec.data_id for spec in catalog.list()) == (
        "aggregate_trades_tick",
        "depth_levels_tick",
        "market_bars_1m",
    )
    market_bars = catalog.get("market_bars_1m")
    trades = catalog.get("aggregate_trades_tick")
    depth = catalog.get("depth_levels_tick")

    assert market_bars.source_id == SourceId("binance")
    assert market_bars.frequency is DataFrequency.MINUTE
    assert market_bars.grain is DataGrain.ASSET_TIMESTAMP
    assert market_bars.primary_key == ("source_asset_id", "timestamp")
    assert trades.frequency is DataFrequency.TICK
    assert trades.grain is DataGrain.TRADE_EVENT
    assert trades.primary_key == (
        "source_asset_id",
        "timestamp",
        "aggregate_trade_id",
    )
    assert depth.frequency is DataFrequency.TICK
    assert depth.grain is DataGrain.ORDER_BOOK_LEVEL
    assert depth.primary_key == ("source_asset_id", "timestamp", "side", "level")


def test_binance_interim_fields_are_source_local_only() -> None:
    catalog = interim_catalog()

    for spec in catalog.list():
        assert "source_asset_id" in spec.required_columns
        assert "market_id" not in spec.required_columns
        assert "asset_id" not in spec.required_columns


@pytest.mark.parametrize(
    "captured_at",
    ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00"],
)
def test_binance_interim_accepts_strict_depth_captured_at(captured_at: str) -> None:
    envelope = raw_exports._validate_envelope(
        {
            "source": "binance",
            "market_type": "spot",
            "data_kind": "depth",
            "symbol": "BTCUSDT",
            "captured_at": captured_at,
            "payload": {
                "lastUpdateId": 1,
                "bids": [["100.0", "1.0"]],
                "asks": [["101.0", "1.5"]],
            },
        }
    )
    frame = _normalize_depth((envelope,))

    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00")


def test_binance_interim_rejects_naive_depth_captured_at() -> None:
    with pytest.raises(ValueError, match="invalid_binance_raw_envelope_captured_at"):
        raw_exports._validate_envelope(
            {
                "source": "binance",
                "market_type": "spot",
                "data_kind": "depth",
                "symbol": "BTCUSDT",
                "captured_at": "2026-01-01T00:00:00",
                "payload": {
                    "lastUpdateId": 1,
                    "bids": [["100.0", "1.0"]],
                    "asks": [["101.0", "1.5"]],
                },
            }
        )


@pytest.mark.parametrize(
    ("payload_patch", "error"),
    (
        (
            {"bids": [["99.0", "1.0"], ["100.0", "1.0"]]},
            "invalid_binance_depth_bid",
        ),
        (
            {"bids": [["101.0", "1.0"]], "asks": [["100.0", "1.0"]]},
            "crossed_binance_depth_book",
        ),
        ({"bids": None}, "invalid_binance_depth_bid"),
    ),
)
def test_binance_raw_owner_rejects_invalid_depth_semantics(
    payload_patch: dict[str, object],
    error: str,
) -> None:
    payload = {
        "lastUpdateId": 1,
        "bids": [["100.0", "1.0"]],
        "asks": [["101.0", "1.5"]],
    }
    payload.update(payload_patch)

    with pytest.raises(ValueError, match=error):
        raw_exports._validate_envelope(
            {
                "source": "binance",
                "market_type": "spot",
                "data_kind": "depth",
                "symbol": "BTCUSDT",
                "captured_at": "2026-01-01T00:00:00Z",
                "payload": payload,
            }
        )


def test_binance_interim_maps_millisecond_range_failure_to_owner_error() -> None:
    maximum = pd.Timestamp.max.value // pd.Timedelta(milliseconds=1).value
    envelope = {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "agg_trades",
        "symbol": "BTCUSDT",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": {
            "a": 1,
            "p": "100.0",
            "q": "1.0",
            "f": 1,
            "l": 1,
            "T": maximum + 1,
            "m": False,
        },
    }

    with pytest.raises(ValueError, match="invalid_binance_agg_trade_timestamp"):
        raw_exports._validate_envelope(envelope)


def test_binance_interim_materialization_rejects_bare_str_data_ids(tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid_binance_interim_data_ids"):
        materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            data_ids="market_bars_1m",  # type: ignore[arg-type]
        )


def test_binance_interim_rejects_unknown_data_id_with_stable_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported_binance_data_id"):
        materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            data_ids=("unknown",),
        )


def test_binance_interim_rejects_missing_payload_field_with_value_error() -> None:
    with pytest.raises(ValueError, match="missing_binance_raw_payload_field"):
        raw_exports._validate_envelope(
            {
                "source": "binance",
                "market_type": "spot",
                "data_kind": "agg_trades",
                "symbol": "BTCUSDT",
                "captured_at": "2026-07-06T00:00:02Z",
                "payload": {
                    "a": 1,
                    "p": "100.1",
                    "q": "2.0",
                    "f": 10,
                    "T": 1783296001000,
                    "m": False,
                },
            }
        )
