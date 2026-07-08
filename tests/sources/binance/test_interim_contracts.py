from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.sources.binance.interim import (
    interim_catalog,
    iter_interim_frames,
    materialize_interim_data,
    normalize_raw_envelopes,
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


def test_binance_interim_rejects_data_id_raw_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="binance_raw_data_id_kind_mismatch"):
        normalize_raw_envelopes(
            (
                {
                    "source": "binance",
                    "market_type": "spot",
                    "data_kind": "depth",
                    "symbol": "BTCUSDT",
                    "captured_at": "2026-07-06T00:00:02Z",
                    "payload": {
                        "lastUpdateId": 1,
                        "bids": [["100.0", "1.0"]],
                        "asks": [],
                    },
                },
            ),
            data_id="market_bars_1m",
        )


@pytest.mark.parametrize(
    "captured_at",
    ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00+00:00"],
)
def test_binance_interim_accepts_strict_depth_captured_at(captured_at: str) -> None:
    frame = normalize_raw_envelopes(
        (
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
            },
        ),
        data_id="depth_levels_tick",
    )

    assert frame["timestamp"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00")


def test_binance_interim_rejects_naive_depth_captured_at() -> None:
    with pytest.raises(ValueError, match="invalid_binance_raw_envelope_captured_at"):
        normalize_raw_envelopes(
            (
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
                },
            ),
            data_id="depth_levels_tick",
        )


@pytest.mark.parametrize("row_batch_size", [True, 1.5, "100"])
def test_binance_interim_rejects_non_strict_row_batch_size(
    row_batch_size: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_binance_raw_row_batch_size"):
        tuple(
            iter_interim_frames(
                data_id="market_bars_1m",
                input_files=(),
                row_batch_size=row_batch_size,  # type: ignore[arg-type]
            )
        )


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
        normalize_raw_envelopes(
            (
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
                },
            ),
            data_id="aggregate_trades_tick",
        )
