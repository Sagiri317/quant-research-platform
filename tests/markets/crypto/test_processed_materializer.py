from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialize_factors,
    materialize_labels,
    materialize_panel,
    materializer,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_profiles import (
    build_research_dataset_catalog,
)
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.research.factors.microstructure import (
    calculate_microstructure_factors,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import DatasetReadMode, read_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_crypto_processed_materializer_builds_panel_factors_and_labels(
    tmp_path: Path,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=3)

    result = materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        kinds=frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )

    assert sorted(result) == ["forward_returns", "market_panel_1m", "microstructure"]
    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    factors = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.FACTOR, "microstructure")
    )
    labels = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.LABEL, "forward_returns")
    )

    assert panel.loc[0, "market_id"] == "crypto"
    assert panel.loc[0, "asset_id"] == "BTCUSDT"
    assert panel["eligible"].all()
    assert "source_asset_id" not in panel.columns
    assert {
        "spread",
        "relative_spread",
        "microprice",
        "microprice_deviation",
        "depth_imbalance",
        "trade_intensity_5m",
        "signed_volume_5m",
        "flow_imbalance_5m",
    }.issubset(set(factors.columns))
    assert {"forward_return_1m", "forward_return_5m", "forward_return_15m"}.issubset(
        set(labels.columns)
    )


def test_crypto_processed_materializer_uses_chunked_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=2)
    calls: list[tuple[str, str, DatasetReadMode]] = []
    real_read_dataset = read_dataset

    def spy(module_name: str):
        def wrapped(dataset_root, **kwargs):
            mode = kwargs.get("mode", DatasetReadMode.MATERIALIZED)
            root_name = Path(dataset_root).name
            calls.append((module_name, root_name, mode))
            if root_name in {
                "market_bars_1m",
                "aggregate_trades_tick",
                "depth_levels_tick",
                "market_panel_1m",
            } and mode is not DatasetReadMode.CHUNKED:
                raise AssertionError(
                    f"crypto_processed_requires_chunked_read:{module_name}:{root_name}"
                )
            return real_read_dataset(dataset_root, **kwargs)

        return wrapped

    monkeypatch.setattr(materialize_panel, "read_dataset", spy("panel"))
    monkeypatch.setattr(materialize_factors, "read_dataset", spy("factors"))
    monkeypatch.setattr(materialize_labels, "read_dataset", spy("labels"))

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        kinds=frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )

    assert ("panel", "market_bars_1m", DatasetReadMode.CHUNKED) in calls
    assert ("panel", "aggregate_trades_tick", DatasetReadMode.CHUNKED) in calls
    assert ("panel", "depth_levels_tick", DatasetReadMode.CHUNKED) in calls
    assert ("factors", "market_panel_1m", DatasetReadMode.CHUNKED) in calls
    assert ("labels", "market_panel_1m", DatasetReadMode.CHUNKED) in calls


def test_crypto_panel_reads_matching_trade_and_depth_once_per_bars_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=18)
    call_counts = {
        "market_bars_1m": 0,
        "aggregate_trades_tick": 0,
        "depth_levels_tick": 0,
    }
    yielded_chunks = {"market_bars_1m": 0}
    real_read_dataset = read_dataset

    def wrapped_read_dataset(dataset_root, **kwargs):
        root_name = Path(dataset_root).name
        if kwargs.get("mode") is DatasetReadMode.CHUNKED and root_name in call_counts:
            call_counts[root_name] += 1
        result = real_read_dataset(dataset_root, **kwargs)
        if (
            kwargs.get("mode") is DatasetReadMode.CHUNKED
            and root_name in yielded_chunks
        ):
            return _count_yielded_chunks(result, yielded_chunks, root_name)
        return result

    monkeypatch.setattr(materialize_panel, "read_dataset", wrapped_read_dataset)

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=RuntimeResourcePolicy(
            available_memory_bytes=1024,
            chunk_target_bytes=1,
            memory_safety_factor=1.0,
        ),
    )

    assert call_counts["market_bars_1m"] == 1
    assert yielded_chunks["market_bars_1m"] > 1
    assert call_counts["aggregate_trades_tick"] == yielded_chunks["market_bars_1m"]
    assert call_counts["depth_levels_tick"] == yielded_chunks["market_bars_1m"]


def _count_yielded_chunks(iterator, counts: dict[str, int], root_name: str):
    for chunk in iterator:
        counts[root_name] += 1
        yield chunk


def test_crypto_processed_chunk_boundaries_match_full_frame_calculations(
    tmp_path: Path,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=2)
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        kinds=frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )
    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    ).sort_values(["market_id", "asset_id", "timestamp"], kind="mergesort")
    factors = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.FACTOR, "microstructure")
    ).sort_values(["market_id", "asset_id", "timestamp"], kind="mergesort")
    labels = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.LABEL, "forward_returns")
    ).sort_values(["market_id", "asset_id", "timestamp"], kind="mergesort")
    catalog = processed_catalog()
    expected_factors = calculate_microstructure_factors(panel).loc[
        :,
        catalog.get("microstructure").required_columns,
    ]
    expected_labels = add_forward_return_labels(
        panel,
        horizons=CRYPTO_FORWARD_RETURN_HORIZONS,
        frequency="1m",
        execution_price_column="close",
    ).loc[:, catalog.get("forward_returns").required_columns]

    pd.testing.assert_frame_equal(
        factors.reset_index(drop=True),
        expected_factors.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        labels.reset_index(drop=True),
        expected_labels.reset_index(drop=True),
    )


def test_crypto_research_dataset_includes_full_market_context_columns(
    tmp_path: Path,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=3)
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        kinds=frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )

    root = assemble_research_dataset(
        layout=layout,
        dataset_catalog=build_research_dataset_catalog(),
        processed_catalog=processed_catalog(),
        dataset_id="microstructure_forward_returns",
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
    )

    research = read_dataset(root)
    assert {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "mid_price",
    }.issubset(research.columns)
    assert "source_asset_id" not in research.columns


def test_crypto_processed_materializer_stabilizes_schema_when_depth_is_missing(
    tmp_path: Path,
) -> None:
    _write_binance_fixture(tmp_path, depth_symbols=("BTCUSDT",))
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=2)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        data_ids=("market_panel_1m",),
    )

    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    missing_depth_rows = panel.loc[panel["asset_id"].eq("ETHUSDT")]
    assert not missing_depth_rows.empty
    assert missing_depth_rows["best_bid"].isna().all()
    assert missing_depth_rows["bid_depth"].isna().all()


def _write_binance_fixture(
    tmp_path: Path,
    *,
    depth_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    for symbol in ("BTCUSDT", "ETHUSDT"):
        start_ms = 1783296000000
        for offset in range(18):
            open_time = start_ms + offset * 60_000
            _append_jsonl(
                raw_root / "spot" / "klines" / symbol / "1m" / "2026-07-06.jsonl",
                _kline_envelope(symbol=symbol, offset=offset, open_time=open_time),
            )
            _append_jsonl(
                raw_root / "spot" / "agg_trades" / symbol / "2026-07-06.jsonl",
                _trade_envelope(symbol=symbol, offset=offset, open_time=open_time),
            )
            if symbol in depth_symbols:
                _append_jsonl(
                    raw_root / "spot" / "depth" / symbol / "2026-07-06.jsonl",
                    _depth_envelope(symbol=symbol, offset=offset, open_time=open_time),
                )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _kline_envelope(
    *,
    symbol: str,
    offset: int,
    open_time: int,
) -> dict[str, object]:
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "klines",
        "symbol": symbol,
        "interval": "1m",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": [
            open_time,
            str(100 + offset),
            str(101 + offset),
            str(99 + offset),
            str(100.5 + offset),
            "10.0",
            open_time + 59_999,
            "1005.0",
            7,
            "4.0",
            "402.0",
            "0",
        ],
    }


def _trade_envelope(
    *,
    symbol: str,
    offset: int,
    open_time: int,
) -> dict[str, object]:
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "agg_trades",
        "symbol": symbol,
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": {
            "a": offset + 1,
            "p": str(100.1 + offset),
            "q": "2.0",
            "f": offset * 2,
            "l": offset * 2 + 1,
            "T": open_time + 1_000,
            "m": offset % 2 == 0,
            "M": True,
        },
    }


def _depth_envelope(
    *,
    symbol: str,
    offset: int,
    open_time: int,
) -> dict[str, object]:
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "depth",
        "symbol": symbol,
        "captured_at": pd.Timestamp(open_time + 2_000, unit="ms", tz="UTC").isoformat(),
        "payload": {
            "lastUpdateId": offset + 1,
            "bids": [[str(100 + offset), "1.5"], [str(99.9 + offset), "2.0"]],
            "asks": [[str(100.2 + offset), "1.0"]],
        },
    }
