from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
)
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
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
    write_dataset,
)
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
    assert panel.loc[0, "asset_id"] == "btc_usdt"
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


def test_crypto_factor_time_window_uses_prior_warmup_minutes() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")

    read_window = materialization_dependencies.factor_read_window(core)

    assert read_window == TimeWindow("2026-07-06T00:06:00", "2026-07-06T00:12:00")


def test_crypto_factor_date_only_dependency_window_excludes_next_midnight() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    read_window = materialization_dependencies.factor_read_window(
        TimeWindow("2026-01-01", "2026-01-01")
    )

    assert read_window == TimeWindow("2025-12-31T23:56:00", "2026-01-01T23:59:00")


def test_crypto_factor_only_panel_dependency_window_uses_factor_warmup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, TimeWindow]] = []

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **__: Any) -> object:
            calls.append((name, time_window))
            return object()

        return wrapped

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_factors, "write_factors", record("factors"))

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")
    materializer.materialize_processed_data(
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("synthetic_crypto"),
        time_window=core,
        kinds=frozenset({ProcessedDataKind.FACTOR}),
    )

    assert calls == [
        ("panel", TimeWindow("2026-07-06T00:06:00", "2026-07-06T00:12:00")),
        ("factors", core),
    ]


def test_crypto_label_time_window_uses_future_lookahead_minutes() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")

    read_window = materialization_dependencies.label_read_window(core)

    assert read_window == TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:28:00")


def test_crypto_label_date_only_dependency_window_excludes_next_midnight() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    read_window = materialization_dependencies.label_read_window(
        TimeWindow("2026-01-01", "2026-01-01")
    )

    assert read_window == TimeWindow("2026-01-01T00:00:00", "2026-01-02T00:15:00")


def test_crypto_label_only_panel_dependency_window_uses_label_lookahead(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, TimeWindow]] = []

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **__: Any) -> object:
            calls.append((name, time_window))
            return object()

        return wrapped

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_labels, "write_labels", record("labels"))

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")
    materializer.materialize_processed_data(
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("synthetic_crypto"),
        time_window=core,
        kinds=frozenset({ProcessedDataKind.LABEL}),
    )

    assert calls == [
        ("panel", TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:28:00")),
        ("labels", core),
    ]


def test_crypto_materializer_rejects_bare_str_data_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid_crypto_processed_data_ids"):
        materializer.materialize_processed_data(
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("binance"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            data_ids="market_panel_1m",  # type: ignore[arg-type]
        )


def test_crypto_materializer_rejects_invalid_kinds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid_crypto_processed_kinds"):
        materializer.materialize_processed_data(
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("binance"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            kinds=frozenset({"panel"}),  # type: ignore[arg-type]
        )


def test_crypto_materializer_materializes_panel_dependency_window_for_factor_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, TimeWindow]] = []

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **__: Any) -> object:
            calls.append((name, time_window))
            return object()

        return wrapped

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_factors, "write_factors", record("factors"))
    monkeypatch.setattr(materialize_labels, "write_labels", record("labels"))

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")
    materializer.materialize_processed_data(
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("synthetic_crypto"),
        time_window=core,
        kinds=frozenset({ProcessedDataKind.FACTOR, ProcessedDataKind.LABEL}),
    )

    assert calls == [
        ("panel", TimeWindow("2026-07-06T00:06:00", "2026-07-06T00:28:00")),
        ("factors", core),
        ("labels", core),
    ]


def test_crypto_factor_label_date_only_dependency_union_is_precise() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    read_window = materialization_dependencies.panel_dependency_window(
        TimeWindow("2026-01-01", "2026-01-01")
    )

    assert read_window == TimeWindow("2025-12-31T23:56:00", "2026-01-02T00:15:00")


def test_crypto_datetime_dependency_window_keeps_closed_end() -> None:
    from quant_research_platform.markets.implementations.crypto.processed import (
        materialization_dependencies,
    )

    core = TimeWindow("2026-01-01T09:30:00", "2026-01-01T09:31:00")

    assert materialization_dependencies.factor_read_window(core) == TimeWindow(
        "2026-01-01T09:26:00",
        "2026-01-01T09:31:00",
    )
    assert materialization_dependencies.label_read_window(core) == TimeWindow(
        "2026-01-01T09:30:00",
        "2026-01-01T09:47:00",
    )


@pytest.mark.parametrize(
    ("data_id", "kind", "writer_name"),
    [
        ("market_panel_1m", ProcessedDataKind.PANEL, "panel"),
        ("microstructure", ProcessedDataKind.FACTOR, "factors"),
        ("forward_returns", ProcessedDataKind.LABEL, "labels"),
    ],
)
def test_crypto_materializer_requested_existing_target_fails_before_dependencies(
    tmp_path: Path,
    monkeypatch,
    data_id: str,
    kind: ProcessedDataKind,
    writer_name: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    catalog = processed_catalog()
    spec = catalog.get(data_id)
    write_dataset(
        layout.processed_data_root("crypto", kind, data_id),
        _existing_processed_frame(spec, minutes=30),
        spec=spec,
    )

    def fail_dependency_write(*_: Any, **__: Any) -> object:
        raise AssertionError(f"unexpected_crypto_{writer_name}_write")

    monkeypatch.setattr(materialize_panel, "write_panel", fail_dependency_write)
    monkeypatch.setattr(materialize_factors, "write_factors", fail_dependency_write)
    monkeypatch.setattr(materialize_labels, "write_labels", fail_dependency_write)

    with pytest.raises(ValueError, match="dataset_exists"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
            data_ids=(data_id,),
        )


def test_crypto_materializer_reuses_sufficient_existing_panel_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    catalog = processed_catalog()
    panel_spec = catalog.get("market_panel_1m")
    write_dataset(
        layout.processed_data_root(
            "crypto",
            ProcessedDataKind.PANEL,
            "market_panel_1m",
        ),
        _panel_frame(minutes=30),
        spec=panel_spec,
    )
    calls: list[tuple[str, TimeWindow]] = []

    def fail_panel_write(*_: Any, **__: Any) -> object:
        raise AssertionError("unexpected_crypto_dependency_panel_write")

    def record_factors(*_: Any, time_window: TimeWindow, **__: Any) -> object:
        calls.append(("factors", time_window))
        return object()

    monkeypatch.setattr(materialize_panel, "write_panel", fail_panel_write)
    monkeypatch.setattr(materialize_factors, "write_factors", record_factors)

    result = materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("synthetic_crypto"),
        time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
        data_ids=("microstructure",),
    )

    assert sorted(result) == ["market_panel_1m", "microstructure"]
    assert calls == [
        ("factors", TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"))
    ]


def test_crypto_materializer_rejects_insufficient_existing_panel_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = processed_catalog().get("market_panel_1m")
    write_dataset(
        layout.processed_data_root(
            "crypto",
            ProcessedDataKind.PANEL,
            "market_panel_1m",
        ),
        _panel_frame(minutes=12),
        spec=panel_spec,
    )

    def fail_panel_write(*_: Any, **__: Any) -> object:
        raise AssertionError("unexpected_crypto_dependency_panel_write")

    monkeypatch.setattr(materialize_panel, "write_panel", fail_panel_write)

    with pytest.raises(ValueError, match="insufficient_crypto_dependency_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
            data_ids=("microstructure",),
        )


def test_crypto_materializer_rejects_panel_dependency_with_middle_minute_gap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = processed_catalog().get("market_panel_1m")
    gapped = pd.concat(
        (
            _panel_frame(minutes=1).assign(timestamp=pd.Timestamp("2026-07-06 00:06")),
            _panel_frame(minutes=1).assign(timestamp=pd.Timestamp("2026-07-06 00:12")),
        ),
        ignore_index=True,
    )
    write_dataset(
        layout.processed_data_root(
            "crypto",
            ProcessedDataKind.PANEL,
            "market_panel_1m",
        ),
        gapped,
        spec=panel_spec,
    )

    def fail_panel_write(*_: Any, **__: Any) -> object:
        raise AssertionError("unexpected_crypto_dependency_panel_write")

    monkeypatch.setattr(materialize_panel, "write_panel", fail_panel_write)

    with pytest.raises(ValueError, match="insufficient_crypto_dependency_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
            data_ids=("microstructure",),
        )


def test_crypto_materializer_materializes_missing_panel_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, TimeWindow]] = []

    def record_panel(*_: Any, time_window: TimeWindow, **__: Any) -> object:
        calls.append(("panel", time_window))
        return object()

    def record_factors(*_: Any, time_window: TimeWindow, **__: Any) -> object:
        calls.append(("factors", time_window))
        return object()

    monkeypatch.setattr(materialize_panel, "write_panel", record_panel)
    monkeypatch.setattr(materialize_factors, "write_factors", record_factors)

    materializer.materialize_processed_data(
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("synthetic_crypto"),
        time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
        data_ids=("microstructure",),
    )

    assert calls == [
        ("panel", TimeWindow("2026-07-06T00:06:00", "2026-07-06T00:12:00")),
        ("factors", TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")),
    ]


def test_crypto_factor_label_emit_only_core_window(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    catalog = processed_catalog()
    panel_spec = catalog.get("market_panel_1m")
    factor_spec = catalog.get("microstructure")
    label_spec = catalog.get("forward_returns")
    panel_root = layout.processed_data_root(
        "crypto",
        ProcessedDataKind.PANEL,
        "market_panel_1m",
    )
    write_dataset(panel_root, _panel_frame(minutes=30), spec=panel_spec)

    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")
    materialize_factors.write_factors(
        factor_spec,
        layout=layout,
        time_window=core,
        resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
    )
    materialize_labels.write_labels(
        label_spec,
        layout=layout,
        time_window=core,
        resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
    )

    factors = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.FACTOR, "microstructure")
    )
    labels = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.LABEL, "forward_returns")
    )

    expected_timestamps = [
        pd.Timestamp("2026-07-06 00:10:00"),
        pd.Timestamp("2026-07-06 00:11:00"),
        pd.Timestamp("2026-07-06 00:12:00"),
    ]
    assert factors["timestamp"].tolist() == expected_timestamps
    assert labels["timestamp"].tolist() == expected_timestamps
    assert not factors["trade_intensity_5m"].isna().any()
    assert not labels["forward_return_15m"].isna().any()


@pytest.mark.parametrize("index", [[10, 20, 30], [0, 0, 1]])
def test_crypto_core_mask_uses_original_row_positions(index: list[int]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:09:00",
                    "2026-07-06 00:10:00",
                    "2026-07-06 00:11:00",
                ]
            )
        },
        index=index,
    )
    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:10:00")

    assert materialize_factors._core_mask(frame, core).tolist() == [False, True, False]
    assert materialize_labels._core_mask(frame, core).tolist() == [False, True, False]


@pytest.mark.parametrize("index", [[100 + index for index in range(30)], [0] * 30])
def test_crypto_factor_label_materialization_core_selection_uses_positions(
    monkeypatch,
    index: list[int],
) -> None:
    panel = _panel_frame(minutes=30)
    panel.index = index
    core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")

    def read_panel(*_: Any, **__: Any):
        return iter((panel.copy(),))

    monkeypatch.setattr(materialize_factors, "read_dataset", read_panel)
    monkeypatch.setattr(materialize_labels, "read_dataset", read_panel)

    factors = pd.concat(
        materialize_factors._iter_factor_chunks(
            processed_catalog().get("microstructure"),
            panel_root=Path("unused"),
            time_window=core,
        ),
        ignore_index=True,
    )
    labels = pd.concat(
        materialize_labels._iter_label_chunks(
            processed_catalog().get("forward_returns"),
            panel_root=Path("unused"),
            time_window=core,
        ),
        ignore_index=True,
    )

    expected = pd.to_datetime(
        ["2026-07-06 00:10:00", "2026-07-06 00:11:00", "2026-07-06 00:12:00"]
    ).tolist()
    assert factors["timestamp"].tolist() == expected
    assert labels["timestamp"].tolist() == expected


@pytest.mark.parametrize(
    ("frame_name", "column", "error"),
    [
        ("bars", "timestamp", "invalid_crypto_matching_bar_timestamp"),
        ("bars", "close_time", "invalid_crypto_matching_bar_close_time"),
        ("trades", "timestamp", "invalid_crypto_matching_trade_timestamp"),
        ("depth", "timestamp", "invalid_crypto_matching_depth_timestamp"),
    ],
)
def test_crypto_panel_matching_rejects_invalid_timestamps(
    tmp_path: Path,
    monkeypatch,
    frame_name: str,
    column: str,
    error: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    frames = _matching_panel_frames()
    frames[frame_name][column] = frames[frame_name][column].astype(object)
    frames[frame_name].loc[0, column] = "bad"

    def fake_read_dataset(dataset_root, **kwargs):
        root_name = Path(dataset_root).name
        if kwargs.get("mode") is not DatasetReadMode.CHUNKED:
            raise AssertionError("crypto_panel_matching_requires_chunked_reads")
        if root_name == "market_bars_1m":
            return iter((frames["bars"],))
        if root_name == "aggregate_trades_tick":
            return iter((frames["trades"],))
        if root_name == "depth_levels_tick":
            return iter((frames["depth"],))
        raise AssertionError(f"unexpected_dataset_root:{root_name}")

    monkeypatch.setattr(materialize_panel, "read_dataset", fake_read_dataset)

    with pytest.raises(ValueError, match=error):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )


def test_synthetic_crypto_normalized_interim_can_generate_non_binance_panel(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_synthetic_crypto_interim(layout, source_id)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
        data_ids=("market_panel_1m",),
    )

    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    assert panel["asset_id"].tolist() == ["btc_usdt"]
    assert "source_asset_id" not in panel.columns


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


def _panel_frame(*, minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-06 00:00:00", periods=minutes, freq="min")
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * minutes,
            "asset_id": ["btc_usdt"] * minutes,
            "timestamp": timestamps,
            "open": [100.0 + index for index in range(minutes)],
            "high": [101.0 + index for index in range(minutes)],
            "low": [99.0 + index for index in range(minutes)],
            "close": [100.5 + index for index in range(minutes)],
            "volume": [10.0] * minutes,
            "quote_volume": [1005.0] * minutes,
            "trade_count": [2.0] * minutes,
            "trade_flow_observed": [True] * minutes,
            "taker_buy_volume": [4.0] * minutes,
            "taker_sell_volume": [6.0] * minutes,
            "signed_volume": [1.0] * minutes,
            "best_bid": [100.0 + index for index in range(minutes)],
            "best_ask": [100.2 + index for index in range(minutes)],
            "quote_observed": [True] * minutes,
            "quote_age_seconds": [1.0] * minutes,
            "bid_quantity": [1.5] * minutes,
            "ask_quantity": [1.0] * minutes,
            "bid_depth": [1.5] * minutes,
            "ask_depth": [1.0] * minutes,
            "mid_price": [100.1 + index for index in range(minutes)],
            "eligible": [True] * minutes,
        }
    )


def _matching_panel_frames() -> dict[str, pd.DataFrame]:
    bars = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:00"]),
            "close_time": pd.to_datetime(["2026-07-06 00:00:59"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
            "quote_volume": [1005.0],
            "trade_count": [1],
            "taker_buy_volume": [4.0],
            "taker_buy_quote_volume": [402.0],
        }
    )
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [2.0],
        }
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )
    return {"bars": bars, "trades": trades, "depth": depth}


def _existing_processed_frame(spec, *, minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-06 00:00:00", periods=minutes, freq="min")
    values: dict[str, object] = {}
    for column in spec.required_columns:
        if column == "market_id":
            values[column] = ["crypto"] * minutes
        elif column == "asset_id":
            values[column] = ["btc_usdt"] * minutes
        elif column == "timestamp":
            values[column] = timestamps
        else:
            values[column] = [1.0] * minutes
    return pd.DataFrame(values)


def _write_synthetic_crypto_interim(layout: StorageLayout, source_id: SourceId) -> None:
    write_dataset(
        layout.interim_data_root(source_id, "market_bars_1m"),
        pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"],
                "timestamp": pd.to_datetime(["2026-07-06 00:00:00"]),
                "open_time": pd.to_datetime(["2026-07-06 00:00:00"]),
                "close_time": pd.to_datetime(["2026-07-06 00:00:59"]),
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
        ),
        spec=_interim_spec(
            source_id,
            data_id="market_bars_1m",
            columns=(
                "source_asset_id",
                "timestamp",
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trade_count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "source_market_type",
                "interval",
            ),
            primary_key=("source_asset_id", "timestamp"),
            frequency=DataFrequency.MINUTE,
            grain=DataGrain.ASSET_TIMESTAMP,
        ),
    )
    write_dataset(
        layout.interim_data_root(source_id, "aggregate_trades_tick"),
        pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"],
                "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
                "aggregate_trade_id": [1],
                "price": [100.1],
                "quantity": [2.0],
                "quote_quantity": [200.2],
                "first_trade_id": [1],
                "last_trade_id": [1],
                "is_buyer_maker": [False],
                "taker_side": ["buy"],
                "signed_quantity": [2.0],
                "source_market_type": ["spot"],
            }
        ),
        spec=_interim_spec(
            source_id,
            data_id="aggregate_trades_tick",
            columns=(
                "source_asset_id",
                "timestamp",
                "aggregate_trade_id",
                "price",
                "quantity",
                "quote_quantity",
                "first_trade_id",
                "last_trade_id",
                "is_buyer_maker",
                "taker_side",
                "signed_quantity",
                "source_market_type",
            ),
            primary_key=("source_asset_id", "timestamp", "aggregate_trade_id"),
            frequency=DataFrequency.TICK,
            grain=DataGrain.TRADE_EVENT,
        ),
    )
    write_dataset(
        layout.interim_data_root(source_id, "depth_levels_tick"),
        pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT", "BTCUSDT"],
                "timestamp": pd.to_datetime(
                    ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
                ),
                "side": ["bid", "ask"],
                "level": [1, 1],
                "price": [100.0, 100.2],
                "quantity": [1.5, 1.0],
                "source_market_type": ["spot", "spot"],
            }
        ),
        spec=_interim_spec(
            source_id,
            data_id="depth_levels_tick",
            columns=(
                "source_asset_id",
                "timestamp",
                "side",
                "level",
                "price",
                "quantity",
                "source_market_type",
            ),
            primary_key=("source_asset_id", "timestamp", "side", "level"),
            frequency=DataFrequency.TICK,
            grain=DataGrain.ORDER_BOOK_LEVEL,
        ),
    )


def _interim_spec(
    source_id: SourceId,
    *,
    data_id: str,
    columns: tuple[str, ...],
    primary_key: tuple[str, ...],
    frequency: DataFrequency,
    grain: DataGrain,
) -> InterimDataSpec:
    return InterimDataSpec(
        data_id=data_id,
        source_id=source_id,
        frequency=frequency,
        grain=grain,
        primary_key=primary_key,
        partition_columns=("year",),
        column_roles={
            column: (
                ColumnRole.TIMESTAMP
                if column == "timestamp"
                else ColumnRole.KEY
                if column in primary_key
                else ColumnRole.CONTEXT
            )
            for column in columns
        },
    )


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
        "trade_flow_observed",
        "quote_observed",
        "quote_age_seconds",
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
    missing_depth_rows = panel.loc[panel["asset_id"].eq("eth_usdt")]
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
