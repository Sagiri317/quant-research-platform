from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
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
    materialization_resources,
    materialization_validation,
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
    inspect_dataset,
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
    panel_inspection = inspect_dataset(
        layout.processed_data_root(
            "crypto", ProcessedDataKind.PANEL, "market_panel_1m"
        )
    )
    factor_inspection = inspect_dataset(
        layout.processed_data_root(
            "crypto", ProcessedDataKind.FACTOR, "microstructure"
        )
    )
    label_inspection = inspect_dataset(
        layout.processed_data_root(
            "crypto", ProcessedDataKind.LABEL, "forward_returns"
        )
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
    assert set(panel_inspection.manifest["input_dataset_identities"]) == {
        "aggregate_trades_tick",
        "depth_levels_tick",
        "market_bars_1m",
    }
    assert set(factor_inspection.manifest["input_dataset_identities"]) == {
        "market_panel_1m"
    }
    assert set(label_inspection.manifest["input_dataset_identities"]) == {
        "market_panel_1m"
    }


def test_crypto_processed_materializer_uses_policy_bounded_physical_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout, row_batch_size=2)
    calls: list[tuple[str, str, int]] = []
    policy = RuntimeResourcePolicy(
        available_memory_bytes=1024**3,
        chunk_target_bytes=128 * 1024,
        memory_safety_factor=1.0,
    )

    def spy(module_name: str, module):
        real_iterator = module.iter_dataset_chunks

        def wrapped(context, *, target_chunk_bytes):
            calls.append((module_name, context.root.name, target_chunk_bytes))
            return real_iterator(
                context,
                target_chunk_bytes=target_chunk_bytes,
            )

        return wrapped

    for name, module in (
        ("panel", materialize_panel),
        ("factors", materialize_factors),
        ("labels", materialize_labels),
        ("validation", materialization_validation),
    ):
        monkeypatch.setattr(module, "iter_dataset_chunks", spy(name, module))

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
        resource_policy=policy,
    )

    assert {item[:2] for item in calls}.issuperset(
        {
            ("panel", "market_bars_1m"),
            ("panel", "aggregate_trades_tick"),
            ("panel", "depth_levels_tick"),
            ("factors", "market_panel_1m"),
            ("labels", "market_panel_1m"),
            ("validation", "market_panel_1m"),
        }
    )
    assert {target for _, _, target in calls} == {
        materialization_resources.MaterializationBudget.from_policy(
            policy
        ).physical_batch_bytes
    }


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
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=30),
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
        source_id=source_id,
        time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
        data_ids=("microstructure",),
    )

    assert sorted(result) == ["market_panel_1m", "microstructure"]
    assert calls == [
        ("factors", TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"))
    ]


def test_crypto_materializer_rejects_cross_source_panel_dependency_reuse(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    first_source = SourceId("synthetic_crypto_a")
    second_source = SourceId("synthetic_crypto_b")
    _write_existing_panel_dependency(
        layout,
        first_source,
        _panel_frame(minutes=30),
    )
    _write_synthetic_crypto_interim(layout, second_source, minutes=30)

    with pytest.raises(
        ValueError,
        match="crypto_panel_input_dataset_identity_mismatch",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=second_source,
            time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
            data_ids=("microstructure",),
        )


def test_crypto_materializer_rejects_insufficient_existing_panel_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=12),
        source_minutes=30,
    )

    def fail_panel_write(*_: Any, **__: Any) -> object:
        raise AssertionError("unexpected_crypto_dependency_panel_write")

    monkeypatch.setattr(materialize_panel, "write_panel", fail_panel_write)

    with pytest.raises(ValueError, match="invalid_crypto_panel_key_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
            data_ids=("microstructure",),
        )


def test_crypto_materializer_invalid_dependency_timestamp_is_not_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=30),
    )
    real_iterator = materialization_validation.iter_dataset_chunks

    def invalid_target_read(context, **kwargs):
        if context.root.name == "market_panel_1m":
            return iter(
                (
                    pd.DataFrame(
                        {
                            "market_id": ["crypto"],
                            "asset_id": ["btc_usdt"],
                            "timestamp": ["bad"],
                        }
                    ),
                )
            )
        return real_iterator(context, **kwargs)

    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_chunks",
        invalid_target_read,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(
            ValueError,
            match="invalid_crypto_panel_key_coverage",
        ):
            materializer.materialize_processed_data(
                layout=layout,
                source_id=source_id,
                time_window=TimeWindow(
                    "2026-07-06T00:10:00",
                    "2026-07-06T00:12:00",
                ),
                data_ids=("microstructure",),
            )


def test_crypto_materializer_rejects_panel_dependency_with_middle_minute_gap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    gapped = pd.concat(
        (
            _panel_frame(minutes=1).assign(timestamp=pd.Timestamp("2026-07-06 00:06")),
            _panel_frame(minutes=1).assign(timestamp=pd.Timestamp("2026-07-06 00:12")),
        ),
        ignore_index=True,
    )
    _write_existing_panel_dependency(
        layout,
        source_id,
        gapped,
        source_minutes=30,
    )

    def fail_panel_write(*_: Any, **__: Any) -> object:
        raise AssertionError("unexpected_crypto_dependency_panel_write")

    monkeypatch.setattr(materialize_panel, "write_panel", fail_panel_write)

    with pytest.raises(ValueError, match="invalid_crypto_panel_key_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
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
    full_panel = _panel_frame(minutes=30)
    expected_factors = calculate_microstructure_factors(full_panel).loc[
        lambda frame: frame["timestamp"].between(
            pd.Timestamp("2026-07-06 00:10:00"),
            pd.Timestamp("2026-07-06 00:12:00"),
        ),
        factor_spec.required_columns,
    ]
    expected_labels = add_forward_return_labels(
        full_panel,
        horizons=CRYPTO_FORWARD_RETURN_HORIZONS,
        frequency=DataFrequency.MINUTE,
        execution_price_column="close",
        expected_interval=pd.Timedelta(minutes=1),
        require_contiguous=True,
    ).loc[
        lambda frame: frame["timestamp"].between(
            pd.Timestamp("2026-07-06 00:10:00"),
            pd.Timestamp("2026-07-06 00:12:00"),
        ),
        label_spec.required_columns,
    ]
    pd.testing.assert_frame_equal(
        factors.reset_index(drop=True),
        expected_factors.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        labels.reset_index(drop=True),
        expected_labels.reset_index(drop=True),
        check_dtype=False,
    )


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

    for module in (materialize_factors, materialize_labels):
        monkeypatch.setattr(
            module,
            "build_dataset_read_context",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(module, "iter_dataset_chunks", read_panel)
        monkeypatch.setattr(
            module,
            "build_dataset_equality_group_plan",
            lambda *_args, **_kwargs: SimpleNamespace(
                keys=(("crypto", "btc_usdt"),)
            ),
        )
        monkeypatch.setattr(
            module,
            "dataset_context_for_equality_group",
            lambda context, _plan, _key: context,
        )
    policy = RuntimeResourcePolicy(available_memory_bytes=1024**3)

    factors = pd.concat(
        materialize_factors._iter_factor_chunks(
            processed_catalog().get("microstructure"),
            panel_root=Path("unused"),
            time_window=core,
            resource_policy=policy,
        ),
        ignore_index=True,
    )
    labels = pd.concat(
        materialize_labels._iter_label_chunks(
            processed_catalog().get("forward_returns"),
            panel_root=Path("unused"),
            time_window=core,
            resource_policy=policy,
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

    _patch_matching_panel_reads(monkeypatch, frames)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match=error):
            materialize_panel.write_panel(
                processed_catalog().get("market_panel_1m"),
                layout=layout,
                source_id=SourceId("synthetic_crypto"),
                time_window=TimeWindow(
                    "2026-07-06T00:00:00",
                    "2026-07-06T00:00:00",
                ),
                resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
            )


@pytest.mark.parametrize(
    ("frame_name", "error"),
    [
        ("bars", "invalid_crypto_matching_bar_source_asset_id"),
        ("trades", "invalid_crypto_matching_trade_source_asset_id"),
        ("depth", "invalid_crypto_matching_depth_source_asset_id"),
    ],
)
@pytest.mark.parametrize(
    "bad_source_asset_id",
    [None, pd.NA, "", [], {}, set()],
    ids=["none", "pd_na", "empty", "list", "dict", "set"],
)
def test_crypto_panel_matching_rejects_invalid_source_asset_id(
    tmp_path: Path,
    monkeypatch,
    frame_name: str,
    error: str,
    bad_source_asset_id: object,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    frames = _matching_panel_frames()
    frames[frame_name]["source_asset_id"] = frames[frame_name][
        "source_asset_id"
    ].astype(object)
    frames[frame_name].at[0, "source_asset_id"] = bad_source_asset_id
    _patch_matching_panel_reads(monkeypatch, frames)

    with pytest.raises(ValueError, match=error):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )


@pytest.mark.parametrize(
    ("frame_name", "error"),
    [
        ("bars", "invalid_crypto_matching_bar_source_asset_id"),
        ("trades", "invalid_crypto_matching_trade_source_asset_id"),
        ("depth", "invalid_crypto_matching_depth_source_asset_id"),
    ],
)
def test_crypto_panel_matching_rejects_missing_source_asset_id_column(
    tmp_path: Path,
    monkeypatch,
    frame_name: str,
    error: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    frames = _matching_panel_frames()
    frames[frame_name] = frames[frame_name].drop(columns=["source_asset_id"])
    _patch_matching_panel_reads(monkeypatch, frames)

    with pytest.raises(ValueError, match=error):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )


def test_crypto_panel_matching_accepts_valid_source_asset_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _patch_matching_panel_reads(monkeypatch, _matching_panel_frames())

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("synthetic_crypto"),
        time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
        resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
    )

    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    assert panel["asset_id"].tolist() == ["btc_usdt"]
    assert panel["trade_flow_observed"].tolist() == [True]
    assert panel["quote_observed"].tolist() == [True]
    assert "source_asset_id" not in panel.columns


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


def test_crypto_panel_reads_auxiliary_data_once_per_core_window(
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
    bar_scans = {"asset_discovery": 0, "core": 0}
    group_plan_calls = {"count": 0}
    real_iterator = materialize_panel.iter_dataset_chunks
    real_group_plan = materialize_panel.build_dataset_equality_group_plan

    def wrapped_iterator(context, *, target_chunk_bytes):
        root_name = context.root.name
        if root_name in call_counts:
            call_counts[root_name] += 1
        result = real_iterator(
            context,
            target_chunk_bytes=target_chunk_bytes,
        )
        if root_name == "market_bars_1m":
            if context.logical_columns == ("source_asset_id",):
                bar_scans["asset_discovery"] += 1
                return result
            bar_scans["core"] += 1
            return _count_yielded_chunks(result, yielded_chunks, root_name)
        return result

    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", wrapped_iterator)

    def wrapped_group_plan(*args, **kwargs):
        group_plan_calls["count"] += 1
        return real_group_plan(*args, **kwargs)

    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_equality_group_plan",
        wrapped_group_plan,
    )

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=RuntimeResourcePolicy(
            available_memory_bytes=1024**2,
            chunk_target_bytes=64 * 1024,
            memory_safety_factor=1.0,
        ),
    )

    assert call_counts["market_bars_1m"] == 2
    assert bar_scans == {"asset_discovery": 0, "core": 2}
    assert group_plan_calls["count"] == 1
    assert yielded_chunks["market_bars_1m"] > 1
    assert call_counts["aggregate_trades_tick"] == yielded_chunks["market_bars_1m"]
    assert call_counts["depth_levels_tick"] == yielded_chunks["market_bars_1m"]


def test_crypto_panel_materializer_never_concatenates_raw_auxiliary_chunks() -> None:
    source = inspect.getsource(materialize_panel)
    assert "pd.concat(" not in source


def test_crypto_panel_materializer_validates_complete_trades_across_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames = _matching_panel_frames()
    frames["bars"].loc[0, "trade_count"] = 2
    trades = pd.concat(
        (
            frames["trades"].assign(
                quantity=6.0,
                signed_quantity=6.0,
            ),
            frames["trades"].assign(
                timestamp=pd.Timestamp("2026-07-06 00:00:02"),
                aggregate_trade_id=2,
                quantity=4.0,
                first_trade_id=2,
                last_trade_id=2,
                signed_quantity=-4.0,
            ),
        ),
        ignore_index=True,
    )
    _patch_panel_physical_chunks(
        monkeypatch,
        {
            "market_bars_1m": (frames["bars"],),
            "aggregate_trades_tick": (
                trades.iloc[:1].copy(),
                trades.iloc[1:].copy(),
            ),
            "depth_levels_tick": (frames["depth"],),
        },
    )
    _patch_panel_lifecycle_validation(monkeypatch)
    layout = StorageLayout(project_root=tmp_path)

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("synthetic_crypto"),
        time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
        resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
    )

    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    assert panel.loc[0, "signed_volume"] == 2.0
    assert bool(panel.loc[0, "trade_flow_observed"]) is True


def test_crypto_panel_materializer_rejects_incomplete_trade_range_across_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames = _matching_panel_frames()
    frames["bars"].loc[0, "trade_count"] = 2
    second = frames["trades"].assign(
        timestamp=pd.Timestamp("2026-07-06 00:00:02"),
        aggregate_trade_id=2,
        quantity=4.0,
        first_trade_id=3,
        last_trade_id=3,
        signed_quantity=-4.0,
    )
    first = frames["trades"].assign(quantity=6.0, signed_quantity=6.0)
    _patch_panel_physical_chunks(
        monkeypatch,
        {
            "market_bars_1m": (frames["bars"],),
            "aggregate_trades_tick": (first, second),
            "depth_levels_tick": (frames["depth"],),
        },
    )
    _patch_panel_lifecycle_validation(monkeypatch)

    with pytest.raises(ValueError, match="invalid_crypto_trade_flow_coverage"):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow(
                "2026-07-06T00:00:00",
                "2026-07-06T00:00:00",
            ),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )


def test_crypto_panel_materializer_merges_depth_snapshot_across_storage_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames = _matching_panel_frames()
    level_two = frames["depth"].assign(
        level=2,
        price=[100.3, 99.9],
        quantity=[1.0, 1.0],
    )
    depth = pd.concat((frames["depth"], level_two), ignore_index=True)
    depth = depth.sort_values(
        ["source_asset_id", "timestamp", "side", "level"],
        kind="mergesort",
    ).reset_index(drop=True)

    _patch_panel_physical_chunks(
        monkeypatch,
        {
            "market_bars_1m": (frames["bars"],),
            "aggregate_trades_tick": (frames["trades"],),
            "depth_levels_tick": (
                depth.iloc[:2].copy(),
                depth.iloc[2:].copy(),
            ),
        },
    )
    _patch_panel_lifecycle_validation(monkeypatch)
    layout = StorageLayout(project_root=tmp_path)

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("synthetic_crypto"),
        time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
        resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
    )

    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    assert panel.loc[0, "bid_depth"] == 2.5
    assert panel.loc[0, "ask_depth"] == 2.0


def test_crypto_panel_bar_contract_fails_before_auxiliary_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frames = _matching_panel_frames()
    frames["bars"].loc[0, "interval"] = "5m"
    auxiliary_reads: list[str] = []

    def fake_iterator(context, *, target_chunk_bytes):
        assert target_chunk_bytes > 0
        root_name = Path(context).name
        if root_name == "market_bars_1m":
            return iter((frames["bars"],))
        auxiliary_reads.append(root_name)
        return iter(())

    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_read_context",
        lambda dataset_root, **_kwargs: Path(dataset_root),
    )
    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", fake_iterator)
    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_equality_group_plan",
        lambda *_args, **_kwargs: SimpleNamespace(keys=(("BTCUSDT",),)),
    )
    monkeypatch.setattr(
        materialize_panel,
        "dataset_context_for_equality_group",
        lambda context, _plan, _key: context,
    )
    _patch_panel_lifecycle_validation(monkeypatch)

    with pytest.raises(ValueError, match="invalid_crypto_bar_interval"):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )

    assert auxiliary_reads == []


def test_crypto_panel_small_and_large_resource_windows_match(tmp_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for name, policy in (
        (
            "large",
            RuntimeResourcePolicy(available_memory_bytes=1024**3),
        ),
        (
            "small",
                RuntimeResourcePolicy(
                    available_memory_bytes=1024**2,
                    chunk_target_bytes=64 * 1024,
                    memory_safety_factor=1.0,
                ),
        ),
    ):
        project_root = tmp_path / name
        _write_binance_fixture(project_root)
        layout = StorageLayout(project_root=project_root)
        materialize_interim_data(layout=layout, row_batch_size=3)
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=SourceId("binance"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
        )
        frames.append(
            read_dataset(
                layout.processed_data_root(
                    "crypto",
                    ProcessedDataKind.PANEL,
                    "market_panel_1m",
                )
            )
        )

    pd.testing.assert_frame_equal(frames[0], frames[1])


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
            "open_time": pd.to_datetime(["2026-07-06 00:00:00"]),
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
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "aggregate_trade_id": [1],
            "quantity": [10.0],
            "first_trade_id": [1],
            "last_trade_id": [1],
            "signed_quantity": [10.0],
            "source_market_type": ["spot"],
        }
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["ask", "bid"],
            "level": [1, 1],
            "price": [100.2, 100.0],
            "quantity": [1.0, 1.5],
            "source_market_type": ["spot", "spot"],
        }
    )
    return {"bars": bars, "trades": trades, "depth": depth}


def _patch_matching_panel_reads(monkeypatch, frames: dict[str, pd.DataFrame]) -> None:
    _patch_panel_physical_chunks(
        monkeypatch,
        {
            "market_bars_1m": (frames["bars"],),
            "aggregate_trades_tick": (frames["trades"],),
            "depth_levels_tick": (frames["depth"],),
        },
    )
    _patch_panel_lifecycle_validation(monkeypatch)


def _patch_panel_physical_chunks(
    monkeypatch,
    chunks_by_root: dict[str, tuple[pd.DataFrame, ...]],
) -> None:
    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_read_context",
        lambda dataset_root, **_kwargs: Path(dataset_root),
    )

    bars = chunks_by_root["market_bars_1m"]
    source_asset_ids = tuple(
        (value,)
        for chunk in bars
        for value in (
            chunk["source_asset_id"].tolist()
            if "source_asset_id" in chunk
            else [None]
        )
    )
    plan = SimpleNamespace(keys=source_asset_ids)
    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_equality_group_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        materialize_panel,
        "dataset_context_for_equality_group",
        lambda context, _plan, key: (context, key[0]),
    )

    def fake_iterator(context, *, target_chunk_bytes):
        assert target_chunk_bytes > 0
        dataset_context, source_asset_id = (
            context if isinstance(context, tuple) else (context, None)
        )
        root_name = Path(dataset_context).name
        if root_name not in chunks_by_root:
            raise AssertionError(f"unexpected_dataset_root:{root_name}")
        chunks = chunks_by_root[root_name]
        if source_asset_id is None:
            return iter(chunks)
        return iter(
            tuple(
                chunk.loc[chunk["source_asset_id"].eq(source_asset_id)].copy()
                for chunk in chunks
                if chunk["source_asset_id"].eq(source_asset_id).any()
            )
        )

    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", fake_iterator)


def _patch_panel_lifecycle_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        materialization_validation,
        "panel_input_inspections",
        lambda **_kwargs: {"synthetic": object()},
    )
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        lambda _inspections: {
            "synthetic": {
                "data_id": "synthetic",
                "manifest_hash": "0" * 64,
                "spec_hash": "1" * 64,
            }
        },
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_materialization",
        lambda *_args, **_kwargs: None,
    )


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


def _write_synthetic_crypto_interim(
    layout: StorageLayout,
    source_id: SourceId,
    *,
    minutes: int = 1,
) -> None:
    timestamps = pd.date_range("2026-07-06 00:00:00", periods=minutes, freq="min")
    write_dataset(
        layout.interim_data_root(source_id, "market_bars_1m"),
        pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"] * minutes,
                "timestamp": timestamps,
                "open_time": timestamps,
                "close_time": timestamps + pd.Timedelta(milliseconds=59_999),
                "open": [100.0] * minutes,
                "high": [101.0] * minutes,
                "low": [99.0] * minutes,
                "close": [100.5] * minutes,
                "volume": [10.0] * minutes,
                "quote_volume": [1005.0] * minutes,
                "trade_count": [1] * minutes,
                "taker_buy_volume": [4.0] * minutes,
                "taker_buy_quote_volume": [402.0] * minutes,
                "source_market_type": ["spot"] * minutes,
                "interval": ["1m"] * minutes,
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
                "quantity": [10.0],
                "quote_quantity": [1001.0],
                "first_trade_id": [1],
                "last_trade_id": [1],
                "is_buyer_maker": [False],
                "taker_side": ["buy"],
                "signed_quantity": [10.0],
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
                "side": ["ask", "bid"],
                "level": [1, 1],
                "price": [100.2, 100.0],
                "quantity": [1.0, 1.5],
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


def _write_existing_panel_dependency(
    layout: StorageLayout,
    source_id: SourceId,
    panel: pd.DataFrame,
    *,
    source_minutes: int | None = None,
) -> None:
    _write_synthetic_crypto_interim(
        layout,
        source_id,
        minutes=source_minutes if source_minutes is not None else len(panel),
    )
    panel_spec = processed_catalog().get("market_panel_1m")
    identities = materialization_validation.input_dataset_identities(
        materialization_validation.panel_input_inspections(
            layout=layout,
            source_id=source_id,
            spec=panel_spec,
        )
    )
    write_dataset(
        layout.processed_data_root(
            "crypto",
            ProcessedDataKind.PANEL,
            "market_panel_1m",
        ),
        panel,
        spec=panel_spec,
        instance_metadata={"input_dataset_identities": identities},
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


def test_panel_chunks_are_globally_ordered_after_canonical_asset_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get("market_panel_1m")
    timestamps = pd.to_datetime(["2026-07-06 00:00"])
    bar_chunks = (
        pd.DataFrame({"source_asset_id": ["ETHFIUSDT"], "timestamp": timestamps}),
        pd.DataFrame({"source_asset_id": ["ETHUSDT"], "timestamp": timestamps}),
    )
    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_read_context",
        lambda *_args, **kwargs: kwargs,
    )

    def iter_bars(context, **_kwargs):
        filters = context.get("row_filters", ())
        if not filters:
            return iter(tuple(chunk.copy() for chunk in bar_chunks))
        source_asset_id = str(filters[0][2])
        return iter(
            tuple(
                chunk.loc[chunk["source_asset_id"].eq(source_asset_id)].copy()
                for chunk in bar_chunks
                if chunk["source_asset_id"].eq(source_asset_id).any()
            )
        )

    monkeypatch.setattr(
        materialize_panel,
        "iter_dataset_chunks",
        iter_bars,
    )
    group_plan = SimpleNamespace(keys=(("ETHFIUSDT",), ("ETHUSDT",)))
    monkeypatch.setattr(
        materialize_panel,
        "build_dataset_equality_group_plan",
        lambda *_args, **_kwargs: group_plan,
    )

    def group_context(context, _plan, key):
        return {
            **context,
            "row_filters": (("source_asset_id", "==", key[0]),),
        }

    monkeypatch.setattr(
        materialize_panel,
        "dataset_context_for_equality_group",
        group_context,
    )
    monkeypatch.setattr(
        materialize_panel,
        "_validate_matching_bar_keys",
        lambda _x: None,
    )

    def prepare(bars: pd.DataFrame) -> pd.DataFrame:
        asset_id = {
            "ETHFIUSDT": "ethfi_usdt",
            "ETHUSDT": "eth_usdt",
        }[str(bars.loc[0, "source_asset_id"])]
        output = pd.DataFrame(
            {
                "market_id": ["crypto"],
                "asset_id": [asset_id],
                "timestamp": bars["timestamp"].reset_index(drop=True),
            }
        )
        for column in spec.required_columns:
            if column not in output:
                output[column] = 1.0
        return output

    monkeypatch.setattr(materialize_panel.panels, "prepare_panel_bars", prepare)
    monkeypatch.setattr(
        materialize_panel,
        "_iter_matching_trade_chunks",
        lambda *_args, **_kwargs: iter(()),
    )
    monkeypatch.setattr(
        materialize_panel,
        "_iter_matching_depth_chunks",
        lambda *_args, **_kwargs: iter(()),
    )
    monkeypatch.setattr(
        materialize_panel.panels,
        "calculate_market_panel_from_chunks",
        lambda panel, **_kwargs: panel.sort_values(
            ["market_id", "asset_id", "timestamp"], kind="mergesort"
        ).reset_index(drop=True),
    )

    chunks = list(
        materialize_panel._iter_panel_chunks(
            spec,
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("binance"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=RuntimeResourcePolicy(available_memory_bytes=1024**3),
        )
    )
    keys = pd.concat(chunks, ignore_index=True).loc[
        :, ["market_id", "asset_id", "timestamp"]
    ]

    pd.testing.assert_frame_equal(
        keys,
        keys.sort_values(
            ["market_id", "asset_id", "timestamp"], kind="mergesort"
        ).reset_index(drop=True),
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
            "q": "10.0",
            "f": offset * 7,
            "l": offset * 7 + 6,
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
