from __future__ import annotations

import json
import warnings
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace as _SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import quant_research_platform.storage.atomic as storage_atomic
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    materializer,
    source_ids,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_profiles import (
    build_research_dataset_catalog,
)
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.factors.microstructure.calculations import (
    calculate_microstructure_factors,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage._dataset_reading import open_dataset_session
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


class SimpleNamespace(_SimpleNamespace):
    """Dataset-session test double with the production context protocol."""

    retained_memory_bytes = 0

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *_args: object) -> None:
        close = getattr(self, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize(
    ("module", "writer_name", "data_id"),
    (
        (materialize_panel, "write_panel", MARKET_PANEL_DATA_ID),
        (materialize_factors, "write_factors", MICROSTRUCTURE_PROCESSED_DATA_ID),
        (materialize_labels, "write_labels", FORWARD_RETURNS_PROCESSED_DATA_ID),
    ),
)
def test_crypto_writer_requires_artifact_root_before_scratch_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    data_id: str,
) -> None:  # noqa: ANN001
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    monkeypatch.setattr(
        module,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
    )
    spec = processed_catalog().get(data_id)
    kwargs = {
        "layout": StorageLayout(project_root=tmp_path),
        "time_window": TimeWindow("2024-01-01", "2024-01-02"),
        "resource_policy": policy,
        "retained_payload_bytes": 0,
    }
    if writer_name == "write_panel":
        kwargs["source_id"] = SourceId("binance")
    else:
        kwargs["panel_session"] = SimpleNamespace(resource_policy=policy)

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        getattr(module, writer_name)(spec, **kwargs)


def test_existing_panel_validation_keeps_policy_active_across_both_iterators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    observed: list[tuple[str, int, bool]] = []
    bar_session = SimpleNamespace(
        name="bar", resource_policy=policy, retained_memory_bytes=100
    )
    panel_session = SimpleNamespace(
        name="panel",
        resource_policy=policy,
        manifest={},
        retained_memory_bytes=200,
    )

    def chunks(
        session: SimpleNamespace,
        *,
        target_chunk_bytes: int,
        operation_headroom_bytes: int,
    ):
        assert target_chunk_bytes == policy.remaining_memory_bytes
        expected_external_bytes = 200 if session.name == "bar" else 100
        assert operation_headroom_bytes == (
            policy.task_memory_budget_bytes - expected_external_bytes
        )
        count = 1 if session.name == "bar" else 2
        with runtime_resource_policy_scope(session.resource_policy):
            for index in range(count):
                observed.append(
                    (
                        session.name,
                        index,
                        current_runtime_resource_policy() is policy,
                    )
                )
                yield pd.DataFrame({"value": [index]})

    def validate(expected, actual, **_kwargs):  # noqa: ANN001
        next(expected)
        next(actual)
        with pytest.raises(StopIteration):
            next(expected)
        next(actual)
        with pytest.raises(StopIteration):
            next(actual)

    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(materialization_validation, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        materialization_validation,
        "_iter_panel_primary_key_chunks",
        lambda frames, **_kwargs: (frame for frame in frames),
    )
    monkeypatch.setattr(
        materialization_validation,
        "_validate_ordered_key_stream_coverage",
        validate,
    )

    materialization_validation.validate_existing_panel_materialization(
        panel_session,
        bar_session=bar_session,
        spec=SimpleNamespace(primary_key=("value",)),
        expected_input_identities={},
        resource_policy=policy,
        retained_payload_bytes=0,
    )

    assert observed == [
        ("bar", 0, True),
        ("panel", 0, True),
        ("panel", 1, True),
    ]
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_existing_panel_validation_retires_policy_after_every_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    primary = primary_type("validation_primary")
    session = SimpleNamespace(resource_policy=policy, manifest={})

    def chunks(*_args: object, **_kwargs: object):
        with runtime_resource_policy_scope(policy):
            assert current_runtime_resource_policy() is policy
            yield pd.DataFrame({"value": [1]})

    def fail(expected, actual, **_kwargs):  # noqa: ANN001
        next(expected)
        next(actual)
        assert current_runtime_resource_policy() is policy
        raise primary

    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(materialization_validation, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(
        materialization_validation,
        "_iter_panel_primary_key_chunks",
        lambda frames, **_kwargs: (frame for frame in frames),
    )
    monkeypatch.setattr(
        materialization_validation,
        "_validate_ordered_key_stream_coverage",
        fail,
    )

    with pytest.raises(primary_type) as raised:
        materialization_validation.validate_existing_panel_materialization(
            session,
            bar_session=session,
            spec=SimpleNamespace(primary_key=("value",)),
            expected_input_identities={},
            resource_policy=policy,
            retained_payload_bytes=0,
        )
    assert raised.value is primary
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


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
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    factor_inspection = inspect_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.FACTOR, "microstructure")
    )
    label_inspection = inspect_dataset(
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
    monkeypatch.setattr(
        materializer,
        "dataset_inspection_retained_memory_bytes",
        lambda _inspection: 1,
    )

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **__: Any) -> object:
            calls.append((name, time_window))
            return object()

        return wrapped

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_factors, "write_factors", record("factors"))
    monkeypatch.setattr(
        materializer,
        "open_dataset_session",
        lambda *_args, **_kwargs: SimpleNamespace(verify=lambda: None),
    )

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
    monkeypatch.setattr(
        materializer,
        "dataset_inspection_retained_memory_bytes",
        lambda _inspection: 1,
    )

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **__: Any) -> object:
            calls.append((name, time_window))
            return object()

        return wrapped

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_labels, "write_labels", record("labels"))
    monkeypatch.setattr(
        materializer,
        "open_dataset_session",
        lambda *_args, **_kwargs: SimpleNamespace(verify=lambda: None),
    )

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
    monkeypatch.setattr(
        materializer,
        "dataset_inspection_retained_memory_bytes",
        lambda _inspection: 1,
    )
    shared_panel_session = SimpleNamespace(verify=lambda: None)
    panel_opens = 0

    def record(name: str):
        def wrapped(*_: Any, time_window: TimeWindow, **kwargs: Any) -> object:
            if name != "panel":
                assert kwargs["panel_session"] is shared_panel_session
            calls.append((name, time_window))
            return object()

        return wrapped

    def open_panel(*_args, **_kwargs):
        nonlocal panel_opens
        panel_opens += 1
        return shared_panel_session

    monkeypatch.setattr(materialize_panel, "write_panel", record("panel"))
    monkeypatch.setattr(materialize_factors, "write_factors", record("factors"))
    monkeypatch.setattr(materialize_labels, "write_labels", record("labels"))
    monkeypatch.setattr(materializer, "open_dataset_session", open_panel)

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
    assert panel_opens == 1


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


@pytest.mark.parametrize(
    ("data_id", "kind"),
    (
        ("microstructure", ProcessedDataKind.FACTOR),
        ("forward_returns", ProcessedDataKind.LABEL),
    ),
)
def test_crypto_materializer_does_not_reverify_panel_after_dependent_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_id: str,
    kind: ProcessedDataKind,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=70),
    )
    _write_existing_panel_dependency(
        replacement_layout,
        source_id,
        _panel_frame(minutes=71),
    )
    panel_spec = processed_catalog().get("market_panel_1m")
    current_root = layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_materialize_one = materializer._materialize_one
    replaced = False

    def replace_after_dependent_publish(spec, **kwargs):
        nonlocal replaced
        inspection = real_materialize_one(spec, **kwargs)
        if spec.data_id == data_id:
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True
        return inspection

    monkeypatch.setattr(
        materializer,
        "_materialize_one",
        replace_after_dependent_publish,
    )

    result = materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00"),
        data_ids=(data_id,),
    )

    final_root = layout.processed_data_root("crypto", kind, data_id)
    assert replaced
    assert sorted(result) == sorted(("market_panel_1m", data_id))
    assert final_root.exists()
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(final_root)
    assert not list(final_root.parent.glob(f"{scratch_prefix}*"))


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

            def invalid_chunks():
                yield pd.DataFrame(
                    {
                        "market_id": ["crypto"],
                        "asset_id": ["btc_usdt"],
                        "timestamp": ["bad"],
                    }
                )

            return invalid_chunks()
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
    monkeypatch.setattr(
        materializer,
        "dataset_inspection_retained_memory_bytes",
        lambda _inspection: 1,
    )
    shared_panel_session = SimpleNamespace(verify=lambda: None)

    def record_panel(*_: Any, time_window: TimeWindow, **__: Any) -> object:
        calls.append(("panel", time_window))
        return object()

    def record_factors(
        *_: Any,
        panel_session,
        time_window: TimeWindow,
        **__: Any,
    ) -> object:
        assert panel_session is shared_panel_session
        calls.append(("factors", time_window))
        return object()

    monkeypatch.setattr(materialize_panel, "write_panel", record_panel)
    monkeypatch.setattr(materialize_factors, "write_factors", record_factors)
    monkeypatch.setattr(
        materializer,
        "open_dataset_session",
        lambda *_args, **_kwargs: shared_panel_session,
    )

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
    with (
        open_dataset_session(
            panel_root,
            columns=None,
            time_window=None,
        ) as panel_session,
        runtime_resource_policy_scope(panel_session.resource_policy),
    ):
        core = TimeWindow("2026-07-06T00:10:00", "2026-07-06T00:12:00")
        materialize_factors.write_factors(
            factor_spec,
            layout=layout,
            panel_session=panel_session,
            time_window=core,
            resource_policy=panel_session.resource_policy,
            retained_payload_bytes=0,
        )
        materialize_labels.write_labels(
            label_spec,
            layout=layout,
            panel_session=panel_session,
            time_window=core,
            resource_policy=panel_session.resource_policy,
            retained_payload_bytes=0,
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

    assert materialization_validation.core_time_window_mask(frame, core).tolist() == [
        False,
        True,
        False,
    ]


def test_crypto_panel_source_parts_require_one_asset_per_physical_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_path = Path("part-000.parquet")
    monkeypatch.setattr(
        materialize_panel,
        "open_dataset_session",
        lambda *_args, **kwargs: SimpleNamespace(
            parquet_filters=(),
            resource_policy=kwargs["resource_policy"],
            verify=lambda: None,
        ),
    )
    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_file_batches",
        lambda *_args, **_kwargs: iter(
            (
                (shared_path, pd.DataFrame({"source_asset_id": ["BTCUSDT"]})),
                (shared_path, pd.DataFrame({"source_asset_id": ["ETHUSDT"]})),
            )
        ),
    )

    with _panel_test_sessions(
        StorageLayout(project_root=tmp_path),
        SourceId("synthetic_crypto"),
        processed_catalog().get("market_panel_1m"),
        TimeWindow("2026-07-06", "2026-07-06"),
    ) as (input_sessions, policy):
        with pytest.raises(
            ValueError,
            match="invalid_crypto_panel_source_physical_asset_grouping",
        ):
            next(
                materialize_panel._iter_panel_chunks(
                    processed_catalog().get("market_panel_1m"),
                    input_sessions=input_sessions,
                    time_window=TimeWindow("2026-07-06", "2026-07-06"),
                    resource_policy=policy,
                    retained_payload_bytes=0,
                )
            )


def test_crypto_session_views_prune_asset_and_window_paths_in_shared_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "crypto_physical_views")
    frame = make_default_dataset_frame(
        asset_ids=("BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"),
        dates=("2025-01-02", "2026-01-02", "2025-01-02", "2026-01-02"),
        values=(1.0, 2.0, 3.0, 4.0),
        market_id="crypto",
    )
    write_dataset(
        root,
        tuple(frame.iloc[index : index + 1].copy() for index in range(len(frame))),
        spec=make_processed_dataset_spec("crypto_physical_views"),
    )

    with (
        open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ) as session,
        runtime_resource_policy_scope(session.resource_policy),
    ):
        with materialization_validation.crypto_asset_session_views(
            session,
            columns=("asset_id",),
            target_chunk_bytes=session.resource_policy.remaining_memory_bytes,
            error="invalid_crypto_test_asset_grouping",
            operation_headroom_bytes=session.resource_policy.task_memory_budget_bytes,
        ) as views:
            normal_selections = tuple(view.path_selection for _key, view in views)
            assert {key[0]: len(view.read_paths) for key, view in views} == {
                "BTCUSDT": 2,
                "ETHUSDT": 2,
            }
            assert set(views[0][1].read_paths).isdisjoint(set(views[1][1].read_paths))
        assert all(
            session.inventory.path_count(
                selected=True,
                path_selection=selection,
            )
            == 0
            for selection in normal_selections
        )

        consumer_selections: tuple[int | None, ...] = ()
        with pytest.raises(RuntimeError, match="crypto_view_consumer_failed"):
            with materialization_validation.crypto_asset_session_views(
                session,
                columns=("asset_id",),
                target_chunk_bytes=session.resource_policy.remaining_memory_bytes,
                error="invalid_crypto_test_asset_grouping",
                operation_headroom_bytes=session.resource_policy.task_memory_budget_bytes,
            ) as views:
                consumer_selections = tuple(view.path_selection for _key, view in views)
                raise RuntimeError("crypto_view_consumer_failed")
        assert all(
            session.inventory.path_count(
                selected=True,
                path_selection=selection,
            )
            == 0
            for selection in consumer_selections
        )

        allocated: list[int] = []
        inventory_type = type(session.inventory)
        original_allocate = inventory_type.new_path_selection

        def allocate(current) -> int:  # noqa: ANN001
            selection = original_allocate(current)
            allocated.append(selection)
            return selection

        monkeypatch.setattr(inventory_type, "new_path_selection", allocate)
        original_verify = type(session).verify
        verify_calls = 0

        def fail_verify(current) -> None:  # noqa: ANN001
            nonlocal verify_calls
            if current is session:
                verify_calls += 1
            if current is session and verify_calls == 3:
                raise RuntimeError("crypto_view_verify_failed")
            original_verify(current)

        monkeypatch.setattr(type(session), "verify", fail_verify)
        with pytest.raises(RuntimeError, match="crypto_view_verify_failed"):
            with materialization_validation.crypto_asset_session_views(
                session,
                columns=("asset_id",),
                target_chunk_bytes=session.resource_policy.remaining_memory_bytes,
                error="invalid_crypto_test_asset_grouping",
                operation_headroom_bytes=session.resource_policy.task_memory_budget_bytes,
            ):
                raise AssertionError("unreachable")
        assert allocated
        assert all(
            session.inventory.path_count(
                selected=True,
                path_selection=selection,
            )
            == 0
            for selection in allocated
        )

        window_view = materialize_panel._matching_session_view(
            session,
            time_window=TimeWindow("2026-01-01", "2026-12-31"),
            asset_ids={"BTCUSDT"},
        )
        assert len(window_view.read_paths) == 2
        assert {path.parent.name for path in window_view.read_paths} == {"year=2026"}
        assert window_view.parquet_filters == (
            ("timestamp", ">=", pd.Timestamp("2026-01-01")),
            ("timestamp", "<", pd.Timestamp("2027-01-01")),
            ("source_asset_id", "==", "BTCUSDT"),
        )
    assert not list(root.parent.glob(".dataset-inventory-*"))


def test_matching_session_view_releases_selection_on_construction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "crypto_matching_view_failure")
    write_dataset(
        root,
        make_default_dataset_frame(
            asset_ids=("BTCUSDT", "BTCUSDT"),
            dates=("2025-01-02", "2026-01-02"),
            values=(1.0, 2.0),
            market_id="crypto",
        ),
        spec=make_processed_dataset_spec("crypto_matching_view_failure"),
    )

    with open_dataset_session(root, columns=None, time_window=None) as session:
        allocated: list[int] = []
        inventory_type = type(session.inventory)
        original_allocate = inventory_type.new_path_selection

        def allocate(current) -> int:  # noqa: ANN001
            selection = original_allocate(current)
            allocated.append(selection)
            return selection

        monkeypatch.setattr(inventory_type, "new_path_selection", allocate)
        session_type = type(session)
        original_verify = session_type.verify

        def fail_view_verify(current) -> None:  # noqa: ANN001
            if current is not session and current.path_selection == allocated[-1]:
                raise RuntimeError("matching_view_verify_failed")
            original_verify(current)

        monkeypatch.setattr(session_type, "verify", fail_view_verify)
        with pytest.raises(RuntimeError, match="matching_view_verify_failed"):
            materialize_panel._matching_session_view(
                session,
                time_window=TimeWindow("2026-01-01", "2026-12-31"),
                asset_ids={"BTCUSDT"},
            )

        assert allocated
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=allocated[-1],
            )
            == 0
        )


def test_matching_chunk_caller_releases_selection_on_all_exit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "crypto_matching_caller_cleanup")
    write_dataset(
        root,
        make_default_dataset_frame(
            asset_ids=("BTCUSDT", "BTCUSDT"),
            dates=("2025-01-02", "2026-01-02"),
            values=(1.0, 2.0),
            market_id="crypto",
        ),
        spec=make_processed_dataset_spec("crypto_matching_caller_cleanup"),
    )

    with open_dataset_session(root, columns=None, time_window=None) as session:
        allocated: list[int] = []
        inventory_type = type(session.inventory)
        original_allocate = inventory_type.new_path_selection

        def allocate(current) -> int:  # noqa: ANN001
            selection = original_allocate(current)
            allocated.append(selection)
            return selection

        monkeypatch.setattr(inventory_type, "new_path_selection", allocate)
        timestamp = pd.Timestamp("2026-01-02")
        kwargs = {
            "asset_ids": {"BTCUSDT"},
            "time_window": TimeWindow("2026-01-01", "2026-12-31"),
            "source_asset_error": "invalid_test_source_asset_id",
            "timestamp_error": "invalid_test_timestamp",
            "target_chunk_bytes": session.resource_policy.remaining_memory_bytes,
            "operation_headroom_bytes": (
                session.resource_policy.task_memory_budget_bytes
            ),
        }

        def empty_chunks(*_args, **_kwargs):
            yield from ()

        monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", empty_chunks)
        assert list(materialize_panel._iter_matching_chunks(session, **kwargs)) == []
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=allocated[-1],
            )
            == 0
        )

        def failing_chunks(*_args, **_kwargs):
            raise RuntimeError("matching_consumer_failed")
            yield  # pragma: no cover

        monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", failing_chunks)
        with pytest.raises(RuntimeError, match="matching_consumer_failed"):
            list(materialize_panel._iter_matching_chunks(session, **kwargs))
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=allocated[-1],
            )
            == 0
        )

        def one_chunk(*_args, **_kwargs):
            yield pd.DataFrame(
                {
                    "source_asset_id": ["BTCUSDT"],
                    "timestamp": [timestamp],
                }
            )

        monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", one_chunk)
        chunks = materialize_panel._iter_matching_chunks(session, **kwargs)
        next(chunks)
        chunks.close()
        assert (
            session.inventory.path_count(
                selected=True,
                path_selection=allocated[-1],
            )
            == 0
        )


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
@pytest.mark.parametrize("failure_site", ("construction", "verify", "body"))
def test_asset_views_preserve_primary_and_attempt_all_selection_release(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    failure_site: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    primary = primary_type("selection_primary")

    class Inventory:
        def __init__(self) -> None:
            self.next_selection = 0
            self.released: list[int] = []

        def new_path_selection(self) -> int:
            self.next_selection += 1
            return self.next_selection

        def select_path(self, _selection: int, _path: Path) -> None:
            if failure_site == "construction":
                raise primary

        def release_path_selection(self, selection: int) -> None:
            self.released.append(selection)
            raise OSError(f"release_{selection}")

    inventory = Inventory()
    session = SimpleNamespace(
        resource_policy=policy,
        parquet_filters=(),
        inventory=inventory,
        verify=lambda: (
            (_ for _ in ()).throw(primary) if failure_site == "verify" else None
        ),
    )

    def replace_session(current, **changes):  # noqa: ANN001
        values = vars(current).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    monkeypatch.setattr(materialization_validation, "replace", replace_session)
    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_file_batches",
        lambda *_args, **_kwargs: iter(
            (
                (Path("a.parquet"), pd.DataFrame({"asset_id": ["a"]})),
                (Path("b.parquet"), pd.DataFrame({"asset_id": ["b"]})),
            )
        ),
    )

    with pytest.raises(primary_type) as raised:
        with materialization_validation.crypto_asset_session_views(
            session,
            columns=("asset_id",),
            target_chunk_bytes=policy.remaining_memory_bytes,
            error="invalid_test_asset_grouping",
            operation_headroom_bytes=session.resource_policy.task_memory_budget_bytes,
        ):
            if failure_site == "body":
                raise primary
            raise AssertionError("unreachable")
    assert raised.value is primary
    assert inventory.released == ([1] if failure_site == "construction" else [1, 2])


def test_asset_views_propagate_first_release_failure_after_attempting_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    first_failure = OSError("release_1")

    class Inventory:
        def __init__(self) -> None:
            self.next_selection = 0
            self.released: list[int] = []

        def new_path_selection(self) -> int:
            self.next_selection += 1
            return self.next_selection

        def select_path(self, _selection: int, _path: Path) -> None:
            pass

        def release_path_selection(self, selection: int) -> None:
            self.released.append(selection)
            if selection == 1:
                raise first_failure
            raise OSError("release_2")

    inventory = Inventory()
    session = SimpleNamespace(
        resource_policy=policy,
        parquet_filters=(),
        inventory=inventory,
        verify=lambda: None,
    )

    def replace_session(current, **changes):  # noqa: ANN001
        values = vars(current).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    monkeypatch.setattr(materialization_validation, "replace", replace_session)
    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_file_batches",
        lambda *_args, **_kwargs: iter(
            (
                (Path("a.parquet"), pd.DataFrame({"asset_id": ["a"]})),
                (Path("b.parquet"), pd.DataFrame({"asset_id": ["b"]})),
            )
        ),
    )

    with pytest.raises(OSError) as raised:
        with materialization_validation.crypto_asset_session_views(
            session,
            columns=("asset_id",),
            target_chunk_bytes=policy.remaining_memory_bytes,
            error="invalid_test_asset_grouping",
            operation_headroom_bytes=session.resource_policy.task_memory_budget_bytes,
        ):
            pass
    assert raised.value is first_failure
    assert inventory.released == [1, 2]


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_matching_view_construction_preserves_primary_over_release_failure(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("matching_construction_primary")

    class Inventory:
        def new_path_selection(self) -> int:
            return 1

        def select_path(self, _selection: int, _path: Path) -> None:
            raise primary

        def release_path_selection(self, _selection: int) -> None:
            raise OSError("matching_release_secondary")

    session = SimpleNamespace(
        inventory=Inventory(),
        all_paths=(Path("part.parquet"),),
        data_root=Path("data"),
        spec={},
        timestamp_column=None,
    )
    monkeypatch.setattr(
        materialize_panel,
        "prune_paths_for_time_window",
        lambda *_args, **_kwargs: (Path("part.parquet"),),
    )

    with pytest.raises(primary_type) as raised:
        materialize_panel._matching_session_view(
            session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            asset_ids={"BTCUSDT"},
        )
    assert raised.value is primary


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_matching_chunks_preserve_body_primary_over_release_failure(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("matching_body_primary")

    class Inventory:
        def release_path_selection(self, _selection: int) -> None:
            raise OSError("matching_release_secondary")

    session = SimpleNamespace(inventory=Inventory())
    monkeypatch.setattr(
        materialize_panel,
        "_matching_session_view",
        lambda *_args, **_kwargs: SimpleNamespace(path_selection=1),
    )

    def fail_chunks(*_args, **_kwargs):
        raise primary
        yield  # pragma: no cover

    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", fail_chunks)

    with pytest.raises(primary_type) as raised:
        list(
            materialize_panel._iter_matching_chunks(
                session,
                asset_ids={"BTCUSDT"},
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                source_asset_error="invalid_test_source_asset_id",
                timestamp_error="invalid_test_timestamp",
                target_chunk_bytes=1024,
                operation_headroom_bytes=1024,
            )
        )
    assert raised.value is primary


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
        monkeypatch.setattr(module, "iter_dataset_chunks", read_panel)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(
            ((("crypto", "btc_usdt"), session),)
        ),
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    with runtime_resource_policy_scope(policy):
        factors = pd.concat(
            materialize_factors._iter_factor_chunks(
                processed_catalog().get("microstructure"),
                panel_session=SimpleNamespace(
                    resource_policy=policy,
                    verify=lambda: None,
                ),
                time_window=core,
                resource_policy=policy,
                retained_payload_bytes=0,
            ),
            ignore_index=True,
        )
        labels = pd.concat(
            materialize_labels._iter_label_chunks(
                processed_catalog().get("forward_returns"),
                panel_session=SimpleNamespace(
                    resource_policy=policy,
                    verify=lambda: None,
                ),
                time_window=core,
                resource_policy=policy,
                retained_payload_bytes=0,
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
                resource_policy=capture_test_policy(
                    available_memory_bytes=1024**3,
                ),
                retained_payload_bytes=0,
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
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
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
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
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
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
        retained_payload_bytes=0,
    )

    final_root = layout.processed_data_root(
        "crypto",
        ProcessedDataKind.PANEL,
        "market_panel_1m",
    )
    panel = read_dataset(final_root)
    assert panel["asset_id"].tolist() == ["btc_usdt"]
    assert panel["trade_flow_observed"].tolist() == [True]
    assert panel["quote_observed"].tolist() == [True]
    assert "source_asset_id" not in panel.columns
    assert not list(final_root.parent.rglob(".k-*"))


def test_crypto_panel_validation_failure_never_publishes_final_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _patch_matching_panel_reads(monkeypatch, _matching_panel_frames())
    observed_roots: list[Path] = []

    def reject_staged_dataset(receipt, **_kwargs) -> None:
        observed_roots.append(Path(receipt.root))
        raise ValueError("synthetic_staged_crypto_validation_failure")

    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_materialization",
        reject_staged_dataset,
    )
    final_root = layout.processed_data_root(
        "crypto",
        ProcessedDataKind.PANEL,
        "market_panel_1m",
    )

    with pytest.raises(
        ValueError,
        match="synthetic_staged_crypto_validation_failure",
    ):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow(
                "2026-07-06T00:00:00",
                "2026-07-06T00:00:00",
            ),
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
        )

    assert len(observed_roots) == 1
    assert observed_roots[0] != final_root
    assert observed_roots[0].parent == final_root.parent
    assert not final_root.exists()
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(final_root)
    assert not list(final_root.parent.glob(f"{scratch_prefix}*"))


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
    real_asset_discovery = materialization_validation.crypto_asset_session_views

    def wrapped_iterator(context, *, target_chunk_bytes, operation_headroom_bytes):
        root_name = context.root.name
        if root_name in call_counts:
            call_counts[root_name] += 1
        result = real_iterator(
            context,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
        )
        if root_name == "market_bars_1m":
            if context.logical_columns == ("source_asset_id",):
                bar_scans["asset_discovery"] += 1
                return result
            bar_scans["core"] += 1
            return _count_yielded_chunks(result, yielded_chunks, root_name)
        return result

    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", wrapped_iterator)

    def wrapped_asset_discovery(*args, **kwargs):
        group_plan_calls["count"] += 1
        return real_asset_discovery(*args, **kwargs)

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        wrapped_asset_discovery,
    )

    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=capture_storage_test_policy(
            working_headroom_bytes=12 * 1024**2,
        ),
        retained_payload_bytes=0,
    )

    assert call_counts["market_bars_1m"] == 2
    assert bar_scans == {"asset_discovery": 0, "core": 2}
    assert group_plan_calls["count"] == 1
    assert yielded_chunks["market_bars_1m"] > 1
    assert call_counts["aggregate_trades_tick"] == yielded_chunks["market_bars_1m"]
    assert call_counts["depth_levels_tick"] == yielded_chunks["market_bars_1m"]


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
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
        retained_payload_bytes=0,
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
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
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
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
        retained_payload_bytes=0,
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

    def fake_iterator(context, *, target_chunk_bytes, operation_headroom_bytes):
        assert target_chunk_bytes > 0
        assert operation_headroom_bytes > 0
        root_name = context.root.name
        if root_name == "market_bars_1m":
            return iter((frames["bars"],))
        auxiliary_reads.append(root_name)
        return iter(())

    monkeypatch.setattr(
        materialize_panel,
        "open_dataset_session",
        lambda dataset_root, **kwargs: SimpleNamespace(
            root=Path(dataset_root),
            row_filters=kwargs.get("row_filters", ()),
            resource_policy=kwargs["resource_policy"],
            verify=lambda: None,
        ),
    )
    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", fake_iterator)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(((("BTCUSDT",), session),)),
    )
    _patch_panel_lifecycle_validation(monkeypatch)

    with pytest.raises(ValueError, match="invalid_crypto_bar_interval"):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06T00:00:00", "2026-07-06T00:00:00"),
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
        )

    assert auxiliary_reads == []


def test_crypto_panel_small_and_large_resource_windows_match(tmp_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for name, policy in (
        (
            "large",
            capture_test_policy(
                available_memory_bytes=1024**3,
            ),
        ),
        (
            "small",
            capture_storage_test_policy(
                working_headroom_bytes=12 * 1024**2,
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
            retained_payload_bytes=0,
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


@contextmanager
def _asset_views_context(views):  # noqa: ANN001
    yield views


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
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58", "2026-07-06 00:00:58"]),
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
    class SelectionInventory:
        def __init__(self) -> None:
            self.next_selection = 0
            self.active_selections: set[int] = set()

        def allocate(self) -> int:
            self.next_selection += 1
            self.active_selections.add(self.next_selection)
            return self.next_selection

        def release_path_selection(self, selection: int) -> None:
            self.active_selections.remove(selection)

    inventory = SelectionInventory()

    def assert_selections_released() -> None:
        assert inventory.active_selections == set()

    monkeypatch.setattr(
        materialize_panel,
        "open_dataset_session",
        lambda dataset_root, **kwargs: SimpleNamespace(
            root=Path(dataset_root),
            row_filters=kwargs.get("row_filters", ()),
            resource_policy=kwargs["resource_policy"],
            inventory=inventory,
            path_selection=None,
            retained_memory_bytes=0,
            verify=lambda: None,
            close=assert_selections_released,
        ),
    )

    bars = chunks_by_root["market_bars_1m"]
    source_asset_ids = tuple(
        (value,)
        for chunk in bars
        for value in (
            chunk["source_asset_id"].tolist() if "source_asset_id" in chunk else [None]
        )
    )
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(
            tuple(
                (
                    key,
                    SimpleNamespace(
                        root=session.root,
                        row_filters=(("source_asset_id", "==", key[0]),),
                        resource_policy=session.resource_policy,
                        inventory=session.inventory,
                        path_selection=None,
                        verify=lambda: None,
                    ),
                )
                for key in source_asset_ids
            )
        ),
    )

    def matching_session_view(session, *, asset_ids, **_kwargs):
        return SimpleNamespace(
            root=session.root,
            row_filters=(
                (("source_asset_id", "==", next(iter(asset_ids))),)
                if len(asset_ids) == 1
                else ()
            ),
            resource_policy=session.resource_policy,
            inventory=session.inventory,
            path_selection=session.inventory.allocate(),
            verify=lambda: None,
        )

    monkeypatch.setattr(
        materialize_panel,
        "_matching_session_view",
        matching_session_view,
    )

    def fake_iterator(context, *, target_chunk_bytes, operation_headroom_bytes):
        assert target_chunk_bytes > 0
        assert operation_headroom_bytes > 0
        root_name = context.root.name
        filters = dict(
            (column, value)
            for column, operator, value in context.row_filters
            if operator == "=="
        )
        source_asset_id = filters.get("source_asset_id")
        if root_name not in chunks_by_root:
            raise AssertionError(f"unexpected_dataset_root:{root_name}")
        chunks = chunks_by_root[root_name]
        if source_asset_id is None or root_name != "market_bars_1m":
            return iter(chunks)
        return iter(
            tuple(
                chunk.loc[chunk["source_asset_id"].eq(source_asset_id)].copy()
                for chunk in chunks
                if chunk["source_asset_id"].eq(source_asset_id).any()
            )
        )

    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", fake_iterator)


@contextmanager
def _panel_test_sessions(
    layout: StorageLayout,
    source_id: SourceId,
    spec,
    time_window: TimeWindow,
):  # noqa: ANN202
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    with ExitStack() as stack:
        sessions = {
            data_id: stack.enter_context(
                materialize_panel.open_dataset_session(
                    layout.interim_data_root(source_id, data_id),
                    columns=None,
                    time_window=time_window,
                    resource_policy=policy,
                )
            )
            for data_id in (
                source_ids.market_bars_source_data_id(spec),
                source_ids.aggregate_trades_source_data_id(spec),
                source_ids.depth_levels_source_data_id(spec),
            )
        }
        yield sessions, policy


def _patch_panel_lifecycle_validation(monkeypatch) -> None:
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
    with ExitStack() as stack:
        sessions = {
            data_id: stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, data_id),
                    columns=None,
                    time_window=None,
                )
            )
            for data_id in (
                source_ids.market_bars_source_data_id(panel_spec),
                source_ids.aggregate_trades_source_data_id(panel_spec),
                source_ids.depth_levels_source_data_id(panel_spec),
            )
        }
        identities = materialization_validation.input_dataset_identities(sessions)
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


def test_panel_writer_acquires_inputs_with_cumulative_live_session_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_synthetic_crypto_interim(layout, source_id, minutes=3)
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    real_open = materialize_panel.open_dataset_session
    real_write = materialize_panel.write_dataset
    live_sessions: list[object] = []
    acquisitions: list[int] = []
    opened_retained: list[int] = []
    publication_envelopes: list[int] = []

    @contextmanager
    def record_open(*args, retained_payload_bytes=0, **kwargs):  # noqa: ANN002, ANN003
        acquisitions.append(retained_payload_bytes)
        assert retained_payload_bytes == sum(
            session.retained_memory_bytes for session in live_sessions
        )
        with real_open(
            *args,
            retained_payload_bytes=retained_payload_bytes,
            **kwargs,
        ) as session:
            live_sessions.append(session)
            opened_retained.append(session.retained_memory_bytes)
            try:
                yield session
            finally:
                live_sessions.remove(session)

    def record_write(root, data, **kwargs):  # noqa: ANN001, ANN202
        def publications():  # noqa: ANN202
            for publication in data:
                assert isinstance(publication, materialize_panel.DatasetWriteChunk)
                publication_envelopes.append(publication.retained_payload_bytes)
                yield publication

        return real_write(root, publications(), **kwargs)

    monkeypatch.setattr(materialize_panel, "open_dataset_session", record_open)
    monkeypatch.setattr(materialize_panel, "write_dataset", record_write)
    materialize_panel.write_panel(
        processed_catalog().get("market_panel_1m"),
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=policy,
        retained_payload_bytes=0,
    )

    assert len(acquisitions) == 3
    assert publication_envelopes
    assert set(publication_envelopes) == {sum(opened_retained)}


def test_panel_chunk_keeps_output_admitted_through_consumer_yield(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_synthetic_crypto_interim(layout, source_id, minutes=3)
    spec = processed_catalog().get("market_panel_1m")

    with _panel_test_sessions(
        layout,
        source_id,
        spec,
        TimeWindow("2026-07-06", "2026-07-06"),
    ) as (input_sessions, policy):
        chunks = materialize_panel._iter_panel_chunks(
            spec,
            input_sessions=input_sessions,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
            retained_payload_bytes=0,
        )
        output = next(chunks)
        assert policy._ledger.by_owner["crypto.panel.output"] >= (  # noqa: SLF001
            materialize_panel.pandas_frame_payload_bytes(output)
        )
        chunks.close()
        assert "crypto.panel.output" not in policy._ledger.by_owner  # noqa: SLF001


def test_existing_panel_dependency_acquires_panel_after_all_live_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(layout, source_id, _panel_frame(minutes=3))
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    real_open = materializer.open_dataset_session
    live_sessions: list[object] = []
    acquisitions: list[int] = []

    @contextmanager
    def record_open(*args, retained_payload_bytes=0, **kwargs):  # noqa: ANN002, ANN003
        acquisitions.append(retained_payload_bytes)
        assert retained_payload_bytes == sum(
            session.retained_memory_bytes for session in live_sessions
        )
        with real_open(
            *args,
            retained_payload_bytes=retained_payload_bytes,
            **kwargs,
        ) as session:
            live_sessions.append(session)
            try:
                yield session
            finally:
                live_sessions.remove(session)

    monkeypatch.setattr(materializer, "open_dataset_session", record_open)
    with ExitStack() as stack:
        _inspection, panel_session = materializer._inspect_existing_dependency(
            layout,
            processed_catalog().get("market_panel_1m"),
            source_id=source_id,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
            session_stack=stack,
            retained_payload_bytes=0,
        )
        assert panel_session.retained_memory_bytes > 0

    assert len(acquisitions) == 4


@pytest.mark.parametrize(
    ("module", "writer_name", "iterator_name", "data_id"),
    (
        (
            materialize_factors,
            "write_factors",
            "_iter_factor_chunks",
            "microstructure",
        ),
        (
            materialize_labels,
            "write_labels",
            "_iter_label_chunks",
            "forward_returns",
        ),
    ),
)
def test_panel_dependent_publication_carries_live_panel_session_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    iterator_name: str,
    data_id: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(layout, source_id, _panel_frame(minutes=3))
    panel_spec = processed_catalog().get("market_panel_1m")
    publications: list[object] = []
    write_baselines: list[int] = []
    monkeypatch.setattr(
        module,
        iterator_name,
        lambda *_args, **_kwargs: iter((pd.DataFrame({"value": [1]}),)),
    )

    def record_write(_root, data, **kwargs):  # noqa: ANN001, ANN202
        write_baselines.append(kwargs["retained_payload_bytes"])
        publications.extend(data)
        return SimpleNamespace()

    monkeypatch.setattr(module, "write_dataset", record_write)
    with open_dataset_session(
        layout.processed_data_root(
            panel_spec.market_id,
            panel_spec.kind,
            panel_spec.data_id,
        ),
        columns=None,
        time_window=None,
    ) as panel_session:
        getattr(module, writer_name)(
            processed_catalog().get(data_id),
            layout=layout,
            panel_session=panel_session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=panel_session.resource_policy,
            retained_payload_bytes=123,
        )

        assert write_baselines == [panel_session.retained_memory_bytes + 123]
        assert publications
        assert all(
            isinstance(publication, module.DatasetWriteChunk)
            and publication.retained_payload_bytes
            == panel_session.retained_memory_bytes + 123
            for publication in publications
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
        "open_dataset_session",
        lambda *_args, **kwargs: SimpleNamespace(
            row_filters=kwargs.get("row_filters", ()),
            resource_policy=kwargs["resource_policy"],
            verify=lambda: None,
        ),
    )

    def iter_bars(context, **_kwargs):
        filters = context.row_filters
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
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(
            (
                (
                    ("ETHFIUSDT",),
                    SimpleNamespace(
                        row_filters=(("source_asset_id", "==", "ETHFIUSDT"),),
                        verify=lambda: None,
                    ),
                ),
                (
                    ("ETHUSDT",),
                    SimpleNamespace(
                        row_filters=(("source_asset_id", "==", "ETHUSDT"),),
                        verify=lambda: None,
                    ),
                ),
            )
        ),
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
        lambda *_args, **_kwargs: (_ for _ in ()),
    )
    monkeypatch.setattr(
        materialize_panel,
        "_iter_matching_depth_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()),
    )
    monkeypatch.setattr(
        materialize_panel.panels,
        "calculate_market_panel_from_chunks",
        lambda panel, **_kwargs: panel.sort_values(
            ["market_id", "asset_id", "timestamp"], kind="mergesort"
        ).reset_index(drop=True),
    )

    with _panel_test_sessions(
        StorageLayout(project_root=tmp_path),
        SourceId("binance"),
        spec,
        TimeWindow("2026-07-06", "2026-07-06"),
    ) as (input_sessions, policy):
        chunks = list(
            materialize_panel._iter_panel_chunks(
                spec,
                input_sessions=input_sessions,
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                resource_policy=policy,
                retained_payload_bytes=0,
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


def test_panel_chunks_reject_same_asset_cross_chunk_time_reversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get("market_panel_1m")
    chunks = tuple(
        pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"],
                "timestamp": [timestamp],
            }
        )
        for timestamp in pd.to_datetime(["2026-07-06 00:01", "2026-07-06 00:00"])
    )
    monkeypatch.setattr(
        materialize_panel,
        "open_dataset_session",
        lambda *_args, **kwargs: SimpleNamespace(
            row_filters=kwargs.get("row_filters", ()),
            resource_policy=kwargs["resource_policy"],
            verify=lambda: None,
        ),
    )
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(((("BTCUSDT",), session),)),
    )
    monkeypatch.setattr(
        materialize_panel,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(chunk.copy() for chunk in chunks),
    )
    monkeypatch.setattr(
        materialize_panel,
        "_validate_matching_bar_keys",
        lambda _x: None,
    )

    def prepare(bars: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                "market_id": ["crypto"],
                "asset_id": ["btc_usdt"],
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
        lambda *_args, **_kwargs: (_ for _ in ()),
    )
    monkeypatch.setattr(
        materialize_panel,
        "_iter_matching_depth_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()),
    )
    monkeypatch.setattr(
        materialize_panel.panels,
        "calculate_market_panel_from_chunks",
        lambda panel, **_kwargs: panel,
    )

    with _panel_test_sessions(
        StorageLayout(project_root=tmp_path),
        SourceId("binance"),
        spec,
        TimeWindow("2026-07-06", "2026-07-06"),
    ) as (input_sessions, policy):
        with pytest.raises(ValueError, match="out_of_order_crypto_panel_dependency"):
            list(
                materialize_panel._iter_panel_chunks(
                    spec,
                    input_sessions=input_sessions,
                    time_window=TimeWindow("2026-07-06", "2026-07-06"),
                    resource_policy=policy,
                    retained_payload_bytes=0,
                )
            )


def test_panel_writer_rejects_source_replacement_after_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    source_id = SourceId("synthetic_crypto")
    _write_synthetic_crypto_interim(layout, source_id, minutes=3)
    _write_synthetic_crypto_interim(replacement_layout, source_id, minutes=4)
    spec = processed_catalog().get("market_panel_1m")
    bar_data_id = source_ids.market_bars_source_data_id(spec)
    current_root = layout.interim_data_root(source_id, bar_data_id)
    replacement_root = replacement_layout.interim_data_root(source_id, bar_data_id)
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_identity = materialization_validation.input_dataset_identities
    real_open = materialize_panel.open_dataset_session
    opened_roots: list[Path] = []
    replaced = False

    def record_open(root: Path, **kwargs):
        opened_roots.append(Path(root))
        return real_open(root, **kwargs)

    def replace_after_identity(sessions):  # noqa: ANN001
        nonlocal replaced
        identities = real_identity(sessions)
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        replaced = True
        return identities

    monkeypatch.setattr(materialize_panel, "open_dataset_session", record_open)
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        replace_after_identity,
    )

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
        )

    assert replaced
    assert len(opened_roots) == 3
    assert len(opened_roots) == len(set(opened_roots))


@pytest.mark.parametrize(
    "data_id",
    ("aggregate_trades_tick", "depth_levels_tick"),
)
def test_panel_writer_rejects_auxiliary_replacement_during_staged_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_id: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    source_id = SourceId("synthetic_crypto")
    replacement_source_id = SourceId("replacement_crypto")
    _write_synthetic_crypto_interim(layout, source_id, minutes=3)
    _write_synthetic_crypto_interim(
        replacement_layout,
        replacement_source_id,
        minutes=3,
    )
    current_root = layout.interim_data_root(source_id, data_id)
    replacement_root = replacement_layout.interim_data_root(
        replacement_source_id,
        data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    real_validation = materialization_validation.validate_panel_materialization
    replaced = False

    def replace_after_staged_validation(receipt, **kwargs) -> None:
        nonlocal replaced
        real_validation(receipt, **kwargs)
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        replaced = True

    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_materialization",
        replace_after_staged_validation,
    )
    final_root = layout.processed_data_root(
        "crypto",
        ProcessedDataKind.PANEL,
        "market_panel_1m",
    )

    with pytest.raises(ValueError, match="^dataset_spec_identity_changed$"):
        materialize_panel.write_panel(
            processed_catalog().get("market_panel_1m"),
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=0,
        )

    assert replaced
    _assert_atomic_dataset_not_published(final_root)


@pytest.mark.parametrize(
    ("module", "writer_name", "data_id"),
    (
        (materialize_factors, "write_factors", "microstructure"),
        (materialize_labels, "write_labels", "forward_returns"),
    ),
)
def test_panel_dependent_writer_rejects_replacement_after_identity_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    data_id: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=70),
    )
    _write_existing_panel_dependency(
        replacement_layout,
        source_id,
        _panel_frame(minutes=71),
    )
    panel_spec = processed_catalog().get("market_panel_1m")
    current_root = layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    with open_dataset_session(
        current_root,
        columns=None,
        time_window=None,
    ) as panel_session:
        real_identity = materialization_validation.input_dataset_identities
        replaced = False

        def replace_after_identity(sessions):  # noqa: ANN001
            nonlocal replaced
            identities = real_identity(sessions)
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True
            return identities

        monkeypatch.setattr(
            materialization_validation,
            "input_dataset_identities",
            replace_after_identity,
        )

        with pytest.raises(
            ValueError,
            match="dataset_(spec|manifest)_identity_changed",
        ):
            getattr(module, writer_name)(
                processed_catalog().get(data_id),
                layout=layout,
                panel_session=panel_session,
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                resource_policy=panel_session.resource_policy,
                retained_payload_bytes=0,
            )

    assert replaced
    assert panel_session.root == current_root


@pytest.mark.parametrize(
    ("module", "writer_name", "data_id", "kind"),
    (
        (
            materialize_factors,
            "write_factors",
            "microstructure",
            ProcessedDataKind.FACTOR,
        ),
        (
            materialize_labels,
            "write_labels",
            "forward_returns",
            ProcessedDataKind.LABEL,
        ),
    ),
)
def test_panel_dependent_writer_rejects_replacement_during_staged_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    data_id: str,
    kind: ProcessedDataKind,
) -> None:
    layout = StorageLayout(project_root=tmp_path / "current")
    replacement_layout = StorageLayout(project_root=tmp_path / "replacement")
    source_id = SourceId("synthetic_crypto")
    _write_existing_panel_dependency(
        layout,
        source_id,
        _panel_frame(minutes=70),
    )
    _write_existing_panel_dependency(
        replacement_layout,
        source_id,
        _panel_frame(minutes=71),
    )
    panel_spec = processed_catalog().get("market_panel_1m")
    current_root = layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    replacement_root = replacement_layout.processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")
    with open_dataset_session(
        current_root,
        columns=None,
        time_window=None,
    ) as panel_session:
        real_validation = (
            materialization_validation.validate_panel_dependent_materialization
        )
        real_write = module.write_dataset
        real_verify = type(panel_session).verify
        replaced = False
        write_returned = False
        post_publication_verifications = 0

        def replace_after_staged_validation(receipt, **kwargs) -> None:
            nonlocal replaced
            real_validation(receipt, **kwargs)
            current_root.rename(backup_root)
            replacement_root.rename(current_root)
            replaced = True

        def record_write(*args, **kwargs):
            nonlocal write_returned
            result = real_write(*args, **kwargs)
            write_returned = True
            return result

        def record_verify(session) -> None:
            nonlocal post_publication_verifications
            if session is panel_session and write_returned:
                post_publication_verifications += 1
            real_verify(session)

        monkeypatch.setattr(
            materialization_validation,
            "validate_panel_dependent_materialization",
            replace_after_staged_validation,
        )
        monkeypatch.setattr(module, "write_dataset", record_write)
        monkeypatch.setattr(type(panel_session), "verify", record_verify)
        final_root = layout.processed_data_root("crypto", kind, data_id)

        with pytest.raises(
            ValueError,
            match="^dataset_(spec|manifest)_identity_changed$",
        ):
            getattr(module, writer_name)(
                processed_catalog().get(data_id),
                layout=layout,
                panel_session=panel_session,
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                resource_policy=panel_session.resource_policy,
                retained_payload_bytes=0,
            )

    assert replaced
    assert post_publication_verifications == 0
    _assert_atomic_dataset_not_published(final_root)


def _assert_atomic_dataset_not_published(final_root: Path) -> None:
    assert not final_root.exists()
    scratch_prefix = storage_atomic._atomic_directory_scratch_prefix(final_root)
    assert not list(final_root.parent.glob(f"{scratch_prefix}*"))


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
