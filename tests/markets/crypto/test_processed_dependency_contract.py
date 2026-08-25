from __future__ import annotations

import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    materializer,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    open_dataset_session,
)
from quant_research_platform.storage._dataset_types import DatasetInspection
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import capture_test_policy

RESOURCE_POLICY = capture_test_policy(
    available_memory_bytes=1024**3,
)


def _collect_bound_first(data, policy, *, requested_bounds=None):  # noqa: ANN001, ANN202
    frames = []
    with runtime_resource_policy_scope(policy):
        bound = next(data)
        if requested_bounds is not None:
            requested_bounds.append(bound)
        while True:
            lease = policy.lease("test.crypto.consumer", bound)
            try:
                while True:
                    try:
                        owned = data.send(lease)
                    except StopIteration:
                        lease.close()
                        return frames
                    if not isinstance(owned, _OwnedValue):
                        raise AssertionError("invalid_bound_first_output")
                    break
                frames.append(owned.value)
                try:
                    bound = data.send(None)
                except StopIteration:
                    lease.close()
                    return frames
                if requested_bounds is not None:
                    requested_bounds.append(bound)
            except BaseException:
                lease.close()
                raise
            lease.close()


@pytest.mark.parametrize("owner", ("core", "matching"))
def test_crypto_projection_reuses_storage_filter_admission(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
) -> None:
    frame = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"] * 16,
            "timestamp": pd.date_range("2026-07-06", periods=16, tz="UTC"),
        }
    )
    closed: list[bool] = []
    observed: list[object] = []

    def source_chunks(context, **_kwargs):
        observed.append(context)
        try:
            yield frame
        finally:
            closed.append(True)

    headroom = pandas_frame_payload_bytes(frame)
    if owner == "core":
        monkeypatch.setattr(
            materialization_validation,
            "iter_dataset_chunks",
            source_chunks,
        )
        session = SimpleNamespace(
            parquet_filters=(
                ("timestamp", ">=", pd.Timestamp("2026-01-01")),
                ("market_id", "==", "crypto"),
            ),
            timestamp_column="timestamp",
            time_window=TimeWindow("2026-01-01", "2026-12-31"),
            owns_inventory=True,
            retained_memory_bytes=17,
            verify=lambda: None,
        )

        def replace_session(current, **changes):  # noqa: ANN001, ANN202
            return SimpleNamespace(**(vars(current) | changes))

        monkeypatch.setattr(materialization_validation, "replace", replace_session)

        def projection():
            return materialization_validation._iter_core_window_chunks(
                session,
                time_window=TimeWindow(
                    "2026-07-06T00:00:00",
                    "2026-07-06T00:15:00",
                ),
                target_chunk_bytes=headroom,
            )

    else:
        monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", source_chunks)
        monkeypatch.setattr(
            materialize_panel,
            "_matching_session_view",
            lambda *_args, **_kwargs: SimpleNamespace(
                path_selection=None,
                parquet_filters=(
                    ("timestamp", ">=", frame["timestamp"].min()),
                    ("source_asset_id", "==", "BTCUSDT"),
                ),
            ),
        )

        def projection():
            return materialize_panel._iter_matching_chunks(
                SimpleNamespace(),
                asset_ids={"BTCUSDT"},
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                source_asset_error="probe_source_asset",
                timestamp_error="probe_timestamp",
                target_chunk_bytes=headroom,
            )

    completed = list(projection())
    assert len(completed) == 1
    assert completed[0] is frame
    iterator = projection()
    assert next(iterator) is frame
    iterator.close()
    assert len(closed) == 2
    assert len(observed) == 2
    assert all(context.parquet_filters for context in observed)
    if owner == "core":
        assert all(
            context.parquet_filters[0] == ("market_id", "==", "crypto")
            for context in observed
        )


def test_dependent_validation_closes_expected_stream_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    closed: list[bool] = []

    def expected_chunks(*_args, **_kwargs):
        try:
            yield pd.DataFrame({"asset_id": ["BTCUSDT"]})
        finally:
            closed.append(True)

    def fail_after_one(expected, *_args, **_kwargs):  # noqa: ANN001
        next(iter(expected))
        raise RuntimeError("expected_stream_probe")

    monkeypatch.setattr(
        materialization_validation,
        "_iter_core_window_chunks",
        expected_chunks,
    )
    monkeypatch.setattr(
        materialization_validation,
        "_validate_receipt_key_coverage",
        fail_after_one,
    )
    receipt_session = SimpleNamespace(
        resource_policy=policy,
        manifest={"input_dataset_identities": {}},
    )
    with pytest.raises(RuntimeError, match="^expected_stream_probe$"):
        materialization_validation.validate_panel_dependent_materialization(
            SimpleNamespace(session=receipt_session, retained_memory_bytes=0),
            panel_session=SimpleNamespace(resource_policy=policy),
            spec=SimpleNamespace(primary_key=("asset_id",)),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            expected_input_identities={},
            resource_policy=policy,
        )
    assert closed == [True]


def test_asset_discovery_closes_file_batches_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    closed: list[bool] = []

    def file_batches(*_args, **_kwargs):
        try:
            yield Path("part.parquet"), pd.DataFrame({"asset_id": [None]})
        finally:
            closed.append(True)

    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_file_batches",
        file_batches,
    )
    session = SimpleNamespace(
        resource_policy=policy,
        parquet_filters=(),
        inventory=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="^invalid_asset$"):
        with materialization_validation.crypto_asset_session_views(
            session,
            columns=("asset_id",),
            target_chunk_bytes=policy.available_memory_bytes,
            error="invalid_asset",
        ):
            raise AssertionError("unreachable")
    assert closed == [True]


def test_write_chunk_owner_closes_source_chunks_on_early_close() -> None:
    closed: list[bool] = []

    def source_chunks():
        try:
            lease = yield 128 * 1024
            yield _OwnedValue(pd.DataFrame({"value": [1]}), lease)
        finally:
            closed.append(True)

    with runtime_resource_policy_scope(RESOURCE_POLICY):
        iterator = source_chunks()
        bound = next(iterator)
        lease = RESOURCE_POLICY.lease("test.crypto.consumer", bound)
        assert iterator.send(lease).value.to_dict("list") == {"value": [1]}
        iterator.close()
        lease.close()
    assert closed == [True]


def test_panel_iterator_closes_bar_traversal_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    spec = processed_catalog().get("market_panel_1m")
    data_ids = (
        materialize_panel.source_ids.market_bars_source_data_id(spec),
        materialize_panel.source_ids.aggregate_trades_source_data_id(spec),
        materialize_panel.source_ids.depth_levels_source_data_id(spec),
    )
    sessions = {data_id: _dataset_session_stub(policy) for data_id in data_ids}
    closed: list[bool] = []
    pulled: list[bool] = []

    @contextmanager
    def asset_views(current, **_kwargs):  # noqa: ANN001
        yield ((("BTCUSDT",), current),)

    def bar_chunks(*_args, **_kwargs):
        try:
            pulled.append(True)
            yield pd.DataFrame({"placeholder": [1]})
        finally:
            closed.append(True)

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        asset_views,
    )
    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", bar_chunks)
    monkeypatch.setattr(
        materialize_panel,
        "_validate_matching_bar_keys",
        lambda _frame: None,
    )
    monkeypatch.setattr(
        materialize_panel.panels,
        "prepare_panel_bars",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validated_ordered_dependency_chunk",
        lambda frame, **_kwargs: frame,
    )
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
        lambda *_args, **_kwargs: pd.DataFrame(
            {column: [0] for column in spec.required_columns}
        ),
    )

    iterator = materialize_panel._iter_panel_chunks(
        spec,
        input_sessions=sessions,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        assert pulled == []
        lease = policy.lease("test.crypto.panel", bound)
        iterator.send(lease)
        iterator.close()
        lease.close()
    assert closed == [True]


def test_panel_empty_asset_transition_reuses_admitted_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("market_panel_1m")
    data_ids = (
        materialize_panel.source_ids.market_bars_source_data_id(spec),
        materialize_panel.source_ids.aggregate_trades_source_data_id(spec),
        materialize_panel.source_ids.depth_levels_source_data_id(spec),
    )
    first_session = _dataset_session_stub(policy)
    second_session = _dataset_session_stub(policy)
    sessions = {data_id: _dataset_session_stub(policy) for data_id in data_ids}
    closed: list[str] = []

    @contextmanager
    def asset_views(_current, **_kwargs):  # noqa: ANN001
        yield (
            (("BTCUSDT",), first_session),
            (("ETHUSDT",), second_session),
        )

    def bar_chunks(session, **_kwargs):  # noqa: ANN001, ANN003
        name = "first" if session is first_session else "second"
        try:
            yield (
                pd.DataFrame()
                if session is first_session
                else pd.DataFrame({"placeholder": [1]})
            )
        finally:
            closed.append(name)

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        asset_views,
    )
    monkeypatch.setattr(materialize_panel, "iter_dataset_chunks", bar_chunks)
    monkeypatch.setattr(
        materialize_panel,
        "_validate_matching_bar_keys",
        lambda _frame: None,
    )
    monkeypatch.setattr(
        materialize_panel.panels,
        "prepare_panel_bars",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validated_ordered_dependency_chunk",
        lambda frame, **_kwargs: frame,
    )
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
        lambda *_args, **_kwargs: pd.DataFrame(
            {column: [0] for column in spec.required_columns}
        ),
    )

    iterator = materialize_panel._iter_panel_chunks(
        spec,
        input_sessions=sessions,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        lease = policy.lease("test.crypto.panel.transition", bound)
        try:
            owned = iterator.send(lease)
            assert isinstance(owned, _OwnedValue)
            iterator.close()
        finally:
            lease.close()

    assert closed == ["first", "second"]
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("owner", ("panel", "factor", "label"))
def test_crypto_producer_admits_before_panel_pull(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    pulled: list[bool] = []
    frame = pd.DataFrame({"placeholder": [1]})

    @contextmanager
    def asset_views(current, **_kwargs):  # noqa: ANN001
        yield ((
            ("BTCUSDT",) if owner == "panel" else ("crypto", "btc_usdt"),
            current,
        ),)

    def panels(*_args, **_kwargs):  # noqa: ANN002, ANN003
        pulled.append(True)
        yield frame

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        asset_views,
    )
    module = {
        "panel": materialize_panel,
        "factor": materialize_factors,
        "label": materialize_labels,
    }[owner]
    monkeypatch.setattr(module, "iter_dataset_chunks", panels)
    if owner == "panel":
        spec = processed_catalog().get("market_panel_1m")
        data_ids = (
            materialize_panel.source_ids.market_bars_source_data_id(spec),
            materialize_panel.source_ids.aggregate_trades_source_data_id(spec),
            materialize_panel.source_ids.depth_levels_source_data_id(spec),
        )
        sessions = {data_id: _dataset_session_stub(policy) for data_id in data_ids}
        iterator = materialize_panel._iter_panel_chunks(
            spec,
            input_sessions=sessions,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
        )
    else:
        data_id = "microstructure" if owner == "factor" else "forward_returns"
        spec = processed_catalog().get(data_id)
        iterator = (
            materialize_factors._iter_factor_chunks(
                spec,
                panel_session=_dataset_session_stub(policy),
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                resource_policy=policy,
            )
            if owner == "factor"
            else materialize_labels._iter_label_chunks(
                spec,
                panel_session=_dataset_session_stub(policy),
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                resource_policy=policy,
            )
        )

    with runtime_resource_policy_scope(policy):
        blocker = policy.lease("test.crypto.blocker", policy.memory_budget_bytes)
        try:
            bound = next(iterator)
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                policy.lease("test.crypto.consumer", bound)
            assert pulled == []
        finally:
            blocker.close()


def test_crypto_factor_admits_each_segment_before_carry_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("microstructure")
    first = _panel_frame(start=0, minutes=5)
    second = first.copy()
    second["timestamp"] = second["timestamp"] + pd.Timedelta(minutes=5)
    frames = (first, second)
    bounds_seen = 0
    append_bounds: list[int] = []
    carry_seen = False
    carry_references: list[weakref.ReferenceType[pd.DataFrame]] = []
    carry_close_results: list[bool] = []
    panel_reference: list[weakref.ReferenceType[pd.DataFrame]] = []
    first_resume = True

    @contextmanager
    def asset_views(current, **_kwargs):  # noqa: ANN001
        yield ((
            ("crypto", "btc_usdt"),
            current,
        ),)

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        asset_views,
    )
    monkeypatch.setattr(
        materialize_factors,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(frames),
    )
    real_close = MemoryLease.close

    def record_close(lease: MemoryLease) -> None:
        if lease.owner == "crypto.factor.carry" and carry_references:
            carry_close_results.append(carry_references.pop(0)() is None)
        real_close(lease)

    monkeypatch.setattr(MemoryLease, "close", record_close)
    def capture_core_mask(current, *_args, **_kwargs):  # noqa: ANN001, ANN003
        panel_reference.append(weakref.ref(current))
        return pd.Series(True, index=current.index)

    monkeypatch.setattr(
        materialization_validation,
        "core_time_window_mask",
        capture_core_mask,
    )

    def record_append(segment, carry):  # noqa: ANN001, ANN202
        append_bounds.append(bounds_seen)
        if carry is not None:
            carry_references.append(weakref.ref(carry))
        return (
            segment
            if carry is None
            else pd.concat(
                (
                    carry.assign(**{materialize_factors._CORE_ROW: False}),
                    segment,
                ),
                ignore_index=True,
            )
        )

    monkeypatch.setattr(
        materialize_factors,
        "_append_active_factor_carry",
        record_append,
    )
    monkeypatch.setattr(
        materialize_factors,
        "calculate_microstructure_factors",
        lambda current, **_kwargs: current.assign(
            **{column: 0.0 for column in spec.required_columns if column not in current}
        ),
    )
    iterator = materialize_factors._iter_factor_chunks(
        spec,
        panel_session=_dataset_session_stub(policy),
        time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
        resource_policy=policy,
    )

    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        while True:
            bounds_seen += 1
            lease = policy.lease("test.crypto.factor", bound)
            try:
                owned = iterator.send(lease)
            except StopIteration:
                lease.close()
                break
            if not isinstance(owned, _OwnedValue):
                raise AssertionError("invalid_bound_first_output")
            carry_seen = carry_seen or policy.used_memory_bytes > lease.reserved_bytes
            try:
                bound = iterator.send(None)
            except StopIteration:
                lease.close()
                break
            if first_resume:
                assert panel_reference and panel_reference[0]() is None
                first_resume = False
            lease.close()

    assert len(append_bounds) >= 2
    assert append_bounds[1] > append_bounds[0]
    assert carry_seen
    assert carry_close_results and all(carry_close_results)
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("module_name", ("factor", "label"))
def test_crypto_carry_and_buffer_cap_fixed_allowance_to_admitted_bound(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024 * 1024)
    frame = _panel_frame(start=0, minutes=70)
    target_headroom = 2_000
    producer_bound = policy.memory_budget_bytes - target_headroom - 1
    reader_reserve = policy.memory_budget_bytes - producer_bound - target_headroom
    assert (
        pandas_frame_payload_bytes(
            frame.tail(materialize_labels.LABEL_LOOKAHEAD_BARS)
        )
        > target_headroom
    )
    session = _dataset_session_stub(policy)
    reader_lease: list[MemoryLease | None] = [None]
    buffer_bounds: list[int] = []
    requested_bounds: list[int] = []
    real_close = MemoryLease.close
    real_lease = RuntimeResourcePolicy.lease

    def lease(root, owner, reserved_bytes):  # noqa: ANN001
        result = real_lease(root, owner, reserved_bytes)
        if root is policy and owner == "crypto.label.buffer":
            buffer_bounds.append(reserved_bytes)
        return result

    def close(lease: MemoryLease) -> None:
        real_close(lease)
        if lease.owner == "test.crypto.consumer" and reader_lease[0] is not None:
            reader_lease[0].close()

    monkeypatch.setattr(MemoryLease, "close", close)
    monkeypatch.setattr(RuntimeResourcePolicy, "lease", lease)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: _asset_views_context(
            ((("crypto", "btc_usdt"), session),),
        ),
    )

    def chunks(*_args, **_kwargs):  # noqa: ANN002, ANN003
        lease = policy.lease("test.crypto.reader", reader_reserve)
        reader_lease[0] = lease
        try:
            yield frame
        finally:
            lease.close()

    monkeypatch.setattr(
        materialize_factors if module_name == "factor" else materialize_labels,
        "iter_dataset_chunks",
        chunks,
    )
    spec = processed_catalog().get(
        "microstructure" if module_name == "factor" else "forward_returns"
    )
    iterator = (
        materialize_factors._iter_factor_chunks(
            spec,
            panel_session=session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
        )
        if module_name == "factor"
        else materialize_labels._iter_label_chunks(
            spec,
            panel_session=session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
        )
    )

    outputs = _collect_bound_first(
        iterator,
        policy,
        requested_bounds=requested_bounds,
    )
    if module_name == "label":
        assert len(buffer_bounds) == 1
        assert requested_bounds[1] + buffer_bounds[0] <= producer_bound
        assert buffer_bounds[0] > target_headroom
        assert sum(len(output) for output in outputs) == len(frame)
    assert policy.used_memory_bytes == 0


def test_crypto_label_buffer_lease_survives_output_and_closes_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("forward_returns")
    first = _panel_frame(start=0, minutes=70)
    second = first.copy()
    second["timestamp"] = second["timestamp"] + pd.Timedelta(minutes=70)
    panel_references: list[weakref.ReferenceType[pd.DataFrame]] = []
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(
            ((("crypto", "btc_usdt"), session),)
        ),
    )
    monkeypatch.setattr(
        materialize_labels,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((first, second)),
    )
    def capture_core_mask(current, *_args, **_kwargs):  # noqa: ANN001, ANN003
        panel_references.append(weakref.ref(current))
        return pd.Series(True, index=current.index)

    monkeypatch.setattr(
        materialization_validation,
        "core_time_window_mask",
        capture_core_mask,
    )
    iterator = materialize_labels._iter_label_chunks(
        spec,
        panel_session=_dataset_session_stub(policy),
        time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 02:19"),
        resource_policy=policy,
    )

    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        saw_buffer_owner = False
        while True:
            lease = policy.lease("test.crypto.label", bound)
            try:
                owned = iterator.send(lease)
                assert isinstance(owned, _OwnedValue)
                if not saw_buffer_owner:
                    assert policy.used_memory_bytes > lease.reserved_bytes
                    saw_buffer_owner = True
                try:
                    bound = iterator.send(None)
                except StopIteration:
                    break
                assert panel_references and panel_references[-1]() is None
            finally:
                lease.close()
        iterator.close()

    assert len(panel_references) >= 2
    assert all(reference() is None for reference in panel_references)
    assert saw_buffer_owner
    assert policy.used_memory_bytes == 0


def test_crypto_label_finalizes_asset_before_pulling_next_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("forward_returns")
    btc_session = _dataset_session_stub(policy)
    eth_session = _dataset_session_stub(policy)
    btc = _panel_frame(start=0, minutes=70)
    eth = btc.copy()
    eth["asset_id"] = "eth_usdt"
    eth["timestamp"] = eth["timestamp"] + pd.Timedelta(minutes=70)
    events: list[str] = []

    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: _asset_views_context(
            (
                (("crypto", "btc_usdt"), btc_session),
                (("crypto", "eth_usdt"), eth_session),
            )
        ),
    )

    def chunks(session, **_kwargs):  # noqa: ANN001, ANN003
        if session is btc_session:
            events.append("pull:btc")
            yield btc
        else:
            events.append("pull:eth")
            yield eth

    monkeypatch.setattr(materialize_labels, "iter_dataset_chunks", chunks)
    real_finalize = materialize_labels._final_label_output

    def record_finalize(buffer, current_spec):  # noqa: ANN001
        events.append("finalize")
        return real_finalize(buffer, current_spec)

    monkeypatch.setattr(
        materialize_labels,
        "_final_label_output",
        record_finalize,
    )
    iterator = materialize_labels._iter_label_chunks(
        spec,
        panel_session=_dataset_session_stub(policy),
        time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 02:19"),
        resource_policy=policy,
    )

    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        outputs = []
        while True:
            lease = policy.lease("test.crypto.label.boundary", bound)
            try:
                owned = iterator.send(lease)
                assert isinstance(owned, _OwnedValue)
                outputs.append(owned.value)
                try:
                    bound = iterator.send(None)
                except StopIteration:
                    break
            finally:
                lease.close()
        iterator.close()

    assert outputs
    assert events.index("finalize") < events.index("pull:eth")
    assert policy.used_memory_bytes == 0


def test_crypto_label_closes_terminal_empty_asset_producer_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("forward_returns")
    session = _dataset_session_stub(policy)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: _asset_views_context(
            ((("crypto", "btc_usdt"), session),)
        ),
    )
    monkeypatch.setattr(
        materialize_labels,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((pd.DataFrame(),)),
    )
    iterator = materialize_labels._iter_label_chunks(
        spec,
        panel_session=session,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=policy,
    )

    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        lease = policy.lease("test.crypto.label.empty-terminal", bound)
        try:
            with pytest.raises(StopIteration):
                iterator.send(lease)
            assert not lease.active
            assert policy.used_memory_bytes == 0
        finally:
            if lease.active:
                lease.close()


def test_crypto_key_normalizers_require_active_policy() -> None:
    assert current_runtime_resource_policy() is None
    rows = materialization_validation._iter_normalized_primary_key_rows(
        iter((_key_frame().iloc[:1],)),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
    )
    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        next(rows)

    panel_rows = materialization_validation._iter_panel_primary_key_chunks(
        iter(
            (
                pd.DataFrame(
                    {
                        "source_asset_id": ["BTCUSDT"],
                        "timestamp": [pd.Timestamp("2026-07-06")],
                    }
                ),
            )
        ),
        primary_key=("market_id", "asset_id", "timestamp"),
    )
    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        next(panel_rows)


def test_crypto_label_buffer_replacement_clears_old_payload_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    spec = processed_catalog().get("forward_returns")
    first = _panel_frame(start=0, minutes=70)
    second = first.copy()
    second["timestamp"] = second["timestamp"] + pd.Timedelta(minutes=70)
    old_buffer_references: list[weakref.ReferenceType[pd.DataFrame]] = []
    close_results: list[bool] = []
    real_concat = materialize_labels.pd.concat
    real_close = MemoryLease.close

    def record_concat(objects, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        old_buffer_references.append(weakref.ref(objects[0]))
        return real_concat(objects, *args, **kwargs)

    def record_close(lease: MemoryLease) -> None:
        if lease.owner == "crypto.label.buffer" and old_buffer_references:
            close_results.append(old_buffer_references.pop(0)() is None)
        real_close(lease)

    monkeypatch.setattr(materialize_labels.pd, "concat", record_concat)
    monkeypatch.setattr(MemoryLease, "close", record_close)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda session, **_kwargs: _asset_views_context(
            ((("crypto", "btc_usdt"), session),)
        ),
    )
    monkeypatch.setattr(
        materialize_labels,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((first, second)),
    )

    output = _collect_bound_first(
        materialize_labels._iter_label_chunks(
            spec,
            panel_session=_dataset_session_stub(policy),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 02:19"),
            resource_policy=policy,
        ),
        policy,
    )

    assert output
    assert close_results and all(close_results)
    assert policy.used_memory_bytes == 0


def _dataset_session_stub(
    resource_policy: RuntimeResourcePolicy,
    *,
    retained_memory_bytes: int = 0,
    **values: object,
) -> SimpleNamespace:
    values.setdefault("verify", lambda: None)
    return SimpleNamespace(
        resource_policy=resource_policy,
        retained_memory_bytes=retained_memory_bytes,
        **values,
    )


def test_input_dataset_identities_distinguish_source_and_manifest() -> None:
    first = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 3},
    )
    other_source = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_b"},
        manifest_extra={"row_count": 3},
    )
    changed_manifest = _inspection(
        "market_bars_1m",
        spec_extra={"source_id": "source_a"},
        manifest_extra={"row_count": 4},
    )

    first_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": first}
    )
    other_source_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": other_source}
    )
    changed_manifest_identity = materialization_validation.input_dataset_identities(
        {"market_bars_1m": changed_manifest}
    )

    assert first_identity != other_source_identity
    assert first_identity != changed_manifest_identity
    assert set(first_identity["market_bars_1m"]) == {
        "data_id",
        "spec_hash",
        "manifest_hash",
    }


@pytest.mark.parametrize(
    ("module", "writer_name", "data_id"),
    (
        (materialize_factors, "write_factors", "microstructure"),
        (materialize_labels, "write_labels", "forward_returns"),
    ),
)
def test_panel_dependent_writer_rejects_policy_mismatch_before_input_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    writer_name: str,
    data_id: str,
) -> None:
    session_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    explicit_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    panel_session = _dataset_session_stub(session_policy)
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        lambda *_args, **_kwargs: pytest.fail("input identity reached"),
    )
    monkeypatch.setattr(
        module,
        "_write_dataset_owned",
        lambda *_args, **_kwargs: pytest.fail("write reached"),
    )

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        getattr(module, writer_name)(
            processed_catalog().get(data_id),
            layout=StorageLayout(project_root=tmp_path),
            panel_session=panel_session,
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=explicit_policy,
        )


@pytest.mark.parametrize(
    ("module", "iterator_name", "data_id"),
    (
        (materialize_factors, "_iter_factor_chunks", "microstructure"),
        (materialize_labels, "_iter_label_chunks", "forward_returns"),
    ),
)
def test_panel_dependent_iterator_rejects_policy_mismatch_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    module,
    iterator_name: str,
    data_id: str,
) -> None:
    session_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    explicit_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    panel_session = _dataset_session_stub(session_policy)
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: pytest.fail("selection reached"),
    )

    iterator = getattr(module, iterator_name)(
        processed_catalog().get(data_id),
        panel_session=panel_session,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=explicit_policy,
    )
    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        next(iterator)


def test_panel_iterator_rejects_any_session_policy_mismatch_before_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get("market_panel_1m")
    explicit_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    other_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    data_ids = (
        materialize_panel.source_ids.market_bars_source_data_id(spec),
        materialize_panel.source_ids.aggregate_trades_source_data_id(spec),
        materialize_panel.source_ids.depth_levels_source_data_id(spec),
    )
    sessions = {
        data_id: _dataset_session_stub(other_policy if index == 1 else explicit_policy)
        for index, data_id in enumerate(data_ids)
    }
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: pytest.fail("selection reached"),
    )

    iterator = materialize_panel._iter_panel_chunks(
        spec,
        input_sessions=sessions,
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        resource_policy=explicit_policy,
    )
    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        next(iterator)


@pytest.mark.parametrize(
    ("owner", "mismatch"),
    (
        ("panel", "receipt"),
        ("panel", "bar"),
        ("dependent", "receipt"),
        ("dependent", "panel"),
        ("existing", "panel"),
        ("existing", "bar"),
    ),
)
def test_validation_rejects_session_policy_mismatch_before_identity_or_traversal(
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    mismatch: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    other_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def session(name: str) -> SimpleNamespace:
        return _dataset_session_stub(
            other_policy if mismatch == name else policy,
            name=name,
            manifest={},
        )

    receipt = SimpleNamespace(session=session("receipt"))
    panel_session = session("panel")
    bar_session = session("bar")
    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: pytest.fail("identity validation reached"),
    )
    monkeypatch.setattr(
        materialization_validation,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: pytest.fail("traversal reached"),
    )

    with pytest.raises(RuntimeError, match="^runtime_resource_policy_conflict$"):
        if owner == "panel":
            materialization_validation.validate_panel_materialization(
                receipt,
                bar_session=bar_session,
                spec=SimpleNamespace(primary_key=("value",)),
                expected_input_identities={},
                resource_policy=policy,
            )
        elif owner == "dependent":
            materialization_validation.validate_panel_dependent_materialization(
                receipt,
                panel_session=panel_session,
                spec=SimpleNamespace(primary_key=("value",)),
                time_window=TimeWindow("2026-07-06", "2026-07-06"),
                expected_input_identities={},
                resource_policy=policy,
            )
        else:
            materialization_validation.validate_existing_panel_materialization(
                panel_session,
                bar_session=bar_session,
                spec=SimpleNamespace(primary_key=("value",)),
                expected_input_identities={},
                resource_policy=policy,
            )


def test_panel_dependent_validation_uses_live_policy_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 750,
    )
    observed: list[int | None] = []
    receipt = SimpleNamespace(
        session=_dataset_session_stub(policy, manifest={}),
        retained_memory_bytes=300,
    )
    panel_session = _dataset_session_stub(policy, retained_memory_bytes=200)
    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: None,
    )

    def core_chunks(*_args, **kwargs):  # noqa: ANN002, ANN003
        assert current_runtime_resource_policy() is policy
        observed.append(kwargs)
        return iter(())

    monkeypatch.setattr(
        materialization_validation,
        "_iter_core_window_chunks",
        core_chunks,
    )
    monkeypatch.setattr(
        materialization_validation,
        "_validate_receipt_key_coverage",
        lambda *_args, **_kwargs: None,
    )

    materialization_validation.validate_panel_dependent_materialization(
        receipt,
        panel_session=panel_session,
        spec=SimpleNamespace(primary_key=("value",)),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        expected_input_identities={},
        resource_policy=policy,
    )

    assert len(observed) == 1
    assert observed[0]["time_window"] == TimeWindow("2026-07-06", "2026-07-06")
    assert observed[0]["columns"] == ("value",)
    assert 0 < observed[0]["target_chunk_bytes"] <= policy.memory_budget_bytes


def test_panel_validation_projects_input_before_key_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2,
        explicit_cap_bytes=4 * 1024**2,
    )
    row_count = 4_000
    full_frame = pd.DataFrame(
        {
            "market_id": ["crypto"] * row_count,
            "asset_id": ["asset"] * row_count,
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="s"),
            "value": ["x" * 1_100] * row_count,
        }
    )
    primary_key = ("market_id", "asset_id", "timestamp")
    projected = full_frame.loc[:, list(primary_key)]
    assert pandas_frame_payload_bytes(full_frame) > policy.memory_budget_bytes
    assert 2 * pandas_frame_payload_bytes(projected) <= policy.memory_budget_bytes
    session = DatasetSession(
        root=Path("panel"),
        data_root=Path("panel/data"),
        spec_bytes=b"{}",
        spec_identity_bytes=b"",
        manifest_bytes=b"{}",
        manifest_identity_bytes=b"",
        inventory=SimpleNamespace(),
        logical_columns=tuple(full_frame.columns),
        physical_columns=tuple(full_frame.columns),
        timestamp_column="timestamp",
        time_window=None,
        schema_hash="schema",
        arrow_schema=None,
        logical_dtype_bytes=b"",
        parquet_filters=None,
        resource_policy=policy,
        owns_inventory=False,
        _lease=None,
        plan_base_retained_bytes=0,
    )
    observed: list[tuple[tuple[str, ...], int]] = []

    class Index:
        def add_frame(self, frame: pd.DataFrame, *, namespace: str) -> int:
            assert namespace == "crypto_expected"
            assert tuple(frame.columns) == primary_key
            return 0

        def difference_counts(
            self,
            *,
            expected_namespace: str,
            actual_namespace: str,
        ) -> tuple[int, int]:
            assert expected_namespace == "crypto_expected"
            assert actual_namespace == "dataset"
            return 0, 0

    def chunks(view, *, target_chunk_bytes: int, **_kwargs):  # noqa: ANN001
        observed.append((tuple(view.logical_columns), target_chunk_bytes))
        frame = (
            full_frame
            if "value" in view.logical_columns
            else projected
        )
        with policy.lease(
            "storage.arrow.to_pandas",
            pandas_frame_payload_bytes(frame),
        ):
            yield frame

    monkeypatch.setattr(
        materialization_validation,
        "validate_materialization_input_identities",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(materialization_validation, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(DatasetSession, "verify", lambda _session: None)
    receipt = SimpleNamespace(
        session=SimpleNamespace(
            manifest={},
            resource_policy=policy,
            verify=lambda: None,
        ),
        exact_primary_key_index=Index(),
    )

    materialization_validation.validate_panel_dependent_materialization(
        receipt,
        panel_session=session,
        spec=SimpleNamespace(primary_key=primary_key),
        time_window=TimeWindow("2026-01-01", "2026-01-01"),
        expected_input_identities={},
        resource_policy=policy,
    )

    assert len(observed) == 1
    assert observed[0][0] == primary_key
    assert 0 < observed[0][1] <= policy.memory_budget_bytes
    assert policy.used_memory_bytes == 0


def test_materialization_input_identity_validation_is_exact() -> None:
    source = _inspection("market_bars_1m")
    expected = materialization_validation.input_dataset_identities(
        {"market_bars_1m": source}
    )
    target = _inspection(
        "market_panel_1m",
        manifest_extra={"input_dataset_identities": expected},
    )

    materialization_validation.validate_materialization_input_identities(
        target.manifest,
        expected=expected,
        error="crypto_panel_input_identity_mismatch",
    )

    with pytest.raises(ValueError, match="crypto_panel_input_identity_mismatch"):
        materialization_validation.validate_materialization_input_identities(
            _inspection("market_panel_1m").manifest,
            expected=expected,
            error="crypto_panel_input_identity_mismatch",
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_asset", "missing_middle", "extra", "market_mismatch", "duplicate"],
)
def test_exact_dependency_key_coverage_rejects_any_full_key_delta(
    mutation: str,
    tmp_path: Path,
) -> None:
    expected = _key_frame()
    actual = expected.copy()
    if mutation == "missing_asset":
        actual = actual.loc[actual["asset_id"].ne("asset_b")].copy()
    elif mutation == "missing_middle":
        actual = actual.loc[
            ~(
                actual["asset_id"].eq("asset_a")
                & actual["timestamp"].eq(pd.Timestamp("2026-07-06 00:01"))
            )
        ].copy()
    elif mutation == "extra":
        actual = pd.concat(
            [
                actual,
                actual.iloc[[0]].assign(timestamp=pd.Timestamp("2026-07-06 00:03")),
            ],
            ignore_index=True,
        )
    elif mutation == "market_mismatch":
        actual.loc[0, "market_id"] = "other_market"
    else:
        actual = pd.concat([actual, actual.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            iter((expected.iloc[:2].copy(), expected.iloc[2:].copy())),
            iter((actual.iloc[:3].copy(), actual.iloc[3:].copy())),
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_exact_dependency_key_coverage_accepts_equal_bounded_streams(
    tmp_path: Path,
) -> None:
    expected = _key_frame()

    _validate_receipt_key_coverage(
        iter((expected.iloc[:1], expected.iloc[1:4], expected.iloc[4:])),
        iter((expected.iloc[:2], expected.iloc[2:5], expected.iloc[5:])),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_exact_dependency_key_coverage_is_order_and_chunk_independent(
    chunk_size: int,
    tmp_path: Path,
) -> None:
    expected = _key_frame()
    actual = expected.iloc[[5, 1, 3, 0, 4, 2]].reset_index(drop=True)

    _validate_receipt_key_coverage(
        (
            expected.iloc[start : start + chunk_size].copy()
            for start in range(0, len(expected), chunk_size)
        ),
        (
            actual.iloc[start : start + chunk_size].copy()
            for start in range(0, len(actual), chunk_size)
        ),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


def test_panel_key_coverage_accepts_non_order_preserving_asset_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {"SOURCE_A": "asset_z", "SOURCE_B": "asset_a"}
    monkeypatch.setattr(
        materialization_validation,
        "canonical_crypto_asset_id",
        mapping.__getitem__,
    )
    bars = pd.DataFrame(
        {
            "source_asset_id": ["SOURCE_A", "SOURCE_B"],
            "timestamp": [pd.Timestamp("2026-07-06")] * 2,
        }
    )
    primary_key = ("market_id", "asset_id", "timestamp")
    expected = materialization_validation._panel_primary_keys(
        bars,
        primary_key=primary_key,
    )
    actual = expected.iloc[::-1].reset_index(drop=True)

    _validate_receipt_key_coverage(
        iter((expected,)),
        iter((actual,)),
        primary_key=primary_key,
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )

    assert list(tmp_path.iterdir()) == []


def test_factor_carry_processing_is_linear_in_asset_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_count = 64
    rows_per_asset = 6
    chunks = tuple(
        _panel_frame(start=0, minutes=rows_per_asset).assign(
            asset_id=f"asset_{asset_index:04d}"
        )
        for asset_index in range(asset_count)
    )
    processed_rows = 0

    def record_rows(frame: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        nonlocal processed_rows
        processed_rows += len(frame)
        output = frame.copy()
        for column in processed_catalog().get("microstructure").required_columns:
            if column not in output:
                output[column] = 0.0
        return output

    def read_asset_chunks(context, **_kwargs):
        filters = dict(
            (column, value)
            for column, operator, value in context.row_filters
            if operator == "=="
        )
        if not filters:
            return iter(chunks)
        return iter(
            tuple(
                chunk.loc[
                    chunk["market_id"].eq(filters["market_id"])
                    & chunk["asset_id"].eq(filters["asset_id"])
                ].copy()
                for chunk in chunks
                if chunk["asset_id"].eq(filters["asset_id"]).any()
            )
        )

    monkeypatch.setattr(
        materialize_factors,
        "iter_dataset_chunks",
        read_asset_chunks,
    )
    _patch_mock_group_plan(
        monkeypatch,
        tuple(("crypto", f"asset_{index:04d}") for index in range(asset_count)),
    )
    monkeypatch.setattr(
        materialize_factors,
        "calculate_microstructure_factors",
        record_rows,
    )
    spec = processed_catalog().get("microstructure")
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    _collect_bound_first(
        materialize_factors._iter_factor_chunks(
            spec,
            panel_session=_dataset_session_stub(policy),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:05"),
            resource_policy=policy,
        ),
        policy,
    )

    assert processed_rows <= asset_count * (
        rows_per_asset + materialize_factors.MICROSTRUCTURE_ROLLING_CARRY_BARS
    )


@pytest.mark.parametrize(
    ("module", "output_data_id", "minutes"),
    [
        (materialize_factors, "microstructure", 6),
        (materialize_labels, "forward_returns", 61),
    ],
)
def test_panel_dependents_reject_mixed_asset_physical_parts_before_repeated_reads(
    tmp_path: Path,
    module,
    output_data_id: str,
    minutes: int,
) -> None:
    catalog = processed_catalog()
    panel_spec = catalog.get("market_panel_1m")
    panel_root = StorageLayout(tmp_path).processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    mixed = pd.concat(
        tuple(
            _panel_frame(start=0, minutes=minutes).assign(asset_id=f"asset_{index:04d}")
            for index in range(8)
        ),
        ignore_index=True,
    )
    write_dataset(panel_root, mixed, spec=panel_spec)
    iterator_name = (
        "_iter_factor_chunks"
        if output_data_id == "microstructure"
        else "_iter_label_chunks"
    )
    iterator = getattr(module, iterator_name)
    with open_dataset_session(
        panel_root,
        columns=None,
        time_window=None,
    ) as panel_session:
        with pytest.raises(
            ValueError,
            match="invalid_crypto_panel_physical_asset_grouping",
        ):
            list(
                iterator(
                    catalog.get(output_data_id),
                    panel_session=panel_session,
                    time_window=TimeWindow(
                        pd.Timestamp("2026-07-06 00:00:00"),
                        pd.Timestamp("2026-07-06 00:00:00")
                        + pd.Timedelta(minutes=minutes - 1),
                    ),
                    resource_policy=panel_session.resource_policy,
                )
            )


def test_canonical_asset_normalization_is_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_count = 200
    work = {"len_calls": 0}

    class CountingString(str):
        def __len__(self) -> int:
            work["len_calls"] += 1
            return super().__len__()

    keys = tuple(
        (CountingString(f"source_{index:04d}"),) for index in range(asset_count)
    )
    monkeypatch.setattr(
        materialize_panel,
        "canonical_crypto_asset_id",
        lambda value: CountingString(f"canonical_{value}"),
    )

    observed = materialize_panel._canonical_source_assets(keys)

    assert len(observed) == asset_count
    assert work["len_calls"] <= asset_count * 5


@pytest.mark.parametrize(
    ("module", "iterator_name", "data_id"),
    [
        (materialize_factors, "_iter_factor_chunks", "microstructure"),
        (materialize_labels, "_iter_label_chunks", "forward_returns"),
    ],
)
def test_panel_dependencies_remain_asset_major_across_year_partitions(
    tmp_path: Path,
    module,
    iterator_name: str,
    data_id: str,
) -> None:
    timestamps = pd.date_range("2025-12-31 23:55", periods=10, freq="min")
    asset_a = _panel_frame(start=0, minutes=10).assign(
        asset_id="asset_a",
        timestamp=timestamps,
    )
    asset_b = _panel_frame(start=0, minutes=10).assign(
        asset_id="asset_b",
        timestamp=timestamps,
    )
    panel_root = StorageLayout(project_root=tmp_path).processed_data_root(
        "crypto",
        processed_catalog().get("market_panel_1m").kind,
        "market_panel_1m",
    )
    write_dataset(
        panel_root,
        (asset_a, asset_b),
        spec=processed_catalog().get("market_panel_1m"),
    )

    with (
        open_dataset_session(
            panel_root,
            columns=None,
            time_window=None,
        ) as panel_session,
        runtime_resource_policy_scope(panel_session.resource_policy),
    ):
        outputs = _collect_bound_first(
            getattr(module, iterator_name)(
                processed_catalog().get(data_id),
                panel_session=panel_session,
                time_window=TimeWindow(
                    "2025-12-31 23:55",
                    "2026-01-01 00:04",
                ),
                resource_policy=panel_session.resource_policy,
            ),
            panel_session.resource_policy,
        )

    assert outputs
    observed = pd.concat(outputs, ignore_index=True).loc[
        :, ["market_id", "asset_id", "timestamp"]
    ]
    pd.testing.assert_frame_equal(
        observed,
        observed.sort_values(
            ["market_id", "asset_id", "timestamp"],
            kind="mergesort",
        ).reset_index(drop=True),
    )


@pytest.mark.parametrize("empty_side", ["expected", "actual"])
def test_exact_dependency_key_coverage_rejects_one_empty_stream(
    empty_side: str,
    tmp_path: Path,
) -> None:
    populated = iter((_key_frame(),))
    expected = iter(()) if empty_side == "expected" else populated
    actual = iter(()) if empty_side == "actual" else populated

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            expected,
            actual,
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_exact_dependency_key_coverage_accepts_two_empty_streams(
    tmp_path: Path,
) -> None:
    _validate_receipt_key_coverage(
        iter(()),
        iter(()),
        primary_key=("market_id", "asset_id", "timestamp"),
        error="invalid_crypto_dependency_key_coverage",
        resource_policy=RESOURCE_POLICY,
        scratch_parent=tmp_path,
    )


@pytest.mark.parametrize("stream_name", ["expected", "actual"])
def test_exact_dependency_key_coverage_rejects_invalid_individual_streams(
    stream_name: str,
    tmp_path: Path,
) -> None:
    baseline = _key_frame()
    mutated = iter((baseline.iloc[:2], baseline.iloc[1:].copy()))
    expected = mutated if stream_name == "expected" else iter((baseline,))
    actual = mutated if stream_name == "actual" else iter((baseline,))

    with pytest.raises(ValueError, match="invalid_crypto_dependency_key_coverage"):
        _validate_receipt_key_coverage(
            expected,
            actual,
            primary_key=("market_id", "asset_id", "timestamp"),
            error="invalid_crypto_dependency_key_coverage",
            resource_policy=RESOURCE_POLICY,
            scratch_parent=tmp_path,
        )


def test_dependency_chunk_order_rejects_unsorted_and_overlap() -> None:
    state = materialization_validation.OrderedDependencyState()
    first = _key_frame().iloc[[2, 0, 1, 5, 3, 4]].copy()

    with pytest.raises(ValueError, match="out_of_order_crypto_factor_dependency"):
        materialization_validation.validated_ordered_dependency_chunk(
            first,
            state=state,
            error="out_of_order_crypto_factor_dependency",
        )

    ordered = first.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    result = materialization_validation.validated_ordered_dependency_chunk(
        ordered,
        state=state,
        error="out_of_order_crypto_factor_dependency",
    )
    pd.testing.assert_frame_equal(result, ordered, check_dtype=False)
    assert state.last_key == (
        "crypto",
        "asset_b",
        pd.Timestamp("2026-07-06 00:02"),
    )

    with pytest.raises(ValueError, match="out_of_order_crypto_factor_dependency"):
        materialization_validation.validated_ordered_dependency_chunk(
            _key_frame().iloc[[2]].copy(),
            state=state,
            error="out_of_order_crypto_factor_dependency",
        )


def test_existing_dependency_reuse_passes_same_resource_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 3072,
    )
    inspection = _inspection("market_panel_1m")
    observed: list[RuntimeResourcePolicy] = []
    monkeypatch.setattr(
        materializer,
        "open_dataset_session",
        lambda *_args, **_kwargs: nullcontext(_dataset_session_stub(policy)),
    )
    def owned_inspection(*_args, resource_policy, **_kwargs):  # noqa: ANN001, ANN003
        return _OwnedValue(
            inspection,
            resource_policy.lease("test.crypto.inspection", 0),
        )

    monkeypatch.setattr(
        materializer,
        "_inspect_dataset_session_owned",
        owned_inspection,
    )
    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        lambda _sessions: {},
    )

    def record_validation(*_args, resource_policy, **_kwargs) -> None:
        observed.append(resource_policy)

    monkeypatch.setattr(
        materialization_validation,
        "validate_existing_panel_materialization",
        record_validation,
    )

    with runtime_resource_policy_scope(policy), ExitStack() as session_stack:
        result, panel_session = materializer._inspect_existing_dependency(
            StorageLayout(project_root=tmp_path),
            processed_catalog().get("market_panel_1m"),
            source_id=SourceId("synthetic_crypto"),
            time_window=TimeWindow("2026-07-06", "2026-07-06"),
            resource_policy=policy,
            session_stack=session_stack,
        )

        assert result is inspection
        panel_session.verify()
    assert observed == [policy]


@pytest.mark.parametrize(
    ("module", "function_name", "iterator_name", "data_id"),
    [
        (
            materialize_panel,
            "write_panel",
            "_iter_panel_chunks",
            "market_panel_1m",
        ),
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
    ],
)
def test_crypto_direct_dataset_writers_keep_one_policy_through_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    function_name: str,
    iterator_name: str,
    data_id: str,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
    )
    inspection = _inspection(data_id)
    observed: list[RuntimeResourcePolicy] = []

    def capture_input_identities(_sessions):
        observed.append(current_runtime_resource_policy())
        return {}

    def capture_write(
        *_args,
        staged_validator,
        **_kwargs,
    ) -> DatasetInspection:
        resource_policy = current_runtime_resource_policy()
        assert resource_policy is policy
        observed.append(current_runtime_resource_policy())
        chunks = list(_args[1])
        assert len(chunks) == 1
        assert "retained_payload_bytes" not in _kwargs
        staged_validator(SimpleNamespace())
        return _OwnedValue(
            inspection,
            resource_policy.lease("test.crypto.writer.result", 0),
        )

    monkeypatch.setattr(
        materialization_validation,
        "input_dataset_identities",
        capture_input_identities,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_materialization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_panel_dependent_materialization",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        iterator_name,
        lambda *_args, **_kwargs: iter((pd.DataFrame({"value": [1]}),)),
    )
    monkeypatch.setattr(module, "_write_dataset_owned", capture_write)
    kwargs = {
        "layout": StorageLayout(project_root=tmp_path),
        "time_window": TimeWindow("2026-07-06", "2026-07-06"),
        "resource_policy": policy,
    }
    if module is materialize_panel:
        kwargs["source_id"] = SourceId("synthetic_crypto")
        monkeypatch.setattr(
            module,
            "open_dataset_session",
            lambda *_args, **_kwargs: nullcontext(
                _dataset_session_stub(policy, retained_memory_bytes=37)
            ),
        )
    else:
        kwargs["panel_session"] = _dataset_session_stub(
            policy,
            retained_memory_bytes=37,
        )

    result = getattr(module, function_name)(
        processed_catalog().get(data_id),
        **kwargs,
    )

    assert result is inspection
    assert observed == [policy, policy]


@pytest.mark.parametrize(
    ("module_name", "error"),
    [
        ("factor", "out_of_order_crypto_factor_dependency"),
        ("label", "out_of_order_crypto_label_dependency"),
    ],
)
@pytest.mark.parametrize("second_start", [0, 4], ids=["reversed", "overlap"])
def test_factor_and_label_materializers_reject_non_monotonic_chunks(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    error: str,
    second_start: int,
) -> None:
    if second_start == 0:
        first = _panel_frame(start=5, minutes=5)
    else:
        first = _panel_frame(start=0, minutes=5)
    second = _panel_frame(start=second_start, minutes=5)
    spec = processed_catalog().get(
        "microstructure" if module_name == "factor" else "forward_returns"
    )
    module = materialize_factors if module_name == "factor" else materialize_labels
    monkeypatch.setattr(
        module,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((first, second)),
    )
    _patch_mock_group_plan(monkeypatch, (("crypto", "btc_usdt"),))
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    iterator = (
        module._iter_factor_chunks(
            spec,
            panel_session=_dataset_session_stub(policy),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
            resource_policy=policy,
        )
        if module_name == "factor"
        else module._iter_label_chunks(
            spec,
            panel_session=_dataset_session_stub(policy),
            time_window=TimeWindow("2026-07-06 00:00", "2026-07-06 00:10"),
            resource_policy=policy,
        )
    )

    with pytest.raises(ValueError, match=error):
        _collect_bound_first(iterator, policy)


def _inspection(
    data_id: str,
    *,
    spec_extra: dict[str, object] | None = None,
    manifest_extra: dict[str, object] | None = None,
) -> DatasetInspection:
    spec = {"schema_id": "qrp-dataset-spec", "data_id": data_id}
    manifest = {"schema_id": "qrp-dataset-manifest", "data_id": data_id}
    spec.update(spec_extra or {})
    manifest.update(manifest_extra or {})
    return DatasetInspection(
        root=Path(data_id),
        data_root=Path(data_id) / "data",
        spec=spec,
        columns=("market_id", "asset_id", "timestamp"),
        schema_hash="schema",
        file_count=1,
        row_count=int(manifest.get("row_count", 0)),
        partitions=("year=2026",),
        partition_row_counts={"year=2026": int(manifest.get("row_count", 0))},
        date_min="2026-07-06",
        date_max="2026-07-06",
        manifest=manifest,
    )


def _key_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-06", periods=3, freq="min")
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * 6,
            "asset_id": ["asset_a"] * 3 + ["asset_b"] * 3,
            "timestamp": [*timestamps, *timestamps],
        }
    )


def _validate_receipt_key_coverage(
    expected,
    actual,
    *,
    primary_key: tuple[str, ...],
    error: str,
    resource_policy: RuntimeResourcePolicy,
    scratch_parent: Path,
) -> None:
    test_policy = capture_test_policy(
        available_memory_bytes=resource_policy.captured_available_memory_bytes,
    )
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=tuple(
            column for column in primary_key if column == "timestamp"
        ),
        scratch_parent=scratch_parent,
    ) as index:
        for frame in actual:
            duplicates = index.add_frame(
                frame.loc[:, list(primary_key)],
                namespace="dataset",
            )
            if duplicates:
                raise ValueError(error)
        receipt = SimpleNamespace(
            session=_dataset_session_stub(test_policy),
            exact_primary_key_index=index,
        )
        with runtime_resource_policy_scope(test_policy):
            materialization_validation._validate_receipt_key_coverage(
                expected,
                receipt,
                primary_key=primary_key,
                error=error,
                resource_policy=test_policy,
            )


def _panel_frame(*, start: int, minutes: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-07-06 00:00:00",
        periods=start + minutes,
        freq="min",
    )[start:]
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * minutes,
            "asset_id": ["btc_usdt"] * minutes,
            "timestamp": timestamps,
            "open": [100.0] * minutes,
            "high": [101.0] * minutes,
            "low": [99.0] * minutes,
            "close": [100.5] * minutes,
            "volume": [10.0] * minutes,
            "quote_volume": [1005.0] * minutes,
            "trade_count": [2.0] * minutes,
            "trade_flow_observed": [True] * minutes,
            "taker_buy_volume": [4.0] * minutes,
            "taker_sell_volume": [6.0] * minutes,
            "signed_volume": [1.0] * minutes,
            "best_bid": [100.0] * minutes,
            "best_ask": [100.2] * minutes,
            "quote_observed": [True] * minutes,
            "quote_age_seconds": [1.0] * minutes,
            "bid_quantity": [1.5] * minutes,
            "ask_quantity": [1.0] * minutes,
            "bid_depth": [1.5] * minutes,
            "ask_depth": [1.0] * minutes,
            "mid_price": [100.1] * minutes,
            "eligible": [True] * minutes,
        }
    )


def _patch_mock_group_plan(
    monkeypatch: pytest.MonkeyPatch,
    keys: tuple[tuple[str, str], ...],
) -> None:
    monkeypatch.setattr(
        materialization_validation,
        "crypto_asset_session_views",
        lambda *_args, **_kwargs: _asset_views_context(
            tuple(
                (
                    key,
                    _dataset_session_stub(
                        RESOURCE_POLICY,
                        row_filters=(
                            ("market_id", "==", key[0]),
                            ("asset_id", "==", key[1]),
                        ),
                    ),
                )
                for key in keys
            )
        ),
    )


@contextmanager
def _asset_views_context(views):  # noqa: ANN001
    yield views
