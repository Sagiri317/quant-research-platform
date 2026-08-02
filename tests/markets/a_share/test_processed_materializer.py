from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as materialization_planning,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_factors as materialize_factors,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_labels as materialize_labels,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_panel as materialize_panel,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as materializer,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.storage.datasets import (
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.processed_chunk_parity import (
    assert_forced_tiny_processed_chunks_match_default,
)
from tests.helpers.synthetic_a_share import (
    processed_a_share_root,
    write_synthetic_interim_inputs,
)

SYNTHETIC_SOURCE_ID = SourceId("synthetic_vendor_a")
MARKET_BARS_DATA_ID = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
MARKET_PANEL_DATA_ID = market_panel_data_id()
MARKET_CONTEXT_DATA_ID = market_context_data_id()


def _publication_policy() -> RuntimeResourcePolicy:
    return RuntimeResourcePolicy.from_environment(
        available_memory_bytes=2 * 1024**3,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1024",
            "QRP_PAYLOAD_TARGET_MB": "256",
        },
    )


def test_a_share_output_publication_rejects_unordered_chunks_and_boundaries() -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)

    def frame(timestamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: (
                    pd.to_datetime(timestamps)
                    if column == "timestamp"
                    else ["a_share" if column == "market_id" else "value"]
                    * len(timestamps)
                )
                for column in spec.required_columns
            }
        )

    with pytest.raises(ValueError, match="a_share_output_primary_key_order_regression"):
        list(
            materialization_io.iter_sorted_output_chunks(
                frame(["2014-01-03", "2014-01-02"]),
                spec,
                resource_policy=_publication_policy(),
            )
        )
    with pytest.raises(ValueError, match="a_share_output_primary_key_order_regression"):
        list(
            materialization_io.iter_sorted_output_chunks(
                (frame(["2014-01-03"]), frame(["2014-01-02"])),
                spec,
                resource_policy=_publication_policy(),
            )
        )


def test_a_share_output_publication_rejects_near_headroom_before_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1",
            "QRP_PAYLOAD_TARGET_MB": "1",
        },
    )
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02", "2014-01-03"])
                if column == "timestamp"
                else ["a_share", "a_share"]
            )
            for column in spec.required_columns
        }
    )
    temporary_bytes = len(frame) * (len(spec.primary_key) * 8 + 2 * 8)
    monkeypatch.setattr(
        materialization_io,
        "pandas_frame_payload_bytes",
        lambda _frame: policy.task_memory_budget_bytes - temporary_bytes + 1,
    )

    def reject_sort(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("np.lexsort must not run before capacity admission")

    monkeypatch.setattr(materialization_io.np, "lexsort", reject_sort)

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        list(
            materialization_io.iter_sorted_output_chunks(
                frame,
                spec,
                resource_policy=policy,
            )
        )


def test_sorted_output_chunks_bind_creation_policy_and_activate_before_source_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _publication_policy()
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    observed: list[RuntimeResourcePolicy | None] = []

    class Source:
        def __init__(self) -> None:
            self.remaining = 1

        def __iter__(self):  # noqa: ANN204
            observed.append(current_runtime_resource_policy())
            return self

        def __next__(self) -> pd.DataFrame:
            observed.append(current_runtime_resource_policy())
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            return frame

        def close(self) -> None:
            observed.append(current_runtime_resource_policy())

    source = Source()
    wrapped = materialization_io.iter_sorted_output_chunks(
        source,
        spec,
        resource_policy=policy,
    )

    assert observed == []
    assert list(wrapped) == [frame]
    assert observed and all(item is policy for item in observed)
    assert current_runtime_resource_policy() is None


def test_sorted_output_chunks_reject_different_creation_and_activation_policies() -> (
    None
):
    policy = _publication_policy()
    creation_other_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=2 * 1024**3,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "512",
            "QRP_PAYLOAD_TARGET_MB": "128",
        },
    )
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    source_pulls = 0

    class Source:
        def __iter__(self):  # noqa: ANN204
            nonlocal source_pulls
            source_pulls += 1
            return iter(())

    with runtime_resource_policy_scope(creation_other_policy):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            materialization_io.iter_sorted_output_chunks(
                Source(), spec, resource_policy=policy
            )

    wrapped = materialization_io.iter_sorted_output_chunks(
        Source(), spec, resource_policy=policy
    )
    activation_other_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=2 * 1024**3,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "512",
            "QRP_PAYLOAD_TARGET_MB": "128",
        },
    )
    with runtime_resource_policy_scope(activation_other_policy):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            next(iter(wrapped))

    assert source_pulls == 0


@pytest.mark.parametrize("failure", [KeyboardInterrupt])
def test_sorted_output_chunks_closes_source_inside_bound_policy_on_failure(
    failure: type[BaseException],
) -> None:
    policy = _publication_policy()
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Source:
        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            raise failure("source_failure")

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())

    wrapped = materialization_io.iter_sorted_output_chunks(
        Source(), spec, resource_policy=policy
    )
    with pytest.raises(failure, match="source_failure"):
        next(iter(wrapped))

    assert closed_under == [policy]
    assert current_runtime_resource_policy() is None


@pytest.mark.parametrize(
    "exit_mode",
    ("success", "runtime_error", "keyboard_interrupt", "system_exit", "close"),
)
def test_a_share_output_wrapper_close_cascades_to_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    closed = False

    def source():  # noqa: ANN202
        nonlocal closed
        try:
            yield frame.iloc[:0].copy()
            yield frame
            yield frame.copy()
        finally:
            closed = True

    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
    }
    failure_type = failure_types.get(exit_mode)

    def disk_check(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        if failure_type is not None:
            raise failure_type("disk_exit")

    monkeypatch.setattr(materialization_io, "require_free_disk_headroom", disk_check)
    source_chunks = source()
    sorted_chunks = materialization_io.iter_sorted_output_chunks(
        source_chunks, spec, resource_policy=_publication_policy()
    )
    chunks = materialization_io._iter_disk_guarded_output_chunks(
        sorted_chunks,
        scratch_parent=tmp_path,
    )
    if exit_mode == "success":
        assert len(tuple(chunks)) == 2
    elif exit_mode == "close":
        assert len(next(chunks)) == 1
        chunks.close()
    else:
        assert failure_type is not None
        with pytest.raises(failure_type, match="disk_exit"):
            next(chunks)

    assert closed


_IO_WRAPPER_FAILURE_TYPES = (
    RuntimeError,
    OSError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


@pytest.mark.parametrize(
    "wrapper_name",
    ("sorted", "disk_guarded", "dataset_session"),
)
@pytest.mark.parametrize("failure_type", _IO_WRAPPER_FAILURE_TYPES)
@pytest.mark.parametrize("failure_after_yield", (False, True))
def test_a_share_io_wrapper_preserves_body_primary_over_close_failure(
    wrapper_name: str,
    failure_type: type[BaseException],
    failure_after_yield: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    primary = failure_type("body_failure")
    close_failure = OSError("close_failure")
    cleanup_calls = 0

    class Source:
        def __init__(self) -> None:
            self.yielded = False

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            if failure_after_yield and not self.yielded:
                self.yielded = True
                return frame
            raise primary

        def close(self) -> None:
            nonlocal cleanup_calls
            if wrapper_name == "dataset_session":
                return
            cleanup_calls += 1
            raise close_failure

    source = Source()
    if wrapper_name == "sorted":
        wrapped = materialization_io.iter_sorted_output_chunks(
            source, spec, resource_policy=_publication_policy()
        )
    elif wrapper_name == "disk_guarded":
        monkeypatch.setattr(
            materialization_io,
            "require_free_disk_headroom",
            lambda *_args, **_kwargs: None,
        )
        wrapped = materialization_io._iter_disk_guarded_output_chunks(
            source,
            scratch_parent=tmp_path,
        )
    else:
        policy = RuntimeResourcePolicy.from_environment(
            available_memory_bytes=16 * 1024 * 1024,
            environ={},
        )

        class Session:
            resource_policy = policy

            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> None:
                if exc_type is None:
                    self.close()
                    return
                try:
                    self.close()
                except BaseException:
                    pass

            def close(self) -> None:
                nonlocal cleanup_calls
                cleanup_calls += 1
                raise close_failure

        session = Session()
        monkeypatch.setattr(
            materialization_io,
            "open_dataset_session",
            lambda *_args, **_kwargs: session,
        )
        monkeypatch.setattr(
            materialization_io,
            "iter_dataset_chunks",
            lambda *_args, **_kwargs: source,
        )
        wrapped = materialization_io.iter_dataset_chunks_with_policy(
            tmp_path,
            resource_policy=policy,
        )

    observed: BaseException | None = None
    try:
        if failure_after_yield:
            next(wrapped)
        next(wrapped)
    except BaseException as raised:
        observed = raised

    assert observed is primary
    assert cleanup_calls == 1


@pytest.mark.parametrize(
    "wrapper_name",
    ("sorted", "disk_guarded", "dataset_session"),
)
@pytest.mark.parametrize(
    ("item_count", "close_fails", "early_close"),
    (
        (0, False, False),
        (1, False, False),
        (0, True, False),
        (1, True, False),
        (1, True, True),
    ),
)
def test_a_share_io_wrapper_cleanup_lifecycle(
    wrapper_name: str,
    item_count: int,
    close_fails: bool,
    early_close: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    frame = pd.DataFrame(
        {
            column: (
                pd.to_datetime(["2014-01-02"])
                if column == "timestamp"
                else ["a_share" if column == "market_id" else "value"]
            )
            for column in spec.required_columns
        }
    )
    close_failure = OSError("close_failure")
    cleanup_calls = 0

    class Source:
        def __init__(self) -> None:
            self.remaining = item_count

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            if self.remaining == 0:
                raise StopIteration
            self.remaining -= 1
            return frame

        def close(self) -> None:
            nonlocal cleanup_calls
            if wrapper_name == "dataset_session":
                return
            cleanup_calls += 1
            if close_fails:
                raise close_failure

    source = Source()
    if wrapper_name == "sorted":
        wrapped = materialization_io.iter_sorted_output_chunks(
            source, spec, resource_policy=_publication_policy()
        )
    elif wrapper_name == "disk_guarded":
        monkeypatch.setattr(
            materialization_io,
            "require_free_disk_headroom",
            lambda *_args, **_kwargs: None,
        )
        wrapped = materialization_io._iter_disk_guarded_output_chunks(
            source,
            scratch_parent=tmp_path,
        )
    else:
        policy = RuntimeResourcePolicy.from_environment(
            available_memory_bytes=16 * 1024 * 1024,
            environ={},
        )

        class Session:
            resource_policy = policy

            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: object,
            ) -> None:
                if exc_type is None:
                    self.close()
                    return
                try:
                    self.close()
                except BaseException:
                    pass

            def close(self) -> None:
                nonlocal cleanup_calls
                cleanup_calls += 1
                if close_fails:
                    raise close_failure

        session = Session()
        monkeypatch.setattr(
            materialization_io,
            "open_dataset_session",
            lambda *_args, **_kwargs: session,
        )
        monkeypatch.setattr(
            materialization_io,
            "iter_dataset_chunks",
            lambda *_args, **_kwargs: source,
        )
        wrapped = materialization_io.iter_dataset_chunks_with_policy(
            tmp_path,
            resource_policy=policy,
        )

    observed: BaseException | None = None
    outputs: list[pd.DataFrame] = []
    try:
        if early_close:
            outputs.append(next(wrapped))
            wrapped.close()
        else:
            outputs.extend(wrapped)
    except BaseException as raised:
        observed = raised

    expected = close_failure if close_fails and not early_close else None
    assert observed is expected
    assert len(outputs) == item_count
    assert cleanup_calls == 1


def test_write_and_inspect_activates_explicit_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={},
    )
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)
    inspection = object()
    observed: list[tuple[RuntimeResourcePolicy, int]] = []

    def capture_write(*_args: object, **kwargs: object) -> None:
        observed.append(
            (active_runtime_resource_policy(), kwargs["retained_payload_bytes"])
        )

    def capture_inspection(*_args: object, **kwargs: object) -> object:
        observed.append(
            (active_runtime_resource_policy(), kwargs["retained_payload_bytes"])
        )
        return inspection

    monkeypatch.setattr(materialization_io, "_write_dataset", capture_write)
    monkeypatch.setattr(materialization_io, "inspect_dataset", capture_inspection)
    monkeypatch.setattr(
        materialization_io,
        "dataset_inspection_retained_memory_bytes",
        lambda _inspection: 123,
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_sorted_output_chunks",
        lambda *_args, **_kwargs: (),
    )
    prior = object()
    written = {"prior": prior}

    result = materialization_io.write_and_inspect(
        spec,
        pd.DataFrame(),
        layout=StorageLayout(project_root=tmp_path),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        written=written,
        resource_policy=policy,
    )

    assert observed == [(policy, 123), (policy, 123)]
    assert result is inspection
    assert written == {"prior": prior, spec.data_id: inspection}


FORWARD_RETURN_1D_COLUMN = forward_return_label_column(1)
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)


def test_materialize_processed_data_writes_requested_panel_and_dependencies(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    assert set(written) == {MARKET_PANEL_DATA_ID}
    root = tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    panel = read_dataset(root)
    assert len(panel) == 2
    assert panel["asset_id"].unique().tolist() == ["000001"]
    assert "source_asset_id" not in panel.columns


def test_panel_materializer_reuses_market_bar_inspection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    market_bars_root = layout.interim_data_root(
        SYNTHETIC_SOURCE_ID,
        MARKET_BARS_DATA_ID,
    )
    calls = 0
    original = materialize_panel.inspect_dataset

    def spy_inspect_dataset(*args, **kwargs):
        nonlocal calls
        if Path(args[0]) == market_bars_root and "time_window" not in kwargs:
            calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(materialize_panel, "inspect_dataset", spy_inspect_dataset)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    assert calls == 1


def test_panel_materializer_does_not_preload_full_market_bars(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        end_date="2014-01-10",
    )
    market_bars_root = layout.interim_data_root(
        SYNTHETIC_SOURCE_ID,
        MARKET_BARS_DATA_ID,
    )
    original = materialize_panel.materialization_io.read_materialized_dataset
    full_market_bars_reads = 0
    windowed_market_bars_reads = 0

    def spy_read_dataset(root, **kwargs):
        nonlocal full_market_bars_reads, windowed_market_bars_reads
        if Path(root) == market_bars_root:
            if kwargs.get("time_window") is None:
                full_market_bars_reads += 1
            else:
                windowed_market_bars_reads += 1
        return original(root, **kwargs)

    monkeypatch.setattr(
        materialize_panel.materialization_io,
        "read_materialized_dataset",
        spy_read_dataset,
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        data_ids=(MARKET_PANEL_DATA_ID,),
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=16 * 1024 * 1024,
            environ={
                "QRP_TASK_MEMORY_BUDGET_MB": "5",
                "QRP_PAYLOAD_TARGET_MB": "1",
            },
        ),
    )

    assert full_market_bars_reads == 0
    assert windowed_market_bars_reads > 0


def test_materialize_processed_data_rejects_existing_requested_root(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    kwargs = {
        "layout": layout,
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
        "data_ids": (MARKET_PANEL_DATA_ID,),
    }

    materializer.materialize_processed_data(**kwargs)

    with pytest.raises(ValueError, match="dataset_exists"):
        materializer.materialize_processed_data(**kwargs)


def test_materialize_processed_data_requires_exactly_one_selector(tmp_path) -> None:
    common = {
        "layout": StorageLayout(project_root=tmp_path),
        "source_id": SYNTHETIC_SOURCE_ID,
        "time_window": TimeWindow("2014-01-02", "2014-01-03"),
    }

    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(**common)
    with pytest.raises(ValueError, match="data_ids_xor_kinds_required"):
        materializer.materialize_processed_data(
            **common,
            data_ids=(MARKET_PANEL_DATA_ID,),
            kinds=frozenset({ProcessedDataKind.PANEL}),
        )


@pytest.mark.parametrize("frequency", [DataFrequency.MINUTE, DataFrequency.TICK])
def test_materialize_processed_data_rejects_non_daily_active_profile_before_io(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    frequency: DataFrequency,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    catalog = ProcessedDataCatalog()
    catalog.register(
        replace(
            processed_catalog().get(MARKET_PANEL_DATA_ID),
            frequency=frequency,
        )
    )

    def fail_if_sessions_are_read(
        *args: object,
        **kwargs: object,
    ) -> pd.DatetimeIndex:
        raise AssertionError("frequency validation should run before source reads")

    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)
    monkeypatch.setattr(
        materialization_planning,
        "available_trading_sessions",
        fail_if_sessions_are_read,
    )

    with pytest.raises(
        ValueError,
        match=(
            "unsupported_processed_frequency: "
            f"{MARKET_PANEL_DATA_ID}: {frequency.value}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=(MARKET_PANEL_DATA_ID,),
        )

    assert not (
        tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    ).exists()


def test_materialize_processed_data_by_kind_writes_dependency_and_target(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        kinds=frozenset({ProcessedDataKind.FACTOR}),
    )

    assert set(written) == {MARKET_PANEL_DATA_ID, "price_volume"}
    assert (
        tmp_path / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
    ).exists()
    assert (
        tmp_path / "data" / "processed" / "a_share" / "factor" / "price_volume"
    ).exists()


@pytest.mark.parametrize(
    "data_ids",
    [
        ("forward_returns",),
        (MARKET_PANEL_DATA_ID, "forward_returns"),
    ],
)
def test_label_materializer_expands_panel_dependency_for_future_halo(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    data_ids: tuple[str, ...],
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2014-01-10",
    )
    monkeypatch.setattr(
        materializer,
        "processed_catalog",
        lambda: _catalog_with_one_step_forward_return_only(),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=data_ids,
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    labels = read_dataset(processed_a_share_root(tmp_path, "label", "forward_returns"))
    assert pd.to_datetime(panel["timestamp"]).dt.date.max().isoformat() == "2014-01-07"
    jan3_label = labels.loc[
        pd.to_datetime(labels["timestamp"]).dt.date.astype(str).eq("2014-01-03"),
        FORWARD_RETURN_1D_COLUMN,
    ].iloc[0]
    assert pd.notna(jan3_label)


@pytest.mark.parametrize(
    "data_ids",
    [
        ("price_volume",),
        (MARKET_PANEL_DATA_ID, "price_volume"),
    ],
)
def test_factor_materializer_expands_panel_dependency_for_history_halo(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    data_ids: tuple[str, ...],
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-01",
        end_date="2014-01-31",
    )
    monkeypatch.setattr(
        materializer,
        "processed_catalog",
        lambda: _catalog_with_momentum_20d_only(),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-31", "2014-01-31"),
        data_ids=data_ids,
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    factors = read_dataset(processed_a_share_root(tmp_path, "factor", "price_volume"))
    assert pd.to_datetime(panel["timestamp"]).dt.date.min().isoformat() == "2014-01-03"
    assert pd.notna(factors[MOMENTUM_20D_COLUMN].iloc[0])


def test_materializer_empty_target_window_still_fails_closed(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    with pytest.raises(ValueError, match="no_trading_sessions_in_time_window"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2009-01-01", "2009-01-31"),
            data_ids=(MARKET_PANEL_DATA_ID,),
        )


def test_factor_materializer_rejects_unexpected_declared_processed_input(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog = _catalog_with_processed_inputs(
        "price_volume",
        (MARKET_PANEL_DATA_ID, MARKET_CONTEXT_DATA_ID),
    )
    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match=(
            "unexpected_processed_input_for_factor: "
            f"price_volume: {MARKET_CONTEXT_DATA_ID}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


def test_factor_materializer_requires_family_contract_input_declared_by_spec(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog = _catalog_with_processed_inputs("price_volume", ())
    monkeypatch.setattr(materializer, "processed_catalog", lambda: catalog)

    with pytest.raises(
        ValueError,
        match=(
            f"missing_processed_input_for_factor: price_volume: {MARKET_PANEL_DATA_ID}"
        ),
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=("price_volume",),
        )


@pytest.mark.parametrize(
    (
        "definition_owner",
        "catalog_name",
        "families",
        "data_ids",
        "error_match",
    ),
    [
        (
            materialize_factors.factor_definitions,
            "factor_catalog",
            ("alternative_price_family",),
            ("price_volume",),
            "unknown_factor_family: alternative_price_family",
        ),
        (
            materialize_factors.factor_definitions,
            "factor_catalog",
            ("price_volume", "other_factor"),
            ("price_volume",),
            "mixed_factor_families: price_volume, other_factor",
        ),
        (
            materialize_labels.label_definitions,
            "label_catalog",
            ("alternative_label_family",),
            ("forward_returns",),
            "unknown_label_family: alternative_label_family",
        ),
        (
            materialize_labels.label_definitions,
            "label_catalog",
            ("forward_return", "other_label"),
            ("forward_returns",),
            "mixed_label_families: forward_return, other_label",
        ),
    ],
)
def test_materializer_family_contract_failures_are_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    definition_owner,
    catalog_name: str,
    families: tuple[str, ...],
    data_ids: tuple[str, ...],
    error_match: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    catalog_builder = (
        _factor_catalog_with_families
        if catalog_name == "factor_catalog"
        else _label_catalog_with_families
    )
    monkeypatch.setattr(
        definition_owner,
        catalog_name,
        lambda: catalog_builder(families),
    )

    with pytest.raises(ValueError, match=error_match):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-03"),
            data_ids=data_ids,
        )


def test_materializer_reuses_same_run_dependency_for_empty_window_tail(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        kinds=frozenset({ProcessedDataKind.ANNOTATION}),
    )

    assert {
        MARKET_PANEL_DATA_ID,
        MARKET_CONTEXT_DATA_ID,
        "default",
        "as_of",
        "execution",
        "evaluation",
    }.issubset(written)


def test_materialize_processed_data_existing_dependency_accepts_non_trading_end(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout, SYNTHETIC_SOURCE_ID, market_bars_data_id=MARKET_BARS_DATA_ID
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    written = materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-04"),
        data_ids=("execution",),
    )

    assert "execution" in written


def test_materialize_processed_data_writes_only_core_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2017-04-28",
        end_date="2017-05-04",
        closed_dates=("2017-05-01",),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2017-04-28", "2017-05-03"),
        data_ids=("price_volume",),
    )

    panel = read_dataset(
        processed_a_share_root(tmp_path, "panel", MARKET_PANEL_DATA_ID)
    )
    factors = read_dataset(processed_a_share_root(tmp_path, "factor", "price_volume"))

    assert panel["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]
    assert factors["timestamp"].dt.date.astype(str).tolist() == [
        "2017-04-28",
        "2017-05-02",
        "2017-05-03",
    ]


def test_materialize_processed_data_existing_dependency_must_cover_trading_sessions(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        end_date="2014-01-06",
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    with pytest.raises(ValueError, match="insufficient_dataset_coverage"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2014-01-06"),
            data_ids=("execution",),
        )


def test_existing_dependency_coverage_rejects_missing_middle_sessions(
    tmp_path,
) -> None:
    source_layout = StorageLayout(project_root=tmp_path / "source")
    write_synthetic_interim_inputs(
        source_layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    materializer.materialize_processed_data(
        layout=source_layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-02", "2016-01-04"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )
    source_panel_root = processed_a_share_root(
        tmp_path / "source",
        "panel",
        MARKET_PANEL_DATA_ID,
    )
    panel = read_dataset(source_panel_root)
    panel_without_2015 = panel.loc[pd.to_datetime(panel["timestamp"]).dt.year.ne(2015)]

    layout = StorageLayout(project_root=tmp_path / "coverage")
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-02",
        end_date="2016-01-04",
    )
    panel_root = processed_a_share_root(
        tmp_path / "coverage", "panel", MARKET_PANEL_DATA_ID
    )
    write_dataset(
        panel_root,
        panel_without_2015,
        spec=processed_catalog().get(MARKET_PANEL_DATA_ID),
    )

    with pytest.raises(
        ValueError,
        match=r"insufficient_dataset_coverage: missing_sessions=",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=SYNTHETIC_SOURCE_ID,
            time_window=TimeWindow("2014-01-02", "2016-01-04"),
            data_ids=("execution",),
        )


def test_existing_dependency_coverage_accepts_non_trading_end_date(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2016-12-29",
        end_date="2017-01-03",
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-30"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2016-12-29", "2016-12-31"),
        data_ids=("execution",),
    )

    annotation_root = processed_a_share_root(tmp_path, "annotation", "execution")
    annotation = read_dataset(annotation_root)
    assert annotation["timestamp"].dt.date.astype(str).tolist() == [
        "2016-12-29",
        "2016-12-30",
    ]


def test_annotation_materializer_uses_dependency_lookahead_session(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        SYNTHETIC_SOURCE_ID,
        market_bars_data_id=MARKET_BARS_DATA_ID,
        start_date="2014-01-30",
        end_date="2014-02-07",
        closed_dates=(
            "2014-01-31",
            "2014-02-03",
            "2014-02-04",
            "2014-02-05",
            "2014-02-06",
        ),
    )
    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-02-07"),
        data_ids=(MARKET_PANEL_DATA_ID,),
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=SYNTHETIC_SOURCE_ID,
        time_window=TimeWindow("2014-01-30", "2014-01-30"),
        data_ids=("execution",),
    )

    execution = read_dataset(
        processed_a_share_root(tmp_path, "annotation", "execution")
    )
    assert execution["timestamp"].dt.date.astype(str).tolist() == ["2014-01-30"]
    assert execution["next_bar_is_trading"].tolist() == [True]
    assert execution["can_enter"].tolist() == [True]
    assert execution["can_exit"].tolist() == [True]


def test_materialize_processed_data_forced_tiny_chunks_equal_default_output(
    tmp_path,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=MARKET_PANEL_DATA_ID,
        kind="panel",
        end_date="2014-02-28",
        time_window=TimeWindow("2014-01-02", "2014-02-28"),
        asset_count=256,
    )


@pytest.mark.parametrize(
    ("data_id", "kind", "end_date", "time_window", "asset_count"),
    [
        (
            "execution",
            "annotation",
            "2014-02-28",
            TimeWindow("2014-01-02", "2014-02-28"),
            256,
        ),
        pytest.param(
            "price_volume",
            "factor",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            64,
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "forward_returns",
            "label",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            64,
            marks=pytest.mark.slow,
        ),
    ],
)
def test_materialize_processed_data_downstream_forced_tiny_chunks_equal_default_output(
    tmp_path,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
    asset_count: int,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date=end_date,
        time_window=time_window,
        asset_count=asset_count,
    )


def _catalog_with_processed_inputs(
    target_data_id: str,
    processed_input_data_ids: tuple[str, ...],
) -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == target_data_id:
            spec = replace(spec, processed_input_data_ids=processed_input_data_ids)
        catalog.register(spec)
    return catalog


def _catalog_with_one_step_forward_return_only() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == "forward_returns":
            spec = replace(
                spec,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    FORWARD_RETURN_1D_COLUMN: ColumnRole.LABEL,
                },
            )
        catalog.register(spec)
    return catalog


def _catalog_with_momentum_20d_only() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in processed_catalog().list():
        if spec.data_id == "price_volume":
            spec = replace(
                spec,
                column_roles={
                    "market_id": ColumnRole.KEY,
                    "asset_id": ColumnRole.KEY,
                    "timestamp": ColumnRole.TIMESTAMP,
                    MOMENTUM_20D_COLUMN: ColumnRole.FACTOR,
                },
            )
        catalog.register(spec)
    return catalog


def _factor_catalog_with_families(families: tuple[str, ...]) -> FactorCatalog:
    catalog = FactorCatalog()
    definitions = materialize_factors.factor_definitions.factor_definitions()
    for index, definition in enumerate(definitions):
        catalog.register(
            replace(definition, family=families[min(index, len(families) - 1)])
        )
    return catalog


def _label_catalog_with_families(families: tuple[str, ...]) -> LabelCatalog:
    catalog = LabelCatalog()
    definitions = materialize_labels.label_definitions.label_definitions()
    for index, definition in enumerate(definitions):
        catalog.register(
            replace(definition, family=families[min(index, len(families) - 1)])
        )
    return catalog
