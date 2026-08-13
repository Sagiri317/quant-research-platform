from __future__ import annotations

import gc
import sys
import tempfile
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.storage._dataset_writing as dataset_writing
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    AS_OF_MASK_ID,
    DEFAULT_UNIVERSE_ID,
    EVALUATION_MASK_ID,
    EXECUTION_MASK_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_dependencies as materialization_dependencies,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as materialization_planning,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_validation as materialization_validation,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_annotations as materialize_annotations,
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
    source_ids as source_ids,
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
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import capture_test_policy

MARKET_BARS_DATA_ID = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
MARKET_PANEL_DATA_ID = market_panel_data_id()
MARKET_CONTEXT_DATA_ID = market_context_data_id()
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)
FORWARD_RETURN_20D_COLUMN = forward_return_label_column(20)


def _scoped_active_materialization_batches(
    *args: object,
    **kwargs: object,
) -> Iterator[dataset_writing.DatasetWriteChunk]:
    policy = kwargs["resource_policy"]
    assert isinstance(policy, RuntimeResourcePolicy)
    policy_scope = runtime_resource_policy_scope(policy)
    policy_scope.__enter__()
    try:
        batches = materialization_planning.iter_active_materialization_batches(
            *args,
            **kwargs,
        )
    except BaseException:
        policy_scope.__exit__(*sys.exc_info())
        raise

    def consume() -> Iterator[dataset_writing.DatasetWriteChunk]:
        try:
            yield from batches
        finally:
            primary = sys.exception()
            try:
                batches.close()
            except BaseException:
                if primary is None:
                    raise
            finally:
                try:
                    policy_scope.__exit__(None, None, None)
                except BaseException:
                    if primary is None:
                        raise

    return consume()


def test_a_share_processed_specs_declare_exact_direct_dependencies() -> None:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_catalog,
    )

    catalog = processed_catalog.processed_catalog()

    dependencies = {
        spec.data_id: (spec.source_input_data_ids, spec.processed_input_data_ids)
        for spec in catalog.list()
    }

    assert dependencies == {
        MARKET_PANEL_DATA_ID: (
            (MARKET_BARS_DATA_ID, "stock_master", "market_cap", "trading_calendar"),
            (),
        ),
        MARKET_CONTEXT_DATA_ID: ((), (MARKET_PANEL_DATA_ID,)),
        DEFAULT_UNIVERSE_ID: ((), (MARKET_PANEL_DATA_ID,)),
        AS_OF_MASK_ID: ((), (MARKET_PANEL_DATA_ID,)),
        EXECUTION_MASK_ID: ((), (MARKET_PANEL_DATA_ID,)),
        EVALUATION_MASK_ID: ((), (MARKET_PANEL_DATA_ID,)),
        PRICE_VOLUME_PROCESSED_DATA_ID: ((), (MARKET_PANEL_DATA_ID,)),
        FORWARD_RETURNS_PROCESSED_DATA_ID: ((), (MARKET_PANEL_DATA_ID,)),
    }


def test_panel_materializer_skips_market_bars_source_preload(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_vendor_a")
    spec = _processed_spec(
        MARKET_PANEL_DATA_ID,
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=(MARKET_BARS_DATA_ID, "trading_calendar"),
    )
    _write_interim_frame(layout, source_id, MARKET_BARS_DATA_ID)
    _write_interim_frame(layout, source_id, "trading_calendar")

    inputs = materialization_io.read_source_inputs_for_spec(
        spec,
        layout=layout,
        source_id=source_id,
        market_bars_data_id=source_ids.market_bars_source_data_id(spec),
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
    )

    assert set(inputs) == {"trading_calendar"}
    assert all(isinstance(frame, pd.DataFrame) for frame in inputs.values())


def test_source_input_physical_reads_use_owner_resource_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_vendor_a")
    spec = _processed_spec(
        MARKET_PANEL_DATA_ID,
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=(MARKET_BARS_DATA_ID, "trading_calendar"),
    )
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024,
    )
    observed_targets: list[int] = []

    monkeypatch.setattr(
        materialization_io,
        "open_dataset_session",
        lambda root, **_kwargs: nullcontext(root),
        raising=False,
    )

    def fake_materialize_dataset_session(  # noqa: ANN001
        context,
        *,
        target_chunk_bytes,
        materialized_budget_bytes,
    ):
        assert materialized_budget_bytes == policy.task_memory_budget_bytes
        observed_targets.append(target_chunk_bytes)
        return _key_frame()

    monkeypatch.setattr(
        materialization_io,
        "materialize_dataset_session",
        fake_materialize_dataset_session,
        raising=False,
    )

    inputs = materialization_io.read_source_inputs_for_spec(
        spec,
        layout=layout,
        source_id=source_id,
        market_bars_data_id=source_ids.market_bars_source_data_id(spec),
        resource_policy=policy,
    )

    assert set(inputs) == {"trading_calendar"}
    assert observed_targets == [policy.remaining_memory_bytes]


def test_materialized_read_preserves_target_and_reserves_retained_owner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=10 * 1024**2,
    )
    observed: list[tuple[int, int]] = []
    opened_retained: list[int] = []

    def open_session(root, **kwargs):  # noqa: ANN001, ANN202
        opened_retained.append(kwargs["retained_payload_bytes"])
        return nullcontext(root)

    monkeypatch.setattr(
        materialization_io,
        "open_dataset_session",
        open_session,
    )

    def fake_materialize_dataset_session(  # noqa: ANN001
        _context,
        *,
        target_chunk_bytes,
        materialized_budget_bytes,
    ):
        observed.append((target_chunk_bytes, materialized_budget_bytes))
        return _key_frame()

    monkeypatch.setattr(
        materialization_io,
        "materialize_dataset_session",
        fake_materialize_dataset_session,
    )

    retained_payload_bytes = 2 * 1024**2
    materialization_io.read_materialized_dataset(
        Path("dataset"),
        resource_policy=policy,
        retained_payload_bytes=retained_payload_bytes,
    )

    available = policy.task_memory_budget_bytes - retained_payload_bytes
    assert observed == [(policy.remaining_memory_bytes, available)]
    assert opened_retained == [retained_payload_bytes]


@pytest.mark.parametrize("operation", ("materialized", "chunked"))
def test_dataset_read_scope_remains_active_through_session_exit(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    competing = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class Session:
        resource_policy = policy

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            assert current_runtime_resource_policy() is policy
            with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
                with runtime_resource_policy_scope(competing):
                    pass

    @contextmanager
    def open_session(*_args, **_kwargs):  # noqa: ANN202
        with runtime_resource_policy_scope(policy):
            with Session() as session:
                yield session

    monkeypatch.setattr(materialization_io, "open_dataset_session", open_session)
    monkeypatch.setattr(
        materialization_io,
        "materialize_dataset_session",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter(()),
    )

    if operation == "materialized":
        materialization_io.read_materialized_dataset(
            Path("dataset"),
            resource_policy=policy,
        )
    else:
        list(
            materialization_io.iter_dataset_chunks_with_policy(
                Path("dataset"),
                resource_policy=policy,
            )
        )


@pytest.mark.parametrize("operation", ("materialized", "chunked"))
def test_dataset_read_does_not_construct_session_when_policy_entry_fails(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    primary = RuntimeError("runtime_resource_policy_concurrent_root")

    @contextmanager
    def failing_scope(_policy):  # noqa: ANN001, ANN202
        raise primary
        yield

    @contextmanager
    def open_session(*_args, **_kwargs):  # noqa: ANN202
        with failing_scope(policy):
            pytest.fail("session constructed after policy entry failure")
            yield

    monkeypatch.setattr(materialization_io, "open_dataset_session", open_session)

    with pytest.raises(RuntimeError) as raised:
        if operation == "materialized":
            materialization_io.read_materialized_dataset(
                Path("dataset"),
                resource_policy=policy,
            )
        else:
            list(
                materialization_io.iter_dataset_chunks_with_policy(
                    Path("dataset"),
                    resource_policy=policy,
                )
            )

    assert raised.value is primary


def test_materialized_read_rejects_exhausted_retained_state_before_decoder() -> None:
    policy = capture_test_policy(
        available_memory_bytes=10 * 1024**2,
    )

    with pytest.raises(
        ValueError,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        materialization_io.read_materialized_dataset(
            Path("dataset"),
            resource_policy=policy,
            retained_payload_bytes=policy.task_memory_budget_bytes,
        )


def test_materialized_read_rejects_negative_retained_state() -> None:
    with pytest.raises(ValueError, match="invalid_a_share_retained_payload_bytes"):
        materialization_io.read_materialized_dataset(
            Path("dataset"),
            resource_policy=capture_test_policy(
                available_memory_bytes=1024**3,
            ),
            retained_payload_bytes=-1,
        )


def test_chunk_reader_rejects_exhausted_retained_state_before_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    opened = False

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("dataset_session_opened_after_retained_exhaustion")

    monkeypatch.setattr(materialization_io, "open_dataset_session", forbidden_open)
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        Path("dataset"),
        resource_policy=policy,
        retained_payload_bytes=policy.task_memory_budget_bytes,
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        next(chunks)

    assert opened is False


def test_chunk_reader_passes_caller_retained_state_to_session_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
    )
    observed: list[int] = []

    @contextmanager
    def open_session(*_args, **kwargs):  # noqa: ANN001, ANN202
        observed.append(kwargs["retained_payload_bytes"])
        yield SimpleNamespace(resource_policy=policy)

    monkeypatch.setattr(materialization_io, "open_dataset_session", open_session)
    monkeypatch.setattr(
        materialization_io, "iter_dataset_chunks", lambda *_args, **_kwargs: iter(())
    )

    retained = 321
    list(
        materialization_io.iter_dataset_chunks_with_policy(
            Path("dataset"),
            resource_policy=policy,
            retained_payload_bytes=retained,
        )
    )

    assert observed == [retained]


@pytest.mark.parametrize("operation", ("chunks", "assets"))
def test_lazy_a_share_factories_bind_active_policy_at_creation(
    operation: str,
) -> None:
    active = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    explicit = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(active):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            if operation == "chunks":
                materialization_io.iter_dataset_chunks_with_policy(
                    Path("dataset"),
                    resource_policy=explicit,
                )
            else:
                materialization_planning.iter_active_materialization_batches(
                    Path("dataset"),
                    time_window=TimeWindow("2024-01-01", "2024-01-01"),
                    sessions=(pd.Timestamp("2024-01-01"),),
                    resource_policy=explicit,
                    materialize=lambda *_args: pd.DataFrame(),
                )


def test_session_batch_window_uses_indexed_bounds_without_full_session_scan() -> None:
    class IndexedSessions(tuple[pd.Timestamp, ...]):
        def __iter__(self):
            raise AssertionError("session_bounds_must_not_scan_all_sessions")

        def index(self, *_args: object, **_kwargs: object) -> int:
            raise AssertionError("session_bounds_must_not_use_linear_index_lookup")

    sessions = IndexedSessions(pd.date_range("2014-01-01", periods=10_000))
    current = (sessions[5_000], sessions[5_002])

    result = materialization_planning.session_batch_window(
        sessions,
        current,
        warmup_sessions=2,
        lookahead_sessions=3,
    )

    assert result == TimeWindow(sessions[4_998].date(), sessions[5_005].date())


def test_dependency_window_uses_sorted_indexed_sessions_without_selection_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IndexedSessions(tuple[pd.Timestamp, ...]):
        def __iter__(self):
            raise AssertionError("dependency_bounds_must_not_scan_all_sessions")

        def index(self, *_args: object, **_kwargs: object) -> int:
            raise AssertionError("dependency_bounds_must_not_use_linear_index_lookup")

    sessions = IndexedSessions(pd.date_range("2014-01-01", periods=10_000))
    target = _processed_spec(
        "factor",
        kind=ProcessedDataKind.FACTOR,
        processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
    )
    monkeypatch.setattr(
        materialization_dependencies,
        "_processed_dependency_session_offsets",
        lambda _spec: (2, 3),
    )

    result = materialization_dependencies.planned_processed_dependency_windows(
        (target,),
        layout=SimpleNamespace(),  # type: ignore[arg-type]
        source_id="source",  # type: ignore[arg-type]
        time_window=TimeWindow(sessions[5_000].date(), sessions[5_002].date()),
        sessions=sessions,
    )

    assert result == {
        MARKET_PANEL_DATA_ID: TimeWindow(sessions[4_998].date(), sessions[5_005].date())
    }


@pytest.mark.parametrize("target_chunk_bytes", (0, -1, False))
def test_invalid_chunk_target_rejection_happens_before_dataset_session_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    target_chunk_bytes: object,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    acquisitions = 0

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        nonlocal acquisitions
        acquisitions += 1
        raise AssertionError("dataset_session_acquired_before_target_validation")

    monkeypatch.setattr(materialization_io, "open_dataset_session", forbidden_open)
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        Path("dataset"),
        resource_policy=policy,
        target_chunk_bytes=target_chunk_bytes,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="invalid_target_chunk_bytes"):
        next(chunks)

    assert acquisitions == 0


def test_a_share_indivisible_output_rejects_hard_budget() -> None:
    policy = capture_test_policy(
        available_memory_bytes=100,
    )
    output = pd.DataFrame({"value": ["x" * 256]})

    with pytest.raises(
        ValueError,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        materialization_planning.require_known_materialization_payload_bound(
            frames=(),
            output=output,
            resource_policy=policy,
        )


def test_panel_materializer_rejects_unapproved_market_bars_source_variant() -> None:
    spec = _processed_spec(
        MARKET_PANEL_DATA_ID,
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=("market_bars_unsupported", "trading_calendar"),
    )

    with pytest.raises(ValueError, match="panel_market_bars_source_input_not_unique"):
        source_ids.market_bars_source_data_id(spec)


def test_materializer_reads_processed_inputs_from_spec_processed_input_data_ids(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = _processed_spec(MARKET_PANEL_DATA_ID, kind=ProcessedDataKind.PANEL)
    target = _processed_spec(
        "price_volume",
        kind=ProcessedDataKind.FACTOR,
        processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
    )
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id,
            panel_spec.kind,
            panel_spec.data_id,
        ),
        _key_frame(),
        spec=panel_spec,
    )

    inputs = materialization_io.read_processed_inputs_for_spec(
        target,
        layout=layout,
        processed_catalog=_catalog(panel_spec, target),
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
    )

    assert set(inputs) == {MARKET_PANEL_DATA_ID}
    assert inputs[MARKET_PANEL_DATA_ID]["asset_id"].tolist() == ["asset_a"]


def test_materializer_validates_unknown_processed_dependency() -> None:
    target = _processed_spec(
        "price_volume",
        kind=ProcessedDataKind.FACTOR,
        processed_input_data_ids=("missing_panel",),
    )

    with pytest.raises(ValueError, match="unknown_processed_dependency: missing_panel"):
        materialization_validation.validate_processed_dependency_graph(
            _catalog(target),
            (target,),
        )


def test_materializer_validates_processed_dependency_cycle() -> None:
    left = _processed_spec(
        "left",
        kind=ProcessedDataKind.ANNOTATION,
        processed_input_data_ids=("right",),
    )
    right = _processed_spec(
        "right",
        kind=ProcessedDataKind.ANNOTATION,
        processed_input_data_ids=("left",),
    )

    with pytest.raises(ValueError, match="processed_dependency_cycle"):
        materialization_validation.validate_processed_dependency_graph(
            _catalog(left, right),
            (left,),
        )


def test_materializer_selects_factor_definitions_from_required_columns_only() -> None:
    selected = _factor_definition("mom_5d", lookback_window=5)
    unselected = _factor_definition("mom_252d", lookback_window=252)
    catalog = FactorCatalog()
    catalog.register(selected)
    catalog.register(unselected)
    spec = _processed_spec(
        "price_volume",
        kind=ProcessedDataKind.FACTOR,
        columns=(*platform_entity_time_key(), "mom_5d"),
    )

    definitions = materialize_factors.select_factor_definitions_for_spec(spec, catalog)

    assert definitions == (selected,)
    assert max(definition.lookback_window or 0 for definition in definitions) == 5


def test_materializer_selects_label_definitions_from_required_columns_only() -> None:
    selected = _label_definition("forward_return_2d", horizon=2, lookahead=3)
    unselected = _label_definition("forward_return_120d", horizon=120, lookahead=121)
    catalog = LabelCatalog()
    catalog.register(selected)
    catalog.register(unselected)
    spec = _processed_spec(
        "forward_returns",
        kind=ProcessedDataKind.LABEL,
        columns=(*platform_entity_time_key(), "forward_return_2d"),
    )

    definitions = materialize_labels.select_label_definitions_for_spec(spec, catalog)

    assert definitions == (selected,)
    assert max(definition.lookahead for definition in definitions) == 3


def test_sparse_valid_bar_windows_clamp_across_long_suspensions() -> None:
    factor = _factor_definition("mom_2d", lookback_window=2)
    label = _label_definition("forward_return_2d", horizon=2, lookahead=2)
    coverage = {
        "suspended": tuple(
            pd.Timestamp(value) for value in ("2024-01-02", "2024-01-03", "2024-06-03")
        ),
        "short_history": (pd.Timestamp("2024-06-03"),),
    }

    factor_window = factor_calculations.factor_read_window_for_panel(
        coverage,
        definitions=(factor,),
        read_window=TimeWindow("2024-06-03", "2024-06-03"),
        core_sessions=(pd.Timestamp("2024-06-03"),),
    )
    label_window = label_calculations.label_read_window_for_panel(
        coverage,
        definitions=(label,),
        read_window=TimeWindow("2024-01-02", "2024-01-02"),
        core_sessions=(pd.Timestamp("2024-01-02"),),
    )

    assert factor_window == TimeWindow("2024-01-02", "2024-06-03")
    assert label_window == TimeWindow("2024-01-02", "2024-06-03")


@pytest.mark.parametrize("kind", ("factor", "label"))
def test_valid_bar_window_rejects_duplicate_panel_keys(kind: str) -> None:
    keys = pd.DataFrame(
        {
            "asset_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        }
    )
    kwargs = {
        "read_window": TimeWindow("2024-01-02", "2024-01-02"),
        "core_sessions": (pd.Timestamp("2024-01-02"),),
    }

    with pytest.raises(ValueError, match="duplicate_panel_keys"):
        if kind == "factor":
            factor_calculations.factor_read_window_for_panel(
                keys,
                definitions=(_factor_definition("mom_2d", lookback_window=2),),
                **kwargs,
            )
        else:
            label_calculations.label_read_window_for_panel(
                keys,
                definitions=(
                    _label_definition("forward_return_2d", horizon=2, lookahead=2),
                ),
                **kwargs,
            )


def test_future_tradable_forward_return_spec_depends_on_execution_mask() -> None:
    spec = _processed_spec(
        "tradable_forward_return",
        kind=ProcessedDataKind.LABEL,
        columns=(*platform_entity_time_key(), "tradable_forward_return_1d"),
        processed_input_data_ids=(MARKET_PANEL_DATA_ID, "execution"),
    )

    assert spec.processed_input_data_ids == (MARKET_PANEL_DATA_ID, "execution")


def test_write_paths_read_processed_inputs_via_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    window = TimeWindow("2024-01-02", "2024-01-02")
    sessions = (pd.Timestamp("2024-01-02"),)
    calls = {
        "annotation": 0,
        "factor": 0,
        "factor_key_batch": 0,
        "label": 0,
        "label_key_batch": 0,
    }
    asset_filters: list[tuple[object, ...] | None] = []
    key_asset_filters: list[tuple[object, ...] | None] = []
    key_time_windows: list[TimeWindow | None] = []

    def fake_inspect_dataset(*args: object, **kwargs: object):
        return SimpleNamespace(
            manifest={"file_size_bytes": 1},
            row_count=1,
            columns=("asset_id", "timestamp", "adj_close", "adj_open"),
        )

    def fake_write_and_inspect(spec: ProcessedDataSpec, frame, **kwargs: object):
        list(frame)
        return fake_inspect_dataset()

    def fake_read_processed_inputs(
        spec: ProcessedDataSpec,
        **kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        asset_filters.append(kwargs.get("asset_ids"))
        if spec.kind is ProcessedDataKind.ANNOTATION:
            calls["annotation"] += 1
        elif spec.kind is ProcessedDataKind.FACTOR:
            calls["factor"] += 1
        elif spec.kind is ProcessedDataKind.LABEL:
            calls["label"] += 1
        return {MARKET_PANEL_DATA_ID: _panel_frame_for_write_path()}

    def fake_derive_annotations(
        panel: pd.DataFrame,
        *,
        data_ids: tuple[str, ...],
        **kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        output = _key_frame().assign(
            next_bar_is_trading=[True],
            next_bar_limit_status=[0],
            can_enter=[True],
            can_exit=[True],
        )
        return {data_ids[0]: output}

    def fake_calculate_factors_for_panel(
        panel: pd.DataFrame,
        *,
        definitions: tuple[FactorDefinition, ...],
    ) -> pd.DataFrame:
        return _key_frame().assign(**{definitions[0].output_column: [1.0]})

    def fake_factor_read_window_for_panel(
        panel_keys: object,
        *,
        definitions: tuple[FactorDefinition, ...],
        read_window: TimeWindow,
        core_sessions: tuple[pd.Timestamp, ...],
    ) -> TimeWindow:
        del definitions, core_sessions
        assert list(panel_keys.itertuples(index=False, name=None)) == [
            ("asset_a", pd.Timestamp("2024-01-02"))
        ]
        calls["factor_key_batch"] += 1
        return read_window

    def fake_calculate_labels_for_panel(
        panel: pd.DataFrame,
        *,
        definitions: tuple[LabelDefinition, ...],
    ) -> pd.DataFrame:
        return _key_frame().assign(**{definitions[0].output_column: [0.01]})

    def fake_label_read_window_for_panel(
        panel_keys: object,
        *,
        definitions: tuple[LabelDefinition, ...],
        read_window: TimeWindow,
        core_sessions: tuple[pd.Timestamp, ...],
    ) -> TimeWindow:
        del definitions, core_sessions
        assert list(panel_keys.itertuples(index=False, name=None)) == [
            ("asset_a", pd.Timestamp("2024-01-02"))
        ]
        calls["label_key_batch"] += 1
        return read_window

    def iter_dataset_chunks_with_policy(*args: object, **kwargs: object):
        del args
        if kwargs.get("columns") == ("asset_id", "timestamp"):
            key_asset_filters.append(kwargs.get("asset_ids"))
            key_time_windows.append(kwargs.get("time_window"))
        return (_key_frame() for _ in range(1))

    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        fake_inspect_dataset,
    )
    monkeypatch.setattr(materialization_io, "write_and_inspect", fake_write_and_inspect)
    monkeypatch.setattr(
        materialization_io,
        "read_processed_inputs_for_spec",
        fake_read_processed_inputs,
    )
    monkeypatch.setattr(
        materialization_io,
        "read_materialized_dataset",
        lambda *_args, **_kwargs: _key_frame().loc[:, ["asset_id", "timestamp"]],
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        iter_dataset_chunks_with_policy,
    )
    monkeypatch.setattr(
        materialize_annotations.annotations,
        "derive_annotations",
        fake_derive_annotations,
    )
    monkeypatch.setattr(
        materialize_factors.factor_calculations,
        "calculate_factors_for_panel",
        fake_calculate_factors_for_panel,
    )
    monkeypatch.setattr(
        materialize_factors.factor_calculations,
        "factor_read_window_for_panel",
        fake_factor_read_window_for_panel,
    )
    monkeypatch.setattr(
        materialize_labels.label_calculations,
        "calculate_labels_for_panel",
        fake_calculate_labels_for_panel,
    )
    monkeypatch.setattr(
        materialize_labels.label_calculations,
        "label_read_window_for_panel",
        fake_label_read_window_for_panel,
    )
    resource_policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    with runtime_resource_policy_scope(resource_policy):
        materialize_annotations.write_annotations(
            _processed_spec(
                "execution",
                kind=ProcessedDataKind.ANNOTATION,
                columns=(
                    *platform_entity_time_key(),
                    "next_bar_is_trading",
                    "next_bar_limit_status",
                    "can_enter",
                    "can_exit",
                ),
                processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
            ),
            layout=layout,
            time_window=window,
            policy=resource_policy,
            sessions=sessions,
            written={},
        )
        materialize_factors.write_factors(
            _processed_spec(
                "price_volume",
                kind=ProcessedDataKind.FACTOR,
                columns=(*platform_entity_time_key(), MOMENTUM_20D_COLUMN),
                processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
            ),
            layout=layout,
            time_window=window,
            policy=resource_policy,
            sessions=sessions,
            written={},
        )
        materialize_labels.write_labels(
            _processed_spec(
                "forward_returns",
                kind=ProcessedDataKind.LABEL,
                columns=(*platform_entity_time_key(), FORWARD_RETURN_20D_COLUMN),
                processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
            ),
            layout=layout,
            time_window=window,
            policy=resource_policy,
            sessions=sessions,
            written={},
        )

    assert calls == {
        "annotation": 1,
        "factor": 1,
        "factor_key_batch": 1,
        "label": 1,
        "label_key_batch": 1,
    }
    assert asset_filters == [("asset_a",)] * 3
    assert key_asset_filters == [("asset_a",), ("asset_a",)]
    assert key_time_windows == [window, window]


@pytest.mark.parametrize("writer_kind", ("panel", "factor", "label", "annotation"))
def test_write_paths_reserve_registered_output_before_transform(
    writer_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = processed_catalog()
    spec_by_kind = {
        "panel": catalog.get(MARKET_PANEL_DATA_ID),
        "factor": catalog.get(PRICE_VOLUME_PROCESSED_DATA_ID),
        "label": catalog.get(FORWARD_RETURNS_PROCESSED_DATA_ID),
        "annotation": catalog.get(EXECUTION_MASK_ID),
    }
    spec = spec_by_kind[writer_kind]
    window = TimeWindow("2024-01-02", "2024-01-02")
    sessions = pd.DatetimeIndex(["2024-01-02"])
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    current_keys = _key_frame()
    panel = _panel_frame_for_write_path()
    market_bars = pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [1.0],
            "amount": [10.0],
        }
    )
    source_inputs = {
        "stock_master": pd.DataFrame({"source_asset_id": ["asset_a"]}),
        "market_cap": pd.DataFrame({"source_asset_id": ["asset_a"]}),
        "trading_calendar": pd.DataFrame({"calendar_date": ["2024-01-02"]}),
    }
    if writer_kind == "panel":
        live_frames = (*source_inputs.values(), market_bars)
    elif writer_kind in {"factor", "label"}:
        live_frames = (panel, current_keys)
    else:
        live_frames = (panel,)
    retained_bytes = 0
    live_frame_bytes = sum(
        pandas_frame_payload_bytes(frame) for frame in live_frames
    )

    monkeypatch.setattr(
        materialization_planning,
        "session_batch_window",
        lambda *_args, **_kwargs: window,
    )
    monkeypatch.setattr(
        materialization_planning,
        "bounded_valid_bar_key_coverage",
        lambda *_args, **_kwargs: current_keys,
    )

    def invoke_materialize(*_args: object, **kwargs: object):
        materialize = kwargs["materialize"]
        arguments = (("asset_a",), tuple(sessions), tuple(sessions), retained_bytes)
        with policy.reserve("test.upstream.retained", retained_bytes):
            if kwargs.get("valid_bar_offsets") is not None:
                yield materialize(*arguments, current_keys)
            else:
                yield materialize(*arguments)

    monkeypatch.setattr(
        materialization_planning,
        "iter_active_materialization_batches",
        invoke_materialize,
    )
    def consume_frames(_spec, frames, **_kwargs):  # noqa: ANN001, ANN202
        with runtime_resource_policy_scope(policy):
            return tuple(frames)

    monkeypatch.setattr(materialization_io, "write_and_inspect", consume_frames)
    monkeypatch.setattr(
        materialization_io,
        "read_processed_inputs_for_spec",
        lambda *_args, **_kwargs: {MARKET_PANEL_DATA_ID: panel},
    )
    monkeypatch.setattr(
        materialization_io,
        "read_source_inputs_for_spec",
        lambda *_args, **_kwargs: source_inputs,
    )
    monkeypatch.setattr(
        materialization_io,
        "read_materialized_dataset",
        lambda *_args, **_kwargs: market_bars,
    )
    monkeypatch.setattr(
        materialization_validation,
        "require_source_input_columns_from_schema",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_validation,
        "require_source_inputs_for_spec",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_validation,
        "require_source_input_columns",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_validation,
        "validate_processed_input_contract",
        lambda *_args, **_kwargs: None,
    )
    inspection = SimpleNamespace(
        manifest={"file_size_bytes": 1},
        row_count=1,
        columns=tuple(market_bars.columns),
    )
    monkeypatch.setattr(
        materialize_panel,
        "inspect_dataset",
        lambda *_args, **_kwargs: inspection,
    )
    monkeypatch.setattr(
        materialize_factors.factor_calculations,
        "factor_read_window_for_panel",
        lambda *_args, **_kwargs: window,
    )
    monkeypatch.setattr(
        materialize_labels.label_calculations,
        "label_read_window_for_panel",
        lambda *_args, **_kwargs: window,
    )

    transform_owner, transform_name = {
        "panel": (materialize_panel.panels, "compute_market_panel"),
        "factor": (
            materialize_factors.factor_calculations,
            "calculate_factors_for_panel",
        ),
        "label": (
            materialize_labels.label_calculations,
            "calculate_labels_for_panel",
        ),
        "annotation": (materialize_annotations.annotations, "derive_annotations"),
    }[writer_kind]

    class TransformObserved(RuntimeError):
        pass

    def reject_transform(*_args: object, **_kwargs: object) -> object:
        assert policy.reserved_memory_bytes > live_frame_bytes
        raise TransformObserved

    monkeypatch.setattr(transform_owner, transform_name, reject_transform)
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(TransformObserved):
        if writer_kind == "panel":
            materialize_panel.write_panel(
                spec,
                layout=layout,
                source_id=SourceId("synthetic_vendor_a"),
                market_bars_data_id=MARKET_BARS_DATA_ID,
                time_window=window,
                policy=policy,
                sessions=sessions,
                written={},
            )
        elif writer_kind == "factor":
            materialize_factors.write_factors(
                spec,
                layout=layout,
                time_window=window,
                policy=policy,
                sessions=sessions,
                written={},
            )
        elif writer_kind == "label":
            materialize_labels.write_labels(
                spec,
                layout=layout,
                time_window=window,
                policy=policy,
                sessions=sessions,
                written={},
            )
        else:
            materialize_annotations.write_annotations(
                spec,
                layout=layout,
                time_window=window,
                policy=policy,
                sessions=sessions,
                written={},
            )


def test_materialized_dataset_asset_filter_is_forwarded_outside_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    class Session:
        resource_policy = policy

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        materialization_io,
        "open_dataset_session",
        lambda _root, **kwargs: observed.append(kwargs) or Session(),
    )
    monkeypatch.setattr(
        materialization_io,
        "materialize_dataset_session",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"timestamp": pd.Series(dtype="datetime64[ns]")}
        ),
    )
    result = materialization_io.read_materialized_dataset(
        Path("panel"),
        resource_policy=policy,
        columns=("timestamp",),
        asset_ids=("a", "b"),
    )

    assert result.empty
    assert observed[0]["columns"] == ("timestamp",)
    assert observed[0]["row_filters"] == (("asset_id", "in", ("a", "b")),)


@pytest.mark.parametrize("reverse_chunks", (False, True))
def test_available_trading_sessions_is_order_independent_across_chunks(
    monkeypatch: pytest.MonkeyPatch,
    reverse_chunks: bool,
) -> None:
    market_type = next(iter(materialization_planning.CURRENT_MARKET_TYPES))
    frames = (
        pd.DataFrame(
            {
                "source_market_type": [market_type, market_type, market_type],
                "is_trading_day": [True, False, True],
                "timestamp": pd.Series(
                    ["2024-01-05", "2024-01-02", "2024-01-02"],
                    dtype="datetime64[ns]",
                ),
            }
        ),
        pd.DataFrame(
            {
                "source_market_type": [market_type, market_type, market_type],
                "is_trading_day": [True, True, True],
                "timestamp": pd.Series(
                    ["2024-01-01", "2024-01-04", "2024-01-03"],
                    dtype="datetime64[ns]",
                ),
            }
        ),
    )
    chunks = iter(tuple(reversed(frames)) if reverse_chunks else frames)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(
            row_count=6,
            manifest={"file_size_bytes": 10**9},
        ),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: observed.append(kwargs) or chunks,
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=5 * 1024 * 1024,
    )
    result = materialization_planning.available_trading_sessions(
        SimpleNamespace(interim_data_root=lambda *_args: Path("calendar")),
        "source",  # type: ignore[arg-type]
        time_window=TimeWindow("2024-01-02", "2024-01-04"),
        lookback_sessions=1,
        lookahead_sessions=1,
        resource_policy=policy,
    )

    assert result.equals(
        pd.DatetimeIndex(pd.Timestamp(f"2024-01-0{day}") for day in range(1, 6))
    )
    retained_bytes = 6 * np.dtype(np.int64).itemsize
    fixed_row_bytes = (
        np.dtype(np.int64).itemsize
        + np.dtype(np.bool_).itemsize
        + np.dtype("datetime64[ns]").itemsize
    )
    transform_row_bytes = 2 * np.dtype(np.bool_).itemsize + np.dtype(np.int64).itemsize
    target_bytes = min(
        policy.remaining_memory_bytes,
        retained_bytes + (10**9 + 5) // 6,
        (policy.task_memory_budget_bytes - retained_bytes)
        * fixed_row_bytes
        // (fixed_row_bytes + transform_row_bytes),
    )
    assert observed[0]["target_chunk_bytes"] == target_bytes
    assert observed[0]["retained_payload_bytes"] == (
        retained_bytes + ceil(target_bytes / fixed_row_bytes) * transform_row_bytes
    )


def test_available_trading_sessions_small_fit_reaches_exact_working_set_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def read(*_args, **_kwargs):  # noqa: ANN001, ANN202
        nonlocal reads
        reads += 1
        return iter(
            (
                pd.DataFrame(
                    {
                        "source_market_type": [
                            next(iter(materialization_planning.CURRENT_MARKET_TYPES))
                        ],
                        "is_trading_day": [True],
                        "timestamp": pd.Series(
                            [pd.Timestamp("2024-01-02")],
                            dtype="datetime64[ns]",
                        ),
                    }
                ),
            )
        )

    monkeypatch.setattr(materialization_io, "iter_dataset_chunks_with_policy", read)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(
            row_count=1, manifest={"file_size_bytes": 1}
        ),
    )
    result = materialization_planning.available_trading_sessions(
        SimpleNamespace(interim_data_root=lambda *_args: Path("calendar")),
        "source",  # type: ignore[arg-type]
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        lookback_sessions=0,
        lookahead_sessions=0,
        resource_policy=policy,
    )

    assert tuple(result) == (pd.Timestamp("2024-01-02"),)
    assert reads == 1


def test_available_trading_sessions_retains_only_selected_boundary_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_type = next(iter(materialization_planning.CURRENT_MARKET_TYPES))
    before = tuple(pd.date_range("2020-01-01", periods=80, freq="D"))
    core = tuple(pd.date_range("2024-06-03", periods=2, freq="D"))
    after = tuple(pd.date_range("2026-01-01", periods=80, freq="D"))
    dates = (*before[::2], *after[1::2], *core, *after[::2], *before[1::2])

    def chunks():
        for session in dates:
            yield pd.DataFrame(
                {
                    "source_market_type": [market_type],
                    "is_trading_day": [True],
                    "timestamp": pd.Series([session], dtype="datetime64[ns]"),
                }
            )

    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(
            row_count=len(dates),
            manifest={"file_size_bytes": 10**9},
        ),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: chunks(),
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    result = materialization_planning.available_trading_sessions(
        SimpleNamespace(interim_data_root=lambda *_args: Path("calendar")),
        "source",  # type: ignore[arg-type]
        time_window=TimeWindow("2024-06-03", "2024-06-04"),
        lookback_sessions=2,
        lookahead_sessions=2,
        resource_policy=policy,
    )

    assert result.equals(pd.DatetimeIndex((*before[-2:], *core, *after[:2])))
    assert result.nbytes == len(result) * np.dtype(np.int64).itemsize
    assert isinstance(result.asi8.base, np.ndarray)
    assert result.asi8.base.nbytes == result.nbytes


def test_dataset_coverage_reader_accounts_compact_retained_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = pd.date_range("2024-01-01", periods=3, freq="D")
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(columns=("timestamp",)),
    )

    def read(*_args, **kwargs):  # noqa: ANN001, ANN202
        observed.append(kwargs)
        return iter((pd.DataFrame({"timestamp": sessions}),))

    monkeypatch.setattr(materialization_io, "iter_dataset_chunks_with_policy", read)

    materialization_planning.require_dataset_coverage(
        Path("panel"),
        TimeWindow("2024-01-01", "2024-01-03"),
        sessions=sessions,
        resource_policy=policy,
    )

    retained_bytes = sessions.nbytes + len(sessions) * np.dtype(np.bool_).itemsize
    assert observed == [
        {
            "columns": ["timestamp"],
            "time_window": TimeWindow("2024-01-01", "2024-01-03"),
            "resource_policy": policy,
            "target_chunk_bytes": min(
                policy.remaining_memory_bytes,
                policy.task_memory_budget_bytes - retained_bytes,
            ),
            "retained_payload_bytes": retained_bytes,
        }
    ]


def test_dataset_coverage_preflights_int64_normalization_before_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    chunk = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"])})
    retained_state_bytes = 17
    observed: list[int] = []
    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(columns=("timestamp",)),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter((chunk,)),
    )

    def reject_bound(*, retained_payload_bytes, **_kwargs):  # noqa: ANN001, ANN202
        observed.append(retained_payload_bytes)
        raise materialization_io.MaterializationCapacityExceeded(
            "a_share_known_payloads_exceed_task_memory_budget"
        )

    monkeypatch.setattr(
        materialization_planning,
        "require_known_materialization_payload_bound",
        reject_bound,
    )
    monkeypatch.setattr(
        materialization_planning.strict_values,
        "strict_timestamp_value_series",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("timestamp_normalized_before_capacity_preflight")
        ),
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        materialization_planning._mark_dataset_session_coverage(
            Path("dataset"),
            TimeWindow("2024-01-02", "2024-01-03"),
            expected_sessions=pd.DatetimeIndex(chunk["timestamp"]),
            covered=np.zeros(2, dtype=np.bool_),
            retained_state_bytes=retained_state_bytes,
            resource_policy=policy,
        )
    assert observed == [retained_state_bytes + 2 * np.dtype(np.int64).itemsize]


def test_adaptive_materialization_publication_carries_upstream_retained_bytes() -> None:
    frame = pd.DataFrame({"asset_id": ["a"], "timestamp": [pd.Timestamp("2024-01-02")]})
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        publications = list(
            materialization_planning._iter_adaptive_materialization_batches(
                asset_ids=("a",),
                sessions=(pd.Timestamp("2024-01-02"),),
                session_context=(pd.Timestamp("2024-01-02"),),
                retained_payload_bytes=123,
                materialize=lambda *_args: frame,
            )
        )

    assert len(publications) == 1
    assert publications[0].frame is frame
    assert publications[0].retained_payload_bytes == 123


def test_adaptive_tuple_slice_is_admitted_before_allocation() -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    class SliceRejectingTuple(tuple):
        def __getitem__(self, item):  # noqa: ANN001, ANN204
            if isinstance(item, slice):
                raise AssertionError("tuple slice allocated before admission")
            return super().__getitem__(item)

    def materialize(*_args):  # noqa: ANN002, ANN202
        raise materialization_io.MaterializationCapacityExceeded("split")

    with (
        runtime_resource_policy_scope(policy),
        pytest.raises(
            materialization_io.MaterializationCapacityExceeded,
            match="a_share_known_payloads_exceed_task_memory_budget",
        ),
    ):
        next(
            materialization_planning._iter_adaptive_materialization_batches(
                asset_ids=SliceRejectingTuple(("a", "b")),
                sessions=(pd.Timestamp("2024-01-02"),),
                session_context=(pd.Timestamp("2024-01-02"),),
                retained_payload_bytes=policy.task_memory_budget_bytes - 1,
                materialize=materialize,
            )
        )


def test_sorted_publication_reserves_upstream_before_key_temporaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _processed_spec(
        MARKET_PANEL_DATA_ID,
        kind=ProcessedDataKind.PANEL,
        columns=(*platform_entity_time_key(), "payload"),
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset"],
            "timestamp": [pd.Timestamp("2024-01-02")],
            "payload": [1.0],
        }
    )
    publication = dataset_writing.DatasetWriteChunk(
        frame, policy.task_memory_budget_bytes - 1
    )
    monkeypatch.setattr(
        materialization_io.np,
        "lexsort",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("key temporaries allocated before retained admission")
        ),
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        next(
            iter(
                materialization_io.iter_sorted_output_chunks(
                    (publication,), spec, resource_policy=policy
                )
            )
        )


def test_active_materialization_rejects_missing_outer_policy_before_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    bounds_hits: list[Sequence[pd.Timestamp]] = []
    sessions = (pd.Timestamp("2024-01-02"),)
    real_bounds = materialization_planning._target_session_bounds

    def bounds(values, window):  # noqa: ANN001, ANN202
        bounds_hits.append(values)
        return real_bounds(values, window)

    monkeypatch.setattr(materialization_planning, "_target_session_bounds", bounds)

    with pytest.raises(
        RuntimeError, match="^a_share_materialization_policy_scope_required$"
    ):
        materialization_planning.iter_active_materialization_batches(
            Path("panel"),
            time_window=TimeWindow("2024-01-02", "2024-01-02"),
            sessions=sessions,
            resource_policy=policy,
            materialize=lambda *_args: pd.DataFrame(),
        )

    assert bounds_hits == []


@pytest.mark.parametrize("primary_type", (KeyboardInterrupt,))
def test_active_materialization_outer_scope_owns_factory_snapshot_and_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    sessions = pd.date_range("2024-01-02", periods=2)
    original = tuple(sessions)
    primary = primary_type("materialize_primary")
    observed: list[tuple[str, RuntimeResourcePolicy | None]] = []

    def source():
        try:
            observed.append(("source", current_runtime_resource_policy()))
            yield pd.DataFrame({"asset_id": ["asset_a"]})
        finally:
            observed.append(("close", current_runtime_resource_policy()))

    materialized_sessions: list[tuple[pd.Timestamp, ...]] = []

    def materialize(  # noqa: ANN001, ANN202
        _assets, current_sessions, _session_context, _retained
    ):
        observed.append(("materialize", current_runtime_resource_policy()))
        materialized_sessions.append(tuple(current_sessions))
        raise primary

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )

    with runtime_resource_policy_scope(policy):
        batches = materialization_planning.iter_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            sessions=sessions,
            resource_policy=policy,
            materialize=materialize,
        )
        sessions._data._ndarray[0] = pd.Timestamp("2099-01-01").to_datetime64()  # noqa: SLF001
        with pytest.raises(primary_type) as caught:
            next(batches)

    assert {
        "caught_is_primary": caught.value is primary,
        "snapshot": materialized_sessions,
        "observed": observed,
    } == {
        "caught_is_primary": True,
        "snapshot": [original],
        "observed": [("source", policy), ("close", policy), ("materialize", policy)],
    }


def test_active_materialization_callback_receives_full_owned_session_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    sessions = pd.date_range("2024-01-01", periods=4)
    original = tuple(sessions)
    observed: list[tuple[tuple[pd.Timestamp, ...], tuple[pd.Timestamp, ...]]] = []

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter((pd.DataFrame({"asset_id": ["asset_a"]}),)),
    )

    def materialize(
        asset_ids,
        current_sessions,
        session_context,
        _retained,  # noqa: ANN001
    ):  # noqa: ANN202
        observed.append((tuple(current_sessions), tuple(session_context)))
        return pd.DataFrame(
            {"asset_id": [asset_ids[0]], "timestamp": [current_sessions[0]]}
        )

    with runtime_resource_policy_scope(policy):
        batches = materialization_planning.iter_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            sessions=sessions,
            resource_policy=policy,
            materialize=materialize,
        )
        sessions._data._ndarray[-1] = pd.Timestamp(  # noqa: SLF001
            "2099-01-01"
        ).to_datetime64()
        list(batches)

    assert observed == [(original[1:3], original)]


def test_active_materialization_never_consumed_factory_abandons_without_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    source_hits: list[str] = []
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source_hits.append("source") or iter(()),
    )

    with runtime_resource_policy_scope(policy):
        batches = materialization_planning.iter_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-02", "2024-01-02"),
            sessions=(pd.Timestamp("2024-01-02"),),
            resource_policy=policy,
            materialize=lambda *_args: pd.DataFrame(),
        )
        reference = weakref.ref(batches)
        del batches
        gc.collect()

    assert {"collected": reference() is None, "source_hits": source_hits} == {
        "collected": True,
        "source_hits": [],
    }


def test_active_materialization_owner_deduplicates_planned_batches_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    attempts: list[tuple[tuple[object, ...], int, int]] = []
    sessions = tuple(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def source():
        try:
            yield pd.DataFrame({"asset_id": ["a", "b", "a"]})
        finally:
            closed.append(True)

    def materialize(  # noqa: ANN001, ANN202
        asset_ids, current_sessions, _session_context, retained_payload_bytes
    ):
        attempts.append(
            (
                tuple(asset_ids),
                retained_payload_bytes,
                policy.reserved_memory_bytes,
            )
        )
        return pd.DataFrame({"asset_id": asset_ids, "timestamp": current_sessions})

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )
    publications = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        sessions=sessions,
        resource_policy=policy,
        output_row_bound_bytes=policy.task_memory_budget_bytes // 2,
        materialize=materialize,
    )
    result = pd.concat(
        (publication.frame for publication in publications),
        ignore_index=True,
    )

    assert list(result.itertuples(index=False, name=None)) == [
        ("a", sessions[0]),
        ("a", sessions[1]),
        ("b", sessions[0]),
        ("b", sessions[1]),
    ]
    assert [assets for assets, _retained, _reserved in attempts] == [
        ("a",),
        ("a",),
        ("b",),
        ("b",),
    ]
    assert all(retained == 0 for _assets, retained, _reserved in attempts)
    assert all(reserved > 0 for _assets, _retained, reserved in attempts)
    assert closed == [True]


def test_active_materialization_orders_late_discovered_assets_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=1024**3)
    sessions = tuple(pd.to_datetime(["2024-01-02"]))
    spec = processed_catalog().get(MARKET_PANEL_DATA_ID)

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter(
            (
                pd.DataFrame({"asset_id": ["b"]}),
                pd.DataFrame({"asset_id": ["a"]}),
            )
        ),
    )

    def materialize(asset_ids, current_sessions, *_args):  # noqa: ANN001, ANN202
        rows = tuple(
            (asset_id, timestamp)
            for asset_id in asset_ids
            for timestamp in current_sessions
        )
        return pd.DataFrame(
            {
                column: (
                    ["a_share"] * len(rows)
                    if column == "market_id"
                    else [asset_id for asset_id, _timestamp in rows]
                    if column == "asset_id"
                    else [timestamp for _asset_id, timestamp in rows]
                    if column == "timestamp"
                    else [1.0] * len(rows)
                )
                for column in spec.required_columns
            }
        ).sort_values(list(spec.primary_key))

    publications = _scoped_active_materialization_batches(
        tmp_path / "panel",
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=sessions,
        resource_policy=policy,
        materialize=materialize,
    )
    ordered = materialization_io.iter_sorted_output_chunks(
        publications,
        spec,
        resource_policy=policy,
    )
    result = pd.concat(
        (
            publication.frame
            if isinstance(publication, dataset_writing.DatasetWriteChunk)
            else publication
            for publication in ordered
        ),
        ignore_index=True,
    )

    assert result["asset_id"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    ("container_kind", "window_kind", "mutation_timing"),
    (
        ("datetime_index", "full", "between_pulls"),
        ("list", "full", "before_first_pull"),
        ("indexed_sequence", "sub", "between_pulls"),
    ),
)
def test_active_materialization_owns_stable_target_sessions_across_capacity_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container_kind: str,
    window_kind: str,
    mutation_timing: str,
) -> None:
    class IndexedSessions(Sequence[pd.Timestamp]):
        def __init__(
            self,
            values: list[pd.Timestamp],
            start: int = 0,
            stop: int | None = None,
        ) -> None:
            self.values = values
            self.start = start
            self.stop = len(values) if stop is None else stop

        def __len__(self) -> int:
            return self.stop - self.start

        def __getitem__(self, key):  # noqa: ANN001, ANN202
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self))
                if step != 1:
                    return tuple(self[index] for index in range(start, stop, step))
                return IndexedSessions(
                    self.values,
                    self.start + start,
                    self.start + stop,
                )
            return self.values[self.start + key]

        def __iter__(self):  # noqa: ANN204
            raise AssertionError("custom_sessions_must_use_indexed_snapshot")

    original = list(pd.date_range("2024-01-01", periods=4))
    if container_kind == "datetime_index":
        sessions: Sequence[pd.Timestamp] = pd.DatetimeIndex(original)
    elif container_kind == "tuple":
        sessions = tuple(original)
    elif container_kind == "list":
        sessions = list(original)
    else:
        sessions = IndexedSessions(list(original))
    target = original if window_kind == "full" else original[1:3]
    window = TimeWindow(target[0].date(), target[-1].date())
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    def mutate_source() -> None:
        replacement = pd.Timestamp("2099-01-03")
        position = original.index(target[-1])
        if isinstance(sessions, pd.DatetimeIndex):
            sessions._data._ndarray[position] = replacement.to_datetime64()  # noqa: SLF001
        elif isinstance(sessions, list):
            sessions[position] = replacement
        elif isinstance(sessions, IndexedSessions):
            sessions.values[position] = replacement

    def source():
        yield pd.DataFrame({"asset_id": ["asset_a"]})

    def materialize(  # noqa: ANN001, ANN202
        asset_ids, current_sessions, _session_context, _retained
    ):
        if mutation_timing == "before_first_pull":
            mutate_source()
        return pd.DataFrame(
            {"asset_id": [asset_ids[0]], "timestamp": [current_sessions[0]]}
        )

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )
    output = _scoped_active_materialization_batches(
        tmp_path / "panel",
        time_window=window,
        sessions=sessions,
        resource_policy=policy,
        output_row_bound_bytes=policy.task_memory_budget_bytes // 2,
        materialize=materialize,
    )
    first = next(output).frame["timestamp"].iloc[0]
    if mutation_timing == "between_pulls":
        mutate_source()
    observed = (first, *(item.frame["timestamp"].iloc[0] for item in output))

    assert observed == tuple(target)


def test_active_materialization_rejects_hostile_datetime_index_subclass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AliasingArray(np.ndarray):
        def copy(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            return self

    class DataProxy:
        def __init__(self, owner) -> None:  # noqa: ANN001
            self.owner = owner

        @property
        def _ndarray(self):  # noqa: ANN202
            return self.owner._ndarray.view(AliasingArray)  # noqa: SLF001

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(self.owner, name)

    class AliasingDatetimeIndex(pd.DatetimeIndex):
        @property
        def _data(self):  # noqa: ANN202
            return DataProxy(self.__dict__["_hostile_data"])

        @_data.setter
        def _data(self, value) -> None:  # noqa: ANN001
            self.__dict__["_hostile_data"] = value

    sessions = AliasingDatetimeIndex(pd.date_range("2024-01-01", periods=2))
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: pytest.fail("source_opened_for_hostile_sessions"),
    )
    with pytest.raises(ValueError, match="^invalid_session_dates$"):
        _scoped_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            sessions=sessions,
            resource_policy=policy,
            materialize=lambda *_args: pd.DataFrame(),
        )


@pytest.mark.parametrize("container_kind", ("datetime_index", "mutable_sequence"))
def test_active_materialization_admits_source_and_snapshot_overlap_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    container_kind: str,
) -> None:
    class IndexedSessions(Sequence[pd.Timestamp]):
        def __init__(self, values: tuple[pd.Timestamp, ...]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, key):  # noqa: ANN001, ANN202
            if isinstance(key, slice):
                raise AssertionError("snapshot_allocated_before_overlap_gate")
            return self.values[key]

    values = tuple(pd.date_range("2024-01-01", periods=4))
    sessions: Sequence[pd.Timestamp] = (
        pd.DatetimeIndex(values)
        if container_kind == "datetime_index"
        else IndexedSessions(values)
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    source_bytes = (
        sessions.nbytes
        if isinstance(sessions, pd.Index)
        else len(sessions) * np.dtype(object).itemsize
    )
    target_bytes = len(values) * (
        np.dtype("datetime64[ns]").itemsize
        if isinstance(sessions, pd.Index)
        else np.dtype(object).itemsize
    )
    observed: list[int] = []

    def require_capacity(retained, **_kwargs):  # noqa: ANN001, ANN202
        observed.append(retained)
        if retained == source_bytes + target_bytes:
            raise materialization_io.MaterializationCapacityExceeded("overlap_gate")

    monkeypatch.setattr(
        materialization_planning,
        "_require_retained_materialization_capacity",
        require_capacity,
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: pytest.fail("source_opened_before_overlap_gate"),
    )
    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="overlap_gate",
    ):
        _scoped_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-01", "2024-01-04"),
            sessions=sessions,
            resource_policy=policy,
            materialize=lambda *_args: pd.DataFrame(),
        )
    assert source_bytes + target_bytes in observed


def test_active_materialization_owner_preserves_indexed_target_session_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IndexedSessions(Sequence[pd.Timestamp]):
        def __init__(self, values: tuple[pd.Timestamp, ...]) -> None:
            self._values = values

        def __len__(self) -> int:
            return len(self._values)

        def __getitem__(self, key):  # noqa: ANN001, ANN202
            if isinstance(key, slice):
                return IndexedSessions(self._values[key])
            return self._values[key]

        def __iter__(self):  # noqa: ANN204
            raise AssertionError("full_session_materialization_before_source_admission")

    source_opened: list[bool] = []
    callback_sessions: list[Sequence[pd.Timestamp]] = []
    sessions = IndexedSessions(tuple(pd.date_range("2024-01-01", periods=4)))
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def source():
        source_opened.append(True)
        yield pd.DataFrame({"asset_id": ["asset_a"]})

    def materialize(  # noqa: ANN001, ANN202
        asset_ids, current_sessions, _session_context, _retained
    ):
        callback_sessions.append(current_sessions)
        return pd.DataFrame(
            {
                "asset_id": [asset_ids[0]],
                "timestamp": [current_sessions[0]],
            }
        )

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )

    publications = list(
        _scoped_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            sessions=sessions,
            resource_policy=policy,
            materialize=materialize,
        )
    )

    assert source_opened == [True]
    assert len(callback_sessions) == 1
    assert type(callback_sessions[0]) is tuple
    assert len(callback_sessions[0]) == 2
    assert publications[0].frame["timestamp"].tolist() == [pd.Timestamp("2024-01-02")]


@pytest.mark.parametrize(
    ("writer_kind", "valid_bar_offsets"),
    (
        ("panel", None),
        ("factor", (1, 0)),
    ),
)
def test_active_materialization_admits_sessions_before_slice_index_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_kind: str,
    valid_bar_offsets: tuple[int, int] | None,
) -> None:
    del writer_kind

    class OversizedSessions(Sequence[pd.Timestamp]):
        def __len__(self) -> int:
            return 200_000

        def __getitem__(self, key):  # noqa: ANN001, ANN202
            if isinstance(key, slice):
                raise AssertionError("session_slice_before_capacity_gate")
            return pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=key)

    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        materialization_planning,
        "ExactPrimaryKeyIndex",
        lambda **_kwargs: pytest.fail("index_before_session_capacity_gate"),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: pytest.fail("source_before_session_capacity_gate"),
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        _scoped_active_materialization_batches(
            tmp_path / "panel",
            time_window=TimeWindow("2024-01-01", "2024-01-01"),
            sessions=OversizedSessions(),
            resource_policy=policy,
            valid_bar_offsets=valid_bar_offsets,
            materialize=lambda *_args: pd.DataFrame(),
        )


def test_active_materialization_requires_artifact_root_before_index_or_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    monkeypatch.setattr(
        materialization_planning,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
        raising=False,
    )
    monkeypatch.setattr(
        materialization_planning,
        "ExactPrimaryKeyIndex",
        lambda **_kwargs: pytest.fail("index_before_artifact_root_gate"),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: pytest.fail("source_before_artifact_root_gate"),
    )

    batches = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=(pd.Timestamp("2024-01-02"),),
        resource_policy=policy,
        materialize=lambda *_args: pd.DataFrame(),
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        next(batches)


def test_active_materialization_owner_closes_source_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def source():
        try:
            yield pd.DataFrame({"asset_id": ["asset_a"]})
        finally:
            closed.append(True)

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )
    batches = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=(pd.Timestamp("2024-01-02"),),
        resource_policy=policy,
        materialize=lambda assets, sessions, _session_context, _retained: pd.DataFrame(
            {"asset_id": assets, "timestamp": sessions}
        ),
    )

    next(batches)
    batches.close()

    assert closed == [True]


def test_active_materialization_scope_covers_indexes_source_materialize_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    observed: list[tuple[str, RuntimeResourcePolicy | None]] = []

    class Index:
        retained_memory_bytes = 0

        def __init__(self, **_kwargs: object) -> None:
            self.payloads: dict[str, list[tuple[bytes, bytes]]] = {}
            observed.append(("index_init", current_runtime_resource_policy()))

        def __enter__(self) -> Index:
            observed.append(("index_enter", current_runtime_resource_policy()))
            return self

        def __exit__(self, *_args: object) -> None:
            observed.append(("index_exit", current_runtime_resource_policy()))

        def add_identity_payload_rows(self, *, rows):  # noqa: ANN001, ANN202
            observed.append(("add", current_runtime_resource_policy()))
            inserted = 0
            for namespace, identity, value in rows:
                self.payloads.setdefault(namespace, []).append((identity, value))
                inserted += 1
            return inserted

        def identity_payload_values(self, *, namespace):  # noqa: ANN001, ANN202
            observed.append(("iterate", current_runtime_resource_policy()))
            return iter(
                value for _identity, value in sorted(self.payloads.get(namespace, ()))
            )

    def source():
        try:
            observed.append(("source", current_runtime_resource_policy()))
            yield pd.DataFrame({"asset_id": ["asset_a"]})
        finally:
            observed.append(("source_close", current_runtime_resource_policy()))

    def materialize(  # noqa: ANN001, ANN202
        asset_ids, sessions, _session_context, _retained
    ):
        observed.append(("materialize", current_runtime_resource_policy()))
        return pd.DataFrame({"asset_id": asset_ids, "timestamp": sessions})

    monkeypatch.setattr(materialization_planning, "ExactPrimaryKeyIndex", Index)
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )

    batches = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=(pd.Timestamp("2024-01-02"),),
        resource_policy=policy,
        materialize=materialize,
    )
    next(batches)
    batches.close()

    assert observed
    assert all(active is policy for _, active in observed)


def test_active_materialization_owner_surfaces_close_failure_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def source():
        try:
            yield pd.DataFrame({"asset_id": ["asset_a"]})
        finally:
            raise OSError("close_failure")

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: source(),
    )
    batches = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=(pd.Timestamp("2024-01-02"),),
        resource_policy=policy,
        materialize=lambda assets, sessions, _session_context, _retained: pd.DataFrame(
            {"asset_id": assets, "timestamp": sessions}
        ),
    )

    with pytest.raises(OSError, match="close_failure"):
        list(batches)


def test_valid_bar_owner_reads_with_retained_state_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    retained: list[int] = []
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    def source():
        try:
            yield pd.DataFrame(
                {
                    "asset_id": ["a", "a", "a", "a"],
                    "timestamp": pd.to_datetime(
                        ["2024-01-04", "2024-01-02", "2024-01-05", "2024-01-03"]
                    ),
                }
            )
        finally:
            closed.append(True)

    def read(*_args, **kwargs):  # noqa: ANN001, ANN202
        retained.append(kwargs["retained_payload_bytes"])
        return source()

    monkeypatch.setattr(materialization_io, "iter_dataset_chunks_with_policy", read)
    with materialization_planning.ExactPrimaryKeyIndex(
        primary_key=("asset_id", "timestamp"),
        timestamp_columns=("timestamp",),
        scratch_parent=Path(tempfile.gettempdir()),
    ) as index:
        materialization_planning._build_valid_bar_key_index(
            Path("panel"), index=index, resource_policy=policy
        )
        result = materialization_planning.bounded_valid_bar_key_coverage(
            index,
            asset_ids=("a",),
            core_sessions=pd.DatetimeIndex(["2024-01-03"]),
            lookback_bars=1,
            lookahead_bars=1,
            retained_payload_bytes=23 + index.retained_memory_bytes,
            resource_policy=policy,
        )

    assert list(result.itertuples(index=False, name=None)) == [
        ("a", pd.Timestamp("2024-01-02")),
        ("a", pd.Timestamp("2024-01-03")),
        ("a", pd.Timestamp("2024-01-04")),
    ]
    assert retained[0] >= 23
    assert closed == [True]


def test_valid_bar_index_streams_exact_rows_without_column_validation_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    observed: list[tuple[str, bytes, bytes]] = []

    class Index:
        retained_memory_bytes = 1024

        @staticmethod
        def add_identity_payload_rows(*, rows):  # noqa: ANN001, ANN205
            observed.extend(rows)
            return len(observed)

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    monkeypatch.setattr(
        materialization_planning.strict_values,
        "strict_timestamp_value_series",
        lambda *_args, **_kwargs: pytest.fail("whole_timestamp_column_copy"),
    )

    materialization_planning._build_valid_bar_key_index(
        Path("panel"),
        index=Index(),  # type: ignore[arg-type]
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
    )

    assert [namespace for namespace, _identity, _value in observed] == [
        materialization_planning._valid_bar_asset_namespace("a"),
        materialization_planning._valid_bar_asset_namespace("b"),
    ]


def test_active_asset_discovery_admits_before_string_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    class UnencodableAsset(str):
        def encode(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            pytest.fail("asset_encoding_before_admission")

    asset = UnencodableAsset("a" * policy.task_memory_budget_bytes)
    frame = pd.DataFrame({"asset_id": pd.Series([asset], dtype=object)})
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter((frame,)),
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        list(
            _scoped_active_materialization_batches(
                Path("panel"),
                time_window=TimeWindow("2024-01-02", "2024-01-02"),
                sessions=(pd.Timestamp("2024-01-02"),),
                resource_policy=policy,
                materialize=lambda *_args: pd.DataFrame(),
            )
        )


def test_valid_bar_boundary_requires_strictly_ordered_assets() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    with materialization_planning.ExactPrimaryKeyIndex(
        primary_key=("asset_id", "timestamp"),
        timestamp_columns=("timestamp",),
        scratch_parent=Path(tempfile.gettempdir()),
    ) as index:
        with pytest.raises(ValueError, match="invalid_processed_panel_asset_order"):
            materialization_planning.bounded_valid_bar_key_coverage(
                index,
                asset_ids=("b", "a"),
                core_sessions=(pd.Timestamp("2024-01-03"),),
                lookback_bars=1,
                lookahead_bars=1,
                retained_payload_bytes=index.retained_memory_bytes,
                resource_policy=policy,
            )


def test_valid_bar_boundary_admits_state_before_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        materialization_planning.np,
        "full",
        lambda *_args, **_kwargs: pytest.fail("boundary_array_before_admission"),
    )

    with pytest.raises(
        materialization_io.MaterializationCapacityExceeded,
        match="a_share_known_payloads_exceed_task_memory_budget",
    ):
        materialization_planning.bounded_valid_bar_key_coverage(
            SimpleNamespace(),  # type: ignore[arg-type]
            asset_ids=("a",),
            core_sessions=(pd.Timestamp("2024-01-03"),),
            lookback_bars=1,
            lookahead_bars=1,
            retained_payload_bytes=policy.task_memory_budget_bytes,
            resource_policy=policy,
        )


def test_valid_bar_output_capacity_is_checked_before_dataframe_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )

    class Index:
        @staticmethod
        def identity_payload_values(*, namespace):  # noqa: ANN001, ANN205
            assert namespace
            value = pd.Timestamp("2024-01-03").value
            yield materialization_planning._valid_bar_timestamp_payload(value)

    original_empty = materialization_planning.np.empty
    observed_reservations: list[int] = []

    def observe_empty(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        observed_reservations.append(policy.reserved_memory_bytes)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(materialization_planning.np, "empty", observe_empty)
    with runtime_resource_policy_scope(policy):
        materialization_planning.bounded_valid_bar_key_coverage(
            Index(),  # type: ignore[arg-type]
            asset_ids=("a",),
            core_sessions=(pd.Timestamp("2024-01-03"),),
            lookback_bars=0,
            lookahead_bars=0,
            retained_payload_bytes=0,
            resource_policy=policy,
        )
    assert observed_reservations
    assert min(observed_reservations) > 0


def test_valid_bar_index_builds_once_across_adaptive_asset_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    reads: list[
        tuple[
            tuple[str, ...],
            tuple[object, ...] | None,
            TimeWindow | None,
        ]
    ] = []

    def read(*_args, columns, **kwargs):  # noqa: ANN001, ANN003, ANN202
        reads.append(
            (
                tuple(columns),
                kwargs.get("asset_ids"),
                kwargs.get("time_window"),
            )
        )
        if tuple(columns) == ("asset_id", "timestamp"):
            return iter(
                (
                    pd.DataFrame(
                        {
                            "asset_id": ["a", "a", "a", "b", "b", "b"],
                            "timestamp": pd.to_datetime(
                                [
                                    "2024-01-01",
                                    "2024-01-02",
                                    "2024-01-03",
                                    "2024-01-01",
                                    "2024-01-02",
                                    "2024-01-03",
                                ]
                            ),
                        }
                    ),
                )
            )
        return iter((pd.DataFrame({"asset_id": ["a", "b"]}),))

    monkeypatch.setattr(materialization_io, "iter_dataset_chunks_with_policy", read)

    def materialize(  # noqa: ANN001, ANN202
        assets, sessions, _session_context, retained, keys
    ):
        assert retained == 0
        assert policy.reserved_memory_bytes > 0
        assert len(assets) == 1
        assert set(keys["asset_id"]) == set(assets)
        return pd.DataFrame({"asset_id": assets, "timestamp": sessions})

    chunks = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=pd.date_range("2024-01-01", "2024-01-03"),
        resource_policy=policy,
        valid_bar_offsets=(1, 1),
        output_row_bound_bytes=policy.task_memory_budget_bytes // 2,
        materialize=materialize,
    )

    assert len(list(chunks)) == 2
    assert reads == [
        (
            ("asset_id",),
            None,
            TimeWindow("2024-01-02", "2024-01-02"),
        ),
        (
            ("asset_id", "timestamp"),
            ("a", "b"),
            TimeWindow("2024-01-01", "2024-01-03"),
        ),
    ]


@pytest.mark.parametrize("build_fails", (False, True))
def test_valid_bar_index_build_counts_session_and_cache_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    build_fails: bool,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    active_indexes: list[materialization_planning.ExactPrimaryKeyIndex] = []
    build_observations: list[
        tuple[int, int, int, tuple[object, ...] | None, TimeWindow | None]
    ] = []
    original_index = materialization_planning.ExactPrimaryKeyIndex

    class RecordingIndex(original_index):
        def __enter__(self):  # noqa: ANN204
            opened = super().__enter__()
            active_indexes.append(self)
            return opened

        def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001, ANN204
            try:
                return super().__exit__(exc_type, exc_value, traceback)
            finally:
                active_indexes.remove(self)

    def read(*_args, columns, retained_payload_bytes, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if tuple(columns) == ("asset_id", "timestamp"):
            live_cache_bytes = sum(
                index.retained_memory_bytes for index in active_indexes
            )
            assert policy.reserved_memory_bytes >= live_cache_bytes
            build_observations.append(
                (
                    len(active_indexes),
                    retained_payload_bytes,
                    live_cache_bytes,
                    kwargs.get("asset_ids"),
                    kwargs.get("time_window"),
                )
            )
            if build_fails:

                def fail():  # noqa: ANN202
                    raise RuntimeError("valid_index_build_failed")
                    yield

                return fail()
            return iter(
                (
                    pd.DataFrame(
                        {
                            "asset_id": ["a"],
                            "timestamp": pd.to_datetime(["2024-01-02"]),
                        }
                    ),
                )
            )
        return iter((pd.DataFrame({"asset_id": ["a"]}),))

    monkeypatch.setattr(
        materialization_planning, "ExactPrimaryKeyIndex", RecordingIndex
    )
    monkeypatch.setattr(materialization_io, "iter_dataset_chunks_with_policy", read)
    batches = _scoped_active_materialization_batches(
        Path("panel"),
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        sessions=(pd.Timestamp("2024-01-02"),),
        resource_policy=policy,
        valid_bar_offsets=(0, 0),
        materialize=lambda assets, sessions, _session_context, _retained, _keys: (
            pd.DataFrame({"asset_id": assets, "timestamp": sessions})
        ),
    )

    if build_fails:
        with pytest.raises(RuntimeError, match="valid_index_build_failed"):
            list(batches)
    else:
        assert len(list(batches)) == 1

    expected_cache_bytes = original_index(
        primary_key=("asset_id", "timestamp"),
        timestamp_columns=("timestamp",),
        scratch_parent=Path(tempfile.gettempdir()),
    ).retained_memory_bytes
    assert len(build_observations) == 1
    active_count, retained_bytes, cache_bytes, asset_ids, time_window = (
        build_observations[0]
    )
    assert active_count == 2
    assert retained_bytes == 0
    assert cache_bytes == 2 * expected_cache_bytes
    assert asset_ids == ("a",)
    assert time_window == TimeWindow("2024-01-02", "2024-01-02")
    assert active_indexes == []


def test_active_materialization_indexes_use_dataset_sibling_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "processed" / "panel"
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    observed: list[Path] = []

    class Index:
        retained_memory_bytes = 0

        def __init__(self, *, scratch_parent: Path, **_kwargs: object) -> None:
            observed.append(scratch_parent)

        def __enter__(self) -> Index:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        @staticmethod
        def add_identity_payload_rows(**_kwargs: object) -> int:
            return 0

        @staticmethod
        def identity_payload_values(**_kwargs: object):  # noqa: ANN205
            return iter(())

    monkeypatch.setattr(materialization_planning, "ExactPrimaryKeyIndex", Index)
    monkeypatch.setattr(
        materialization_planning,
        "_build_valid_bar_key_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *_args, **_kwargs: iter((pd.DataFrame({"asset_id": ["asset"]}),)),
    )

    assert (
        list(
            _scoped_active_materialization_batches(
                root,
                time_window=TimeWindow("2024-01-02", "2024-01-02"),
                sessions=(pd.Timestamp("2024-01-02"),),
                resource_policy=policy,
                valid_bar_offsets=(0, 0),
                materialize=lambda *_args: pd.DataFrame(),
            )
        )
        == []
    )
    assert observed == [root.parent]


def test_a_share_direct_consumers_close_retained_iterators_on_all_exit_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    retained: list[object] = []
    closed: list[str] = []

    def iterator(name: str, frames: tuple[pd.DataFrame, ...]):
        def generate():
            try:
                yield from frames
            finally:
                closed.append(name)

        current = generate()
        retained.append(current)
        return current

    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(columns=("asset_id", "timestamp")),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: iterator(
            "normal", (pd.DataFrame({"timestamp": ["2024-01-02"]}),)
        ),
    )
    expected_sessions = pd.DatetimeIndex(("2024-01-02",))
    covered = np.zeros(1, dtype=np.bool_)
    materialization_planning._mark_dataset_session_coverage(
        Path("panel"),
        TimeWindow("2024-01-02", "2024-01-02"),
        expected_sessions=expected_sessions,
        covered=covered,
        retained_state_bytes=expected_sessions.nbytes + covered.nbytes,
        resource_policy=policy,
    )
    assert covered.all()

    market_type = next(iter(materialization_planning.CURRENT_MARKET_TYPES))
    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root, **_kwargs: SimpleNamespace(
            row_count=2,
            manifest={"file_size_bytes": 10**9},
        ),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: iterator(
            "early",
            (
                pd.DataFrame(
                    {
                        "source_market_type": [market_type, market_type],
                        "is_trading_day": [True, True],
                        "timestamp": pd.Series(
                            ["2024-01-02", "2024-01-03"],
                            dtype="datetime64[ns]",
                        ),
                    }
                ),
            ),
        ),
    )
    result = materialization_planning.available_trading_sessions(
        SimpleNamespace(interim_data_root=lambda *_args: Path("calendar")),
        "source",  # type: ignore[arg-type]
        time_window=TimeWindow("2024-01-02", "2024-01-02"),
        lookback_sessions=0,
        lookahead_sessions=0,
        resource_policy=policy,
    )
    assert not result.empty
    assert closed == ["normal", "early"]


def test_multi_processed_input_mapping_then_unknown_tradable_label_fails_closed(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = _processed_spec(MARKET_PANEL_DATA_ID, kind=ProcessedDataKind.PANEL)
    execution_spec = _processed_spec(
        "execution",
        kind=ProcessedDataKind.ANNOTATION,
    )
    target = _processed_spec(
        "tradable_forward_return",
        kind=ProcessedDataKind.LABEL,
        columns=(*platform_entity_time_key(), "tradable_forward_return_1d"),
        processed_input_data_ids=(MARKET_PANEL_DATA_ID, "execution"),
    )
    for spec in (panel_spec, execution_spec):
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            _key_frame(),
            spec=spec,
        )

    inputs = materialization_io.read_processed_inputs_for_spec(
        target,
        layout=layout,
        processed_catalog=_catalog(panel_spec, execution_spec, target),
        resource_policy=capture_test_policy(
            available_memory_bytes=1024**3,
        ),
    )

    assert set(inputs) == {MARKET_PANEL_DATA_ID, "execution"}
    with pytest.raises(
        ValueError,
        match="unsupported_label_family: tradable_forward_return",
    ):
        label_calculations.calculate_labels_for_panel(
            inputs[MARKET_PANEL_DATA_ID],
            definitions=(
                LabelDefinition(
                    label_id="tradable_forward_return_1d",
                    market_id=MarketId("a_share"),
                    family="tradable_forward_return",
                    input_columns=platform_entity_time_key(),
                    output_column="tradable_forward_return_1d",
                    label_timing="example_label_timing",
                    horizon=1,
                    lookahead=2,
                    entry_rule="example_entry_rule",
                    exit_rule="example_exit_rule",
                ),
            ),
        )


def test_source_input_root_keeps_one_policy_active_for_all_children(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    spec = _processed_spec(
        "target",
        kind=ProcessedDataKind.ANNOTATION,
        source_input_data_ids=("left", "right"),
    )
    observed: list[str] = []

    def read_child(root: Path, **_kwargs: object) -> pd.DataFrame:
        assert current_runtime_resource_policy() is policy
        observed.append(root.name)
        return _key_frame()

    monkeypatch.setattr(materialization_io, "read_materialized_dataset", read_child)

    inputs = materialization_io.read_source_inputs_for_spec(
        spec,
        layout=StorageLayout(project_root=tmp_path),
        source_id=SourceId("synthetic_vendor_a"),
        resource_policy=policy,
    )

    assert list(inputs) == ["left", "right"]
    assert observed == ["left", "right"]
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_source_input_root_retires_policy_after_every_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    primary = primary_type("source_child_primary")
    spec = _processed_spec(
        "target",
        kind=ProcessedDataKind.ANNOTATION,
        source_input_data_ids=("left", "right"),
    )

    def fail_child(*_args: object, **_kwargs: object) -> pd.DataFrame:
        assert current_runtime_resource_policy() is policy
        raise primary

    monkeypatch.setattr(materialization_io, "read_materialized_dataset", fail_child)

    with pytest.raises(primary_type) as raised:
        materialization_io.read_source_inputs_for_spec(
            spec,
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("synthetic_vendor_a"),
            resource_policy=policy,
        )
    assert raised.value is primary
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


@pytest.mark.parametrize(
    "existing",
    ((True, True), (False, False), (True, False)),
    ids=("two-existing", "two-missing", "mixed"),
)
def test_dependency_root_keeps_one_policy_active_for_all_children(
    existing: tuple[bool, bool],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    left = _processed_spec("left", kind=ProcessedDataKind.ANNOTATION)
    right = _processed_spec("right", kind=ProcessedDataKind.ANNOTATION)
    target = _processed_spec(
        "target",
        kind=ProcessedDataKind.ANNOTATION,
        processed_input_data_ids=("left", "right"),
    )
    layout = StorageLayout(project_root=tmp_path)
    for dependency, present in zip((left, right), existing, strict=True):
        if present:
            materialization_io.processed_root(layout, dependency).mkdir(parents=True)
    observed: list[tuple[str, str]] = []

    def require_coverage(root: Path, *_args: object, **_kwargs: object) -> None:
        assert current_runtime_resource_policy() is policy
        observed.append((root.name, "existing"))

    def materialize_one(dependency: ProcessedDataSpec, **_kwargs: object) -> None:
        assert current_runtime_resource_policy() is policy
        observed.append((dependency.data_id, "missing"))

    monkeypatch.setattr(
        materialization_dependencies.materialization_planning,
        "require_dataset_coverage",
        require_coverage,
    )

    materialization_dependencies.ensure_processed_dependencies(
        target,
        layout=layout,
        source_id=SourceId("synthetic_vendor_a"),
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        policy=policy,
        sessions=(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
        written={},
        processed_catalog=_catalog(left, right, target),
        dependency_windows={},
        materialize_one=materialize_one,
    )

    assert len(observed) == 2
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_dependency_root_retires_policy_after_every_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024**3,
    )
    primary = primary_type("dependency_child_primary")
    dependency = _processed_spec("left", kind=ProcessedDataKind.ANNOTATION)
    target = _processed_spec(
        "target",
        kind=ProcessedDataKind.ANNOTATION,
        processed_input_data_ids=("left",),
    )

    def fail_child(*_args: object, **_kwargs: object) -> None:
        assert current_runtime_resource_policy() is policy
        raise primary

    with pytest.raises(primary_type) as raised:
        materialization_dependencies.ensure_processed_dependencies(
            target,
            layout=StorageLayout(project_root=tmp_path),
            source_id=SourceId("synthetic_vendor_a"),
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            policy=policy,
            sessions=(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
            written={},
            processed_catalog=_catalog(dependency, target),
            dependency_windows={},
            materialize_one=fail_child,
        )
    assert raised.value is primary
    with pytest.raises(RuntimeError, match="runtime_resource_policy_retired"):
        with runtime_resource_policy_scope(policy):
            pass


def _catalog(*specs: ProcessedDataSpec) -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in specs:
        catalog.register(spec)
    return catalog


def _processed_spec(
    data_id: str,
    *,
    kind: ProcessedDataKind,
    columns: tuple[str, ...] = platform_entity_time_key(),
    source_input_data_ids: tuple[str, ...] = (),
    processed_input_data_ids: tuple[str, ...] = (),
) -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MarketId("a_share"),
        kind=kind,
        source_input_data_ids=source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=_roles(columns),
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _roles(columns: tuple[str, ...]) -> dict[str, ColumnRole]:
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    for column in columns:
        roles.setdefault(column, ColumnRole.CONTEXT)
    return roles


def _key_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
        }
    )


def _panel_frame_for_write_path() -> pd.DataFrame:
    return _key_frame().assign(
        adj_close=[10.0],
        adj_open=[10.0],
    )


def _write_interim_frame(
    layout: StorageLayout,
    source_id: SourceId,
    data_id: str,
) -> None:
    from quant_research_platform.contracts.data import InterimDataSpec

    spec = InterimDataSpec(
        data_id=data_id,
        source_id=source_id,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "source_asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
        },
        primary_key=("source_asset_id", "timestamp"),
        partition_columns=("year",),
    )
    write_dataset(
        layout.interim_data_root(source_id, data_id),
        pd.DataFrame(
            {
                "source_asset_id": ["asset_a"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
            }
        ),
        spec=spec,
    )


@pytest.mark.parametrize("enveloped", (False, True))
def test_sorted_output_releases_previous_publication_before_upstream_next(
    enveloped: bool,
) -> None:
    import weakref

    spec = _processed_spec("publication_liveness", kind=ProcessedDataKind.PANEL)
    first = _key_frame()
    second = _key_frame()
    second["timestamp"] += pd.Timedelta(days=1)
    first_reference = weakref.ref(first)

    def source():
        nonlocal first
        current = first
        first = None
        yield (dataset_writing.DatasetWriteChunk(current, 0) if enveloped else current)
        del current
        assert first_reference() is None
        yield second

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    output = iter(
        materialization_io.iter_sorted_output_chunks(
            source(), spec, resource_policy=policy
        )
    )
    first_output = next(output)
    del first_output
    next(output)


def test_output_session_filter_admits_range_index_copy_growth() -> None:
    timestamps = pd.date_range("2024-01-01", periods=102)
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["a_share"] * 102, dtype="str"),
            "asset_id": pd.Series(["asset"] * 102, dtype=object),
            "timestamp": timestamps,
            "eligible": True,
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    output = materialization_planning.filter_output_sessions(
        frame,
        timestamps.delete([1, 100]),
        resource_policy=policy,
    )

    assert output["timestamp"].tolist() == timestamps.delete([1, 100]).tolist()
    assert tuple(output.columns) == tuple(frame.columns)


def test_adaptive_materialization_does_not_retry_after_capacity_failure() -> None:
    calls = 0
    retained: list[int] = []

    def materialize(  # noqa: ANN001
        _assets, _sessions, _session_context, retained_payload_bytes
    ):
        nonlocal calls
        calls += 1
        retained.append(retained_payload_bytes)
        raise materialization_io.MaterializationCapacityExceeded("capacity")

    base = 100
    policy = capture_test_policy(available_memory_bytes=1024**3)
    with runtime_resource_policy_scope(policy):
        output = materialization_planning._iter_adaptive_materialization_batches(
            asset_ids=("a", "b"),
            sessions=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
            session_context=(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ),
            retained_payload_bytes=base,
            materialize=materialize,
        )
        with pytest.raises(
            materialization_io.MaterializationCapacityExceeded,
            match="a_share_known_payloads_exceed_task_memory_budget",
        ):
            next(output)
    assert calls == 1
    assert retained[0] == base


def _factor_definition(output_column: str, *, lookback_window: int) -> FactorDefinition:
    return FactorDefinition(
        factor_id=output_column,
        market_id=MarketId("a_share"),
        factor_set=FactorSet.STANDARD,
        input_columns=("adj_close",),
        output_column=output_column,
        family="price_volume",
        required_input_data_id=MARKET_PANEL_DATA_ID,
        lookback_window=lookback_window,
        window=lookback_window,
    )


def _label_definition(
    output_column: str,
    *,
    horizon: int,
    lookahead: int,
) -> LabelDefinition:
    return LabelDefinition(
        label_id=output_column,
        market_id=MarketId("a_share"),
        family="forward_return",
        input_columns=("market_id", "asset_id", "timestamp", "adj_open"),
        output_column=output_column,
        label_timing="example_label_timing",
        horizon=horizon,
        lookahead=lookahead,
        entry_rule="example_entry_rule",
        exit_rule="example_exit_rule",
    )


@pytest.mark.parametrize("value", (True, -1))
@pytest.mark.parametrize(
    "entrant",
    (
        "session_batch_window",
        "dependency_window",
        "valid_bar_coverage",
        "available_sessions",
        "active_batches",
    ),
)
def test_a_share_session_offset_entrants_require_exact_non_negative_int_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrant: str,
    value: object,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    sessions = pd.date_range("2024-01-02", periods=2)
    io_hits: list[str] = []

    if entrant == "session_batch_window":
        monkeypatch.setattr(
            materialization_planning,
            "bisect_left",
            lambda *_args: io_hits.append("bisect") or 0,
        )
        operation = lambda: materialization_planning.session_batch_window(  # noqa: E731
            sessions,
            (sessions[0],),
            warmup_sessions=value,  # type: ignore[arg-type]
        )
        error = "invalid_a_share_session_offset"
    elif entrant == "dependency_window":
        monkeypatch.setattr(
            materialization_dependencies,
            "bisect_left",
            lambda *_args: io_hits.append("bisect") or 0,
        )
        operation = lambda: materialization_dependencies._session_offset_time_window(  # noqa: E731
            sessions=sessions,
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            lookback_sessions=value,  # type: ignore[arg-type]
            lookahead_sessions=0,
        )
        error = "invalid_a_share_session_offset"
    elif entrant == "valid_bar_coverage":

        class IndexProbe:
            def identity_payload_values(self, **_kwargs):  # noqa: ANN202
                io_hits.append("index")
                return iter(())

        operation = lambda: materialization_planning.bounded_valid_bar_key_coverage(  # noqa: E731
            IndexProbe(),  # type: ignore[arg-type]
            asset_ids=("asset",),
            core_sessions=(sessions[0],),
            lookback_bars=value,  # type: ignore[arg-type]
            lookahead_bars=0,
            retained_payload_bytes=0,
            resource_policy=policy,
        )
        error = "invalid_valid_bar_offset"
    elif entrant == "available_sessions":
        monkeypatch.setattr(
            materialization_planning,
            "inspect_dataset",
            lambda *_args, **_kwargs: (
                io_hits.append("inspect")
                or pytest.fail("dataset_inspection_before_offset_gate")
            ),
        )
        operation = lambda: materialization_planning.available_trading_sessions(  # noqa: E731
            StorageLayout(project_root=tmp_path),
            SourceId("synthetic_vendor_a"),
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            lookback_sessions=value,  # type: ignore[arg-type]
            lookahead_sessions=0,
            resource_policy=policy,
        )
        error = "invalid_a_share_session_offset"
    else:
        root = tmp_path / "processed"
        root.mkdir()

        def materialize(*_args):  # noqa: ANN202
            io_hits.append("materialize")
            return pd.DataFrame()

        operation = lambda: next(  # noqa: E731
            _scoped_active_materialization_batches(
                root,
                time_window=TimeWindow("2024-01-02", "2024-01-03"),
                sessions=sessions,
                resource_policy=policy,
                valid_bar_offsets=(value, 0),  # type: ignore[arg-type]
                materialize=materialize,
            )
        )
        error = "invalid_valid_bar_offset"

    with pytest.raises(ValueError, match=f"^{error}$"):
        operation()

    assert io_hits == []
    assert not list(tmp_path.glob(".k-*"))
