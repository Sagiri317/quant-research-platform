from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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
    materialization_io,
    materialization_planning,
    materialization_validation,
    materialize_annotations,
    materialize_factors,
    materialize_labels,
    materializer,
    source_ids,
)
from quant_research_platform.markets.session_windows import SessionWindow
from quant_research_platform.research.factors.price_volume import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.sources.csmar import ids as csmar_ids
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

MARKET_BARS_DATA_ID = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
MARKET_PANEL_DATA_ID = market_panel_data_id()
MARKET_CONTEXT_DATA_ID = market_context_data_id()
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)
FORWARD_RETURN_20D_COLUMN = forward_return_label_column(20)


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
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=1024**3,
            environ={},
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
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=32 * 1024,
        environ={},
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
    assert observed_targets == [policy.payload_target_bytes]


def test_materialized_read_preserves_target_and_reserves_retained_owner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )
    observed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        materialization_io,
        "open_dataset_session",
        lambda root, **_kwargs: nullcontext(root),
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
    assert observed == [(policy.payload_target_bytes, available)]


def test_materialized_read_rejects_exhausted_retained_state_before_decoder() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
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
            resource_policy=RuntimeResourcePolicy.from_environment(
                available_memory_bytes=1024**3,
                environ={},
            ),
            retained_payload_bytes=-1,
        )


def test_chunk_target_rejection_happens_before_dataset_session_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
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
        target_chunk_bytes=policy.payload_target_bytes + 1,
    )

    with pytest.raises(
        ValueError,
        match="a_share_indivisible_market_unit_exceeds_payload_target",
    ):
        next(chunks)

    assert acquisitions == 0


@pytest.mark.parametrize("target_chunk_bytes", (0, -1, False))
def test_invalid_chunk_target_rejection_happens_before_dataset_session_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    target_chunk_bytes: object,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
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


def test_a_share_indivisible_output_rejects_payload_over_target() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=100,
        environ={},
    )
    output = pd.DataFrame({"value": ["x" * 256]})

    with pytest.raises(
        ValueError,
        match="a_share_indivisible_market_unit_exceeds_payload_target",
    ):
        materialization_planning.require_known_materialization_payload_bound(
            frames=(),
            output=output,
            resource_policy=policy,
        )


def test_a_share_sizing_probe_materializes_one_core_session_at_a_time() -> None:
    sessions = tuple(pd.date_range("2026-01-01", periods=30, freq="D"))

    windows = materialization_planning._sample_windows(
        sessions,
        time_window=TimeWindow("2026-01-01", "2026-01-30"),
        warmup_sessions=2,
        lookahead_sessions=1,
        reason="sizing_test",
    )

    assert len(windows) == materialization_planning.SIZING_SAMPLE_SESSION_CAP
    assert all(len(window.core_sessions) == 1 for window in windows)


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
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=1024**3,
            environ={},
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


def test_materializer_delegates_factor_label_calculation_details() -> None:
    source = inspect.getsource(materializer)

    assert "forward_returns" not in source
    assert "_forward_return_valid_bar_read_window" not in source
    assert "_factor_read_window" not in source
    assert "_factor_warmup_sessions" not in source


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
    session_window = SessionWindow(
        core=window,
        read=window,
        core_sessions=sessions,
        read_sessions=sessions,
        reason="test",
    )
    calls = {
        "annotation": 0,
        "factor": 0,
        "factor_key_coverage": 0,
        "label": 0,
        "label_key_coverage": 0,
    }

    def fakeplanned_session_windows(*args: object, **kwargs: object):
        return (session_window,)

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
        assert panel_keys == {"asset_a": (pd.Timestamp("2024-01-02"),)}
        calls["factor_key_coverage"] += 1
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
        assert panel_keys == {"asset_a": (pd.Timestamp("2024-01-02"),)}
        calls["label_key_coverage"] += 1
        return read_window

    def iter_dataset_chunks_with_policy(*args: object, **kwargs: object):
        del args, kwargs
        return (_key_frame() for _ in range(1))

    monkeypatch.setattr(
        materialization_planning,
        "planned_session_windows",
        fakeplanned_session_windows,
    )
    for module in (materialize_annotations, materialize_factors, materialize_labels):
        monkeypatch.setattr(module, "inspect_dataset", fake_inspect_dataset)
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
    monkeypatch.setattr(
        materialization_planning,
        "estimate_factor_bytes_per_session",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        materialization_planning,
        "estimate_label_bytes_per_session",
        lambda *args, **kwargs: 1,
    )
    resource_policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3,
        environ={},
    )

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
        "factor_key_coverage": 1,
        "label": 1,
        "label_key_coverage": 1,
    }


def test_processed_panel_key_coverage_rejects_invalid_keys(monkeypatch) -> None:
    monkeypatch.setattr(
        materialization_planning,
        "inspect_dataset",
        lambda _root: SimpleNamespace(columns=("asset_id", "timestamp")),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: (
            frame
            for frame in (
                pd.DataFrame({"asset_id": [[]], "timestamp": ["2024-01-02"]}),
            )
        ),
    )

    with pytest.raises(ValueError, match="invalid_processed_panel_asset_id"):
        materialization_planning.processed_panel_key_coverage(
            Path("panel"),
            resource_policy=RuntimeResourcePolicy.from_environment(
                available_memory_bytes=1024**3,
                environ={},
            ),
        )

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: (
            frame
            for frame in (
                pd.DataFrame(
                    {"asset_id": ["asset_a"], "timestamp": ["not-a-date"]}
                ),
            )
        ),
    )

    with pytest.raises(ValueError, match="invalid_processed_panel_timestamp"):
        materialization_planning.processed_panel_key_coverage(
            Path("panel"),
            resource_policy=RuntimeResourcePolicy.from_environment(
                available_memory_bytes=1024**3,
                environ={},
            ),
        )


def test_a_share_direct_dataset_consumers_use_deterministic_iterator_closing() -> None:
    functions = (
        materialization_planning._dataset_session_dates,
        materialization_planning.processed_panel_key_coverage,
        materialization_planning.available_trading_sessions,
    )

    for function in functions:
        source = inspect.getsource(function)
        assert "with closing(" in source


def test_a_share_direct_consumers_close_retained_iterators_on_all_exit_kinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024**3, environ={}
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
        lambda _root: SimpleNamespace(columns=("asset_id", "timestamp")),
    )
    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: iterator(
            "normal", (pd.DataFrame({"timestamp": ["2024-01-02"]}),)
        ),
    )
    assert materialization_planning._dataset_session_dates(
        Path("panel"), TimeWindow("2024-01-02", "2024-01-02"), resource_policy=policy
    )

    monkeypatch.setattr(
        materialization_io,
        "iter_dataset_chunks_with_policy",
        lambda *args, **kwargs: iterator(
            "error",
            (pd.DataFrame({"asset_id": [[]], "timestamp": ["2024-01-02"]}),),
        ),
    )
    with pytest.raises(ValueError, match="invalid_processed_panel_asset_id"):
        materialization_planning.processed_panel_key_coverage(
            Path("panel"), resource_policy=policy
        )

    market_type = next(iter(materialization_planning.CURRENT_MARKET_TYPES))
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
                        "timestamp": ["2024-01-02", "2024-01-03"],
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
    assert result
    assert closed == ["normal", "error", "early"]


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
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=1024**3,
            environ={},
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
