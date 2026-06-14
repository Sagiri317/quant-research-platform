from __future__ import annotations

import inspect
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
from quant_research_platform.markets.calendars import SessionWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_a_share_processed_specs_declare_exact_direct_dependencies() -> None:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_catalog,
    )

    catalog = processed_catalog.a_share_processed_catalog()

    dependencies = {
        spec.data_id: (spec.source_input_data_ids, spec.processed_input_data_ids)
        for spec in catalog.list()
    }

    assert dependencies == {
        DAILY_PANEL_PROCESSED_DATA_ID: (
            ("daily_market", "stock_master", "market_cap", "trading_calendar"),
            (),
        ),
        DAILY_CONTEXT_PROCESSED_DATA_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        A_SHARE_DEFAULT_UNIVERSE_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        A_SHARE_AS_OF_MASK_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        A_SHARE_EXECUTION_MASK_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        A_SHARE_EVALUATION_MASK_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        PRICE_VOLUME_PROCESSED_DATA_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
        FORWARD_RETURNS_PROCESSED_DATA_ID: ((), (DAILY_PANEL_PROCESSED_DATA_ID,)),
    }


def test_materializer_reads_source_inputs_from_spec_source_input_data_ids(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    source_id = SourceId("synthetic_vendor_a")
    spec = _processed_spec(
        "daily_panel",
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=("daily_market", "trading_calendar"),
    )
    _write_interim_frame(layout, source_id, "daily_market")
    _write_interim_frame(layout, source_id, "trading_calendar")

    inputs = materializer._read_source_inputs_for_spec(
        spec,
        layout=layout,
        source_id=source_id,
    )

    assert set(inputs) == {"daily_market", "trading_calendar"}
    assert all(isinstance(frame, pd.DataFrame) for frame in inputs.values())


def test_materializer_reads_processed_inputs_from_spec_processed_input_data_ids(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = _processed_spec("daily_panel", kind=ProcessedDataKind.PANEL)
    target = _processed_spec(
        "price_volume",
        kind=ProcessedDataKind.FACTOR,
        processed_input_data_ids=("daily_panel",),
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

    inputs = materializer._read_processed_inputs_for_spec(
        target,
        layout=layout,
        processed_catalog=_catalog(panel_spec, target),
    )

    assert set(inputs) == {"daily_panel"}
    assert inputs["daily_panel"]["asset_id"].tolist() == ["asset_a"]


def test_materializer_validates_unknown_processed_dependency() -> None:
    target = _processed_spec(
        "price_volume",
        kind=ProcessedDataKind.FACTOR,
        processed_input_data_ids=("missing_panel",),
    )

    with pytest.raises(ValueError, match="unknown_processed_dependency: missing_panel"):
        materializer._validate_processed_dependency_graph(_catalog(target), (target,))


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
        materializer._validate_processed_dependency_graph(
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

    definitions = materializer._select_factor_definitions_for_spec(spec, catalog)

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

    definitions = materializer._select_label_definitions_for_spec(spec, catalog)

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
        processed_input_data_ids=("daily_panel", "a_share_execution"),
    )

    assert spec.processed_input_data_ids == ("daily_panel", "a_share_execution")


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
    calls = {"annotation": 0, "factor": 0, "label": 0}
    factor_window_calls = {"count": 0}
    label_window_calls = {"count": 0}

    def fake_planned_session_windows(*args: object, **kwargs: object):
        return (session_window,)

    def fake_inspect_dataset(*args: object, **kwargs: object):
        return SimpleNamespace(manifest={"file_size_bytes": 1}, row_count=1)

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
        return {"daily_panel": _panel_frame_for_write_path()}

    def fake_derive_annotations(
        panel: pd.DataFrame,
        *,
        data_ids: tuple[str, ...],
        **kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        output = _key_frame().assign(
            next_day_is_trading=[True],
            next_day_limit_status=[0],
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
        panel_keys: pd.DataFrame,
        *,
        definitions: tuple[FactorDefinition, ...],
        read_window: TimeWindow,
        core_sessions: tuple[pd.Timestamp, ...],
    ) -> TimeWindow:
        factor_window_calls["count"] += 1
        return read_window

    def fake_calculate_labels_for_panel(
        panel: pd.DataFrame,
        *,
        definitions: tuple[LabelDefinition, ...],
    ) -> pd.DataFrame:
        return _key_frame().assign(**{definitions[0].output_column: [0.01]})

    def fake_label_read_window_for_panel(
        panel_keys: pd.DataFrame,
        *,
        definitions: tuple[LabelDefinition, ...],
        read_window: TimeWindow,
        core_sessions: tuple[pd.Timestamp, ...],
    ) -> TimeWindow:
        label_window_calls["count"] += 1
        return read_window

    monkeypatch.setattr(
        materializer,
        "_planned_session_windows",
        fake_planned_session_windows,
    )
    monkeypatch.setattr(materializer, "inspect_dataset", fake_inspect_dataset)
    monkeypatch.setattr(materializer, "_write_and_inspect", fake_write_and_inspect)
    monkeypatch.setattr(
        materializer,
        "_read_processed_inputs_for_spec",
        fake_read_processed_inputs,
    )
    monkeypatch.setattr(
        materializer,
        "read_dataset",
        lambda *args, **kwargs: _key_frame(),
    )
    monkeypatch.setattr(
        materializer.annotations,
        "derive_annotations",
        fake_derive_annotations,
    )
    monkeypatch.setattr(
        materializer.factor_calculations,
        "calculate_factors_for_panel",
        fake_calculate_factors_for_panel,
    )
    monkeypatch.setattr(
        materializer.factor_calculations,
        "factor_read_window_for_panel",
        fake_factor_read_window_for_panel,
    )
    monkeypatch.setattr(
        materializer.label_calculations,
        "calculate_labels_for_panel",
        fake_calculate_labels_for_panel,
    )
    monkeypatch.setattr(
        materializer.label_calculations,
        "label_read_window_for_panel",
        fake_label_read_window_for_panel,
    )
    monkeypatch.setattr(
        materializer,
        "_estimate_factor_bytes_per_session",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        materializer,
        "_estimate_label_bytes_per_session",
        lambda *args, **kwargs: 1,
    )

    materializer._write_annotation(
        _processed_spec(
            "a_share_execution",
            kind=ProcessedDataKind.ANNOTATION,
            columns=(
                *platform_entity_time_key(),
                "next_day_is_trading",
                "next_day_limit_status",
                "can_enter",
                "can_exit",
            ),
            processed_input_data_ids=("daily_panel",),
        ),
        layout=layout,
        time_window=window,
        policy=SimpleNamespace(),
        sessions=sessions,
        written={},
    )
    materializer._write_factors(
        _processed_spec(
            "price_volume",
            kind=ProcessedDataKind.FACTOR,
            columns=(*platform_entity_time_key(), "mom_20d"),
            processed_input_data_ids=("daily_panel",),
        ),
        layout=layout,
        time_window=window,
        policy=SimpleNamespace(),
        sessions=sessions,
        written={},
    )
    materializer._write_labels(
        _processed_spec(
            "forward_returns",
            kind=ProcessedDataKind.LABEL,
            columns=(*platform_entity_time_key(), "forward_return_20d"),
            processed_input_data_ids=("daily_panel",),
        ),
        layout=layout,
        time_window=window,
        policy=SimpleNamespace(),
        sessions=sessions,
        written={},
    )

    assert calls == {"annotation": 1, "factor": 1, "label": 1}
    assert factor_window_calls == {"count": 1}
    assert label_window_calls == {"count": 1}


def test_multi_processed_input_mapping_then_unknown_tradable_label_fails_closed(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    panel_spec = _processed_spec("daily_panel", kind=ProcessedDataKind.PANEL)
    execution_spec = _processed_spec(
        "a_share_execution",
        kind=ProcessedDataKind.ANNOTATION,
    )
    target = _processed_spec(
        "tradable_forward_return",
        kind=ProcessedDataKind.LABEL,
        columns=(*platform_entity_time_key(), "tradable_forward_return_1d"),
        processed_input_data_ids=("daily_panel", "a_share_execution"),
    )
    for spec in (panel_spec, execution_spec):
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            _key_frame(),
            spec=spec,
        )

    inputs = materializer._read_processed_inputs_for_spec(
        target,
        layout=layout,
        processed_catalog=_catalog(panel_spec, execution_spec, target),
    )

    assert set(inputs) == {"daily_panel", "a_share_execution"}
    with pytest.raises(
        ValueError,
        match="unsupported_a_share_label_family: tradable_forward_return",
    ):
        label_calculations.calculate_labels_for_panel(
            inputs["daily_panel"],
            definitions=(
                LabelDefinition(
                    label_id="tradable_forward_return_1d",
                    market_id=MarketId("a_share"),
                    family="tradable_forward_return",
                    input_columns=platform_entity_time_key(),
                    output_column="tradable_forward_return_1d",
                    label_timing="next_open_to_horizon_open",
                    horizon=1,
                    lookahead=2,
                    entry_rule="next_session_open",
                    exit_rule="horizon_session_open",
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
        required_input_data_id="daily_panel",
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
        label_timing="next_open_to_horizon_open",
        horizon=horizon,
        lookahead=lookahead,
        entry_rule="next_session_open",
        exit_rule="horizon_session_open",
    )
