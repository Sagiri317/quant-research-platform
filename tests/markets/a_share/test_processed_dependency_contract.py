from __future__ import annotations

from pathlib import Path

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
        "daily_panel": (
            ("daily_market", "stock_master", "market_cap", "trading_calendar"),
            (),
        ),
        "daily_context": ((), ("daily_panel",)),
        "a_share_default": ((), ("daily_panel",)),
        "a_share_as_of": ((), ("daily_panel",)),
        "a_share_execution": ((), ("daily_panel",)),
        "a_share_evaluation": ((), ("daily_panel",)),
        "price_volume": ((), ("daily_panel",)),
        "forward_returns": ((), ("daily_panel",)),
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


def test_future_tradable_forward_return_spec_depends_on_execution_mask() -> None:
    spec = _processed_spec(
        "tradable_forward_return",
        kind=ProcessedDataKind.LABEL,
        columns=(*platform_entity_time_key(), "tradable_forward_return_1d"),
        processed_input_data_ids=("daily_panel", "a_share_execution"),
    )

    assert spec.processed_input_data_ids == ("daily_panel", "a_share_execution")


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
