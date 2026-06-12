from __future__ import annotations

from dataclasses import fields

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition


def test_processed_data_spec_declares_direct_input_dependencies() -> None:
    spec = ProcessedDataSpec(
        data_id="daily_panel",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=(
            "daily_market",
            "stock_master",
            "market_cap",
            "trading_calendar",
        ),
        processed_input_data_ids=(),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )

    field_names = {field.name for field in fields(ProcessedDataSpec)}
    assert "source_input_data_ids" in field_names
    assert "processed_input_data_ids" in field_names
    assert spec.source_input_data_ids == (
        "daily_market",
        "stock_master",
        "market_cap",
        "trading_calendar",
    )
    assert spec.processed_input_data_ids == ()


def test_label_definition_declares_family_and_input_columns_only() -> None:
    label = LabelDefinition(
        label_id="forward_return_20d",
        market_id=MarketId("a_share"),
        family="forward_return",
        input_columns=("market_id", "asset_id", "timestamp", "adj_open"),
        output_column="forward_return_20d",
        label_timing="next_open_to_horizon_open",
        horizon=20,
        lookahead=21,
        entry_rule="next_session_open",
        exit_rule="horizon_session_open",
    )

    manifest = label.to_manifest()
    assert manifest["family"] == "forward_return"
    assert manifest["input_columns"] == [
        "market_id",
        "asset_id",
        "timestamp",
        "adj_open",
    ]
    assert "required_input_data_id" not in manifest
    assert not hasattr(label, "required_input_data_id")


def test_research_dataset_spec_requires_explicit_base_input() -> None:
    spec = ResearchDatasetSpec(
        data_id="demo",
        market_id=MarketId("dummy_market"),
        input_data_ids=("base", "features", "labels"),
        base_input_data_id="base",
        feature_columns=("feature_1",),
        label_columns=("label_1",),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "feature_1": ColumnRole.FEATURE,
            "label_1": ColumnRole.LABEL,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )

    field_names = {field.name for field in fields(ResearchDatasetSpec)}
    assert "base_input_data_id" in field_names
    assert spec.base_input_data_id == "base"
