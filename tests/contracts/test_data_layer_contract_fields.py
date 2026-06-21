from __future__ import annotations

from dataclasses import fields

import pytest

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition
from tests.helpers import frequency_ids


def test_processed_data_spec_declares_direct_input_dependencies() -> None:
    spec = ProcessedDataSpec(
        data_id=frequency_ids.MARKET_PANEL_DATA_ID,
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.PANEL,
        source_input_data_ids=(
            frequency_ids.MARKET_BARS_DATA_ID,
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
        frequency_ids.MARKET_BARS_DATA_ID,
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
        label_timing="example_label_timing",
        horizon=20,
        lookahead=21,
        entry_rule="example_entry_rule",
        exit_rule="example_exit_rule",
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


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_error"),
    [
        ("label_id", "", "label_id_required"),
        ("family", "", "label_family_required"),
        ("input_columns", (), "label_input_columns_required"),
        ("input_columns", ("market_id", ""), "label_input_column_required"),
        ("output_column", "", "label_output_column_required"),
        ("label_timing", "", "label_timing_required"),
        ("horizon", 0, "label_horizon_must_be_positive"),
        ("lookahead", 0, "label_lookahead_must_be_positive"),
        ("entry_rule", "", "label_entry_rule_required"),
        ("exit_rule", "", "label_exit_rule_required"),
        ("definition_version", "", "label_definition_version_required"),
    ],
)
def test_label_definition_fails_closed_on_invalid_contract_fields(
    field_name: str,
    bad_value: object,
    expected_error: str,
) -> None:
    kwargs = {
        "label_id": "forward_return_20d",
        "market_id": MarketId("a_share"),
        "family": "forward_return",
        "input_columns": ("market_id", "asset_id", "timestamp", "adj_open"),
        "output_column": "forward_return_20d",
        "label_timing": "example_label_timing",
        "horizon": 20,
        "lookahead": 21,
        "entry_rule": "example_entry_rule",
        "exit_rule": "example_exit_rule",
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match=expected_error):
        LabelDefinition(**kwargs)


def test_label_definition_rejects_unsupported_definition_version() -> None:
    with pytest.raises(ValueError, match="unsupported_label_definition_version"):
        LabelDefinition(
            label_id="forward_return_20d",
            market_id=MarketId("a_share"),
            family="forward_return",
            input_columns=("market_id", "asset_id", "timestamp", "adj_open"),
            output_column="forward_return_20d",
            label_timing="example_label_timing",
            horizon=20,
            lookahead=21,
            entry_rule="example_entry_rule",
            exit_rule="example_exit_rule",
            definition_version="9.9",
        )


def test_research_dataset_spec_requires_explicit_base_input() -> None:
    spec = ResearchDatasetSpec(
        data_id="demo",
        market_id=MarketId("dummy_market"),
        input_data_ids=("base", "features", "labels"),
        base_input_data_id="base",
        factor_columns=("factor_1",),
        feature_columns=("feature_1",),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="feature_1_from_factor_1",
                source_columns=("factor_1",),
                output_columns=("feature_1",),
                transform="cross_sectional_zscore",
            ),
        ),
        label_columns=("label_1",),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "factor_1": ColumnRole.FACTOR,
            "feature_1": ColumnRole.FEATURE,
            "label_1": ColumnRole.LABEL,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )

    field_names = {field.name for field in fields(ResearchDatasetSpec)}
    assert "base_input_data_id" in field_names
    assert spec.base_input_data_id == "base"
