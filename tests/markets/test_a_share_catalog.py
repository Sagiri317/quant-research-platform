from __future__ import annotations

import pytest

from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_field_catalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    field_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    AS_OF_MASK_ID,
    DEFAULT_UNIVERSE_ID,
    EVALUATION_MASK_ID,
    EXECUTION_MASK_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID,
    FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    FUNDAMENTAL_RAW_PROCESSED_DATA_ID,
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.research_datasets import (
    build_research_dataset_catalog,
)


def test_platform_field_catalog_does_not_own_a_share_specific_fields() -> None:
    platform_fields = {spec.name for spec in platform_field_catalog().list()}

    assert (
        not {
            "is_st",
            "is_pt",
            "limit_status",
            "can_enter",
            "can_exit",
            "eligible",
            "market_type",
            "float_market_cap",
            "total_market_cap",
        }
        & platform_fields
    )


def test_field_catalog_owns_market_specific_fields() -> None:
    catalog = field_catalog()

    for field_name in (
        "is_st",
        "is_pt",
        "limit_status",
        "can_enter",
        "can_exit",
        "eligible",
        "market_type",
        "float_market_cap",
        "total_market_cap",
    ):
        assert catalog.get(field_name).name == field_name

    assert catalog.get("timestamp").role is ColumnRole.TIMESTAMP
    with pytest.raises(KeyError, match="unknown_field"):
        catalog.get("unknown")


def test_a_share_source_code_fields_are_numeric_null_preserving() -> None:
    catalog = field_catalog()

    for field_name in (
        "market_type",
        "trading_status_code",
        "filling_flag",
        "limit_status",
        "next_bar_limit_status",
    ):
        assert catalog.get(field_name).dtype == "float64"


def test_a_share_processed_specs_use_kind_and_canonical_entity_time_key() -> None:
    catalog = processed_catalog()

    for spec in catalog.list():
        assert spec.market_id == "a_share"
        assert spec.primary_key == platform_entity_time_key()
        assert spec.primary_key == ("market_id", "asset_id", "timestamp")
        assert spec.column_roles["market_id"] is ColumnRole.KEY
        assert spec.column_roles["asset_id"] is ColumnRole.KEY
        assert spec.column_roles["timestamp"] is ColumnRole.TIMESTAMP


def test_a_share_price_volume_spec_is_value_only_and_catalog_driven() -> None:
    catalog = processed_catalog()
    spec = catalog.get(PRICE_VOLUME_PROCESSED_DATA_ID)
    factor_columns = tuple(
        column
        for column, role in spec.column_roles.items()
        if role is ColumnRole.FACTOR
    )
    registered_columns = tuple(
        definition.output_column
        for definition in factor_definitions.factor_catalog().list()
        if definition.family == "price_volume"
    )

    with pytest.raises(KeyError, match="unknown_processed_data_spec"):
        catalog.get("unknown")
    assert spec.kind is ProcessedDataKind.FACTOR
    assert factor_columns == _a_share_factor_columns()
    assert registered_columns == (_a_share_factor_columns())
    assert spec.required_columns == (
        *platform_entity_time_key(),
        *_a_share_factor_columns(),
    )
    assert "mom_20d" in factor_columns
    assert "log_float_market_cap" in factor_columns


def test_a_share_processed_annotation_contracts_are_role_based() -> None:
    catalog = processed_catalog()

    market_panel_daily = catalog.get(market_panel_data_id())
    context = catalog.get(market_context_data_id())
    labels = catalog.get(FORWARD_RETURNS_PROCESSED_DATA_ID)
    universe = catalog.get(DEFAULT_UNIVERSE_ID)
    as_of_mask = catalog.get(AS_OF_MASK_ID)
    execution_mask = catalog.get(EXECUTION_MASK_ID)
    evaluation_mask = catalog.get(EVALUATION_MASK_ID)

    assert market_panel_daily.kind is ProcessedDataKind.PANEL
    assert {"session_date", "available_at"}.issubset(
        market_panel_daily.required_columns
    )
    assert context.kind is ProcessedDataKind.ANNOTATION
    assert context.column_roles["market_type"] is ColumnRole.CONTEXT
    assert labels.kind is ProcessedDataKind.LABEL
    assert labels.column_roles["forward_return_20d"] is ColumnRole.LABEL
    assert universe.kind is ProcessedDataKind.ANNOTATION
    assert universe.column_roles["eligible"] is ColumnRole.UNIVERSE
    assert as_of_mask.kind is ProcessedDataKind.ANNOTATION
    assert as_of_mask.column_roles["eligible"] is ColumnRole.MASK
    assert execution_mask.kind is ProcessedDataKind.ANNOTATION
    assert execution_mask.column_roles["can_enter"] is ColumnRole.MASK
    assert execution_mask.column_roles["can_exit"] is ColumnRole.MASK
    assert evaluation_mask.kind is ProcessedDataKind.ANNOTATION
    assert evaluation_mask.column_roles["eligible"] is ColumnRole.MASK
    assert evaluation_mask.column_roles["can_exit"] is ColumnRole.MASK
    pit = catalog.get(FUNDAMENTAL_PIT_PROCESSED_DATA_ID)
    raw = catalog.get(FUNDAMENTAL_RAW_PROCESSED_DATA_ID)
    assert pit.kind is ProcessedDataKind.ANNOTATION
    assert raw.kind is ProcessedDataKind.FACTOR
    assert raw.processed_input_data_ids == (FUNDAMENTAL_PIT_PROCESSED_DATA_ID,)
    fundamental_dataset = build_research_dataset_catalog().get(
        FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID
    )
    assert fundamental_dataset.input_data_ids[0] == FUNDAMENTAL_RAW_PROCESSED_DATA_ID


def test_a_share_research_dataset_carries_input_data_ids_not_source_columns() -> None:
    dataset = build_research_dataset_catalog().get(
        PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )

    assert dataset.input_data_ids == (
        PRICE_VOLUME_PROCESSED_DATA_ID,
        FORWARD_RETURNS_PROCESSED_DATA_ID,
        market_context_data_id(),
        DEFAULT_UNIVERSE_ID,
        EXECUTION_MASK_ID,
    )
    assert dataset.required_columns == (
        *platform_entity_time_key(),
        *dataset.factor_columns,
        *dataset.feature_columns,
        *dataset.label_columns,
        *dataset.filter_columns,
        *dataset.context_columns,
    )
    assert not {
        "stock_code",
        "trade_date",
        "index_code",
        "source_id",
        "source_asset_id",
    }.intersection(dataset.required_columns)


def _a_share_factor_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in factor_definitions.factor_definitions()
        if definition.family == "price_volume"
    )
