from __future__ import annotations

import pytest

from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_field_catalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_factor_catalog,
    a_share_field_catalog,
    a_share_price_volume_factor_columns,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    PRICE_VOLUME_PROCESSED_DATA_ID,
)


def test_platform_field_catalog_does_not_own_a_share_specific_fields() -> None:
    platform_fields = set(platform_field_catalog().list_names())

    assert not {
        "is_st",
        "is_pt",
        "limit_status",
        "can_enter",
        "can_exit",
        "eligible",
        "market_type",
        "float_market_cap",
        "total_market_cap",
    } & platform_fields


def test_a_share_field_catalog_owns_market_specific_fields() -> None:
    catalog = a_share_field_catalog()

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


def test_a_share_processed_specs_use_canonical_entity_time_key() -> None:
    catalog = a_share_processed_catalog()

    for spec in catalog.list():
        assert spec.market_id == "a_share"
        assert spec.primary_key == platform_entity_time_key()
        assert spec.primary_key == ("market_id", "asset_id", "timestamp")
        assert spec.column_roles["market_id"] is ColumnRole.KEY
        assert spec.column_roles["asset_id"] is ColumnRole.KEY
        assert spec.column_roles["timestamp"] is ColumnRole.TIMESTAMP


def test_a_share_price_volume_spec_is_value_only_and_catalog_driven() -> None:
    catalog = a_share_processed_catalog()
    spec = catalog.get(PRICE_VOLUME_PROCESSED_DATA_ID)
    factor_columns = [
        column
        for column, role in spec.column_roles.items()
        if role is ColumnRole.FACTOR
    ]
    registered_columns = [
        definition.output_column for definition in a_share_factor_catalog().list()
    ]

    with pytest.raises(KeyError, match="unknown_processed_data_spec"):
        catalog.get("price_volume" + "_56")
    assert tuple(factor_columns) == a_share_price_volume_factor_columns()
    assert tuple(registered_columns) == a_share_price_volume_factor_columns()
    assert spec.required_columns == (
        *platform_entity_time_key(),
        *a_share_price_volume_factor_columns(),
    )
    assert "mom_20d" in factor_columns
    assert "log_float_market_cap" in factor_columns
    assert "forward_return_20d" not in factor_columns
    assert not {
        "stock_code",
        "trade_date",
        "eligible",
        "eligible_after_listing_age",
        "can_enter",
        "can_exit",
        "amount",
        "market_cap",
        "turnover",
    }.intersection(spec.required_columns)


def test_a_share_label_mask_and_universe_specs_are_queryable() -> None:
    catalog = a_share_processed_catalog()

    labels = catalog.get("forward_returns")
    universe = catalog.get("a_share_default")
    as_of_mask = catalog.get("a_share_as_of")
    execution_mask = catalog.get("a_share_execution")
    evaluation_mask = catalog.get("a_share_evaluation")

    assert labels.category == "labels"
    assert labels.column_roles["forward_return_20d"] is ColumnRole.LABEL
    assert universe.category == "universes"
    assert universe.column_roles["eligible"] is ColumnRole.UNIVERSE
    assert as_of_mask.category == "masks/as_of"
    assert as_of_mask.column_roles["eligible"] is ColumnRole.MASK
    assert execution_mask.category == "masks/execution"
    assert execution_mask.column_roles["can_enter"] is ColumnRole.MASK
    assert execution_mask.column_roles["can_exit"] is ColumnRole.MASK
    assert evaluation_mask.category == "masks/evaluation"
    assert evaluation_mask.column_roles["eligible"] is ColumnRole.MASK
    assert evaluation_mask.column_roles["can_exit"] is ColumnRole.MASK
