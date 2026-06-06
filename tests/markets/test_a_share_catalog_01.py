from __future__ import annotations

import numpy as np
import pandas as pd
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
    a_share_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.validation import (
    factor_tables,
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
    for retired_alias in ("stock_code", "trade_date", "index_code"):
        with pytest.raises(KeyError, match="unknown_field"):
            catalog.get(retired_alias)


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
        "index_code",
        "source_id",
        "source_asset_id",
        "session_date",
        "available_at",
        "eligible",
        "eligible_after_listing_age",
        "can_enter",
        "can_exit",
        "amount",
        "market_cap",
        "turnover",
    }.intersection(spec.required_columns)


def test_a_share_factor_validator_blocks_mask_columns() -> None:
    factor_column = a_share_price_volume_factor_columns()[0]
    report = factor_tables.check_factor_table(
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["000001"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
                factor_column: [1.0],
                "eligible": [True],
                "can_enter": [True],
                "can_exit": [True],
            }
        )
    )

    assert any(
        issue.check == "panel_only_columns_in_factor_table"
        and issue.severity == "blocking"
        and issue.count == 3
        for issue in report.issues
    )
    assert "eligible_counts" not in report.stats
    assert "can_enter_counts" not in report.stats
    assert "can_exit_counts" not in report.stats


def test_a_share_factor_validator_requires_exact_registered_columns() -> None:
    factor_columns = a_share_price_volume_factor_columns()
    first_factor = factor_columns[0]
    missing_factor = factor_columns[-1]
    base = {
        "market_id": ["a_share"],
        "asset_id": ["000001"],
        "timestamp": pd.to_datetime(["2024-01-02"]),
        first_factor: [1.0],
    }

    missing_report = factor_tables.check_factor_table(pd.DataFrame(base))
    extra_report = factor_tables.check_factor_table(
        pd.DataFrame({**base, missing_factor: [2.0], "unknown_factor": [3.0]})
    )

    assert any(
        issue.check == "missing_required_columns"
        and missing_factor in issue.message
        for issue in missing_report.issues
    )
    assert any(
        issue.check == "unexpected_columns_in_factor_table"
        and "unknown_factor" in issue.message
        for issue in extra_report.issues
    )


def test_a_share_factor_validator_rejects_retired_mixed_schema() -> None:
    factor_column = a_share_price_volume_factor_columns()[0]
    report = factor_tables.check_factor_table(
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "stock_code": ["000001"],
                "trade_date": pd.to_datetime(["2024-01-02"]),
                factor_column: [1.0],
            }
        )
    )

    assert report.min_date is None
    assert report.max_date is None
    assert any(
        issue.check == "missing_required_columns"
        and "asset_id" in issue.message
        and "timestamp" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.check == "unexpected_columns_in_factor_table"
        and "stock_code" in issue.message
        and "trade_date" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.check == "panel_only_columns_in_factor_table"
        and issue.count == 2
        for issue in report.issues
    )


def test_a_share_label_validator_requires_exact_registered_columns() -> None:
    label_columns = (
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "forward_return_60d",
        "forward_return_120d",
    )
    base = {
        "market_id": ["a_share"],
        "asset_id": ["000001"],
        "timestamp": pd.to_datetime(["2024-01-02"]),
        **{column: [0.01] for column in label_columns[:-1]},
    }

    missing_report = factor_tables.check_label_table(pd.DataFrame(base))
    extra_report = factor_tables.check_label_table(
        pd.DataFrame(
            {
                **base,
                label_columns[-1]: [0.02],
                "eligible": [True],
                "unknown_label": [0.0],
            }
        )
    )

    assert any(
        issue.check == "missing_required_columns"
        and label_columns[-1] in issue.message
        for issue in missing_report.issues
    )
    assert any(
        issue.check == "unexpected_columns_in_label_table"
        and "unknown_label" in issue.message
        and "eligible" in issue.message
        for issue in extra_report.issues
    )


def test_a_share_factor_and_label_validators_reject_infinite_numeric_outputs() -> None:
    factor_frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            **{
                column: [0.0]
                for column in a_share_price_volume_factor_columns()
            },
        }
    )
    label_columns = (
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "forward_return_60d",
        "forward_return_120d",
    )
    label_frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            **{column: [0.0] for column in label_columns},
        }
    )
    factor_frame.loc[0, "mom_20d"] = np.inf
    label_frame.loc[0, "forward_return_20d"] = -np.inf

    factor_report = factor_tables.check_factor_table(factor_frame)
    label_report = factor_tables.check_label_table(label_frame)

    assert any(
        issue.check == "non_finite_factor_values" and issue.count == 1
        for issue in factor_report.issues
    )
    assert any(
        issue.check == "non_finite_label_values" and issue.count == 1
        for issue in label_report.issues
    )


def test_a_share_label_validator_rejects_retired_mixed_schema() -> None:
    report = factor_tables.check_label_table(
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "stock_code": ["000001"],
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "forward_return_20d": [0.01],
            }
        )
    )

    assert report.min_date is None
    assert report.max_date is None
    assert any(
        issue.check == "missing_required_columns"
        and "asset_id" in issue.message
        and "timestamp" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.check == "unexpected_columns_in_label_table"
        and "stock_code" in issue.message
        and "trade_date" in issue.message
        for issue in report.issues
    )


def test_a_share_final_processed_artifact_column_contracts_are_exact() -> None:
    catalog = a_share_processed_catalog()

    daily_panel = catalog.get(DAILY_PANEL_PROCESSED_DATA_ID)
    context = catalog.get(DAILY_CONTEXT_PROCESSED_DATA_ID)
    labels = catalog.get(FORWARD_RETURNS_PROCESSED_DATA_ID)
    universe = catalog.get(A_SHARE_DEFAULT_UNIVERSE_ID)
    as_of_mask = catalog.get(A_SHARE_AS_OF_MASK_ID)
    execution_mask = catalog.get(A_SHARE_EXECUTION_MASK_ID)
    evaluation_mask = catalog.get(A_SHARE_EVALUATION_MASK_ID)
    forbidden_after_interim = {
        "stock_code",
        "trade_date",
        "index_code",
        "source_id",
        "source_asset_id",
    }

    assert daily_panel.category == "panels"
    assert {"session_date", "available_at"}.issubset(daily_panel.required_columns)
    assert forbidden_after_interim.isdisjoint(daily_panel.required_columns)
    assert context.required_columns == (
        *platform_entity_time_key(),
        "market_type",
        "is_st",
        "is_pt",
        "limit_status",
        "listing_age_trading_days",
        "csrc_2012_industry_code",
        "csrc_2012_industry_name",
        "total_market_cap",
        "float_market_cap",
        "amount",
        "turnover",
    )
    assert not {
        *forbidden_after_interim,
        "session_date",
        "available_at",
        "eligible",
        "can_enter",
        "can_exit",
    }.intersection(context.required_columns)
    assert labels.category == "labels"
    assert labels.required_columns == (
        *platform_entity_time_key(),
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "forward_return_60d",
        "forward_return_120d",
    )
    assert labels.column_roles["forward_return_20d"] is ColumnRole.LABEL
    assert universe.category == "universes"
    assert universe.required_columns == (*platform_entity_time_key(), "eligible")
    assert universe.column_roles["eligible"] is ColumnRole.UNIVERSE
    assert as_of_mask.category == "masks/as_of"
    assert as_of_mask.required_columns == (*platform_entity_time_key(), "eligible")
    assert as_of_mask.column_roles["eligible"] is ColumnRole.MASK
    assert execution_mask.category == "masks/execution"
    assert execution_mask.required_columns == (
        *platform_entity_time_key(),
        "can_enter",
        "can_exit",
    )
    assert execution_mask.column_roles["can_enter"] is ColumnRole.MASK
    assert execution_mask.column_roles["can_exit"] is ColumnRole.MASK
    assert evaluation_mask.category == "masks/evaluation"
    assert evaluation_mask.required_columns == (
        *platform_entity_time_key(),
        "eligible",
        "can_exit",
    )
    assert evaluation_mask.column_roles["eligible"] is ColumnRole.MASK
    assert evaluation_mask.column_roles["can_exit"] is ColumnRole.MASK


def test_a_share_research_dataset_carries_no_source_columns() -> None:
    dataset = a_share_research_dataset_catalog().get(
        A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )

    assert dataset.source_columns == ()
    assert dataset.required_columns == (
        *platform_entity_time_key(),
        *dataset.context_columns,
        *dataset.filter_columns,
        *dataset.feature_columns,
        *dataset.label_columns,
    )
    assert not {
        "stock_code",
        "trade_date",
        "index_code",
        "source_id",
        "source_asset_id",
    }.intersection(dataset.required_columns)
