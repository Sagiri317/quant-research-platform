"""A-share research dataset profile specs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    a_share_field_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)

RESEARCH_DATASET_FILTER_COLUMNS: tuple[str, ...] = (
    "eligible",
    "can_enter",
    "can_exit",
)
RESEARCH_DATASET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_type",
    "amount",
    "turnover",
    "total_market_cap",
    "float_market_cap",
)


def a_share_research_dataset_catalog() -> ResearchDatasetCatalog:
    """Register A-share research dataset specs assembled from processed data."""

    factor_columns = _registered_factor_columns()
    label_columns = _registered_label_columns()
    columns = (
        *platform_entity_time_key(),
        *factor_columns,
        *label_columns,
        *RESEARCH_DATASET_FILTER_COLUMNS,
        *RESEARCH_DATASET_CONTEXT_COLUMNS,
    )
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            data_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            market_id=A_SHARE_MARKET_ID,
            input_data_ids=(
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
                DAILY_CONTEXT_PROCESSED_DATA_ID,
                A_SHARE_DEFAULT_UNIVERSE_ID,
                A_SHARE_EXECUTION_MASK_ID,
            ),
            base_input_data_id=A_SHARE_DEFAULT_UNIVERSE_ID,
            feature_columns=factor_columns,
            label_columns=label_columns,
            filter_columns=RESEARCH_DATASET_FILTER_COLUMNS,
            context_columns=RESEARCH_DATASET_CONTEXT_COLUMNS,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles=_research_dataset_column_roles(
                columns,
                factor_columns=factor_columns,
                label_columns=label_columns,
            ),
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def _research_dataset_column_roles(
    columns: tuple[str, ...],
    *,
    factor_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> dict[str, ColumnRole]:
    fields = a_share_field_catalog()
    factor_set = set(factor_columns)
    label_set = set(label_columns)
    filter_set = set(RESEARCH_DATASET_FILTER_COLUMNS)
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in factor_set:
            roles[column] = ColumnRole.FEATURE
            continue
        if column in label_set:
            roles[column] = ColumnRole.LABEL
            continue
        if column in filter_set:
            roles[column] = ColumnRole.MASK
            continue
        roles[column] = fields.get(column).role
    return roles


def _registered_factor_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in factor_definitions.a_share_factor_catalog().list()
    )


def _registered_label_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in sorted(
            label_definitions.a_share_label_catalog().list(),
            key=lambda item: item.horizon,
        )
    )


__all__ = [
    "RESEARCH_DATASET_CONTEXT_COLUMNS",
    "RESEARCH_DATASET_FILTER_COLUMNS",
    "a_share_research_dataset_catalog",
]
