"""Crypto research dataset specs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.markets.implementations.crypto.fields import (
    field_catalog,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
)
from quant_research_platform.research.factors.microstructure import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)

RESEARCH_DATASET_FILTER_COLUMNS: tuple[str, ...] = ("eligible",)
RESEARCH_DATASET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "mid_price",
)


def build_research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    columns = (
        *platform_entity_time_key(),
        *MICROSTRUCTURE_FACTOR_COLUMNS,
        *CRYPTO_FORWARD_RETURN_COLUMNS,
        *RESEARCH_DATASET_FILTER_COLUMNS,
        *RESEARCH_DATASET_CONTEXT_COLUMNS,
    )
    return (
        ResearchDatasetSpec(
            data_id=MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
            market_id=MARKET_ID,
            input_data_ids=(
                MARKET_PANEL_DATA_ID,
                MICROSTRUCTURE_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
            ),
            base_input_data_id=MARKET_PANEL_DATA_ID,
            factor_columns=MICROSTRUCTURE_FACTOR_COLUMNS,
            label_columns=CRYPTO_FORWARD_RETURN_COLUMNS,
            filter_columns=RESEARCH_DATASET_FILTER_COLUMNS,
            context_columns=RESEARCH_DATASET_CONTEXT_COLUMNS,
            frequency=DataFrequency.MINUTE,
            grain=DataGrain.RESEARCH_ROW,
            column_roles=_column_roles(columns),
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        ),
    )


def build_research_dataset_catalog() -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    for spec in build_research_dataset_specs():
        catalog.register(spec)
    return catalog


def _column_roles(columns: tuple[str, ...]) -> dict[str, ColumnRole]:
    fields = field_catalog()
    factor_set = set(MICROSTRUCTURE_FACTOR_COLUMNS)
    label_set = set(CRYPTO_FORWARD_RETURN_COLUMNS)
    filter_set = set(RESEARCH_DATASET_FILTER_COLUMNS)
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in factor_set:
            roles[column] = ColumnRole.FACTOR
            continue
        if column in label_set:
            roles[column] = ColumnRole.LABEL
            continue
        if column in filter_set:
            roles[column] = ColumnRole.MASK
            continue
        roles[column] = fields.get(column).role
    return roles


__all__ = [
    "RESEARCH_DATASET_CONTEXT_COLUMNS",
    "RESEARCH_DATASET_FILTER_COLUMNS",
    "build_research_dataset_catalog",
    "build_research_dataset_specs",
]
