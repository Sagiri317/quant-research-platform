"""Crypto research dataset specs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.markets.implementations.crypto.factors.definitions import (
    HFT_FACTOR_COLUMNS,
)
from quant_research_platform.markets.implementations.crypto.fields import (
    field_catalog,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
    HFT_LABEL_COLUMNS,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    HFT_PANEL_CONTEXT_COLUMNS,
    HFT_PRIMARY_KEY,
)
from quant_research_platform.research.factors.microstructure.columns import (
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
    "trade_flow_observed",
    "quote_observed",
    "quote_age_seconds",
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
    return ResearchDatasetCatalog(build_research_dataset_specs())


def build_hft_research_dataset_catalog() -> ResearchDatasetCatalog:
    columns = (
        *HFT_PRIMARY_KEY,
        *HFT_FACTOR_COLUMNS,
        *HFT_LABEL_COLUMNS,
        "eligible",
        *HFT_PANEL_CONTEXT_COLUMNS,
    )
    roles = {
        column: (
            ColumnRole.FACTOR
            if column in HFT_FACTOR_COLUMNS
            else ColumnRole.LABEL
            if column in HFT_LABEL_COLUMNS
            else ColumnRole.UNIVERSE
            if column == "eligible"
            else ColumnRole.TIMESTAMP
            if column == "timestamp"
            else ColumnRole.KEY
            if column in HFT_PRIMARY_KEY
            else ColumnRole.CONTEXT
        )
        for column in columns
    }
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id=HFT_RESEARCH_DATASET_ID,
                market_id=MARKET_ID,
                input_data_ids=(
                    HFT_PANEL_DATA_ID,
                    HFT_FACTOR_DATA_ID,
                    HFT_LABEL_DATA_ID,
                ),
                base_input_data_id=HFT_PANEL_DATA_ID,
                factor_columns=HFT_FACTOR_COLUMNS,
                label_columns=HFT_LABEL_COLUMNS,
                filter_columns=("eligible",),
                context_columns=HFT_PANEL_CONTEXT_COLUMNS,
                frequency=DataFrequency.TICK,
                grain=DataGrain.RESEARCH_ROW,
                column_roles=roles,
                primary_key=HFT_PRIMARY_KEY,
                partition_columns=("year",),
            ),
        )
    )


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
