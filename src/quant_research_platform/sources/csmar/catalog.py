"""CSMAR source catalogs."""

from __future__ import annotations

from quant_research_platform.catalog.fields import (
    platform_entity_time_key,
    platform_field_catalog,
    platform_interim_column_names,
)
from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.sources.csmar.columns import INTERIM_KEEP_COLUMNS
from quant_research_platform.sources.csmar.fields import csmar_field_role
from quant_research_platform.sources.csmar.interim import CSMAR_SOURCE_ID


def csmar_interim_catalog() -> InterimDataCatalog:
    catalog = InterimDataCatalog()
    for interim_data_id, columns in INTERIM_KEEP_COLUMNS.items():
        if interim_data_id == "trading_calendar":
            catalog.register(
                InterimDataSpec(
                    market_id=MarketId("a_share"),
                    source_id=SourceId(CSMAR_SOURCE_ID),
                    interim_data_id=interim_data_id,
                    frequency=DataFrequency.DAILY,
                    grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
                    column_roles=_trading_calendar_column_roles(columns),
                    primary_key=(
                        "market_id",
                        "source_id",
                        "source_market_type",
                        "calendar_date",
                    ),
                    partition_columns=("year",),
                )
            )
            continue
        catalog.register(
            InterimDataSpec(
                market_id=MarketId("a_share"),
                source_id=SourceId(CSMAR_SOURCE_ID),
                interim_data_id=interim_data_id,
                frequency=DataFrequency.DAILY,
                grain=DataGrain.ASSET_TIMESTAMP,
                column_roles=_column_roles(columns),
                primary_key=platform_entity_time_key(),
                partition_columns=("year",),
            )
        )
    return catalog


def _trading_calendar_column_roles(columns: tuple[str, ...]) -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in {"market_id", "source_id", "source_market_type"}:
            roles[column] = ColumnRole.KEY
        elif column in {"timestamp", "calendar_date"}:
            roles[column] = ColumnRole.TIMESTAMP
        else:
            roles[column] = ColumnRole.CONTEXT
    return roles


def _column_roles(columns: tuple[str, ...]) -> dict[str, ColumnRole]:
    platform_catalog = platform_field_catalog()
    platform_columns = set(platform_interim_column_names())
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in platform_columns:
            roles[column] = platform_catalog.get(column).role
        else:
            roles[column] = csmar_field_role(column)
    return roles


__all__ = ["csmar_interim_catalog"]
