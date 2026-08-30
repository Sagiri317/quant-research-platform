"""A-share research dataset profile specs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.markets.implementations.a_share.fields import (
    field_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    DEFAULT_UNIVERSE_ID,
    EXECUTION_MASK_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID,
    FUNDAMENTAL_RAW_PROCESSED_DATA_ID,
    MARKET_ID,
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
    market_context_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    _registered_factor_columns,
    _registered_fundamental_factor_columns,
    _registered_label_columns,
)
from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)

RESEARCH_DATASET_MOMENTUM_COLUMN = price_volume_window_column(
    "mom",
    20,
    DataFrequency.DAILY,
)
RESEARCH_DATASET_MOMENTUM_FEATURE_COLUMN = f"{RESEARCH_DATASET_MOMENTUM_COLUMN}_zscore"
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
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
)
RESEARCH_DATASET_FEATURE_COLUMNS: tuple[str, ...] = (
    RESEARCH_DATASET_MOMENTUM_FEATURE_COLUMN,
)
RESEARCH_DATASET_FEATURE_TRANSFORMS: tuple[FeatureTransformSpec, ...] = (
    FeatureTransformSpec(
        transform_id=RESEARCH_DATASET_MOMENTUM_FEATURE_COLUMN,
        source_columns=(RESEARCH_DATASET_MOMENTUM_COLUMN,),
        output_columns=RESEARCH_DATASET_FEATURE_COLUMNS,
        transform="cross_sectional_zscore",
    ),
)


def build_research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    """Return A-share research dataset specs assembled from processed data."""

    factor_columns = _registered_factor_columns()
    fundamental_columns = _registered_fundamental_factor_columns()
    label_columns = _registered_label_columns()
    columns = (
        *platform_entity_time_key(),
        *factor_columns,
        *RESEARCH_DATASET_FEATURE_COLUMNS,
        *label_columns,
        *RESEARCH_DATASET_FILTER_COLUMNS,
        *RESEARCH_DATASET_CONTEXT_COLUMNS,
    )
    return (
        ResearchDatasetSpec(
            data_id=PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            market_id=MARKET_ID,
            input_data_ids=(
                PRICE_VOLUME_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
                market_context_data_id(),
                DEFAULT_UNIVERSE_ID,
                EXECUTION_MASK_ID,
            ),
            base_input_data_id=DEFAULT_UNIVERSE_ID,
            factor_columns=factor_columns,
            feature_columns=RESEARCH_DATASET_FEATURE_COLUMNS,
            feature_transforms=RESEARCH_DATASET_FEATURE_TRANSFORMS,
            label_columns=label_columns,
            filter_columns=RESEARCH_DATASET_FILTER_COLUMNS,
            context_columns=RESEARCH_DATASET_CONTEXT_COLUMNS,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles=_research_dataset_column_roles(
                columns,
                factor_columns=factor_columns,
                feature_columns=RESEARCH_DATASET_FEATURE_COLUMNS,
                label_columns=label_columns,
            ),
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        ),
        ResearchDatasetSpec(
            data_id=FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID,
            market_id=MARKET_ID,
            input_data_ids=(
                FUNDAMENTAL_RAW_PROCESSED_DATA_ID,
                FORWARD_RETURNS_PROCESSED_DATA_ID,
                market_context_data_id(),
                DEFAULT_UNIVERSE_ID,
                EXECUTION_MASK_ID,
            ),
            base_input_data_id=DEFAULT_UNIVERSE_ID,
            factor_columns=fundamental_columns,
            feature_columns=(),
            feature_transforms=(),
            label_columns=label_columns,
            filter_columns=RESEARCH_DATASET_FILTER_COLUMNS,
            context_columns=RESEARCH_DATASET_CONTEXT_COLUMNS,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles=_research_dataset_column_roles(
                (
                    *platform_entity_time_key(),
                    *fundamental_columns,
                    *label_columns,
                    *RESEARCH_DATASET_FILTER_COLUMNS,
                    *RESEARCH_DATASET_CONTEXT_COLUMNS,
                ),
                factor_columns=fundamental_columns,
                feature_columns=(),
                label_columns=label_columns,
            ),
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        ),
    )


def build_research_dataset_catalog() -> ResearchDatasetCatalog:
    return ResearchDatasetCatalog(build_research_dataset_specs())


def _research_dataset_column_roles(
    columns: tuple[str, ...],
    *,
    factor_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> dict[str, ColumnRole]:
    fields = field_catalog()
    factor_set = set(factor_columns)
    feature_set = set(feature_columns)
    label_set = set(label_columns)
    filter_set = set(RESEARCH_DATASET_FILTER_COLUMNS)
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in factor_set:
            roles[column] = ColumnRole.FACTOR
            continue
        if column in feature_set:
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
