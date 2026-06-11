"""A-share market-specific catalogs and processed data specs."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import (
    FieldCatalog,
    platform_entity_time_key,
    platform_field_catalog,
)
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketDefinition
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import (
    PredictionCatalog,
    default_prediction_column,
)
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)

A_SHARE_FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(name="stock_name", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="market_type", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="trading_status_code", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="filling_flag", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="open", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="high", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="low", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="adj_open", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="adj_close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="prev_close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="amount", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="turnover", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="return_1d", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="limit_status", role=ColumnRole.CONTEXT, dtype="int8"),
    FieldSpec(name="is_limit_up", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_limit_down", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_trading", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_st", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_pt", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_a_share_scope", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="eligible_base", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="eligible", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="can_enter", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="can_exit", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="listing_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="delisting_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="company_status", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="status_change_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="listing_age_days", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(
        name="listing_age_trading_days",
        role=ColumnRole.CONTEXT,
        dtype="int64",
    ),
    FieldSpec(
        name="eligible_after_listing_age",
        role=ColumnRole.UNIVERSE,
        dtype="bool",
    ),
    FieldSpec(name="eligible_liquidity", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(
        name="csrc_2012_industry_code",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(
        name="csrc_2012_industry_name",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(name="share_change_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="total_shares", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="a_float_shares", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="total_market_cap", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="float_market_cap", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="next_day_is_trading", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="next_day_limit_status", role=ColumnRole.CONTEXT, dtype="int8"),
)

_PANEL_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "session_date",
    "available_at",
    "stock_name",
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "prev_close",
    "volume",
    "amount",
    "turnover",
    "return_1d",
    "limit_status",
    "is_limit_up",
    "is_limit_down",
    "is_trading",
    "is_st",
    "is_pt",
    "is_a_share_scope",
    "eligible_base",
    "eligible",
    "can_enter",
    "can_exit",
    "listing_date",
    "delisting_date",
    "company_status",
    "status_change_date",
    "listing_age_days",
    "listing_age_trading_days",
    "eligible_after_listing_age",
    "eligible_liquidity",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "share_change_date",
    "total_shares",
    "a_float_shares",
    "total_market_cap",
    "float_market_cap",
)

_CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
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

_UNIVERSE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
)

_LABEL_CONTROL_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)

_RESEARCH_DATASET_SOURCE_COLUMNS: tuple[str, ...] = ()

_RESEARCH_DATASET_FILTER_COLUMNS: tuple[str, ...] = (
    "eligible",
    "can_enter",
    "can_exit",
)

_RESEARCH_DATASET_CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_type",
    "amount",
    "turnover",
    "total_market_cap",
    "float_market_cap",
)

A_SHARE_MASK_ALLOWED_USES: dict[str, tuple[str, ...]] = {
    "a_share_as_of": ("feature_filter", "universe_filter", "diagnostics"),
    "a_share_execution": ("label_alignment", "execution_feasibility"),
    "a_share_evaluation": ("evaluation_filter", "diagnostics"),
}


def a_share_market_definition() -> MarketDefinition:
    return MarketDefinition(
        market_id=A_SHARE_MARKET_ID,
        display_name="A-share market implementation",
        processed_spec_ids=(
            DAILY_PANEL_PROCESSED_DATA_ID,
            DAILY_CONTEXT_PROCESSED_DATA_ID,
            A_SHARE_DEFAULT_UNIVERSE_ID,
            A_SHARE_AS_OF_MASK_ID,
            A_SHARE_EXECUTION_MASK_ID,
            A_SHARE_EVALUATION_MASK_ID,
            PRICE_VOLUME_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        ),
    )


def a_share_field_catalog() -> FieldCatalog:
    """Return platform canonical fields plus A-share-only processed fields."""

    catalog = platform_field_catalog()
    for spec in A_SHARE_FIELD_SPECS:
        catalog.register(spec)
    return catalog


def a_share_factor_catalog() -> FactorCatalog:
    """Return the A-share factor registry used by processed factor specs."""

    catalog = FactorCatalog()
    for definition in (
        factor_definitions.a_share_price_volume_factor_definitions()
    ):
        catalog.register(definition)
    return catalog


def a_share_label_catalog() -> LabelCatalog:
    """Return the A-share forward-return label registry."""

    catalog = LabelCatalog()
    for definition in (
        label_definitions.a_share_forward_return_label_definitions()
    ):
        catalog.register(definition)
    return catalog


def a_share_processed_catalog() -> ProcessedDataCatalog:
    """Register the minimal real A-share 0.1 processed data specs."""

    catalog = ProcessedDataCatalog()
    for spec in _processed_specs():
        catalog.register(spec)
    return catalog


def a_share_research_dataset_catalog() -> ResearchDatasetCatalog:
    """Register A-share research dataset specs assembled from processed data."""

    factor_columns = _registered_factor_columns()
    label_columns = _registered_label_columns()
    columns = (
        *platform_entity_time_key(),
        *_RESEARCH_DATASET_SOURCE_COLUMNS,
        *_RESEARCH_DATASET_CONTEXT_COLUMNS,
        *_RESEARCH_DATASET_FILTER_COLUMNS,
        *factor_columns,
        *label_columns,
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
            feature_columns=factor_columns,
            label_columns=label_columns,
            filter_columns=_RESEARCH_DATASET_FILTER_COLUMNS,
            context_columns=_RESEARCH_DATASET_CONTEXT_COLUMNS,
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


def a_share_evaluation_catalog() -> EvaluationCatalog:
    """Register minimal A-share evaluation run specs for QRP 0.1."""

    catalog = EvaluationCatalog()
    catalog.register(
        EvaluationSpec(
            run_id="a_share_mom_20d_forward_return_20d_2014",
            market_id=A_SHARE_MARKET_ID,
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
        )
    )
    return catalog


def a_share_model_catalog() -> ModelCatalog:
    """Register minimal non-DL A-share model specs for QRP 0.1."""

    catalog = ModelCatalog()
    catalog.register(
        ModelSpec(
            model_id="a_share_linear_mom_20d_forward_return_20d",
            model_family="linear",
            feature_columns=("mom_20d",),
            label_column="forward_return_20d",
        )
    )
    return catalog


def a_share_prediction_catalog() -> PredictionCatalog:
    """Register minimal A-share prediction run specs for QRP 0.1."""

    catalog = PredictionCatalog()
    catalog.register(
        PredictionSpec(
            prediction_id="a_share_linear_mom_20d_prediction_2014",
            model_id="a_share_linear_mom_20d_forward_return_20d",
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            prediction_column=default_prediction_column(),
            primary_key=platform_entity_time_key(),
            train_end="2014-06-30",
            test_start="2014-07-01",
        )
    )
    return catalog


def _processed_specs() -> tuple[ProcessedDataSpec, ...]:
    factor_columns = _registered_factor_columns()
    label_columns = _registered_label_columns()
    return (
        _processed_spec(
            kind=ProcessedDataKind.PANEL,
            data_id=DAILY_PANEL_PROCESSED_DATA_ID,
            columns=_PANEL_COLUMNS,
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=DAILY_CONTEXT_PROCESSED_DATA_ID,
            columns=_CONTEXT_COLUMNS,
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_DEFAULT_UNIVERSE_ID,
            columns=_UNIVERSE_COLUMNS,
            role_overrides={"eligible": ColumnRole.UNIVERSE},
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_AS_OF_MASK_ID,
            columns=(
                "market_id",
                "asset_id",
                "timestamp",
                "eligible",
            ),
            role_overrides={"eligible": ColumnRole.MASK},
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_EXECUTION_MASK_ID,
            columns=(
                "market_id",
                "asset_id",
                "timestamp",
                "can_enter",
                "can_exit",
            ),
            role_overrides={
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_EVALUATION_MASK_ID,
            columns=(
                "market_id",
                "asset_id",
                "timestamp",
                "eligible",
                "can_exit",
            ),
            role_overrides={
                "eligible": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
        ),
        _processed_spec(
            kind=ProcessedDataKind.FACTOR,
            data_id=PRICE_VOLUME_PROCESSED_DATA_ID,
            columns=(*platform_entity_time_key(), *factor_columns),
            role_overrides={column: ColumnRole.FACTOR for column in factor_columns},
        ),
        _processed_spec(
            kind=ProcessedDataKind.LABEL,
            data_id=FORWARD_RETURNS_PROCESSED_DATA_ID,
            columns=(*_LABEL_CONTROL_COLUMNS, *label_columns),
            role_overrides={column: ColumnRole.LABEL for column in label_columns},
        ),
    )


def _processed_spec(
    *,
    kind: ProcessedDataKind,
    data_id: str,
    columns: tuple[str, ...],
    role_overrides: dict[str, ColumnRole] | None = None,
) -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=A_SHARE_MARKET_ID,
        kind=kind,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=_column_roles(columns, role_overrides=role_overrides),
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _column_roles(
    columns: tuple[str, ...],
    *,
    role_overrides: dict[str, ColumnRole] | None = None,
) -> dict[str, ColumnRole]:
    fields = a_share_field_catalog()
    overrides = role_overrides or {}
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in overrides:
            roles[column] = overrides[column]
        else:
            roles[column] = fields.get(column).role
    return roles


def _research_dataset_column_roles(
    columns: tuple[str, ...],
    *,
    factor_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> dict[str, ColumnRole]:
    fields = a_share_field_catalog()
    factor_set = set(factor_columns)
    label_set = set(label_columns)
    filter_set = set(_RESEARCH_DATASET_FILTER_COLUMNS)
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
    for column in factor_columns:
        roles[column] = ColumnRole.FEATURE
    for column in label_columns:
        roles[column] = ColumnRole.LABEL
    for column in _RESEARCH_DATASET_FILTER_COLUMNS:
        roles[column] = ColumnRole.MASK
    return roles


def _registered_factor_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column for definition in a_share_factor_catalog().list()
    )


def _registered_label_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in sorted(
            a_share_label_catalog().list(),
            key=lambda item: item.horizon,
        )
    )


__all__ = [
    "A_SHARE_FIELD_SPECS",
    "A_SHARE_MARKET_ID",
    "A_SHARE_MASK_ALLOWED_USES",
    "a_share_evaluation_catalog",
    "a_share_factor_catalog",
    "a_share_field_catalog",
    "a_share_label_catalog",
    "a_share_market_definition",
    "a_share_model_catalog",
    "a_share_prediction_catalog",
    "a_share_processed_catalog",
    "a_share_research_dataset_catalog",
]
