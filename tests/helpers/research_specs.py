from __future__ import annotations

from collections.abc import Mapping

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

DEMO_FACTOR_COLUMN = price_volume_window_column("mom", 20)
DEMO_FACTOR_ZSCORE_COLUMN = f"{DEMO_FACTOR_COLUMN}_zscore"
DEMO_LABEL_COLUMN = forward_return_label_column(20)


def make_signal_spec(
    column: str,
    *,
    signal_id: str | None = None,
    source_kind: str = "factor",
    direction: str = "higher_is_better",
    transform: str = "identity",
) -> SignalSpec:
    return SignalSpec(
        signal_id=signal_id or column,
        source_column=column,
        source_kind=source_kind,
        direction=direction,
        transform=transform,
    )


def make_demo_research_dataset_spec(
    *,
    data_id: str = "demo",
    market_id: str | MarketId = "a_share",
    input_data_ids: tuple[str, ...] | None = None,
    base_input_data_id: str | None = None,
    factor_columns: tuple[str, ...] | None = None,
    feature_columns: tuple[str, ...] = (),
    feature_transforms: tuple[object, ...] = (),
    label_columns: tuple[str, ...] | None = None,
    filter_columns: tuple[str, ...] | None = None,
    context_columns: tuple[str, ...] | None = None,
    key_coverage: Mapping[str, str] | None = None,
    filter_column_roles: Mapping[str, ColumnRole] | None = None,
    frequency: DataFrequency = DataFrequency.DAILY,
    grain: DataGrain = DataGrain.RESEARCH_ROW,
    primary_key: tuple[str, ...] = platform_entity_time_key(),
    partition_columns: tuple[str, ...] = ("year",),
) -> ResearchDatasetSpec:
    defaults = _demo_research_dataset_defaults()
    return _research_dataset_spec_from_defaults(
        defaults,
        data_id=data_id,
        market_id=market_id,
        input_data_ids=input_data_ids,
        base_input_data_id=base_input_data_id,
        factor_columns=factor_columns,
        feature_columns=feature_columns,
        feature_transforms=feature_transforms,
        label_columns=label_columns,
        filter_columns=filter_columns,
        context_columns=context_columns,
        key_coverage=key_coverage,
        filter_column_roles=filter_column_roles,
        frequency=frequency,
        grain=grain,
        primary_key=primary_key,
        partition_columns=partition_columns,
    )


def make_dummy_research_dataset_spec(**overrides: object) -> ResearchDatasetSpec:
    return _research_dataset_spec_from_defaults(
        _dummy_research_dataset_defaults(),
        **overrides,
    )


def make_portfolio_research_dataset_spec(**overrides: object) -> ResearchDatasetSpec:
    return _research_dataset_spec_from_defaults(
        _portfolio_research_dataset_defaults(),
        **overrides,
    )


def _research_dataset_spec_from_defaults(
    defaults: Mapping[str, object],
    *,
    data_id: str = "demo",
    market_id: str | MarketId = "a_share",
    input_data_ids: tuple[str, ...] | None = None,
    base_input_data_id: str | None = None,
    factor_columns: tuple[str, ...] | None = None,
    feature_columns: tuple[str, ...] = (),
    feature_transforms: tuple[object, ...] = (),
    label_columns: tuple[str, ...] | None = None,
    filter_columns: tuple[str, ...] | None = None,
    context_columns: tuple[str, ...] | None = None,
    key_coverage: Mapping[str, str] | None = None,
    filter_column_roles: Mapping[str, ColumnRole] | None = None,
    frequency: DataFrequency = DataFrequency.DAILY,
    grain: DataGrain = DataGrain.RESEARCH_ROW,
    primary_key: tuple[str, ...] = platform_entity_time_key(),
    partition_columns: tuple[str, ...] = ("year",),
) -> ResearchDatasetSpec:
    if data_id == "demo" and defaults["data_id"] != "demo":
        data_id = defaults["data_id"]
    if str(market_id) == "a_share" and defaults["market_id"] != "a_share":
        market_id = defaults["market_id"]
    input_data_ids = (
        defaults["input_data_ids"] if input_data_ids is None else input_data_ids
    )
    base_input_data_id = (
        defaults["base_input_data_id"]
        if base_input_data_id is None
        else base_input_data_id
    )
    factor_columns = (
        defaults["factor_columns"] if factor_columns is None else factor_columns
    )
    label_columns = (
        defaults["label_columns"] if label_columns is None else label_columns
    )
    filter_columns = (
        defaults["filter_columns"] if filter_columns is None else filter_columns
    )
    context_columns = (
        defaults["context_columns"] if context_columns is None else context_columns
    )
    roles = _research_column_roles(
        factor_columns=factor_columns,
        feature_columns=feature_columns,
        label_columns=label_columns,
        filter_columns=filter_columns,
        context_columns=context_columns,
        filter_column_roles=filter_column_roles,
    )
    return ResearchDatasetSpec(
        data_id=data_id,
        market_id=MarketId(str(market_id)),
        input_data_ids=input_data_ids,
        base_input_data_id=base_input_data_id,
        factor_columns=factor_columns,
        feature_columns=feature_columns,
        feature_transforms=feature_transforms,
        label_columns=label_columns,
        filter_columns=filter_columns,
        context_columns=context_columns,
        key_coverage=dict(key_coverage or {}),
        frequency=frequency,
        grain=grain,
        column_roles=roles,
        primary_key=primary_key,
        partition_columns=partition_columns,
    )


def make_research_dataset_catalog(
    *specs: ResearchDatasetSpec,
    **spec_kwargs: object,
) -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    registered_specs = specs or (make_demo_research_dataset_spec(**spec_kwargs),)
    for spec in registered_specs:
        catalog.register(spec)
    return catalog


def correctness_research_dataset_catalog(
    *,
    include_feature_transform: bool = False,
) -> ResearchDatasetCatalog:
    feature_columns = (DEMO_FACTOR_ZSCORE_COLUMN,) if include_feature_transform else ()
    feature_transforms = (
        (
            FeatureTransformSpec(
                transform_id=DEMO_FACTOR_ZSCORE_COLUMN,
                source_columns=(DEMO_FACTOR_COLUMN,),
                output_columns=(DEMO_FACTOR_ZSCORE_COLUMN,),
                transform="cross_sectional_zscore",
            ),
        )
        if include_feature_transform
        else ()
    )
    catalog = ResearchDatasetCatalog()
    catalog.register(
        make_demo_research_dataset_spec(
            feature_columns=feature_columns,
            feature_transforms=feature_transforms,
        )
    )
    return catalog


def correctness_processed_catalog():
    from tests.helpers.research_processed_inputs import make_demo_processed_catalog

    return make_demo_processed_catalog()


def _demo_research_dataset_defaults() -> dict[str, object]:
    return {
        "data_id": "demo",
        "market_id": "a_share",
        "input_data_ids": (
            "demo_factors",
            "demo_labels",
            "demo_annotations",
        ),
        "base_input_data_id": "demo_annotations",
        "factor_columns": (DEMO_FACTOR_COLUMN,),
        "label_columns": (DEMO_LABEL_COLUMN,),
        "filter_columns": ("eligible", "can_enter", "can_exit"),
        "context_columns": ("market_type",),
    }


def _dummy_research_dataset_defaults() -> dict[str, object]:
    return {
        "data_id": "dummy_dataset",
        "market_id": "dummy_market",
        "input_data_ids": ("base", "factors", "labels", "filters", "context"),
        "base_input_data_id": "base",
        "factor_columns": ("factor_1",),
        "label_columns": ("label_1",),
        "filter_columns": ("eligible",),
        "context_columns": ("context_1",),
    }


def _portfolio_research_dataset_defaults() -> dict[str, object]:
    return {
        "data_id": "demo",
        "market_id": "a_share",
        "input_data_ids": ("factor", "label"),
        "base_input_data_id": "factor",
        "factor_columns": ("raw_risk",),
        "label_columns": ("forward_return",),
        "filter_columns": (),
        "context_columns": (),
    }


def _research_column_roles(
    *,
    factor_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
    filter_columns: tuple[str, ...],
    context_columns: tuple[str, ...],
    filter_column_roles: Mapping[str, ColumnRole] | None,
) -> dict[str, ColumnRole]:
    roles: dict[str, ColumnRole] = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    roles.update({column: ColumnRole.FACTOR for column in factor_columns})
    roles.update({column: ColumnRole.FEATURE for column in feature_columns})
    roles.update({column: ColumnRole.LABEL for column in label_columns})
    roles.update({column: ColumnRole.MASK for column in filter_columns})
    roles.update({column: ColumnRole.CONTEXT for column in context_columns})
    roles.update(filter_column_roles or {})
    return roles
