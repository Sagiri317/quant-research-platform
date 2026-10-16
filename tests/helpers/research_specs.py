from __future__ import annotations

from collections.abc import Mapping

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

DEMO_FACTOR_COLUMN = "momentum_20d"
DEMO_LABEL_COLUMN = forward_return_label_column(20)


def make_demo_research_dataset_spec(**overrides: object) -> ResearchDatasetSpec:
    return _research_dataset_spec_from_defaults(
        **(_demo_research_dataset_defaults() | overrides)
    )


def make_dummy_research_dataset_spec(**overrides: object) -> ResearchDatasetSpec:
    return _research_dataset_spec_from_defaults(
        **(_dummy_research_dataset_defaults() | overrides)
    )


def _research_dataset_spec_from_defaults(
    *,
    data_id: str,
    market_id: str | MarketId,
    input_data_ids: tuple[str, ...],
    base_input_data_id: str,
    factor_columns: tuple[str, ...],
    feature_columns: tuple[str, ...] = (),
    feature_transforms: tuple[object, ...] = (),
    label_columns: tuple[str, ...],
    filter_columns: tuple[str, ...],
    context_columns: tuple[str, ...],
    key_coverage: Mapping[str, str] | None = None,
    filter_column_roles: Mapping[str, ColumnRole] | None = None,
    frequency: DataFrequency = DataFrequency.DAILY,
    grain: DataGrain = DataGrain.RESEARCH_ROW,
    primary_key: tuple[str, ...] = PLATFORM_ASSET_TIME_KEY,
    partition_columns: tuple[str, ...] = ("year",),
) -> ResearchDatasetSpec:
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
    registered_specs = specs or (make_demo_research_dataset_spec(**spec_kwargs),)
    return ResearchDatasetCatalog(registered_specs)


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


def assemble_research_dataset(
    *,
    inputs,
    output_root,
    dataset_catalog,
    processed_catalog,
    dataset_id,
    time_window,
):
    """Run the public assembly operation for existing explicit scenario fixtures."""
    from quant_research_platform.contracts.time import time_value_isoformat
    from quant_research_platform.research.datasets.assembler import assemble

    spec = dataset_catalog.get(dataset_id)
    config = {
        "market_id": str(spec.market_id),
        "dataset_id": spec.data_id,
        "inputs": {key: str(value) for key, value in inputs.items()},
        "output_root": str(output_root),
        "time_window": {
            "start": time_value_isoformat(time_window.start),
            "end": time_value_isoformat(time_window.end),
        },
        "factor_columns": list(spec.factor_columns),
        "label_columns": list(spec.label_columns),
        "filter_columns": list(spec.filter_columns),
        "context_columns": list(spec.context_columns),
        "base_input_data_id": spec.base_input_data_id,
        "sparse_input_data_ids": [
            key for key, value in spec.key_coverage.items() if value == "sparse"
        ],
        "feature_recipe": None,
        "eligibility_recipe": spec.eligibility_recipe,
    }
    if spec.feature_transforms:
        transform = spec.feature_transforms[0]
        config["feature_recipe"] = transform.recipe_id
        config["feature_transform"] = {
            name: getattr(transform, name)
            for name in (
                "clip",
                "industry_neutralize",
                "size_neutralize",
                "center",
                "scale",
                "impute_missing",
            )
        }
    return assemble(config)
