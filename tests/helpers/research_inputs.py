from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.factors.price_volume import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

DEMO_FACTOR_COLUMN = price_volume_window_column("mom", 20)
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


def make_research_dataset_spec(
    *,
    scenario: str = "demo",
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
    defaults = _research_dataset_defaults(scenario)
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
    registered_specs = specs or (make_research_dataset_spec(**spec_kwargs),)
    for spec in registered_specs:
        catalog.register(spec)
    return catalog


def make_processed_catalog(
    *,
    scenario: str = "demo",
    include_extensions: bool = False,
    duplicate_factor_context: bool = False,
    missing_factor_key: bool = False,
) -> ProcessedDataCatalog:
    if scenario == "demo":
        return _demo_processed_catalog()
    if scenario == "filter":
        return _filter_processed_catalog()
    if scenario == "dummy":
        return _dummy_processed_catalog(
            include_extensions=include_extensions,
            duplicate_factor_context=duplicate_factor_context,
            missing_factor_key=missing_factor_key,
        )
    raise ValueError(f"unknown_processed_catalog_scenario: {scenario}")


def write_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    frame: pd.DataFrame | None = None,
    label_frame: pd.DataFrame | None = None,
    include_extensions: bool = False,
    drop_factor_key: str | None = None,
    reverse_base: bool = False,
    truncate_factors: bool = False,
    extra_factor_key: bool = False,
    factor_values: Sequence[object] | None = None,
) -> None:
    data_ids = frozenset(spec.data_id for spec in processed_catalog.list())
    if {"demo_factors", "demo_labels", "demo_annotations"}.issubset(data_ids):
        _write_demo_processed_inputs(
            layout,
            processed_catalog,
            frame=frame,
            label_frame=label_frame,
        )
        return
    if {"demo_factor", "demo_label", "demo_annotation"}.issubset(data_ids):
        if frame is None:
            raise ValueError("filter_processed_inputs_require_frame")
        _write_filter_processed_inputs(layout, processed_catalog, frame)
        return
    if {"base", "factors", "labels", "filters", "context"}.issubset(data_ids):
        _write_dummy_processed_inputs(
            layout,
            processed_catalog,
            include_extensions=include_extensions,
            drop_factor_key=drop_factor_key,
            reverse_base=reverse_base,
            truncate_factors=truncate_factors,
            extra_factor_key=extra_factor_key,
            factor_values=factor_values,
        )
        return
    raise ValueError(f"unsupported_processed_catalog_ids: {sorted(data_ids)}")


def make_prediction_frame(*, filter_flip: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    if filter_flip:
        return pd.DataFrame(
            {
                "market_id": ["a_share"] * len(dates),
                "asset_id": ["asset_a"] * len(dates),
                "timestamp": dates,
                "factor_a": [0.0, 1.0, 2.0, 100.0, 101.0, 0.0, 7.0, 8.0],
                "factor_b": [0.0, 1.0, 2.0, 0.0, 0.0, 10.0, 7.0, 8.0],
                "eligible": [True, True, True, False, False, True, True, True],
                "label": [0.0, 1.0, 2.0, 100.0, 101.0, 10.0, 7.0, 8.0],
            }
        )

    factor_a = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0]
    factor_b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    label = [1.0 + 2.0 * value for value in factor_b]
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * len(dates),
            "asset_id": ["asset_a"] * len(dates),
            "timestamp": dates,
            "factor_a": factor_a,
            "factor_b": factor_b,
            "eligible": [True] * len(dates),
            "label": label,
        }
    )


def _research_dataset_defaults(scenario: str) -> dict[str, object]:
    if scenario == "demo":
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
    if scenario == "dummy":
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
    if scenario == "portfolio":
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
    raise ValueError(f"unknown_research_dataset_scenario: {scenario}")


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


def _demo_processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        _processed_spec(
            data_id="demo_factors",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                DEMO_FACTOR_COLUMN: ColumnRole.FACTOR,
            },
        )
    )
    catalog.register(
        _processed_spec(
            data_id="demo_labels",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                DEMO_LABEL_COLUMN: ColumnRole.LABEL,
            },
        )
    )
    catalog.register(
        _processed_spec(
            data_id="demo_annotations",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
                "market_type": ColumnRole.CONTEXT,
            },
        )
    )
    return catalog


def _filter_processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        _processed_spec(
            data_id="demo_factor",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "factor": ColumnRole.FACTOR,
            },
        )
    )
    catalog.register(
        _processed_spec(
            data_id="demo_label",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "label": ColumnRole.LABEL,
            },
        )
    )
    catalog.register(
        _processed_spec(
            data_id="demo_annotation",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
        )
    )
    return catalog


def _dummy_processed_catalog(
    *,
    include_extensions: bool,
    duplicate_factor_context: bool,
    missing_factor_key: bool,
) -> ProcessedDataCatalog:
    specs = (
        _dummy_processed_spec(
            "base",
            ProcessedDataKind.PANEL,
            platform_entity_time_key(),
        ),
        _dummy_processed_spec(
            "factors",
            ProcessedDataKind.FACTOR,
            (
                *(
                    ("market_id", "timestamp")
                    if missing_factor_key
                    else platform_entity_time_key()
                ),
                "factor_1",
                *(("factor_2",) if include_extensions else ()),
            ),
            role=ColumnRole.FACTOR,
            primary_key=(
                ("market_id", "timestamp")
                if missing_factor_key
                else platform_entity_time_key()
            ),
        ),
        _dummy_processed_spec(
            "labels",
            ProcessedDataKind.LABEL,
            (
                *platform_entity_time_key(),
                "label_1",
                *(("label_2",) if include_extensions else ()),
            ),
            role=ColumnRole.LABEL,
        ),
        _dummy_processed_spec(
            "filters",
            ProcessedDataKind.ANNOTATION,
            (*platform_entity_time_key(), "eligible"),
            role=ColumnRole.MASK,
        ),
        _dummy_processed_spec(
            "context",
            ProcessedDataKind.ANNOTATION,
            (
                *platform_entity_time_key(),
                "context_1",
                *(("context_2",) if include_extensions else ()),
                *(("factor_1",) if duplicate_factor_context else ()),
            ),
            role=ColumnRole.CONTEXT,
        ),
    )
    catalog = ProcessedDataCatalog()
    for spec in specs:
        catalog.register(spec)
    return catalog


def _processed_spec(
    *,
    data_id: str,
    market_id: MarketId,
    kind: ProcessedDataKind,
    column_roles: Mapping[str, ColumnRole],
    primary_key: tuple[str, ...] = platform_entity_time_key(),
) -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=market_id,
        kind=kind,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=dict(column_roles),
        primary_key=primary_key,
        partition_columns=("year",),
    )


def _dummy_processed_spec(
    data_id: str,
    kind: ProcessedDataKind,
    columns: tuple[str, ...],
    *,
    role: ColumnRole = ColumnRole.CONTEXT,
    primary_key: tuple[str, ...] = platform_entity_time_key(),
) -> ProcessedDataSpec:
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in {"market_id", "asset_id"}:
            roles[column] = ColumnRole.KEY
        elif column == "timestamp":
            roles[column] = ColumnRole.TIMESTAMP
        else:
            roles[column] = role
    return _processed_spec(
        data_id=data_id,
        market_id=MarketId("dummy_market"),
        kind=kind,
        column_roles=roles,
        primary_key=primary_key,
    )


def _write_demo_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    frame: pd.DataFrame | None,
    label_frame: pd.DataFrame | None,
) -> None:
    frame = _demo_research_rows() if frame is None else frame
    labels = frame if label_frame is None else label_frame
    columns_by_id = {
        "demo_factors": ["market_id", "asset_id", "timestamp", DEMO_FACTOR_COLUMN],
        "demo_annotations": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
            "market_type",
        ],
    }
    for data_id, columns in columns_by_id.items():
        _write_processed_frame(
            layout,
            processed_catalog,
            data_id,
            frame.loc[:, columns],
        )
    _write_processed_frame(
        layout,
        processed_catalog,
        "demo_labels",
        labels.loc[
            :,
            ["market_id", "asset_id", "timestamp", DEMO_LABEL_COLUMN],
        ],
    )


def _write_filter_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    frame: pd.DataFrame,
) -> None:
    columns_by_id = {
        "demo_factor": ["market_id", "asset_id", "timestamp", "factor"],
        "demo_label": ["market_id", "asset_id", "timestamp", "label"],
        "demo_annotation": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
        ],
    }
    for data_id, columns in columns_by_id.items():
        _write_processed_frame(
            layout,
            processed_catalog,
            data_id,
            frame.loc[:, columns],
        )


def _write_dummy_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    include_extensions: bool,
    drop_factor_key: str | None,
    reverse_base: bool,
    truncate_factors: bool,
    extra_factor_key: bool,
    factor_values: Sequence[object] | None,
) -> None:
    base = _dummy_base_frame()
    if reverse_base:
        base = base.iloc[::-1].reset_index(drop=True)
    factor_frame = _with_optional_columns(
        _dummy_base_frame().assign(factor_1=list(factor_values or [1.0, 2.0])),
        include_extensions=include_extensions,
        extension_column="factor_2",
        values=[10.0, 20.0],
    )
    if truncate_factors:
        factor_frame = factor_frame.iloc[[0]].reset_index(drop=True)
    if extra_factor_key:
        factor_frame = pd.concat(
            [
                factor_frame,
                pd.DataFrame(
                    {
                        "market_id": ["dummy_market"],
                        "asset_id": ["asset_extra"],
                        "timestamp": [pd.Timestamp("2024-01-03")],
                        "factor_1": [3.0],
                    }
                ),
            ],
            ignore_index=True,
        )
    frames = {
        "base": base,
        "factors": factor_frame,
        "labels": _with_optional_columns(
            _dummy_base_frame().assign(label_1=[0.1, 0.2]),
            include_extensions=include_extensions,
            extension_column="label_2",
            values=[0.3, 0.4],
        ),
        "filters": _dummy_base_frame().assign(eligible=[True, False]),
        "context": _with_optional_columns(
            _dummy_base_frame().assign(context_1=["x", "y"]),
            include_extensions=include_extensions,
            extension_column="context_2",
            values=["u", "v"],
        ),
    }
    if drop_factor_key is not None:
        frames["factors"] = frames["factors"].drop(columns=[drop_factor_key])
    for data_id, frame in frames.items():
        _write_processed_frame(layout, processed_catalog, data_id, frame)


def _write_processed_frame(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    data_id: str,
    frame: pd.DataFrame,
) -> Path:
    spec = processed_catalog.get(data_id)
    return write_dataset(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
        frame,
        spec=spec,
    )


def _demo_research_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            DEMO_FACTOR_COLUMN: [1.0, 2.0, 2.0, 1.0],
            DEMO_LABEL_COLUMN: [0.01, 0.03, 0.02, 0.00],
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "market_type": [4, 4, 4, 4],
        }
    )


def _dummy_base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["dummy_market", "dummy_market"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )


def _with_optional_columns(
    frame: pd.DataFrame,
    *,
    include_extensions: bool,
    extension_column: str,
    values: Sequence[object],
) -> pd.DataFrame:
    if include_extensions:
        return frame.assign(**{extension_column: list(values)})
    return frame
