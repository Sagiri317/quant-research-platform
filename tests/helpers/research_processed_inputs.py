from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.paths import StorageLayout
from tests.helpers.research_specs import DEMO_FACTOR_COLUMN, DEMO_LABEL_COLUMN


def make_demo_processed_catalog() -> ProcessedDataCatalog:
    return ProcessedDataCatalog(
        (
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
            ),
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
            ),
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
            ),
        )
    )


def make_dummy_processed_catalog(
    *,
    include_extensions: bool = False,
    duplicate_factor_context: bool = False,
    missing_factor_key: bool = False,
) -> ProcessedDataCatalog:
    specs = (
        _dummy_processed_spec(
            "base",
            ProcessedDataKind.PANEL,
            PLATFORM_ASSET_TIME_KEY,
        ),
        _dummy_processed_spec(
            "factors",
            ProcessedDataKind.FACTOR,
            (
                *(
                    ("market_id", "timestamp")
                    if missing_factor_key
                    else PLATFORM_ASSET_TIME_KEY
                ),
                "factor_1",
                *(("factor_2",) if include_extensions else ()),
            ),
            role=ColumnRole.FACTOR,
            primary_key=(
                ("market_id", "timestamp")
                if missing_factor_key
                else PLATFORM_ASSET_TIME_KEY
            ),
        ),
        _dummy_processed_spec(
            "labels",
            ProcessedDataKind.LABEL,
            (
                *PLATFORM_ASSET_TIME_KEY,
                "label_1",
                *(("label_2",) if include_extensions else ()),
            ),
            role=ColumnRole.LABEL,
        ),
        _dummy_processed_spec(
            "filters",
            ProcessedDataKind.ANNOTATION,
            (*PLATFORM_ASSET_TIME_KEY, "eligible"),
            role=ColumnRole.MASK,
        ),
        _dummy_processed_spec(
            "context",
            ProcessedDataKind.ANNOTATION,
            (
                *PLATFORM_ASSET_TIME_KEY,
                "context_1",
                *(("context_2",) if include_extensions else ()),
                *(("factor_1",) if duplicate_factor_context else ()),
            ),
            role=ColumnRole.CONTEXT,
        ),
    )
    return ProcessedDataCatalog(specs)


def _processed_spec(
    *,
    data_id: str,
    market_id: MarketId,
    kind: ProcessedDataKind,
    column_roles: Mapping[str, ColumnRole],
    primary_key: tuple[str, ...] = PLATFORM_ASSET_TIME_KEY,
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
    primary_key: tuple[str, ...] = PLATFORM_ASSET_TIME_KEY,
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


def write_demo_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    frame: pd.DataFrame | None = None,
    label_frame: pd.DataFrame | None = None,
) -> None:
    frame = _demo_research_rows() if frame is None else frame
    labels = frame if label_frame is None else label_frame
    frames_by_id = {
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
    _write_processed_frames(
        layout,
        processed_catalog,
        {data_id: frame.loc[:, columns] for data_id, columns in frames_by_id.items()}
        | {
            "demo_labels": labels.loc[
                :,
                [
                    "market_id",
                    "asset_id",
                    "timestamp",
                    DEMO_LABEL_COLUMN,
                ],
            ],
        },
    )


def write_dummy_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    include_extensions: bool = False,
    drop_factor_key: str | None = None,
    truncate_factors: bool = False,
    extra_factor_key: bool = False,
    factor_values: Sequence[object] | None = None,
) -> None:
    base = _dummy_base_frame()
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
    _write_processed_frames(layout, processed_catalog, frames)


def _write_processed_frames(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    frames_by_id: Mapping[str, pd.DataFrame],
) -> None:
    for data_id, frame in frames_by_id.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
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


def assembly_source(
    tmp_path, frame, *, data_id="external_table", units=None, availability=None
):
    roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    roles.update(
        {column: ColumnRole.CONTEXT for column in frame if column not in roles}
    )
    roles.update(
        {
            column: ColumnRole.FACTOR
            for column in (
                "momentum_20d",
                "log_total_market_cap",
                "log_float_market_cap",
            )
            if column in frame
        }
    )
    if "eligible" in frame:
        roles["eligible"] = ColumnRole.UNIVERSE
    for column in ("return_next", "forward_return_20d"):
        if column in frame:
            roles[column] = ColumnRole.LABEL
    spec = ProcessedDataSpec(
        data_id=data_id,
        market_id="a_share",
        kind="panel",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
        column_roles=roles,
        column_units=units or {},
        label_availability_columns=availability or {},
    )
    root = tmp_path / data_id
    write_dataset(
        root, (frame.iloc[i : i + 17] for i in range(0, len(frame), 17)), spec=spec
    )
    return root, spec


def assembly_config(tmp_path, root, **choices):
    return {
        "market_id": "a_share",
        "dataset_id": "external_research",
        "inputs": {root.name: str(root)},
        "factor_columns": ["momentum_20d"],
        "label_columns": [],
        "feature_recipe": None,
        "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
        "output_root": str(tmp_path / "assembled"),
        **choices,
    }


def assembly_frame(n=120):
    return pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i:03d}" for i in range(n)],
            "timestamp": pd.Timestamp("2024-01-02"),
            "momentum_20d": np.sin(np.arange(n)),
        }
    )
