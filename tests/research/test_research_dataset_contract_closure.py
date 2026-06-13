from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_research_dataset_spec_requires_base_input_in_inputs() -> None:
    with pytest.raises(ValueError, match="base_input_data_id_not_in_inputs: base"):
        _dataset_spec(
            input_data_ids=("features", "labels"),
            base_input_data_id="base",
        )


def test_research_dataset_key_coverage_is_defensively_copied() -> None:
    source = {"features": "sparse"}

    spec = _dataset_spec(key_coverage=source)
    source["features"] = "strict"

    assert spec.key_coverage["features"] == "sparse"


def test_research_dataset_validation_rejects_noncanonical_primary_key() -> None:
    from dataclasses import replace

    from quant_research_platform.research.datasets.validation import (
        validate_research_dataset,
    )

    spec = replace(_dataset_spec(), primary_key=("asset_id", "market_id", "timestamp"))

    with pytest.raises(ValueError, match="invalid_research_dataset_primary_key"):
        validate_research_dataset(_base_frame().assign(feature_1=[1.0, 2.0]), spec)


def test_research_dataset_assembler_uses_explicit_base_not_input_order(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        _dataset_spec(
            input_data_ids=("features", "base", "labels", "filters", "context"),
            base_input_data_id="base",
        )
    )
    _write_processed_inputs(layout, processed_catalog)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert result["asset_id"].tolist() == ["asset_a", "asset_b"]
    assert result.columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "feature_1",
        "label_1",
        "eligible",
        "context_1",
    ]


def test_research_dataset_columns_extend_without_assembler_change(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog(include_extensions=True)
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        _dataset_spec(
            feature_columns=("feature_1", "feature_2"),
            label_columns=("label_1", "label_2"),
            context_columns=("context_1", "context_2"),
            include_extensions=True,
        )
    )
    _write_processed_inputs(layout, processed_catalog, include_extensions=True)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert result.columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "feature_1",
        "feature_2",
        "label_1",
        "label_2",
        "eligible",
        "context_1",
        "context_2",
    ]


def test_dummy_market_assembles_without_a_share_import(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert set(result["market_id"]) == {"dummy_market"}


def test_missing_join_key_fails_closed(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog(missing_feature_key=True)
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog, drop_feature_key="asset_id")

    with pytest.raises(
        ValueError,
        match="missing_research_join_key: features: asset_id",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
        )


def test_duplicate_output_column_fails_closed() -> None:
    processed_catalog = _processed_catalog(duplicate_feature_context=True)
    spec = _dataset_spec(context_columns=("feature_1",))

    with pytest.raises(
        ValueError,
        match="duplicate_research_output_columns: feature_1",
    ):
        _validate_inputs(spec, processed_catalog)


@pytest.mark.parametrize(
    ("field_name", "bad_column", "expected_error"),
    [
        ("feature_columns", "eligible", "invalid_research_feature_columns"),
        ("label_columns", "feature_1", "invalid_research_label_columns"),
        ("filter_columns", "context_1", "invalid_research_filter_columns"),
        ("context_columns", "label_1", "invalid_research_context_columns"),
    ],
)
def test_research_dataset_role_separation_is_enforced(
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    processed_catalog = _processed_catalog()
    kwargs = {
        "feature_columns": ("feature_1",),
        "label_columns": ("label_1",),
        "filter_columns": ("eligible",),
        "context_columns": ("context_1",),
    }
    kwargs[field_name] = (bad_column,)
    spec = _dataset_spec(**kwargs)

    with pytest.raises(ValueError, match=expected_error):
        _validate_inputs(spec, processed_catalog)


def test_output_row_order_is_stable_sort_by_primary_key(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog, reverse_base=True)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    ordered_keys = result.loc[:, ["market_id", "asset_id", "timestamp"]]
    assert ordered_keys["market_id"].tolist() == ["dummy_market", "dummy_market"]
    assert ordered_keys["asset_id"].tolist() == ["asset_a", "asset_b"]
    assert ordered_keys["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


def test_research_dataset_fails_closed_on_empty_base_sessions(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog)

    with pytest.raises(ValueError, match="no_research_dataset_sessions"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2025-01-01", "2025-01-02"),
        )


def test_research_dataset_fails_closed_on_empty_inputs() -> None:
    with pytest.raises(ValueError, match="base_input_data_id_not_in_inputs"):
        _dataset_spec(input_data_ids=(), base_input_data_id="")


def test_strict_research_input_missing_base_key_fails_in_assembler(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog, truncate_features=True)

    with pytest.raises(ValueError, match="missing_research_input_keys: features: 1"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
        )


def test_sparse_research_input_missing_base_key_passes_with_nan(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        _dataset_spec(key_coverage={"features": "sparse"})
    )
    _write_processed_inputs(layout, processed_catalog, truncate_features=True)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert result.loc[result["asset_id"].eq("asset_a"), "feature_1"].tolist() == [1.0]
    assert pd.isna(result.loc[result["asset_id"].eq("asset_b"), "feature_1"]).all()


def test_research_dataset_key_coverage_is_persisted_for_all_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec(key_coverage={"features": "sparse"}))
    _write_processed_inputs(layout, processed_catalog, truncate_features=True)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    root = tmp_path / "data/research/dummy_market/dummy_dataset"
    spec_payload = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    manifest_payload = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "base": "strict",
        "features": "sparse",
        "labels": "strict",
        "filters": "strict",
        "context": "strict",
    }
    assert spec_payload["key_coverage"] == expected
    assert manifest_payload["key_coverage"] == expected


def test_strict_research_input_value_nan_does_not_count_as_missing_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    _write_processed_inputs(layout, processed_catalog, feature_values=[1.0, None])

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert pd.isna(result.loc[result["asset_id"].eq("asset_b"), "feature_1"]).all()


@pytest.mark.parametrize("coverage_policy", ["strict", "sparse"])
def test_research_input_extra_key_outside_base_fails_closed(
    tmp_path: Path,
    coverage_policy: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = _processed_catalog()
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(
        _dataset_spec(key_coverage={"features": coverage_policy})
    )
    _write_processed_inputs(layout, processed_catalog, extra_feature_key=True)

    with pytest.raises(ValueError, match="extra_research_input_keys: features: 1"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
        )


def test_research_dataset_reports_incompatible_and_missing_required_specs() -> None:
    incompatible = _processed_catalog()
    bad_features = _processed_spec(
        "features",
        ProcessedDataKind.FACTOR,
        (*platform_entity_time_key(), "feature_1"),
        role=ColumnRole.FACTOR,
        primary_key=("asset_id", "market_id", "timestamp"),
    )
    incompatible = ProcessedDataCatalog()
    for data_id in ("base", "labels", "filters", "context"):
        incompatible.register(_processed_catalog().get(data_id))
    incompatible.register(bad_features)
    spec = _dataset_spec()
    with pytest.raises(ValueError, match="incompatible_research_input_keys: features"):
        _validate_inputs(spec, incompatible)

    missing_label = _processed_catalog()
    spec = _dataset_spec(label_columns=("label_2",))
    with pytest.raises(ValueError, match="missing_research_required_columns: label_2"):
        _validate_inputs(spec, missing_label)


def test_research_dataset_measured_bytes_handles_empty_frame() -> None:
    from quant_research_platform.research.datasets.assembler import (
        _measured_bytes_per_session,
    )

    assert (
        _measured_bytes_per_session(
            pd.DataFrame(),
            sample_sessions=(pd.Timestamp("2024-01-02"),),
            target_session_count=1,
        )
        == 1
    )


def _validate_inputs(
    spec: ResearchDatasetSpec,
    processed_catalog: ProcessedDataCatalog,
) -> None:
    from quant_research_platform.research.datasets.assembler import (
        _validate_research_input_specs,
    )

    _validate_research_input_specs(spec, processed_catalog)


def _dataset_spec(
    *,
    input_data_ids: tuple[str, ...] = (
        "base",
        "features",
        "labels",
        "filters",
        "context",
    ),
    base_input_data_id: str = "base",
    feature_columns: tuple[str, ...] = ("feature_1",),
    label_columns: tuple[str, ...] = ("label_1",),
    filter_columns: tuple[str, ...] = ("eligible",),
    context_columns: tuple[str, ...] = ("context_1",),
    include_extensions: bool = False,
    key_coverage: dict[str, str] | None = None,
) -> ResearchDatasetSpec:
    return ResearchDatasetSpec(
        data_id="dummy_dataset",
        market_id=MarketId("dummy_market"),
        input_data_ids=input_data_ids,
        base_input_data_id=base_input_data_id,
        feature_columns=feature_columns,
        label_columns=label_columns,
        filter_columns=filter_columns,
        context_columns=context_columns,
        key_coverage=key_coverage or {},
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            **_roles(feature_columns, ColumnRole.FEATURE),
            **_roles(label_columns, ColumnRole.LABEL),
            **_roles(filter_columns, ColumnRole.MASK),
            **_roles(context_columns, ColumnRole.CONTEXT),
        },
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _processed_catalog(
    *,
    include_extensions: bool = False,
    duplicate_feature_context: bool = False,
    missing_feature_key: bool = False,
) -> ProcessedDataCatalog:
    specs = (
        _processed_spec("base", ProcessedDataKind.PANEL, platform_entity_time_key()),
        _processed_spec(
            "features",
            ProcessedDataKind.FACTOR,
            (
                *(
                    ("market_id", "timestamp")
                    if missing_feature_key
                    else platform_entity_time_key()
                ),
                "feature_1",
                *(("feature_2",) if include_extensions else ()),
            ),
            role=ColumnRole.FACTOR,
            primary_key=(
                ("market_id", "timestamp")
                if missing_feature_key
                else platform_entity_time_key()
            ),
        ),
        _processed_spec(
            "labels",
            ProcessedDataKind.LABEL,
            (
                *platform_entity_time_key(),
                "label_1",
                *(("label_2",) if include_extensions else ()),
            ),
            role=ColumnRole.LABEL,
        ),
        _processed_spec(
            "filters",
            ProcessedDataKind.ANNOTATION,
            (*platform_entity_time_key(), "eligible"),
            role=ColumnRole.MASK,
        ),
        _processed_spec(
            "context",
            ProcessedDataKind.ANNOTATION,
            (
                *platform_entity_time_key(),
                "context_1",
                *(("context_2",) if include_extensions else ()),
                *(("feature_1",) if duplicate_feature_context else ()),
            ),
            role=ColumnRole.CONTEXT,
        ),
    )
    catalog = ProcessedDataCatalog()
    for spec in specs:
        catalog.register(spec)
    return catalog


def _processed_spec(
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
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MarketId("dummy_market"),
        kind=kind,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=primary_key,
        partition_columns=("year",),
    )


def _roles(columns: tuple[str, ...], role: ColumnRole) -> dict[str, ColumnRole]:
    return {column: role for column in columns}


def _write_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    *,
    include_extensions: bool = False,
    drop_feature_key: str | None = None,
    reverse_base: bool = False,
    truncate_features: bool = False,
    extra_feature_key: bool = False,
    feature_values: list[object] | None = None,
) -> None:
    base = _base_frame()
    if reverse_base:
        base = base.iloc[::-1].reset_index(drop=True)
    feature_frame = _with_optional_columns(
        _base_frame().assign(feature_1=feature_values or [1.0, 2.0]),
        include_extensions=include_extensions,
        extension_column="feature_2",
        values=[10.0, 20.0],
    )
    if truncate_features:
        feature_frame = feature_frame.iloc[[0]].reset_index(drop=True)
    if extra_feature_key:
        feature_frame = pd.concat(
            [
                feature_frame,
                pd.DataFrame(
                    {
                        "market_id": ["dummy_market"],
                        "asset_id": ["asset_extra"],
                        "timestamp": [pd.Timestamp("2024-01-03")],
                        "feature_1": [3.0],
                    }
                ),
            ],
            ignore_index=True,
        )
    frames = {
        "base": base,
        "features": feature_frame,
        "labels": _with_optional_columns(
            _base_frame().assign(label_1=[0.1, 0.2]),
            include_extensions=include_extensions,
            extension_column="label_2",
            values=[0.3, 0.4],
        ),
        "filters": _base_frame().assign(eligible=[True, False]),
        "context": _with_optional_columns(
            _base_frame().assign(context_1=["x", "y"]),
            include_extensions=include_extensions,
            extension_column="context_2",
            values=["u", "v"],
        ),
    }
    if drop_feature_key is not None:
        frames["features"] = frames["features"].drop(columns=[drop_feature_key])
    for data_id, frame in frames.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame,
            spec=spec,
        )


def _base_frame() -> pd.DataFrame:
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
    values: list[object],
) -> pd.DataFrame:
    if include_extensions:
        return frame.assign(**{extension_column: values})
    return frame
