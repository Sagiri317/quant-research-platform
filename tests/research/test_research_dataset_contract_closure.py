from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_dummy_processed_catalog,
    write_dummy_processed_inputs,
)
from tests.helpers.research_specs import (
    make_dummy_research_dataset_spec,
    make_research_dataset_catalog,
)


def test_research_dataset_spec_requires_base_input_in_inputs() -> None:
    with pytest.raises(ValueError, match="base_input_data_id_not_in_inputs: base"):
        make_dummy_research_dataset_spec(
            input_data_ids=("factors", "labels"),
            base_input_data_id="base",
        )


def test_research_dataset_key_coverage_is_defensively_copied() -> None:
    source = {"factors": "sparse"}

    spec = make_dummy_research_dataset_spec(key_coverage=source)
    source["factors"] = "strict"

    assert spec.key_coverage["factors"] == "sparse"


def test_research_dataset_validation_rejects_noncanonical_primary_key() -> None:
    from quant_research_platform.research.datasets.validation import (
        validate_research_dataset,
    )

    spec = replace(
        make_dummy_research_dataset_spec(),
        primary_key=("asset_id", "market_id", "timestamp"),
    )

    frame = pd.DataFrame(
        {
            "market_id": ["dummy_market", "dummy_market"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "factor_1": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_research_dataset_primary_key"):
        validate_research_dataset(frame, spec)


def test_research_dataset_assembler_uses_explicit_base_not_input_order(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec(
            input_data_ids=("factors", "base", "labels", "filters", "context"),
            base_input_data_id="base",
        )
    )
    write_dummy_processed_inputs(layout, processed_catalog)

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
        "factor_1",
        "label_1",
        "eligible",
        "context_1",
    ]


def test_research_dataset_columns_extend_without_assembler_change(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog(
        include_extensions=True,
    )
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec(
            factor_columns=("factor_1", "factor_2"),
            label_columns=("label_1", "label_2"),
            context_columns=("context_1", "context_2"),
        )
    )
    write_dummy_processed_inputs(layout, processed_catalog, include_extensions=True)

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
        "factor_1",
        "factor_2",
        "label_1",
        "label_2",
        "eligible",
        "context_1",
        "context_2",
    ]


def test_dummy_market_assembles_without_a_share_import(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog)

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
    processed_catalog = make_dummy_processed_catalog(
        missing_factor_key=True,
    )
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog, drop_factor_key="asset_id")

    with pytest.raises(
        ValueError,
        match="missing_research_join_key: factors: asset_id",
    ):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
        )


def test_duplicate_output_column_fails_closed() -> None:
    processed_catalog = make_dummy_processed_catalog(
        include_extensions=True,
        duplicate_factor_context=True,
    )
    spec = make_dummy_research_dataset_spec(
        factor_columns=("factor_2",),
        context_columns=("factor_1",),
    )

    with pytest.raises(
        ValueError,
        match="duplicate_research_output_columns: factor_1",
    ):
        _validate_inputs(spec, processed_catalog)


@pytest.mark.parametrize(
    ("field_name", "bad_column", "expected_error"),
    [
        ("factor_columns", "eligible", "research_input_column_overlap"),
        ("label_columns", "factor_1", "research_input_column_overlap"),
        ("filter_columns", "context_1", "research_semantic_column_role_invalid"),
        ("context_columns", "label_1", "research_semantic_column_role_invalid"),
    ],
)
def test_research_dataset_role_separation_is_enforced(
    field_name: str,
    bad_column: str,
    expected_error: str,
) -> None:
    processed_catalog = make_dummy_processed_catalog()
    kwargs = {
        "factor_columns": ("factor_1",),
        "label_columns": ("label_1",),
        "filter_columns": ("eligible",),
        "context_columns": ("context_1",),
    }
    kwargs[field_name] = (bad_column,)

    with pytest.raises(ValueError, match=expected_error):
        spec = make_dummy_research_dataset_spec(**kwargs)
        _validate_inputs(spec, processed_catalog)


def test_output_row_order_is_stable_sort_by_primary_key(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog, reverse_base=True)

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
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog)

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
        make_dummy_research_dataset_spec(
            input_data_ids=(),
            base_input_data_id="",
        )


def test_strict_research_input_missing_base_key_fails_in_assembler(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog, truncate_factors=True)

    with pytest.raises(ValueError, match="missing_research_input_keys: factors: 1"):
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
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec(
            key_coverage={"factors": "sparse"},
        )
    )
    write_dummy_processed_inputs(layout, processed_catalog, truncate_factors=True)

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert result.loc[result["asset_id"].eq("asset_a"), "factor_1"].tolist() == [1.0]
    assert pd.isna(result.loc[result["asset_id"].eq("asset_b"), "factor_1"]).all()


def test_research_dataset_semantic_snapshots_are_persisted_for_all_inputs(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec(
            key_coverage={"factors": "sparse"},
        )
    )
    write_dummy_processed_inputs(layout, processed_catalog, truncate_factors=True)

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
        "factors": "sparse",
        "labels": "strict",
        "filters": "strict",
        "context": "strict",
    }
    assert spec_payload["key_coverage"] == expected
    assert manifest_payload["key_coverage"] == expected
    expected_roles = {
        column: role.value
        for column, role in dataset_catalog.get("dummy_dataset").column_roles.items()
    }
    assert spec_payload["column_roles"] == expected_roles
    assert manifest_payload["column_roles"] == expected_roles


def test_strict_research_input_value_nan_does_not_count_as_missing_key(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec()
    )
    write_dummy_processed_inputs(layout, processed_catalog, factor_values=[1.0, None])

    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="dummy_dataset",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )

    result = read_dataset(tmp_path / "data/research/dummy_market/dummy_dataset")
    assert pd.isna(result.loc[result["asset_id"].eq("asset_b"), "factor_1"]).all()


@pytest.mark.parametrize("coverage_policy", ["strict", "sparse"])
def test_research_input_extra_key_outside_base_fails_closed(
    tmp_path: Path,
    coverage_policy: str,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    processed_catalog = make_dummy_processed_catalog()
    dataset_catalog = make_research_dataset_catalog(
        make_dummy_research_dataset_spec(
            key_coverage={"factors": coverage_policy},
        )
    )
    write_dummy_processed_inputs(layout, processed_catalog, extra_factor_key=True)

    with pytest.raises(ValueError, match="extra_research_input_keys: factors: 1"):
        assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=processed_catalog,
            dataset_id="dummy_dataset",
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
        )


def test_research_dataset_reports_incompatible_and_missing_required_specs() -> None:
    processed_catalog = make_dummy_processed_catalog()
    bad_factors = replace(
        processed_catalog.get("factors"),
        primary_key=("asset_id", "market_id", "timestamp"),
    )
    incompatible = ProcessedDataCatalog()
    for data_id in ("base", "labels", "filters", "context"):
        incompatible.register(processed_catalog.get(data_id))
    incompatible.register(bad_factors)
    spec = make_dummy_research_dataset_spec()
    with pytest.raises(ValueError, match="incompatible_research_input_keys: factors"):
        _validate_inputs(spec, incompatible)

    missing_label = make_dummy_processed_catalog()
    spec = make_dummy_research_dataset_spec(label_columns=("label_2",))
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
