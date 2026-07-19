from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.storage._dataset_schema import (
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
)
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    write_dataset,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.manifests import (
    build_layer_summary,
    build_run_data_summary,
    write_manifest,
)
from tests.storage import low_level_fixtures
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
    write_low_level_parquet_artifact,
)


def _write_arrow_value_artifact(
    root: Path,
    frame: pd.DataFrame,
    value_array: pa.Array,
) -> None:
    write_low_level_parquet_artifact(
        root,
        frame,
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    data_root = root / "data"
    path = next(data_root.rglob("*.parquet"))
    arrays = [
        value_array if column == "value" else pa.array(frame[column])
        for column in frame.columns
    ]
    logical_dtypes = logical_dtype_payload_from_frame(frame)
    table = pa.Table.from_arrays(arrays, names=frame.columns).replace_schema_metadata(
        {b"qrp.logical-dtypes": logical_dtype_metadata_bytes(logical_dtypes)}
    )
    pq.write_table(table, path)
    spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    low_level_fixtures._write_runtime_manifest(
        root,
        data_root,
        [path],
        frame=frame,
        spec=spec,
    )


def test_dataset_writer_rejects_outside_local_only_root_before_iterating_chunks(
    tmp_path: Path,
) -> None:
    iterated = False

    def chunks():
        nonlocal iterated
        iterated = True
        yield make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
        )

    with pytest.raises(ValueError, match="outside_local_only_boundary"):
        write_dataset(
            tmp_path / "outside-local-only" / "x",
            chunks(),
            spec=make_processed_dataset_spec(),
        )

    assert not iterated
    assert not (tmp_path / "public" / "x").exists()


def test_dataset_writer_accepts_private_processed_artifact_root(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)

    write_dataset(
        root,
        make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
        ),
        spec=make_processed_dataset_spec(),
    )

    assert (root / "spec.json").is_file()
    assert (root / "_manifest" / "manifest.json").is_file()
    assert (root / "data" / "year=2014" / "part-000.parquet").is_file()


def test_projected_dataset_fingerprint_includes_primary_key_columns(
    tmp_path: Path,
) -> None:
    left = tmp_path / "data" / "processed" / "a_share" / "factor" / "left"
    right = tmp_path / "data" / "processed" / "a_share" / "factor" / "right"
    right_frame = make_default_dataset_frame(
        asset_ids=("asset_001",),
        dates=("2014-01-02",),
        values=(1.0,),
    ).assign(asset_id="asset_002")

    write_dataset(
        left,
        make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
        ),
        spec=make_processed_dataset_spec("left"),
    )
    write_dataset(right, right_frame, spec=make_processed_dataset_spec("right"))

    comparison = compare_datasets(left, right, columns=["value"])

    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.matches is False


def test_projected_compare_binds_only_selected_and_primary_key_schema(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame(
        asset_ids=("asset_001",),
        dates=("2014-01-02",),
        values=(1.0,),
    )
    spec = make_processed_dataset_spec("projected_left")
    projected_roles = {**spec.column_roles, "ignored": ColumnRole.CONTEXT}
    left = make_dataset_artifact_root(tmp_path, "projected_left")
    ignored = make_dataset_artifact_root(tmp_path, "projected_ignored")
    selected = make_dataset_artifact_root(tmp_path, "projected_selected")
    primary_key = make_dataset_artifact_root(tmp_path, "projected_primary_key")
    write_dataset(
        left,
        base.assign(ignored=1),
        spec=replace(spec, column_roles=projected_roles),
    )
    for root, frame in (
        (ignored, base.assign(ignored="different")),
        (selected, base.assign(value="different", ignored=1)),
        (primary_key, base.assign(asset_id=1, ignored=1)),
    ):
        write_dataset(
            root,
            frame,
            spec=replace(
                make_processed_dataset_spec(root.name),
                column_roles=projected_roles,
            ),
        )

    assert compare_datasets(left, ignored, columns=["value"]).status == "MATCH"
    assert (
        compare_datasets(left, selected, columns=["value"]).status == "SCHEMA_MISMATCH"
    )
    assert (
        compare_datasets(left, primary_key, columns=["value"]).status
        == "SCHEMA_MISMATCH"
    )


def test_arrow_map_entry_order_is_layout_not_content(tmp_path: Path) -> None:
    map_type = pa.map_(pa.string(), pa.float64())

    def write(root: Path, entries: list[tuple[str, float]]) -> None:
        frame = make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(b"placeholder",),
        )
        _write_arrow_value_artifact(
            root,
            frame,
            pa.array([entries], type=map_type),
        )

    left = make_dataset_artifact_root(tmp_path, "map_order_left")
    right = make_dataset_artifact_root(tmp_path, "map_order_right")
    write(left, [("b", 2.0), ("a", 1.0)])
    write(right, [("a", 1.0), ("b", 2.0)])

    assert (
        fingerprint_dataset(left).content_hash
        == fingerprint_dataset(right).content_hash
    )
    assert compare_datasets(left, right).status == "MATCH"


def test_non_ascii_nested_arrow_type_compares_end_to_end(tmp_path: Path) -> None:
    nested_type = pa.struct([pa.field("指标", pa.int64())])

    def write(root: Path, value: int) -> None:
        frame = make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(b"placeholder",),
        )
        _write_arrow_value_artifact(
            root,
            frame,
            pa.array([{"指标": value}], type=nested_type),
        )

    left = make_dataset_artifact_root(tmp_path, "unicode_nested_left")
    equal = make_dataset_artifact_root(tmp_path, "unicode_nested_equal")
    changed = make_dataset_artifact_root(tmp_path, "unicode_nested_changed")
    for root, value in ((left, 1), (equal, 1), (changed, 2)):
        write(root, value)

    assert fingerprint_dataset(left).row_count == 1
    assert compare_datasets(left, left).status == "MATCH"
    assert compare_datasets(left, equal).status == "MATCH"
    assert compare_datasets(left, changed).status == "CONTENT_MISMATCH"


def test_manifest_writer_sanitizes_review_safe_summary_payload(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "outputs" / "run" / "manifest.json",
        {
            "schema_id": "qrp-output-manifest",
            "path": "provider-internal-detail",
            "traceback": "Traceback (most recent call last): provider internals",
            "credential_hint": "redacted credential detail",
        },
    )
    text = manifest.read_text(encoding="utf-8")

    assert "C:" not in text
    assert "/mnt/" not in text
    assert "000001" not in text
    assert "abcdef123456" not in text


def test_manifest_writer_rejects_existing_target_file(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "run" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_manifest(manifest, {"schema_id": "qrp-output-manifest"})

    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_output_already_exists_error_review_safe_summary() -> None:
    assert OutputAlreadyExistsError().to_review_safe_summary() == {
        "status": "failed",
        "failure_code": "output_already_exists",
    }


def test_build_layer_summary_detects_missing_dataset_manifests(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    present = layer_root / "a_share" / "factor" / "present"
    missing = layer_root / "a_share" / "label" / "missing"
    write_dataset(
        present,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("present"),
    )

    payload = build_layer_summary(
        layer_root=layer_root,
        logical_layer="processed",
        dataset_roots=(present, missing),
    )

    assert payload["dataset_count"] == 1
    assert payload["total_rows"] == 3
    assert payload["missing_dataset_manifests"] == [
        "data/processed/a_share/label/missing/_manifest/manifest.json"
    ]
    assert payload["layer_validation_status"] == "FAIL"


def test_build_run_data_summary_aggregates_dynamic_layers(tmp_path: Path) -> None:
    processed = build_layer_summary(
        layer_root=tmp_path / "data" / "processed",
        logical_layer="processed",
        dataset_roots=(),
    )
    research = build_layer_summary(
        layer_root=tmp_path / "data" / "research",
        logical_layer="research",
        dataset_roots=(),
    )

    payload = build_run_data_summary(
        layer_summaries=(processed, research),
    )

    assert payload["layer_count"] == 2
    assert payload["run_validation_status"] == "PASS"
    assert [layer["logical_layer"] for layer in payload["layers"]] == [
        "processed",
        "research",
    ]
