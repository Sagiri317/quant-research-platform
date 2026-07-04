from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage.datasets import compare_datasets, write_dataset
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.manifests import (
    build_layer_summary,
    build_run_data_summary,
    write_internal_artifact_manifest,
    write_manifest,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
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
    write_internal_artifact_manifest(
        present / "_manifest" / "manifest.json",
        _artifact_manifest_payload(
            data_id="present",
            row_count=3,
            date_min="2014-01-02",
            date_max="2014-01-03",
        ),
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
    assert payload["layer_validation_status"] == "fail"


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
    assert payload["run_validation_status"] == "pass"
    assert [layer["logical_layer"] for layer in payload["layers"]] == [
        "processed",
        "research",
    ]


def _artifact_manifest_payload(
    *,
    data_id: str,
    row_count: int,
    date_min: str,
    date_max: str,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-dataset-manifest",
        "data_id": data_id,
        "row_count": row_count,
        "date_min": date_min,
        "date_max": date_max,
        "duplicate_primary_key_count": 0,
        "validation_status": "pass",
    }
