from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.reports.private_evidence import (
    build_private_evidence_summary,
    build_validation_summary,
    write_run_summary_md,
)
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.storage.manifests import write_internal_artifact_manifest


def test_validation_summary_detects_manifest_failures_and_private_safety(tmp_path):
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "demo"
    write_internal_artifact_manifest(
        dataset_root / "_manifest" / "manifest.json",
        {
            "schema_version": "0.1",
            "run_id": "run",
            "logical_layer": "processed",
            "dataset_id": "demo",
            "primary_key": ["market_id", "asset_id", "timestamp"],
            "duplicate_primary_key_count": 1,
            "null_count_summary": {"market_id": 0, "asset_id": 0, "timestamp": 0},
            "non_finite_count_summary": {"factor": 0},
            "date_min": "2014-01-02",
            "date_max": "2014-01-03",
            "row_count": 2,
            "validation_status": "fail",
        },
    )
    unsafe_summary = tmp_path / "summary.md"
    unsafe_summary.write_text("source_asset_id leaked", encoding="utf-8")

    summary = build_validation_summary(
        run_id="run",
        output_root=tmp_path,
        dataset_manifest_paths=(dataset_root / "_manifest" / "manifest.json",),
        layer_manifest_paths=(),
        generated_text_paths=(unsafe_summary,),
    )

    assert summary["status"] == "FAIL"
    assert "dataset_validation_failed" in summary["failure_codes"]
    assert "private_safety_scan_failed" in summary["failure_codes"]


def test_validation_summary_accepts_schema_only_dataset_manifest(tmp_path):
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factors" / "demo"
    write_internal_artifact_manifest(
        dataset_root / "_manifest" / "manifest.json",
        {
            "schema_version": "0.1",
            "market_id": "a_share",
            "category": "factors",
            "processed_data_id": "demo",
            "primary_key": ["market_id", "asset_id", "timestamp"],
            "partition_columns": ["year"],
            "required_columns": ["market_id", "asset_id", "timestamp", "factor"],
            "column_roles": {
                "market_id": "key",
                "asset_id": "key",
                "timestamp": "timestamp",
                "factor": "factor",
            },
        },
    )

    summary = build_validation_summary(
        run_id="s8",
        output_root=tmp_path / ".run" / "s8",
        dataset_manifest_paths=(dataset_root / "_manifest" / "manifest.json",),
        layer_manifest_paths=(),
        generated_text_paths=(),
    )

    assert summary["status"] == "PASS"
    assert "dataset_validation_failed" not in summary["failure_codes"]


def test_private_evidence_summary_and_md_avoid_absolute_paths(tmp_path):
    validation = {"status": "PASS", "failure_codes": []}
    evidence = build_private_evidence_summary(
        run_id="run",
        date_range={"start_date": "2014-01-01", "end_date": "2016-12-31"},
        manifests={"dataset_count": 1, "layer_count": 4},
        timings={"total_duration_seconds": 1.25},
        validation_summary=validation,
        output_root=Path(".run") / "s1",
    )
    md_path = write_run_summary_md(tmp_path / "summary.md", evidence)
    text = md_path.read_text(encoding="utf-8")

    assert evidence["status"] == "PASS"
    assert ".run/s1" in text
    assert str(tmp_path) not in text


@pytest.mark.parametrize(
    "text",
    [
        r"C:\Users\DELL\data\raw\provider\file.csv",
        "vendor_export_abc__dataset__export-20240101.csv",
        "row_sample: value",
        "stock sample 000001.SZ",
        "source_asset_id",
        "token=abcdef123456",
    ],
)
def test_private_safety_scan_rejects_private_markers(tmp_path, text):
    path = tmp_path / "unsafe.md"
    path.write_text(text, encoding="utf-8")

    summary = build_validation_summary(
        run_id="run",
        output_root=tmp_path,
        dataset_manifest_paths=(),
        layer_manifest_paths=(),
        generated_text_paths=(path,),
    )

    assert summary["status"] == "FAIL"
    assert "private_safety_scan_failed" in summary["failure_codes"]


def test_private_safety_scan_does_not_treat_json_numbers_as_stock_samples(tmp_path):
    safe = tmp_path / "safe.json"
    unsafe = tmp_path / "unsafe.json"
    atomic_write_json(safe, {"row_count": 344350, "duration_seconds": 0.000123})
    atomic_write_json(unsafe, {"example": "000001.SZ"})

    safe_summary = build_validation_summary(
        run_id="run",
        output_root=tmp_path,
        dataset_manifest_paths=(),
        layer_manifest_paths=(),
        generated_text_paths=(safe,),
    )
    unsafe_summary = build_validation_summary(
        run_id="run",
        output_root=tmp_path,
        dataset_manifest_paths=(),
        layer_manifest_paths=(),
        generated_text_paths=(unsafe,),
    )

    assert safe_summary["status"] == "PASS"
    assert unsafe_summary["status"] == "FAIL"
