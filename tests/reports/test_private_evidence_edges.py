from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quant_research_platform.reports import private_evidence


def test_private_safety_scan_detects_text_and_json_leaks(tmp_path: Path) -> None:
    safe = tmp_path / "safe.md"
    safe.write_text("aggregate only", encoding="utf-8")
    json_leak = tmp_path / "payload.json"
    json_leak.write_text('{"note": "row_sample"}', encoding="utf-8")
    text_leak = tmp_path / "payload.md"
    text_leak.write_text("source_asset_id", encoding="utf-8")

    result = private_evidence._private_safety_scan(
        (safe, json_leak, text_leak, tmp_path / "missing.md")
    )

    assert result["status"] == "FAIL"
    assert result["checked_count"] == 3
    assert result["missing_count"] == 1
    assert {item["reason"] for item in result["failures"]} == {
        "row_sample",
        "source_asset_id",
    }


def test_private_evidence_unsafe_reason_variants() -> None:
    assert private_evidence._unsafe_text_reason(r"C:\\private\\path") == "absolute_path"
    assert (
        private_evidence._unsafe_text_reason("daily__vendor__export-20240101.csv")
        == "vendor_filename"
    )
    assert private_evidence._unsafe_text_reason("000001.SZ") == "stock_sample"
    assert private_evidence._unsafe_text_reason("token: abcdefghi123") == "credential"
    assert private_evidence._unsafe_json_reason("{bad") == "invalid_json"
    assert private_evidence._unsafe_json_reason('{"safe": ["aggregate"]}') is None


def test_validation_summary_classifies_manifest_and_safety_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "panel" / "daily"
    layer_manifest = tmp_path / "manifests" / "layer.json"
    layer_manifest.parent.mkdir(parents=True)
    layer_manifest.write_text(
        '{"logical_layer": "processed", "layer_validation_status": "fail"}',
        encoding="utf-8",
    )
    text_leak = tmp_path / "summary.md"
    text_leak.write_text("row_sample", encoding="utf-8")

    manifest = {
        "data_id": "daily",
        "validation_status": "fail",
        "primary_key": ["market_id"],
        "null_count_summary": {"market_id": 1},
        "duplicate_primary_key_count": 1,
        "non_finite_count_summary": {"value": 1},
        "logical_layer": "processed",
        "source_asset_id": "source_local",
        "date_min": "2024-01-02",
        "date_max": "2024-01-03",
        "partition_count": 1,
        "factor_nan_profile": {"factor": 2},
        "label_nan_profile": {"label": 3},
    }

    monkeypatch.setattr(
        private_evidence,
        "inspect_dataset",
        lambda _root: SimpleNamespace(manifest=manifest),
    )
    result = private_evidence.build_validation_summary(
        run_id="run_demo",
        output_root=tmp_path / ".run" / "output",
        dataset_roots=(dataset_root,),
        layer_manifest_paths=(layer_manifest, tmp_path / "missing_layer.json"),
        generated_text_paths=(text_leak,),
    )

    assert result["status"] == "FAIL"
    assert set(result["failure_codes"]) >= {
        "dataset_validation_failed",
        "primary_key_nulls",
        "duplicate_primary_key",
        "non_finite_values",
        "source_local_key_leakage",
        "layer_manifest_missing",
        "layer_validation_failed",
        "private_safety_scan_failed",
    }
    assert result["factor_nan_profile"] == {"factor": 2}
    assert result["label_nan_profile"] == {"label": 3}


def test_private_evidence_writers_emit_utf8(tmp_path: Path) -> None:
    payload = {"status": "PASS", "rows": 1}
    run_payload = {
        "run_id": "run_demo",
        "status": "PASS",
        "output_root": ".run/output",
        "validation_summary": {"status": "PASS"},
    }

    json_path = private_evidence.write_validation_summary(
        tmp_path / "summary.json",
        payload,
    )
    md_path = private_evidence.write_run_summary_md(
        tmp_path / "summary.md",
        run_payload,
    )
    run_json_path = private_evidence.write_run_summary_json(
        tmp_path / "run_summary.json",
        run_payload,
    )

    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert run_json_path.read_text(encoding="utf-8").endswith("\n")
    assert "status" in md_path.read_text(encoding="utf-8")


def test_private_evidence_mapping_and_logical_path_helpers() -> None:
    target: dict[str, int] = {"a": 1}
    private_evidence._merge_profile(target, {"a": 2, "b": None})

    assert target == {"a": 3, "b": 0}
    assert private_evidence._int_mapping([("a", 1)]) == {}
    assert private_evidence._logical_path(Path("plain") / "path") == "plain/path"
    assert private_evidence._logical_path(Path(".run") / "demo" / "file.md") == (
        ".run/demo/file.md"
    )
    assert private_evidence._utc_now().endswith("Z")
