from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quant_research_platform.artifacts.evidence import (
    write_run_summary_json,
    write_run_summary_markdown,
    write_validation_summary,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_unsafe_reason,
    evidence_text_unsafe_reason,
)
from quant_research_platform.reports import verification_evidence
from quant_research_platform.runtime.logical_paths import logical_path


def test_share_safety_scan_detects_text_and_json_leaks(tmp_path: Path) -> None:
    safe = tmp_path / "safe.md"
    safe.write_text("aggregate only", encoding="utf-8")
    json_leak = tmp_path / "payload.json"
    json_leak.write_text('{"note": "row_sample"}', encoding="utf-8")
    text_leak = tmp_path / "payload.md"
    text_leak.write_text("source_asset_id", encoding="utf-8")

    result = verification_evidence._share_safety_scan(
        (safe, json_leak, text_leak, tmp_path / "missing.md")
    )

    assert result["status"] == "FAIL"
    assert result["checked_count"] == 3
    assert result["missing_count"] == 1
    assert {item["reason"] for item in result["failures"]} == {
        "row_sample",
        "source_local_key",
    }


def test_verification_evidence_unsafe_reason_variants() -> None:
    assert evidence_text_unsafe_reason("data/raw/csmar/export.csv") == "absolute_path"
    assert (
        evidence_text_unsafe_reason("daily__vendor__export-20240101.csv")
        == "vendor_filename"
    )
    assert evidence_text_unsafe_reason("source_market_id") == "source_local_key"
    assert (
        evidence_text_unsafe_reason("runtime_payload=private-aggregate")
        == "runtime_payload"
    )
    assert evidence_json_unsafe_reason("{bad") == "invalid_json"
    assert evidence_json_unsafe_reason('{"safe": ["aggregate"]}') is None


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
        verification_evidence,
        "inspect_dataset",
        lambda _root: SimpleNamespace(manifest=manifest),
    )
    result = verification_evidence.build_validation_summary(
        run_id="run_demo",
        output_root=tmp_path / "verification-output" / "output",
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
        "share_safety_scan_failed",
    }
    assert result["factor_nan_profile"] == {"factor": 2}
    assert result["label_nan_profile"] == {"label": 3}


def test_verification_evidence_writers_emit_utf8(tmp_path: Path) -> None:
    payload = {"status": "PASS", "rows": 1}
    run_payload = verification_evidence.build_verification_evidence_payload(
        run_id="run_demo",
        date_range={"start_date": "2014-01-01", "end_date": "2014-01-31"},
        manifests={"dataset_count": 1},
        timings={"records": []},
        validation_summary={"status": "PASS"},
        output_root=Path("verification-output"),
    )

    json_path = write_validation_summary(
        tmp_path / "summary.json",
        payload,
    )
    md_path = write_run_summary_markdown(
        tmp_path / "summary.md",
        verification_evidence.render_verification_evidence_markdown(run_payload),
    )
    run_json_path = write_run_summary_json(
        tmp_path / "run_summary.json",
        run_payload,
    )

    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert run_json_path.read_text(encoding="utf-8").endswith("\n")
    assert "status" in md_path.read_text(encoding="utf-8")


def test_verification_evidence_mapping_and_logical_path_helpers() -> None:
    target: dict[str, int] = {"a": 1}
    verification_evidence._merge_profile(target, {"a": 2, "b": None})

    assert target == {"a": 3, "b": 0}
    assert verification_evidence._int_mapping([("a", 1)]) == {}
    assert logical_path(Path("plain") / "path") == "plain/path"
    assert (
        logical_path(Path("verification-output") / "demo" / "file.md")
        == "verification-output/demo/file.md"
    )
    assert verification_evidence._utc_now().endswith("Z")
