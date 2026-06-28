from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.logical_paths import logical_path
from quant_research_platform.storage import manifests


def test_manifest_review_safe_and_internal_writes_are_utf8_and_safe(
    tmp_path: Path,
) -> None:
    review_safe_path = tmp_path / "review-safe" / "manifest.json"
    internal_path = tmp_path / "internal" / "manifest.json"

    manifests.write_manifest(
        review_safe_path,
        {"schema_id": "qrp-output-manifest", "status": "ok", "rows": [1, 2]},
    )
    manifests.write_internal_artifact_manifest(
        internal_path,
        {
            "schema_id": "qrp-dataset-manifest",
            "data_id": "demo",
            "source_id": "csmar",
            "row_count": 2,
        },
    )
    loaded = manifests.read_manifest(internal_path)

    assert review_safe_path.read_text(encoding="utf-8").endswith("\n")
    assert loaded["source_id"] == "csmar"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        # Intentional fake provider/source markers for negative safety coverage.
        ({"path": "data/raw/vendor/file.csv"}, "local_only_path"),
        ({"path": "data/research/a_share/demo"}, "local_only_path"),
        ({"path": "/workspace/qrp/data/research/a_share/demo"}, "local_only_path"),
        ({"file": "daily__vendor__export-20240101.csv"}, "raw_export_filename"),
        ({"row_sample": [{"a": 1}]}, "row_sample"),
    ],
)
def test_internal_manifest_rejects_private_or_row_level_metadata(
    payload: dict[str, object],
    reason: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"unsafe_internal_artifact_metadata: {reason}",
    ):
        manifests.write_internal_artifact_manifest(
            tmp_path / "manifest.json",
            {"schema_id": "qrp-dataset-manifest", **payload},
        )


def test_layer_and_run_manifests_summarize_dataset_manifests(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "panel" / "daily"
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifests.write_internal_artifact_manifest(
        manifest_path,
        {
            "schema_id": "qrp-dataset-manifest",
            "data_id": "daily",
            "row_count": 10,
            "date_min": "2024-01-02",
            "date_max": "2024-01-03",
            "duplicate_primary_key_count": 0,
            "validation_status": "pass",
        },
    )
    missing_root = tmp_path / "data" / "processed" / "a_share" / "panel" / "missing"
    failed_root = tmp_path / "data" / "processed" / "a_share" / "panel" / "failed"
    manifests.write_internal_artifact_manifest(
        failed_root / "_manifest" / "manifest.json",
        {
            "schema_id": "qrp-dataset-manifest",
            "data_id": "failed",
            "row_count": 1,
            "validation_status": "fail",
        },
    )

    layer = manifests.build_layer_summary(
        layer_root=tmp_path / "verification-output" / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root, missing_root, failed_root),
    )
    assert layer["layer_validation_status"] == "fail"
    assert layer["total_rows"] == 11
    assert layer["date_min"] == "2024-01-02"
    assert layer["failed_dataset_validations"] == ["failed"]
    assert layer["missing_dataset_manifests"]

    run = manifests.build_run_data_summary(layer_summaries=(layer,))
    assert run["run_validation_status"] == "fail"
    assert run["failed_layers"] == ["processed"]


def test_manifest_logical_path_falls_back_to_posix(tmp_path: Path) -> None:
    assert (
        manifests._data_id_from_manifest({}, Path("relative/demo")) == "relative/demo"
    )
    assert logical_path(Path("plain") / "path") == "plain/path"


def test_manifest_logical_path_uses_last_generated_marker() -> None:
    path = Path("sandbox/team/qrp/outputs/run/manifest.json")

    assert logical_path(path) == "outputs/run/manifest.json"
