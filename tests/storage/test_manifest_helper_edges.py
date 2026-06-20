from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.logical_paths import logical_path
from quant_research_platform.storage import manifests


def test_manifest_public_and_internal_writes_are_utf8_and_safe(tmp_path: Path) -> None:
    public_path = tmp_path / "public" / "manifest.json"
    internal_path = tmp_path / "internal" / "manifest.json"

    manifests.write_manifest(public_path, {"status": "ok", "rows": [1, 2]})
    manifests.write_internal_artifact_manifest(
        internal_path,
        {"data_id": "demo", "source_id": "csmar", "row_count": 2},
    )
    loaded = manifests.read_manifest(internal_path)

    assert public_path.read_text(encoding="utf-8").endswith("\n")
    assert loaded["source_id"] == "csmar"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"path": "C:" + r"\\private\\data"}, "absolute_path"),
        ({"path": r"\\" + r"\\server\\share"}, "absolute_path"),
        ({"path": "/" + "Users/local/private"}, "absolute_path"),
        ({"path": "data/raw/vendor/file.csv"}, "raw_path"),
        ({"file": "daily__vendor__export-20240101.csv"}, "raw_export_filename"),
        ({"note": "token: " + "abcdefghi123456"}, "credential"),
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
        manifests.write_internal_artifact_manifest(tmp_path / "manifest.json", payload)


def test_layer_and_run_manifests_summarize_dataset_manifests(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "panel" / "daily"
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifests.write_internal_artifact_manifest(
        manifest_path,
        {
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
            "data_id": "failed",
            "row_count": 1,
            "validation_status": "fail",
        },
    )

    layer_path = manifests.write_layer_manifest(
        layer_root=tmp_path / "verification-output" / "layers",
        logical_layer="processed",
        run_id="run_demo",
        dataset_roots=(dataset_root, missing_root, failed_root),
    )
    layer = manifests.read_manifest(layer_path)
    assert layer["layer_validation_status"] == "fail"
    assert layer["total_rows"] == 11
    assert layer["date_min"] == "2024-01-02"
    assert layer["failed_dataset_validations"] == ["failed"]
    assert layer["missing_dataset_manifests"]

    run_path = manifests.write_run_manifest(
        manifests_root=tmp_path / "verification-output" / "manifests",
        run_id="run_demo",
        output_root=tmp_path / "verification-output" / "output",
        layer_manifest_paths=(layer_path,),
    )
    run = manifests.read_manifest(run_path)
    assert run["run_validation_status"] == "fail"
    assert run["failed_layers"] == ["processed"]


def test_manifest_logical_path_falls_back_to_posix(tmp_path: Path) -> None:
    assert (
        manifests._data_id_from_manifest({}, Path("relative/demo"))
        == "relative/demo"
    )
    assert logical_path(Path("plain") / "path") == "plain/path"


def test_manifest_logical_path_uses_last_generated_marker() -> None:
    path = Path("/mnt/data/team/qrp/reports/private_research/run/manifests/run.json")

    assert logical_path(path) == "manifests/run.json"
