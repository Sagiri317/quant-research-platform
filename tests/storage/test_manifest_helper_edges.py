from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.storage import manifests
from quant_research_platform.storage.datasets import write_dataset
from tests.storage.low_level_fixtures import (
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_manifest_review_safe_and_internal_writes_are_utf8_and_safe(
    tmp_path: Path,
) -> None:
    review_safe_path = tmp_path / "review-safe" / "manifest.json"
    internal_path = tmp_path / "internal" / "manifest.json"

    manifests.write_manifest(
        review_safe_path,
        {"schema_id": "qrp-output-manifest", "status": "PASS", "rows": [1, 2]},
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


def test_manifest_replace_does_not_overwrite_unclaimed_temp_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    final_path.write_text('{"schema_id": "old"}\n', encoding="utf-8")
    live_temp = final_path.with_name(f".tmp-{'c' * 12}")
    live_temp.write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(
        manifests,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "c" * 32})(),
    )

    with pytest.raises(FileExistsError):
        manifests.replace_qrp_json(final_path, {"schema_id": "new"})

    assert live_temp.read_text(encoding="utf-8") == "live\n"
    assert manifests.read_manifest(final_path)["schema_id"] == "old"


def test_manifest_replace_cleans_claimed_temp_after_control_flow_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "manifest.json"
    final_path.write_text('{"schema_id": "old"}\n', encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        manifests.replace_qrp_json(final_path, {"schema_id": "new"})

    assert not tuple(tmp_path.glob(".tmp-*"))
    assert manifests.read_manifest(final_path)["schema_id"] == "old"


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
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "daily"
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("daily"),
    )
    missing_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "missing"
    failed_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "failed"
    write_dataset(
        failed_root,
        make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
        ),
        spec=make_processed_dataset_spec("failed"),
    )
    failed_manifest_path = failed_root / "_manifest" / "manifest.json"
    failed_manifest = manifests.read_manifest(failed_manifest_path)
    failed_manifest["validation_status"] = "FAIL"
    manifests.replace_qrp_json(failed_manifest_path, failed_manifest)

    layer = manifests.build_layer_summary(
        layer_root=tmp_path / "local-verification-output" / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root, missing_root, failed_root),
    )
    assert layer["layer_validation_status"] == "FAIL"
    assert layer["total_rows"] == 4
    assert layer["date_min"] == "2014-01-02"
    assert layer["failed_dataset_validations"] == ["failed"]
    assert layer["missing_dataset_manifests"]

    run = manifests.build_run_data_summary(layer_summaries=(layer,))
    assert run["run_validation_status"] == "FAIL"
    assert run["failed_layers"] == ["processed"]


@pytest.mark.parametrize(
    "relative_path",
    (Path("spec.json"), Path("_manifest") / "manifest.json"),
)
def test_layer_summary_distinguishes_present_invalid_dataset_evidence(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "bad"
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("bad"),
    )
    (dataset_root / relative_path).write_text("{", encoding="utf-8")

    layer = manifests.build_layer_summary(
        layer_root=tmp_path / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )

    assert layer["missing_dataset_manifests"] == []
    assert layer["invalid_dataset_manifests"] == [
        "data/processed/a_share/factor/bad/_manifest/manifest.json"
    ]
    assert layer["layer_validation_status"] == "FAIL"


@pytest.mark.parametrize(
    "null_count_summary",
    ({}, {"market_id": 0, "asset_id": 0}),
    ids=("empty", "composite-key-member-omitted"),
)
def test_layer_and_run_summaries_fail_when_primary_key_null_facts_are_missing(
    tmp_path: Path,
    null_count_summary: dict[str, int],
) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "daily"
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("daily"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest["null_count_summary"] = null_count_summary
    manifests.replace_qrp_json(manifest_path, manifest)

    layer = manifests.build_layer_summary(
        layer_root=tmp_path / "local-verification-output" / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )

    assert layer["dataset_count"] == 1
    assert layer["datasets"][0]["data_id"] == "daily"
    assert layer["missing_dataset_manifests"] == []
    assert layer["failed_dataset_validations"] == ["daily"]
    assert layer["datasets"][0]["manifest_summary"]["validation_status"] == "FAIL"
    assert layer["layer_validation_status"] == "FAIL"

    run = manifests.build_run_data_summary(layer_summaries=(layer,))
    assert run["run_validation_status"] == "FAIL"
    assert run["failed_layers"] == ["processed"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_primary_key_count", 1),
        ("non_finite_count_summary", {"value": 1}),
    ),
)
def test_layer_summary_uses_effective_manifest_consumability(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    dataset_root = tmp_path / "data" / "processed" / "a_share" / "factor" / "daily"
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("daily"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest[field] = value
    manifests.replace_qrp_json(manifest_path, manifest)

    layer = manifests.build_layer_summary(
        layer_root=tmp_path / "local-verification-output" / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )

    assert layer["datasets"][0]["manifest_summary"]["validation_status"] == "FAIL"
    assert layer["failed_dataset_validations"] == ["daily"]
    assert layer["layer_validation_status"] == "FAIL"


@pytest.mark.parametrize("status", ("pass", "ok", " PASS ", 1, None))
def test_run_data_summary_rejects_status_normalization(status: object) -> None:
    with pytest.raises(ValueError, match="invalid_check_status"):
        manifests.build_run_data_summary(
            layer_summaries=(
                {
                    "logical_layer": "processed",
                    "layer_validation_status": status,
                },
            )
        )


@pytest.mark.parametrize("logical_layer", (None, "", 1, True))
def test_run_data_summary_rejects_logical_layer_coercion(
    logical_layer: object,
) -> None:
    with pytest.raises(ValueError, match="run_data_summary_layer_invalid"):
        manifests.build_run_data_summary(
            layer_summaries=(
                {
                    "logical_layer": logical_layer,
                    "layer_validation_status": "FAIL",
                },
            )
        )
