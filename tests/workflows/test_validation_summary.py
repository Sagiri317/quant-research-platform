from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from quant_research_platform.storage import manifests
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.workflows.diagnostics.validation_summary import (
    build_validation_summary,
)
from tests.storage.low_level_fixtures import (
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _safe_evidence(tmp_path: Path) -> Path:
    path = tmp_path / "summary.md"
    path.write_text("# Safe summary\n", encoding="utf-8")
    return path


def _valid_dataset(tmp_path: Path, data_id: str) -> Path:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / data_id
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(data_id),
    )
    return root


def _layer_summary(dataset_root: Path) -> dict[str, object]:
    return manifests.build_layer_summary(
        layer_root=dataset_root.parents[3],
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )


def test_build_validation_summary_consumes_embedded_layer_summaries(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "summary_demo")
    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(dataset_root,),
        layer_summaries=(_layer_summary(dataset_root),),
        generated_text_paths=(_safe_evidence(tmp_path),),
    )

    assert summary["status"] == "PASS"
    assert summary["data_layer_summary_completeness"] == {
        "missing": [],
        "checked_count": 1,
    }
    assert summary["share_safety_scan"]["requested_paths"] == ["summary.md"]


@pytest.mark.parametrize("invalid_layer", ("raw", "archive"))
def test_build_validation_summary_rejects_non_run_result_dataset_layers(
    tmp_path: Path,
    invalid_layer: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_layer_invalid",
    ):
        build_validation_summary(
            run_id="invalid_layer_run",
            dataset_roots=(tmp_path / "data" / invalid_layer / "dataset",),
            layer_summaries=(
                {"logical_layer": "processed", "layer_validation_status": "PASS"},
            ),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "dataset_count",
        "dataset_identity",
        "dataset_row_count",
        "dataset_partition_count",
        "layer_total_rows",
        "layer_date_range",
        "missing_manifests",
        "invalid_manifests",
        "failed_datasets",
        "layer_status",
    ),
)
def test_build_validation_summary_rejects_forged_layer_fact_graph(
    tmp_path: Path,
    mutation: str,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "forged_layer_demo")
    layer_summary = deepcopy(_layer_summary(dataset_root))
    if mutation == "dataset_count":
        layer_summary["dataset_count"] = 2
    elif mutation == "dataset_identity":
        layer_summary["datasets"][0]["data_id"] = "substituted_dataset"
    elif mutation == "dataset_row_count":
        layer_summary["datasets"][0]["manifest_summary"]["row_count"] += 1
    elif mutation == "dataset_partition_count":
        layer_summary["datasets"][0]["manifest_summary"]["partition_count"] += 1
    elif mutation == "layer_total_rows":
        layer_summary["total_rows"] += 1
    elif mutation == "layer_date_range":
        layer_summary["date_max"] = "2026-12-31"
    elif mutation == "missing_manifests":
        layer_summary["missing_dataset_manifests"] = [
            "data/processed/missing/_manifest/manifest.json"
        ]
    elif mutation == "invalid_manifests":
        layer_summary["invalid_dataset_manifests"] = [
            "data/processed/invalid/_manifest/manifest.json"
        ]
    elif mutation == "failed_datasets":
        layer_summary["failed_dataset_validations"] = ["forged_layer_demo"]
    else:
        layer_summary["layer_validation_status"] = "FAIL"

    with pytest.raises(
        ValueError,
        match="validation_summary_layer_declaration_mismatch",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


def test_build_validation_summary_rejects_lowercase_layer_status(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "lowercase_status_demo")
    layer_summary = _layer_summary(dataset_root)
    layer_summary["layer_validation_status"] = "pass"
    with pytest.raises(ValueError, match="invalid_check_status"):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


def test_build_validation_summary_classifies_invalid_json_evidence(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "invalid-evidence.json"
    invalid_json.write_text("{", encoding="utf-8")
    dataset_root = _valid_dataset(tmp_path, "invalid_json_demo")

    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(dataset_root,),
        layer_summaries=(_layer_summary(dataset_root),),
        generated_text_paths=(invalid_json,),
    )

    assert summary["status"] == "FAIL"
    assert summary["failure_codes"] == ["share_safety_scan_failed"]
    assert summary["share_safety_scan"]["failures"][0]["reason"] == "invalid_json"


def test_build_validation_summary_consumes_storage_invalid_manifest_classification(
    tmp_path: Path,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "invalid_manifest_demo")
    (dataset_root / "_manifest" / "manifest.json").write_text(
        "{",
        encoding="utf-8",
    )
    layer_summary = _layer_summary(dataset_root)

    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(dataset_root,),
        layer_summaries=(layer_summary,),
        generated_text_paths=(_safe_evidence(tmp_path),),
    )

    expected_path = (
        "data/processed/a_share/factor/invalid_manifest_demo/"
        "_manifest/manifest.json"
    )
    assert summary["failure_codes"] == [
        "dataset_manifest_invalid",
        "layer_validation_failed",
    ]
    assert summary["dataset_manifest_completeness"] == {
        "missing": [],
        "invalid": [expected_path],
        "checked_count": 1,
    }


def test_build_validation_summary_does_not_misclassify_physical_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = _valid_dataset(tmp_path, "inspection_failure_demo")
    layer_summary = _layer_summary(dataset_root)

    def fail_inspection(_root: Path) -> object:
        raise ValueError("physical_inspection_failed")

    monkeypatch.setattr(
        "quant_research_platform.workflows.diagnostics.validation_summary.inspect_dataset",
        fail_inspection,
    )

    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


@pytest.mark.parametrize(
    "null_count_summary",
    ({}, {"market_id": 0, "asset_id": 0}),
    ids=("empty", "composite-key-member-omitted"),
)
def test_build_validation_summary_rejects_forged_primary_key_null_facts(
    tmp_path: Path,
    null_count_summary: dict[str, int],
) -> None:
    dataset_root = (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "workflow_pk_null_fact_demo"
    )
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("workflow_pk_null_fact_demo"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest["null_count_summary"] = null_count_summary
    manifests.replace_qrp_json(manifest_path, manifest)

    layer_summary = _layer_summary(dataset_root)
    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duplicate_primary_key_count", 1),
        ("non_finite_count_summary", {"value": 1}),
    ),
)
def test_validation_summary_rejects_forged_storage_summary(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    dataset_root = (
        tmp_path
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "workflow_consumability_demo"
    )
    write_dataset(
        dataset_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("workflow_consumability_demo"),
    )
    manifest_path = dataset_root / "_manifest" / "manifest.json"
    manifest = manifests.read_manifest(manifest_path)
    manifest[field] = value
    manifests.replace_qrp_json(manifest_path, manifest)
    layer_summary = manifests.build_layer_summary(
        layer_root=tmp_path / "layers",
        logical_layer="processed",
        dataset_roots=(dataset_root,),
    )

    assert layer_summary["failed_dataset_validations"] == [
        "workflow_consumability_demo"
    ]
    assert layer_summary["datasets"][0]["manifest_summary"][
        "validation_status"
    ] == "FAIL"
    with pytest.raises(
        ValueError,
        match="validation_summary_dataset_inspection_failed",
    ):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=(dataset_root,),
            layer_summaries=(layer_summary,),
            generated_text_paths=(_safe_evidence(tmp_path),),
        )


@pytest.mark.parametrize(
    ("empty_field", "error"),
    (
        ("datasets", "validation_summary_dataset_roots_required"),
        ("layers", "validation_summary_layer_summaries_required"),
        ("evidence", "validation_summary_generated_evidence_required"),
    ),
)
def test_build_validation_summary_requires_all_declared_evidence_classes(
    tmp_path: Path,
    empty_field: str,
    error: str,
) -> None:
    dataset_roots = () if empty_field == "datasets" else (tmp_path / "dataset",)
    layer_summaries = (
        ()
        if empty_field == "layers"
        else ({"logical_layer": "processed", "layer_validation_status": "PASS"},)
    )
    evidence_paths = () if empty_field == "evidence" else (tmp_path / "summary.md",)

    with pytest.raises(ValueError, match=error):
        build_validation_summary(
            run_id="demo_run",
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            generated_text_paths=evidence_paths,
        )
