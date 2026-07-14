from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/"
    "compare_run_results.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("compare_run_results", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("logical_layer", ("interim", "processed", "research"))
def test_compare_run_layer_accepts_exact_owner_members(logical_layer: str) -> None:
    module = _module()
    layer = deepcopy(_run_payload("valid_layer_run")["data"]["layers"][0])
    layer["logical_layer"] = logical_layer
    layer["datasets"][0]["manifest_path"] = (
        f"data/{logical_layer}/research_dataset/_manifest/manifest.json"
    )

    assert module._validate_run_layer(layer)[0] == logical_layer


@pytest.mark.parametrize("logical_layer", ("raw", "archive"))
def test_compare_run_layer_rejects_non_owner_members(logical_layer: str) -> None:
    module = _module()
    layer = deepcopy(_run_payload("invalid_layer_run")["data"]["layers"][0])
    layer["logical_layer"] = logical_layer
    layer["datasets"][0]["manifest_path"] = (
        f"data/{logical_layer}/research_dataset/_manifest/manifest.json"
    )

    with pytest.raises(module.CompareBlockedError, match="run_data_layer_invalid"):
        module._validate_run_layer(layer)


def _run_payload(
    run_id: str,
    *,
    output_id: str = "prediction_daily",
    output_definition_id: str = "stage_profile_diagnostic",
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-run",
        "workflow_id": "local_verification",
        "run_id": run_id,
        "root_ownership": {
            "owner": "local_verification",
            "workflow_id": "local_verification",
            "created_by": "quant_research_platform",
            "created_at": "2026-06-26T00:00:00Z",
            "replace_safe": True,
        },
        "status": {
            "overall": "NOT_RUN",
            "validation": "PASS",
            "smoke": "PASS",
            "compare": "NOT_RUN",
            "share_safety": "PASS",
            "full_horizon": "NOT_RUN",
            "validation_matrix": "NOT_RUN",
        },
        "created_at": "2026-06-26T00:00:00Z",
        "completed_at": "2026-06-26T00:00:01Z",
        "code": {},
        "config": {},
        "inputs": {},
        "data": {
            "layer_count": 1,
            "run_validation_status": "PASS",
            "failed_layers": [],
            "layers": [
                {
                    "logical_layer": "research",
                    "dataset_count": 1,
                    "datasets": [
                        {
                            "data_id": "research_dataset",
                            "manifest_path": (
                                "data/research/research_dataset/_manifest/manifest.json"
                            ),
                            "manifest_summary": {
                                "row_count": 3,
                                "date_min": "2024-01-01",
                                "date_max": "2024-01-03",
                                "partition_count": 1,
                                "duplicate_primary_key_count": 0,
                                "validation_status": "PASS",
                            },
                        }
                    ],
                    "total_rows": 3,
                    "date_min": "2024-01-01",
                    "date_max": "2024-01-03",
                    "missing_dataset_manifests": [],
                    "invalid_dataset_manifests": [],
                    "failed_dataset_validations": [],
                    "layer_validation_status": "PASS",
                    "created_at": "2026-06-26T00:00:00Z",
                }
            ],
            "created_at": "2026-06-26T00:00:00Z",
        },
        "outputs": [
            {
                "schema_id": "qrp-output-manifest",
                "output_id": output_id,
                "output_definition_id": output_definition_id,
                "run_id": run_id,
                "metadata": (
                    metadata
                    if metadata is not None
                    else {
                        "market_id": "a_share",
                        "run_type": "diagnostics",
                        "scope": "LOCAL_ONLY",
                    }
                ),
                "files": [
                    "stage_profile.csv",
                    "stage_profile.json",
                    "stage_profile.md",
                ],
                "manifest_path": f"outputs/{output_id}/manifest.json",
            }
        ],
        "checks": {
            "schema_inventory": {
                "schema_id": "qrp-schema-inventory-summary",
                "dataset_count": 1,
                "total_row_count": 3,
                "schema_hash": "0" * 64,
            },
            "validation": {
                "run_id": run_id,
                "status": "PASS",
                "failure_codes": [],
                "required_columns_status": "summarized_by_dataset_manifests",
                "primary_key_null_failures": [],
                "duplicate_primary_key_failures": [],
                "source_local_key_leakage": [],
                "date_ranges": [
                    {
                        "data_id": "research_dataset",
                        "date_min": "2024-01-01",
                        "date_max": "2024-01-03",
                        "partition_count": 1,
                    }
                ],
                "partition_coverage_status": "PASS",
                "non_finite_failures": [],
                "factor_nan_profile": {},
                "label_nan_profile": {},
                "dataset_manifest_completeness": {
                    "missing": [],
                    "invalid": [],
                    "checked_count": 1,
                },
                "data_layer_summary_completeness": {
                    "missing": [],
                    "checked_count": 1,
                },
                "share_safety_scan": {
                    "status": "PASS",
                    "failures": [],
                    "checked_count": 4,
                    "requested_count": 4,
                    "existing_scanned_count": 4,
                    "missing_count": 0,
                    "requested_paths": [
                        f"outputs/{output_id}/manifest.json",
                        f"outputs/{output_id}/stage_profile.csv",
                        f"outputs/{output_id}/stage_profile.json",
                        f"outputs/{output_id}/stage_profile.md",
                    ],
                },
                "created_at": "2026-06-26T00:00:01Z",
            },
        },
        "execution_metrics": {},
        "logs": {},
        "summary": {"markdown_path": "summary.md"},
    }


def _write_run(
    root: Path,
    run_id: str,
    *,
    output_id: str = "prediction_daily",
    output_definition_id: str = "stage_profile_diagnostic",
    metadata: dict[str, object] | None = None,
    file_content: bytes = b"same",
) -> None:
    payload = _run_payload(
        run_id,
        output_id=output_id,
        output_definition_id=output_definition_id,
        metadata=metadata,
    )
    _write_json(root / "run.json", payload)
    (root / "summary.md").write_text("# Run summary\n", encoding="utf-8")
    manifest = dict(payload["outputs"][0])
    manifest.pop("manifest_path")
    _write_json(root / "outputs" / output_id / "manifest.json", manifest)
    for name in ("stage_profile.csv", "stage_profile.json", "stage_profile.md"):
        (root / "outputs" / output_id / name).write_bytes(file_content)


def _write_run_without_outputs(root: Path, run_id: str) -> None:
    payload = _run_payload(run_id)
    payload["outputs"] = []
    payload["checks"]["validation"]["share_safety_scan"] = {
        "status": "PASS",
        "failures": [],
        "checked_count": 0,
        "requested_count": 0,
        "existing_scanned_count": 0,
        "missing_count": 0,
        "requested_paths": [],
    }
    _write_json(root / "run.json", payload)
    (root / "summary.md").write_text("# Run summary\n", encoding="utf-8")


def _append_output(root: Path, *, run_id: str, output_id: str) -> None:
    run_path = root / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_id": "qrp-output-manifest",
        "output_id": output_id,
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": run_id,
        "metadata": {
            "market_id": "a_share",
            "run_type": "diagnostics",
            "scope": "LOCAL_ONLY",
        },
        "files": [
            "stage_profile.csv",
            "stage_profile.json",
            "stage_profile.md",
        ],
    }
    payload["outputs"].append(
        {
            **manifest,
            "manifest_path": f"outputs/{output_id}/manifest.json",
        }
    )
    share_safety = payload["checks"]["validation"]["share_safety_scan"]
    appended_paths = [
        f"outputs/{output_id}/manifest.json",
        f"outputs/{output_id}/stage_profile.csv",
        f"outputs/{output_id}/stage_profile.json",
        f"outputs/{output_id}/stage_profile.md",
    ]
    share_safety["requested_paths"].extend(appended_paths)
    share_safety["checked_count"] += len(appended_paths)
    share_safety["requested_count"] += len(appended_paths)
    share_safety["existing_scanned_count"] += len(appended_paths)
    _write_json(run_path, payload)
    _write_json(root / "outputs" / output_id / "manifest.json", manifest)
    for name in ("stage_profile.csv", "stage_profile.json", "stage_profile.md"):
        (root / "outputs" / output_id / name).write_bytes(b"same")


def test_compare_pairs_same_output_id_without_historical_path_assumptions(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    compare_root = tmp_path / "compare"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")

    payload = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=compare_root,
    )

    assert payload["schema_id"] == "qrp-compare"
    assert payload["overall_status"] == "PASS"
    assert payload["pairing"]["mode"] == "output_id"
    assert payload["pairing"]["pairs"] == [
        {
            "baseline_output_id": "prediction_daily",
            "current_output_id": "prediction_daily",
        }
    ]
    assert payload["output_results"][0]["status"] == "PASS"
    assert (compare_root / "compare.json").is_file()
    assert (compare_root / "summary.md").is_file()
    assert not list(compare_root.glob("*baseline-final*"))


def test_compare_empty_output_pairs_are_blocked(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run_without_outputs(baseline, "baseline_run")
    _write_run_without_outputs(current, "current_run")

    payload = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert payload["overall_status"] == "BLOCKED"
    assert payload["blocked_reason"] == "compare_output_pairs_empty"
    assert payload["pairing"]["pairs"] == []


def test_compare_identical_data_without_outputs_does_not_pass(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run_without_outputs(baseline, "baseline_run")
    _write_run_without_outputs(current, "current_run")

    payload = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert payload["overall_status"] != "PASS"
    assert payload["overall_status"] == "BLOCKED"


@pytest.mark.parametrize(
    "topology",
    ("partial", "disjoint", "baseline_only", "current_only"),
)
def test_compare_implicit_pairing_requires_equal_output_id_topology(
    tmp_path: Path,
    topology: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    if topology == "partial":
        _write_run(baseline, "baseline_run")
        _write_run(current, "current_run")
        _append_output(current, run_id="current_run", output_id="prediction_extra")
    elif topology == "disjoint":
        _write_run(baseline, "baseline_run", output_id="prediction_left")
        _write_run(current, "current_run", output_id="prediction_right")
    elif topology == "baseline_only":
        _write_run(baseline, "baseline_run")
        _write_run_without_outputs(current, "current_run")
    else:
        _write_run_without_outputs(baseline, "baseline_run")
        _write_run(current, "current_run")

    payload = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert payload["overall_status"] == "BLOCKED"
    assert payload["blocked_reason"] == "compare_output_id_topology_mismatch"
    assert payload["output_results"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_run",
        "invalid_json",
        "missing_schema",
        "versioned_schema",
        "unknown_field",
        "wrong_kind_run_id",
        "wrong_kind_output_id",
        "lowercase_manifest_status",
        "backslash_manifest_path",
    ),
)
def test_compare_invalid_run_input_is_blocked(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = baseline / "run.json"
    if mutation == "missing_run":
        run_path.unlink()
    elif mutation == "invalid_json":
        run_path.write_text("{", encoding="utf-8")
    else:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        if mutation == "missing_schema":
            payload["schema_id"] = None
        elif mutation == "versioned_schema":
            payload["schema_id"] = "qrp-run/0.1"
        elif mutation == "unknown_field":
            payload["unknown"] = True
        elif mutation == "wrong_kind_run_id":
            payload["run_id"] = 7
        elif mutation == "wrong_kind_output_id":
            payload["outputs"][0]["output_id"] = 7
        elif mutation == "lowercase_manifest_status":
            payload["data"]["layers"][0]["datasets"][0]["manifest_summary"][
                "validation_status"
            ] = "pass"
        elif mutation == "backslash_manifest_path":
            payload["outputs"][0]["manifest_path"] = (
                "outputs\\prediction_daily\\manifest.json"
            )
        _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["output_results"] == []


def test_compare_blocks_self_consistent_noncurrent_workflow_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = current / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["workflow_id"] = "other_workflow"
    payload["root_ownership"]["workflow_id"] = "other_workflow"
    _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "workflow_id_invalid"
    assert result["output_results"] == []


@pytest.mark.parametrize("field", ("compare", "full_horizon"))
@pytest.mark.parametrize("status", ("PASS", "FAIL", "BLOCKED"))
def test_compare_rejects_unpersisted_run_status_evidence(
    tmp_path: Path,
    field: str,
    status: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = current / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["status"][field] = status
    matrix_status = module.aggregate_validation_matrix_status(
        (
            payload["status"]["smoke"],
            payload["status"]["compare"],
            payload["status"]["full_horizon"],
        )
    )
    payload["status"]["validation_matrix"] = matrix_status
    payload["status"]["overall"] = module.overall_verification_status(
        payload["status"]["validation"],
        {"validation_matrix_status": matrix_status},
    )
    _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "run_status_evidence_missing"
    assert result["output_results"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "status_matrix",
        "status_overall",
        "status_validation_check",
        "status_share_safety_check",
        "validation_failure_codes",
        "share_safety_failure_facts",
        "dataset_noncanonical_date",
        "dataset_reversed_date",
        "dataset_duplicate_pass",
        "layer_date_range",
        "unknown_logical_layer",
        "schema_inventory_bool_count",
        "schema_inventory_dataset_count",
        "schema_inventory_total_rows",
        "schema_inventory_upper_hash",
        "validation_unknown_field",
        "share_safety_counter_mismatch",
        "share_safety_requested_path_substitution",
    ),
)
def test_compare_blocks_cross_field_run_evidence_contradictions(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    path = current / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "status_matrix":
        payload["status"]["validation_matrix"] = "PASS"
    elif mutation == "status_overall":
        payload["status"]["overall"] = "FAIL"
    elif mutation == "status_validation_check":
        payload["checks"]["validation"]["status"] = "FAIL"
    elif mutation == "status_share_safety_check":
        payload["checks"]["validation"]["share_safety_scan"]["status"] = "FAIL"
    elif mutation == "validation_failure_codes":
        payload["checks"]["validation"]["failure_codes"] = ["non_finite_values"]
    elif mutation == "share_safety_failure_facts":
        payload["checks"]["validation"]["share_safety_scan"]["failures"] = [
            {"path": "summary.md", "reason": "unsafe"}
        ]
    elif mutation == "dataset_noncanonical_date":
        payload["data"]["layers"][0]["datasets"][0]["manifest_summary"][
            "date_min"
        ] = "20240101"
    elif mutation == "dataset_reversed_date":
        payload["data"]["layers"][0]["datasets"][0]["manifest_summary"][
            "date_min"
        ] = "2024-01-04"
    elif mutation == "dataset_duplicate_pass":
        payload["data"]["layers"][0]["datasets"][0]["manifest_summary"][
            "duplicate_primary_key_count"
        ] = 1
    elif mutation == "layer_date_range":
        payload["data"]["layers"][0]["date_max"] = "2024-01-04"
    elif mutation == "unknown_logical_layer":
        layer = payload["data"]["layers"][0]
        layer["logical_layer"] = "archive"
        layer["datasets"][0]["manifest_path"] = (
            "data/archive/research_dataset/demo/_manifest/manifest.json"
        )
    elif mutation == "schema_inventory_bool_count":
        payload["checks"]["schema_inventory"]["dataset_count"] = True
    elif mutation == "schema_inventory_dataset_count":
        payload["checks"]["schema_inventory"]["dataset_count"] = 2
    elif mutation == "schema_inventory_total_rows":
        payload["checks"]["schema_inventory"]["total_row_count"] = 4
    elif mutation == "schema_inventory_upper_hash":
        payload["checks"]["schema_inventory"]["schema_hash"] = "A" * 64
    elif mutation == "validation_unknown_field":
        payload["checks"]["validation"]["unknown"] = True
    elif mutation == "share_safety_counter_mismatch":
        payload["checks"]["validation"]["share_safety_scan"]["missing_count"] = 1
    elif mutation == "share_safety_requested_path_substitution":
        payload["checks"]["validation"]["share_safety_scan"][
            "requested_paths"
        ][-1] = "outputs/prediction_daily/opaque.md"
    _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["output_results"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        "dataset_checked_count_high",
        "dataset_missing_mismatch",
        "layer_missing_mismatch",
        "date_range_id_mismatch",
        "date_range_value_mismatch",
        "date_range_partition_count_mismatch",
        "manifest_path_identity_mismatch",
        "failure_detail_unknown_id",
        "failure_code_order",
    ),
)
def test_compare_blocks_addendum_run_fact_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    path = current / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload["checks"]["validation"]
    if mutation == "dataset_checked_count_high":
        validation["dataset_manifest_completeness"]["checked_count"] = 2
    elif mutation == "dataset_missing_mismatch":
        validation["dataset_manifest_completeness"] = {
            "missing": ["data/research/missing/_manifest/manifest.json"],
            "invalid": [],
            "checked_count": 2,
        }
        validation["partition_coverage_status"] = "FAIL"
        validation["failure_codes"] = ["dataset_manifest_missing"]
        validation["status"] = "FAIL"
        payload["status"]["validation"] = "FAIL"
        payload["status"]["smoke"] = "FAIL"
        payload["status"]["validation_matrix"] = "FAIL"
        payload["status"]["overall"] = "FAIL"
    elif mutation == "layer_missing_mismatch":
        validation["data_layer_summary_completeness"]["missing"] = ["research"]
    elif mutation == "date_range_id_mismatch":
        validation["date_ranges"][0]["data_id"] = "other_dataset"
    elif mutation == "date_range_value_mismatch":
        validation["date_ranges"][0]["date_max"] = "2024-01-02"
    elif mutation == "date_range_partition_count_mismatch":
        validation["date_ranges"][0]["partition_count"] = 2
    elif mutation == "manifest_path_identity_mismatch":
        payload["data"]["layers"][0]["datasets"][0]["manifest_path"] = (
            "data/research/other_dataset/_manifest/manifest.json"
        )
    elif mutation == "failure_detail_unknown_id":
        validation["source_local_key_leakage"] = ["other_dataset"]
        validation["failure_codes"] = ["source_local_key_leakage"]
        validation["status"] = "FAIL"
        payload["status"]["validation"] = "FAIL"
        payload["status"]["smoke"] = "FAIL"
        payload["status"]["validation_matrix"] = "FAIL"
        payload["status"]["overall"] = "FAIL"
    else:
        dataset = payload["data"]["layers"][0]["datasets"][0]
        dataset["manifest_summary"]["duplicate_primary_key_count"] = 1
        dataset["manifest_summary"]["validation_status"] = "FAIL"
        payload["data"]["layers"][0]["failed_dataset_validations"] = [
            "research_dataset"
        ]
        payload["data"]["layers"][0]["layer_validation_status"] = "FAIL"
        payload["data"]["failed_layers"] = ["research"]
        payload["data"]["run_validation_status"] = "FAIL"
        validation["duplicate_primary_key_failures"] = ["research_dataset"]
        validation["failure_codes"] = [
            "layer_validation_failed",
            "dataset_validation_failed",
            "duplicate_primary_key",
        ]
        validation["status"] = "FAIL"
        payload["status"]["validation"] = "FAIL"
        payload["status"]["smoke"] = "FAIL"
        payload["status"]["validation_matrix"] = "FAIL"
        payload["status"]["overall"] = "FAIL"
    _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["output_results"] == []


@pytest.mark.parametrize(
    "failure_code",
    (
        "primary_key_nulls",
        "duplicate_primary_key",
        "non_finite_values",
    ),
)
def test_compare_accepts_truthful_failed_dataset_evidence_envelope(
    tmp_path: Path,
    failure_code: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    for root in (baseline, current):
        path = root / "run.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["data"]["layers"][0]["datasets"][0]
        dataset["manifest_summary"]["validation_status"] = "FAIL"
        if failure_code == "duplicate_primary_key":
            dataset["manifest_summary"]["duplicate_primary_key_count"] = 1
        payload["data"]["layers"][0]["failed_dataset_validations"] = [
            "research_dataset"
        ]
        payload["data"]["layers"][0]["layer_validation_status"] = "FAIL"
        payload["data"]["failed_layers"] = ["research"]
        payload["data"]["run_validation_status"] = "FAIL"
        payload["checks"]["validation"]["failure_codes"] = [
            "dataset_validation_failed",
            failure_code,
            "layer_validation_failed",
        ]
        detail_field = {
            "primary_key_nulls": "primary_key_null_failures",
            "duplicate_primary_key": "duplicate_primary_key_failures",
            "non_finite_values": "non_finite_failures",
        }[failure_code]
        payload["checks"]["validation"][detail_field] = ["research_dataset"]
        payload["checks"]["validation"]["status"] = "FAIL"
        payload["status"]["validation"] = "FAIL"
        payload["status"]["smoke"] = "FAIL"
        payload["status"]["validation_matrix"] = "FAIL"
        payload["status"]["overall"] = "FAIL"
        _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "PASS"


def test_compare_accepts_truthful_invalid_manifest_evidence_envelope(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    invalid_path = "data/research/invalid/_manifest/manifest.json"
    for root in (baseline, current):
        path = root / "run.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        layer = payload["data"]["layers"][0]
        layer["invalid_dataset_manifests"] = [invalid_path]
        layer["layer_validation_status"] = "FAIL"
        payload["data"]["failed_layers"] = ["research"]
        payload["data"]["run_validation_status"] = "FAIL"
        validation = payload["checks"]["validation"]
        validation["dataset_manifest_completeness"] = {
            "missing": [],
            "invalid": [invalid_path],
            "checked_count": 2,
        }
        validation["partition_coverage_status"] = "FAIL"
        validation["failure_codes"] = [
            "dataset_manifest_invalid",
            "layer_validation_failed",
        ]
        validation["status"] = "FAIL"
        payload["status"]["validation"] = "FAIL"
        payload["status"]["smoke"] = "FAIL"
        payload["status"]["validation_matrix"] = "FAIL"
        payload["status"]["overall"] = "FAIL"
        _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "PASS"


def test_compare_rejects_failure_codes_with_data_pass(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    path = current / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"]["validation"]["failure_codes"] = ["primary_key_nulls"]
    payload["checks"]["validation"]["primary_key_null_failures"] = [
        "research_dataset"
    ]
    payload["checks"]["validation"]["status"] = "FAIL"
    payload["status"]["validation"] = "FAIL"
    payload["status"]["smoke"] = "FAIL"
    payload["status"]["validation_matrix"] = "FAIL"
    payload["status"]["overall"] = "FAIL"
    _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] in {
        "run_validation_data_facts_mismatch",
        "run_validation_detail_facts_mismatch",
    }


def test_compare_requires_declared_run_summary_artifact(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    (current / "summary.md").unlink()

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "run_summary_missing"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("alternative_summary_path", "run_summary_invalid"),
        ("ownership_created_at_mismatch", "run_root_ownership_invalid"),
    ),
)
def test_compare_blocks_run_root_contract_mutations(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = current / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    if mutation == "alternative_summary_path":
        alternative_summary = current / "alternative-summary.md"
        alternative_summary.write_text("# Alternative summary\n", encoding="utf-8")
        payload["summary"]["markdown_path"] = alternative_summary.name
    else:
        payload["root_ownership"]["created_at"] = "2026-06-26T00:00:02Z"
    _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == expected_reason
    assert result["output_results"] == []


def test_compare_root_file_is_blocked_json_on_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    compare_root = tmp_path / "compare-as-file"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    compare_root.write_text("not a directory\n", encoding="utf-8")

    exit_code = module.main(
        [
            "--baseline-root",
            str(baseline),
            "--current-root",
            str(current),
            "--compare-root",
            str(compare_root),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert exit_code == 1
    assert payload["schema_id"] == "qrp-compare"
    assert payload["overall_status"] == "BLOCKED"
    assert "compare_root_not_directory" in payload["blocked_reason"]


def test_compare_missing_output_manifest_is_fail(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    (current / "outputs" / "prediction_daily" / "manifest.json").unlink()

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert result["output_results"][0]["status"] == "FAIL"
    assert "output_manifest_missing" in result["output_results"][0]["failures"]


@pytest.mark.parametrize(
    "field",
    ("metadata", "files"),
)
def test_compare_rejects_semantically_invalid_run_output_index(
    tmp_path: Path,
    field: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    path = current / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field == "metadata":
        payload["outputs"][0][field]["scope"] = "REVIEW_SAFE"
    else:
        payload["outputs"][0][field] = ["other.parquet"]
    _write_json(path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] in {
        "output_manifest_metadata_invalid",
        "output_manifest_definition_files_mismatch",
    }
    assert result["output_results"] == []


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    (
        (
            "schema_version",
            "output_manifest_unreadable",
        ),
        (
            "contract_version",
            "output_manifest_unreadable",
        ),
        (
            "versioned_schema_id",
            "output_manifest_schema_id_mismatch",
        ),
        ("wrong_output_id", "output_manifest_output_id_mismatch"),
        ("wrong_run_id", "output_manifest_run_id_mismatch"),
        ("equal_output_and_definition_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition_id", "output_manifest_definition_id_invalid"),
        ("path_traversal", "output_file_path_invalid"),
        ("backslash_path", "output_file_path_invalid"),
        ("unknown_field", "output_manifest_fields_invalid"),
        ("wrong_kind_definition", "output_manifest_definition_id_invalid"),
        ("deep_nested_schema", "output_manifest_metadata_invalid"),
    ),
)
def test_compare_rejects_invalid_output_manifest_identity_and_paths(
    tmp_path: Path,
    mutation: str,
    expected_failure: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    manifest_path = current / "outputs" / "prediction_daily" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "schema_version":
        manifest["metadata"]["schema_version"] = "0.1"
    elif mutation == "contract_version":
        manifest["metadata"]["contract_version"] = "0.1"
    elif mutation == "versioned_schema_id":
        manifest["schema_id"] = "qrp-output-manifest/0.1"
    elif mutation == "wrong_output_id":
        manifest["output_id"] = "other_output"
    elif mutation == "wrong_run_id":
        manifest["run_id"] = "other_run"
    elif mutation == "equal_output_and_definition_ids":
        manifest["output_id"] = "stage_profile_diagnostic"
    elif mutation == "unknown_definition_id":
        manifest["output_definition_id"] = "unknown_outputs"
    elif mutation == "path_traversal":
        manifest["files"] = ["../outside.parquet"]
        baseline_manifest_path = (
            baseline / "outputs" / "prediction_daily" / "manifest.json"
        )
        baseline_manifest = json.loads(
            baseline_manifest_path.read_text(encoding="utf-8")
        )
        baseline_manifest["files"] = ["../outside.parquet"]
        _write_json(baseline_manifest_path, baseline_manifest)
        (baseline / "outputs" / "outside.parquet").write_bytes(b"same")
        (current / "outputs" / "outside.parquet").write_bytes(b"same")
    elif mutation == "backslash_path":
        manifest["files"] = ["nested\\prediction.parquet"]
    elif mutation == "unknown_field":
        manifest["unknown"] = True
    elif mutation == "wrong_kind_definition":
        manifest["output_definition_id"] = 7
    elif mutation == "deep_nested_schema":
        manifest["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    _write_json(manifest_path, manifest)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert expected_failure in result["output_results"][0]["failures"]


@pytest.mark.parametrize(
    ("baseline_kwargs", "current_kwargs", "failure"),
    (
        (
            {
                "metadata": {
                    "market_id": "a_share",
                    "run_type": "diagnostics",
                    "scope": "LOCAL_ONLY",
                }
            },
            {
                "metadata": {
                    "market_id": "crypto",
                    "run_type": "diagnostics",
                    "scope": "LOCAL_ONLY",
                }
            },
            "metadata_mismatch",
        ),
        (
            {"file_content": b"left"},
            {"file_content": b"right"},
            "file_hash_mismatch",
        ),
    ),
)
def test_compare_output_contract_mismatch_is_fail(
    tmp_path: Path,
    baseline_kwargs: dict[str, object],
    current_kwargs: dict[str, object],
    failure: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run", **baseline_kwargs)
    _write_run(current, "current_run", **current_kwargs)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert failure in result["output_results"][0]["failures"]


def test_compare_pairs_file_maps_output_ids_only(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run", output_id="prediction_old")
    _write_run(current, "current_run", output_id="prediction_new")
    pairs_file = tmp_path / "pairs.json"
    _write_json(
        pairs_file,
        {
            "schema_id": "qrp-compare-pairs",
            "pairs": [
                {
                    "baseline_output_id": "prediction_old",
                    "current_output_id": "prediction_new",
                }
            ],
        },
    )

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
        pairs_file=pairs_file,
    )

    assert result["overall_status"] == "PASS"
    assert result["pairing"]["mode"] == "pairs_file"
    assert result["output_results"][0]["baseline_output_id"] == "prediction_old"
    assert result["output_results"][0]["current_output_id"] == "prediction_new"


def test_compare_pairs_file_cannot_hide_invalid_declared_output(
    tmp_path: Path,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    _append_output(baseline, run_id="baseline_run", output_id="prediction_extra")
    _append_output(current, run_id="current_run", output_id="prediction_extra")
    current_run_path = current / "run.json"
    current_payload = json.loads(current_run_path.read_text(encoding="utf-8"))
    current_payload["outputs"][1]["metadata"]["scope"] = "REVIEW_SAFE"
    _write_json(current_run_path, current_payload)
    pairs_file = tmp_path / "pairs.json"
    _write_json(
        pairs_file,
        {
            "schema_id": "qrp-compare-pairs",
            "pairs": [
                {
                    "baseline_output_id": "prediction_daily",
                    "current_output_id": "prediction_daily",
                }
            ],
        },
    )

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
        pairs_file=pairs_file,
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "output_manifest_metadata_invalid"


@pytest.mark.parametrize("duplicate_side", ("baseline", "current"))
def test_compare_pairs_file_rejects_duplicate_pair_members(
    tmp_path: Path,
    duplicate_side: str,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    _append_output(baseline, run_id="baseline_run", output_id="prediction_extra")
    _append_output(current, run_id="current_run", output_id="prediction_extra")
    second_pair = {
        "baseline_output_id": "prediction_extra",
        "current_output_id": "prediction_extra",
    }
    second_pair[f"{duplicate_side}_output_id"] = "prediction_daily"
    pairs_file = tmp_path / "pairs.json"
    _write_json(
        pairs_file,
        {
            "schema_id": "qrp-compare-pairs",
            "pairs": [
                {
                    "baseline_output_id": "prediction_daily",
                    "current_output_id": "prediction_daily",
                },
                second_pair,
            ],
        },
    )

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
        pairs_file=pairs_file,
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "pairs_file_pair_duplicate"


@pytest.mark.parametrize(
    "payload",
    (
        {"pairs": []},
        {"schema_id": "qrp-compare-pairs", "pairs": []},
        {"schema_id": "qrp-compare-pairs/0.1", "pairs": []},
        {
            "schema_id": "qrp-compare-pairs",
            "pairs": [
                {
                    "baseline_output_id": "a",
                    "current_output_id": "b",
                    "metadata": {},
                }
            ],
        },
        {
            "schema_id": "qrp-compare-pairs",
            "pairs": [
                {"baseline_output_id": 7, "current_output_id": "prediction_new"}
            ],
        },
    ),
)
def test_compare_invalid_pairs_file_is_blocked(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    pairs_file = tmp_path / "pairs.json"
    _write_json(pairs_file, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
        pairs_file=pairs_file,
    )

    assert result["overall_status"] == "BLOCKED"


def test_compare_cli_not_run_status_returns_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    def fake_compare_runs(**_kwargs: object) -> dict[str, object]:
        return {"schema_id": "qrp-compare", "overall_status": "NOT_RUN"}

    monkeypatch.setattr(module, "compare_runs", fake_compare_runs)

    exit_code = module.main(
        [
            "--baseline-root",
            str(tmp_path / "baseline"),
            "--current-root",
            str(tmp_path / "current"),
            "--compare-root",
            str(tmp_path / "compare"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_status"] == "NOT_RUN"
    assert exit_code == 1


@pytest.mark.parametrize("compare_location", ("inside_baseline", "common_parent"))
def test_compare_root_cannot_overlap_either_input(
    tmp_path: Path,
    compare_location: str,
) -> None:
    module = _module()
    baseline = tmp_path / "runs" / "baseline"
    current = tmp_path / "runs" / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    compare_root = (
        baseline / "compare" if compare_location == "inside_baseline" else tmp_path
    )

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=compare_root,
    )

    assert result["overall_status"] == "BLOCKED"
    assert result["blocked_reason"] == "compare_root_overlaps_input"
    assert not (baseline / "compare").exists()


def test_compare_rejects_extra_output_bundle_member(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    (current / "outputs" / "prediction_daily" / "undeclared.bin").write_bytes(b"x")

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert "output_member_set_mismatch" in result["output_results"][0]["failures"]


def test_compare_hashes_output_files_without_whole_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.suffix == ".parquet":
            raise AssertionError("whole_file_hash_read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "PASS"


def test_compare_output_directory_is_published_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    compare_root = tmp_path / "compare"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")

    def fail_summary(_payload: dict[str, object]) -> str:
        raise RuntimeError("synthetic_summary_failure")

    monkeypatch.setattr(module, "_render_summary", fail_summary)
    with pytest.raises(RuntimeError, match="synthetic_summary_failure"):
        module.compare_runs(
            baseline_root=baseline,
            current_root=current,
            compare_root=compare_root,
        )

    assert not compare_root.exists()


def test_compare_publication_oserror_returns_blocked_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    compare_root = tmp_path / "compare"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    attempts = 0

    def fail_publish(_root: Path, _writer: object) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("synthetic_publish_denial")

    monkeypatch.setattr(module, "atomic_write_directory", fail_publish)

    payload = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=compare_root,
    )

    assert payload["overall_status"] == "BLOCKED"
    assert payload["blocked_reason"] == "compare_output_unavailable:PermissionError"
    assert attempts == 1
    assert not compare_root.exists()


def test_compare_data_summary_mismatch_is_fail(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = current / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["data"]["layers"][0]["datasets"][0]["manifest_summary"]["row_count"] = 4
    payload["data"]["layers"][0]["total_rows"] = 4
    payload["checks"]["schema_inventory"]["total_row_count"] = 4
    _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert result["data_results"][0]["status"] == "FAIL"
