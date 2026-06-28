from __future__ import annotations

import importlib.util
import json
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


def _run_payload(
    run_id: str,
    *,
    output_id: str = "prediction_daily",
    output_definition_id: str = "prediction_outputs",
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
        "status": {"overall": "PASS"},
        "created_at": "2026-06-26T00:00:00Z",
        "completed_at": "2026-06-26T00:00:01Z",
        "code": {},
        "config": {},
        "inputs": {},
        "data": {
            "layer_count": 1,
            "run_validation_status": "pass",
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
                                "duplicate_primary_key_count": 0,
                                "validation_status": "pass",
                            },
                        }
                    ],
                    "total_rows": 3,
                    "date_min": "2024-01-01",
                    "date_max": "2024-01-03",
                    "missing_dataset_manifests": [],
                    "failed_dataset_validations": [],
                    "layer_validation_status": "pass",
                    "created_at": "2026-06-26T00:00:00Z",
                }
            ],
        },
        "outputs": [
            {
                "schema_id": "qrp-output-manifest",
                "output_id": output_id,
                "output_definition_id": output_definition_id,
                "run_id": run_id,
                "metadata": metadata or {"row_count": 3, "partition_count": 1},
                "files": ["prediction.parquet"],
                "manifest_path": f"outputs/{output_id}/manifest.json",
            }
        ],
        "checks": {},
        "execution_metrics": {},
        "logs": {},
        "summary": {"markdown_path": "summary.md"},
    }


def _write_run(
    root: Path,
    run_id: str,
    *,
    output_id: str = "prediction_daily",
    output_definition_id: str = "prediction_outputs",
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
    manifest = dict(payload["outputs"][0])
    manifest.pop("manifest_path")
    _write_json(root / "outputs" / output_id / "manifest.json", manifest)
    (root / "outputs" / output_id / "prediction.parquet").write_bytes(file_content)


def _write_run_without_outputs(root: Path, run_id: str) -> None:
    payload = _run_payload(run_id)
    payload["outputs"] = []
    _write_json(root / "run.json", payload)


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
    "mutation",
    ("missing_run", "invalid_json", "missing_schema", "versioned_schema"),
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
        payload["schema_id"] = None if mutation == "missing_schema" else "qrp-run/0.1"
        _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "BLOCKED"
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
    ("mutation", "expected_failure"),
    (
        (
            "schema_version",
            "output_manifest_identity_invalid:retired_identity_key:schema_version",
        ),
        (
            "contract_version",
            "output_manifest_identity_invalid:retired_identity_key:contract_version",
        ),
        (
            "versioned_schema_id",
            "output_manifest_schema_id_mismatch",
        ),
        ("wrong_output_id", "output_manifest_output_id_mismatch"),
        ("wrong_run_id", "output_manifest_run_id_mismatch"),
        ("path_traversal", "output_file_path_invalid"),
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
            {"metadata": {"row_count": 3, "partition_count": 1}},
            {"metadata": {"row_count": 4, "partition_count": 1}},
            "metadata_mismatch",
        ),
        (
            {"output_definition_id": "prediction_outputs"},
            {"output_definition_id": "evaluation_outputs"},
            "output_definition_id_mismatch",
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


def test_compare_data_summary_mismatch_is_fail(tmp_path: Path) -> None:
    module = _module()
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    _write_run(baseline, "baseline_run")
    _write_run(current, "current_run")
    run_path = current / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["data"]["layers"][0]["datasets"][0]["manifest_summary"]["row_count"] = 4
    _write_json(run_path, payload)

    result = module.compare_runs(
        baseline_root=baseline,
        current_root=current,
        compare_root=tmp_path / "compare",
    )

    assert result["overall_status"] == "FAIL"
    assert result["data_results"][0]["status"] == "FAIL"
