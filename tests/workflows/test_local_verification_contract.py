from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from quant_research_platform.artifacts import outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.diagnostics import validation_summary
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_evaluation_diagnostics,
    make_output_edge_prediction_spec,
    make_output_edge_report_spec,
)
from tests.helpers.output_frames import (
    make_artifact_evaluation_group_spread_summary_frame,
    make_artifact_evaluation_grouped_label_values_frame,
    make_artifact_evaluation_ic_summary_frame,
    make_artifact_evaluation_metrics_frame,
    make_artifact_prediction_output_frame,
    make_artifact_signal_spec,
)

WORKFLOW_MODULE = "quant_research_platform.workflows.local_verification"


@pytest.mark.parametrize("logical_layer", ("interim", "processed", "research"))
def test_local_verification_layer_producer_accepts_exact_owner_members(
    tmp_path: Path,
    logical_layer: str,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    dataset_root = run_root / "data" / logical_layer / "dataset"

    assert module._logical_layer(dataset_root, run_root) == logical_layer


def test_local_verification_output_index_rejects_alias_above_run_root(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    target = tmp_path / "target"
    (target / "run" / "outputs" / "output").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")
    run_root = alias / "run"

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            run_root / "outputs" / "output",
            run_root,
            expected_run_id="run",
            expected_output_definition_id="prediction_outputs",
        )


@pytest.mark.parametrize("mutation", ("raw", "unknown", "outside"))
def test_local_verification_layer_grouping_rejects_non_owner_members_before_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    layout = module.LocalVerificationLayout.from_run_root(run_root)
    if mutation == "raw":
        dataset_root = run_root / "data" / "raw" / "source" / "dataset"
    elif mutation == "unknown":
        dataset_root = run_root / "data" / "archive" / "dataset"
    else:
        dataset_root = tmp_path / "outside" / "data" / "processed" / "dataset"

    def reject_summary_build(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("invalid_layer_reached_storage_summary")

    monkeypatch.setattr(module, "build_layer_summary", reject_summary_build)

    with pytest.raises(
        ValueError,
        match="local_verification_dataset_layer_invalid",
    ):
        module._build_layer_summaries(
            layout=layout,
            dataset_roots=(dataset_root,),
        )


def test_local_verification_run_root_writes_seed_run_json(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".run" / "local" / "local-verification-output"

    module._prepare_run_root(
        run_root,
        run_id="run_000001",
        repo_root=repo_root,
    )

    assert run_root.is_dir()
    payload = module._read_run_json(run_root)
    assert payload["schema_id"] == "qrp-run"
    assert payload["run_id"] == "run_000001"
    assert payload["root_ownership"] == {
        "owner": "local_verification",
        "workflow_id": "local_verification",
        "created_by": "quant_research_platform",
        "created_at": payload["created_at"],
        "replace_safe": True,
    }


def test_local_verification_replace_output_rejects_unowned_directory(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".run" / "local" / "local-verification-output"
    run_root.mkdir(parents=True)
    (run_root / "unowned.txt").write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="run_root_not_owned_by_local_verification_workflow",
    ):
        module._replacement_target_exists(
            run_root,
            replace_output=True,
        )

    assert (run_root / "unowned.txt").is_file()


def test_local_verification_replace_target_validation_preserves_owned_directory(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".run" / "local" / "report-output"

    module._prepare_run_root(
        run_root,
        run_id="report_output",
        repo_root=repo_root,
    )
    (run_root / "old-artifact.txt").write_text("old\n", encoding="utf-8")
    replacing = module._replacement_target_exists(
        run_root,
        replace_output=True,
    )

    assert run_root.is_dir()
    assert replacing is True
    assert (run_root / "old-artifact.txt").is_file()
    assert module._read_run_json(run_root)["root_ownership"]["replace_safe"] is True


def test_local_verification_rejects_owned_run_json_with_nested_identity_violation(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".run" / "local" / "report-output"
    module._prepare_run_root(
        run_root,
        run_id="report_output",
        repo_root=repo_root,
    )
    payload = module._read_run_json(run_root)
    payload["root_ownership"]["schema_id"] = "nested/version"
    (run_root / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="run_root_not_owned_by_local_verification_workflow",
    ):
        module._replacement_target_exists(
            run_root,
            replace_output=True,
        )


def test_local_verification_output_index_rejects_manifest_identity_violation(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    output_bundle = run_root / "outputs" / "prediction_daily"
    output_bundle.mkdir(parents=True)
    (output_bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "qrp-output-manifest",
                "output_id": "prediction_daily",
                "output_definition_id": "prediction_outputs",
                "run_id": "run",
                "metadata": {"schema_id": "nested/version"},
                "files": ["prediction.parquet"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            output_bundle,
            run_root,
            expected_run_id="run",
            expected_output_definition_id="prediction_outputs",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_field",
        "wrong_kind_output_id",
        "wrong_kind_run_id",
        "wrong_run_id",
        "equal_output_and_definition_ids",
        "unknown_definition_id",
        "deep_nested_schema_id",
        "backslash_file",
    ),
)
def test_local_verification_output_index_rejects_noncurrent_manifest_envelope(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        "run",
        "prediction",
    )
    output_bundle = outputs.write_prediction_outputs(
        make_artifact_prediction_output_frame(),
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(project_root=run_root),
    )
    manifest_path = output_bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "unknown_field":
        payload["unknown"] = True
    elif mutation == "wrong_kind_output_id":
        payload["output_id"] = 7
    elif mutation == "wrong_kind_run_id":
        payload["run_id"] = 7
    elif mutation == "wrong_run_id":
        payload["run_id"] = "other_run"
    elif mutation == "equal_output_and_definition_ids":
        payload["output_id"] = "prediction_outputs"
    elif mutation == "unknown_definition_id":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "deep_nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "backslash_file":
        payload["files"] = ["nested\\prediction.parquet"]
    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            output_bundle,
            run_root,
            expected_run_id="run",
            expected_output_definition_id="prediction_outputs",
        )


def test_local_verification_output_index_requires_declared_files(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        "run",
        "prediction",
    )
    output_bundle = outputs.write_prediction_outputs(
        make_artifact_prediction_output_frame(),
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(project_root=run_root),
    )
    payload = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    prediction_path = output_bundle / "prediction.parquet"
    prediction_path.unlink()

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            output_bundle,
            run_root,
            expected_run_id="run",
            expected_output_definition_id="prediction_outputs",
        )

    prediction_path.write_bytes(b"payload")
    index, members = module._output_index(
        output_bundle,
        run_root,
        expected_run_id="run",
        expected_output_definition_id="prediction_outputs",
    )
    assert index == {
        **payload,
        "manifest_path": f"outputs/{output_spec.output_id}/manifest.json",
    }
    assert tuple(path.name for path, _identity in members) == (
        "manifest.json",
        *payload["files"],
    )

    replacement = output_bundle / "replacement.parquet"
    replacement.write_bytes(b"replacement")
    replacement.replace(prediction_path)
    result = validation_summary._share_safety_scan(
        members,
        output_bundle_expectations=(
            (
                output_bundle,
                "run",
                output_spec.output_id,
                "prediction_outputs",
            ),
        ),
    )
    assert result["status"] == "FAIL"
    assert result["failures"][-1] == {
        "path": f"outputs/{output_spec.output_id}/manifest.json",
        "reason": "identity_changed_or_unreadable",
    }


def test_output_index_rejects_prediction_output_substituted_for_evaluation(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    spec = make_output_edge_prediction_spec()
    output_spec = make_output_edge_report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    prediction_root = outputs.write_prediction_outputs(
        make_artifact_prediction_output_frame(),
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(project_root=run_root),
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            prediction_root,
            run_root,
            expected_run_id=output_spec.run_id,
            expected_output_definition_id="evaluation_outputs",
        )


def test_output_index_requires_mandatory_evaluation_summary_member(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    spec = EvaluationSpec(
        run_id="evaluation_required",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    output_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )
    evaluation_bundle = outputs.write_evaluation_outputs(
        make_artifact_evaluation_metrics_frame(),
        make_artifact_evaluation_grouped_label_values_frame(),
        ic_summary=make_artifact_evaluation_ic_summary_frame(),
        group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(project_root=run_root),
    )
    (evaluation_bundle / "summary.md").unlink()

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(
            evaluation_bundle,
            run_root,
            expected_run_id=spec.run_id,
            expected_output_definition_id="evaluation_outputs",
        )


@pytest.mark.parametrize(
    "run_root",
    (
        Path("data") / "team" / "qrp" / ".run" / "s1",
        Path("outputs") / "team" / "qrp" / ".run" / "s2",
        Path("logs") / "team" / "qrp" / ".run" / "s3",
        Path(".run") / "pytest" / "tmp" / "all" / "s5",
    ),
)
def test_local_verification_run_root_logical_name_ignores_host_markers(
    run_root: Path,
) -> None:
    module = _workflow_module()

    assert module._logical_run_root(run_root) == run_root.name


def _workflow_module():
    spec = importlib.util.find_spec(WORKFLOW_MODULE)
    assert spec is not None, "verification_summary workflow module missing"
    return importlib.import_module(WORKFLOW_MODULE)
