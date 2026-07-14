from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

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
        run_id="local_verification_output",
        replace_output=False,
        repo_root=repo_root,
    )

    assert run_root.is_dir()
    payload = module._read_run_json(run_root)
    assert payload["schema_id"] == "qrp-run"
    assert payload["run_id"] == "local_verification_output"
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
        module._prepare_run_root(
            run_root,
            run_id="local_verification_output",
            replace_output=True,
            repo_root=repo_root,
        )

    assert (run_root / "unowned.txt").is_file()


def test_local_verification_replace_output_allows_owned_directory(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / ".run" / "local" / "report-output"

    module._prepare_run_root(
        run_root,
        run_id="report_output",
        replace_output=False,
        repo_root=repo_root,
    )
    (run_root / "old-artifact.txt").write_text("old\n", encoding="utf-8")
    module._prepare_run_root(
        run_root,
        run_id="report_output",
        replace_output=True,
        repo_root=repo_root,
    )

    assert run_root.is_dir()
    assert not (run_root / "old-artifact.txt").exists()
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
        replace_output=False,
        repo_root=repo_root,
    )
    payload = module._read_run_json(run_root)
    payload["root_ownership"]["schema_id"] = "nested/version"
    (run_root / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="run_root_not_owned_by_local_verification_workflow",
    ):
        module._prepare_run_root(
            run_root,
            run_id="report_output",
            replace_output=True,
            repo_root=repo_root,
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
        module._output_index(output_bundle, run_root, expected_run_id="run")


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
    output_bundle = run_root / "outputs" / "prediction_daily"
    output_bundle.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema_id": "qrp-output-manifest",
        "output_id": "prediction_daily",
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": "run",
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
    if mutation == "unknown_field":
        payload["unknown"] = True
    elif mutation == "wrong_kind_output_id":
        payload["output_id"] = 7
    elif mutation == "wrong_kind_run_id":
        payload["run_id"] = 7
    elif mutation == "wrong_run_id":
        payload["run_id"] = "other_run"
    elif mutation == "equal_output_and_definition_ids":
        payload["output_id"] = "stage_profile_diagnostic"
    elif mutation == "unknown_definition_id":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "deep_nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "backslash_file":
        payload["files"] = ["nested\\prediction.parquet"]
    (output_bundle / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(output_bundle, run_root, expected_run_id="run")


def test_local_verification_output_index_requires_declared_files(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    run_root = tmp_path / "run"
    output_bundle = run_root / "outputs" / "prediction_daily"
    output_bundle.mkdir(parents=True)
    payload = {
        "schema_id": "qrp-output-manifest",
        "output_id": "prediction_daily",
        "output_definition_id": "stage_profile_diagnostic",
        "run_id": "run",
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
    (output_bundle / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(output_bundle, run_root, expected_run_id="run")

    for name in payload["files"]:
        (output_bundle / name).write_bytes(b"payload")
    index = module._output_index(output_bundle, run_root, expected_run_id="run")
    assert index == {
        **payload,
        "manifest_path": "outputs/prediction_daily/manifest.json",
    }


def test_local_verification_workflow_exposes_current_run_root_config() -> None:
    module = _workflow_module()

    assert set(module.LocalVerificationRunConfig.__dataclass_fields__) == {
        "run_root",
        "start_date",
        "end_date",
        "profile",
        "prediction_train_end",
        "prediction_test_start",
        "portfolio_frequency",
        "portfolio_execution_lag_steps",
        "evaluation_start_date",
        "evaluation_end_date",
        "prediction_test_end",
        "replace_output",
        "run_id",
    }


def test_local_verification_uses_profile_schema_inventory_columns() -> None:
    module = _workflow_module()
    source = inspect.getsource(module)

    assert "profile.schema_inventory_important_columns()" in source
    assert "important_null_count_columns=" in source
    assert "forward_return_label_column" not in source
    assert '"eligible"' not in source
    assert "markets.implementations.a_share" not in source


@pytest.mark.parametrize(
    "run_root",
    (
        Path("data") / "team" / "qrp" / ".run" / "s1",
        Path("outputs") / "team" / "qrp" / ".run" / "s2",
        Path("logs") / "team" / "qrp" / ".run" / "s3",
        Path("handoff_evidence") / "team" / "qrp" / ".run" / "s4",
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
