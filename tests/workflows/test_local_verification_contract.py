from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pytest

WORKFLOW_MODULE = "quant_research_platform.workflows.local_verification"


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
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_manifest_schema_id_invalid"):
        module._output_index(output_bundle, run_root)


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


def _workflow_module():
    spec = importlib.util.find_spec(WORKFLOW_MODULE)
    assert spec is not None, "verification_summary workflow module missing"
    return importlib.import_module(WORKFLOW_MODULE)
