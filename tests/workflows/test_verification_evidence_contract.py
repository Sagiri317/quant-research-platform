from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

WORKFLOW_MODULE = "quant_research_platform.workflows.verification_evidence"


def test_verification_evidence_output_root_is_explicit_not_run_workspace(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_root = repo_root / ("." + "run") / "private" / "verification-output"

    module._prepare_output_root(
        output_root,
        replace_output=False,
        repo_root=repo_root,
    )

    assert output_root.is_dir()
    assert module._ownership_marker_path(output_root).is_file()


def test_verification_evidence_replace_output_rejects_unowned_directory(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_root = repo_root / ("." + "run") / "private" / "evidence-output"
    output_root.mkdir(parents=True)
    (output_root / "unowned.txt").write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="output_root_not_owned_by_verification_evidence_workflow",
    ):
        module._prepare_output_root(
            output_root,
            replace_output=True,
            repo_root=repo_root,
        )

    assert (output_root / "unowned.txt").is_file()


def test_verification_evidence_replace_output_allows_owned_directory(
    tmp_path: Path,
) -> None:
    module = _workflow_module()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    output_root = repo_root / ("." + "run") / "private" / "report-output"

    module._prepare_output_root(
        output_root,
        replace_output=False,
        repo_root=repo_root,
    )
    (output_root / "old-artifact.txt").write_text("old\n", encoding="utf-8")
    module._prepare_output_root(
        output_root,
        replace_output=True,
        repo_root=repo_root,
    )

    assert output_root.is_dir()
    assert not (output_root / "old-artifact.txt").exists()
    assert module._ownership_marker_path(output_root).is_file()


def test_verification_evidence_workflow_removes_run_workspace_guard() -> None:
    module = _workflow_module()

    assert not hasattr(module, "_assert_output_root_under_run")


def _workflow_module():
    spec = importlib.util.find_spec(WORKFLOW_MODULE)
    assert spec is not None, "verification_evidence workflow module missing"
    return importlib.import_module(WORKFLOW_MODULE)
