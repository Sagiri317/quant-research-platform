from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.policy.share_safety import local_run_root_reason
from quant_research_platform.workflows import local_verification as module


def test_local_verification_run_root_accepts_explicit_locations(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def validate_local_run_root(run_root: Path, root: Path) -> None:
        reason = local_run_root_reason(run_root, root)
        if reason is not None:
            raise ValueError(reason)

    for run_root in (
        repo_root / "local-verification-output" / "s1",
        repo_root / "outputs" / "local" / "a_share" / "s1",
        repo_root / ".run" / "smoke" / "s1",
        repo_root / ".run" / "reviews" / "s1",
        tmp_path / "outside" / "local-verification-output" / "s1",
    ):
        validate_local_run_root(run_root, repo_root)

    with pytest.raises(ValueError, match="run_root_must_not_be_repo_root"):
        validate_local_run_root(repo_root, repo_root)


def test_prepare_run_root_creates_explicit_repo_internal_output(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_root = repo_root / "local-verification-output" / "s1"

    module._prepare_run_root(
        run_root,
        run_id="s1",
        replace_output=False,
        repo_root=repo_root,
    )

    payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert payload["schema_id"] == "qrp-run"
    assert payload["run_id"] == "s1"


def test_prepare_run_root_uses_git_boundary_outside_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").write_text("gitdir: ../repo.git\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root = repo_root / "local-verification-output" / "s1"

    monkeypatch.chdir(outside)

    module._prepare_run_root(
        run_root,
        run_id="s1",
        replace_output=False,
    )

    assert (run_root / "run.json").is_file()


