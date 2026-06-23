from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

HANDOFF_TOOL_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)
TEST_BRANCH = "feature/synthetic-handoff-review"


def _load_handoff_module(name: str):
    script = HANDOFF_TOOL_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _retired_private_member_paths(
    segments: tuple[str, ...],
) -> tuple[str, ...]:
    members: list[str] = []
    for segment in segments:
        if segment.startswith("reports/private_"):
            members.append(f"{segment}/a_share/run/summary.json")
    return tuple(members)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _init_handoff_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "handoff-test@example.invalid")
    _git(repo, "config", "user.name", "Handoff Test")
    _git(repo, "checkout", "-b", TEST_BRANCH)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    fixture = repo / "tests" / "negative_fixtures.py"
    fixture.parent.mkdir()
    fixture.write_text(
        'LEAK_FIXTURE = "E:' + '/negative-fixture/not-real.txt"\n',
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add negative fixture")
    final_head = _git(repo, "rev-parse", "HEAD").strip()
    return base, final_head


def _build_test_package(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    out_dir = tmp_path / "out"
    task = {
        "slug": "handoff-test",
        "mode": "evidence",
        "branch": TEST_BRANCH,
        "base": base,
        "final_head": final_head,
        "validation_evidence": {
            "schema_version": "handoff-validation-evidence/0.1",
            "checks": [
                {
                    "name": "handoff_external_access",
                    "status": "BLOCKED",
                    "evidence": "blocked",
                }
            ],
        },
    }
    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=out_dir,
        task=task,
        verify=False,
    )
    assert result["zip_path"]
    return repo, Path(result["zip_path"]), Path(result["json_path"]), base, final_head


def _rewrite_package_member(package: Path, member: str, data: bytes) -> None:
    temp = package.with_suffix(".rewrite.zip")
    with zipfile.ZipFile(package, "r") as src, zipfile.ZipFile(temp, "w") as dst:
        for info in src.infolist():
            payload = data if info.filename == member else src.read(info.filename)
            dst.writestr(info, payload)
    temp.replace(package)


def _refresh_manifest_and_sidecar(package: Path, sidecar: Path) -> None:
    with zipfile.ZipFile(package, "r") as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name != "artifact_manifest.json"
        )
        entries = [
            {
                "path": name,
                "size": len(archive.read(name)),
                "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
            }
            for name in names
        ]
    manifest = {
        "schema_version": "handoff-artifact-manifest/0.1",
        "entries": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _rewrite_package_member(package, "artifact_manifest.json", manifest_bytes)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["zip_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
    payload["artifact_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_tracked_handoff_tool_source_of_truth_exists() -> None:
    required = {
        "README.md",
        "build_handoff_package.py",
        "verify_handoff_package.py",
        "handoff_safety.py",
        "self_test.py",
    }

    missing = sorted(
        name for name in required if not (HANDOFF_TOOL_ROOT / name).is_file()
    )

    assert missing == []


def test_builder_writes_final_branch_bundle_without_patch_member(
    tmp_path: Path,
) -> None:
    _repo, package, _sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        bundle = archive.read("final-branch.bundle")
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))

    bundle_entry = next(
        item for item in manifest["entries"] if item["path"] == "final-branch.bundle"
    )
    assert bundle
    assert bundle_entry["sha256"] == hashlib.sha256(bundle).hexdigest()


def test_builder_does_not_duplicate_tracked_tools_in_zip(tmp_path: Path) -> None:
    _repo, package, _sidecar, _base, _final_head = _build_test_package(tmp_path)

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()

    assert not any(name.startswith("run_tools/") for name in names)


def test_verifier_rejects_duplicate_tracked_tools_in_zip(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("run_tools/self_test.py", "# duplicate tracked source\n")
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "duplicate_tracked_source_in_zip: run_tools/self_test.py" in failures


def test_verifier_rejects_tracked_repo_path_copies_in_zip(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package, "a") as archive:
        archive.writestr("src/copied.py", "# tracked source copy\n")
        archive.writestr(".agents/copied.py", "# tracked agent source copy\n")
        archive.writestr("docs/copied.md", "tracked doc copy\n")
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "duplicate_tracked_source_in_zip: .agents/copied.py" in failures
    assert "duplicate_tracked_source_in_zip: docs/copied.md" in failures
    assert "duplicate_tracked_source_in_zip: src/copied.py" in failures


def test_verifier_rejects_generation_bundle_hash_mismatch(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        generation = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
    generation["bundle_sha256"] = "0" * 64
    _rewrite_package_member(
        package,
        "handoff_evidence/generation.json",
        (json.dumps(generation, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "generation_bundle_sha256_mismatch" in failures


def test_verify_source_tree_scan_rejects_extra_groups_and_count_mismatch() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    final_head = "1" * 40
    payload = {
        "schema_version": "handoff-source-tree-scan/0.1",
        "status": "PASS",
        "scanned_head": final_head,
        "expected_head": final_head,
        "scanned_tree": "final-branch.bundle clone",
        "finding_count": 0,
        "findings": [{"path": "src/example.py", "rule": "demo"}],
        "scan_groups": {
            "retired_text": {"status": "PASS", "finding_count": 0, "findings": []},
            "retired_governance_paths": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "retired_path_names": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "python_symbols": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "synthesized_strings": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "active_test_literals": {
                "status": "PASS",
                "finding_count": 1,
                "findings": [],
            },
            "structured_keys": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "governance_truth": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "sensitive_negative_literals": {
                "status": "PASS",
                "finding_count": 0,
                "findings": [],
            },
            "extra": {"status": "PASS", "finding_count": 0, "findings": []},
        },
    }

    failures = verifier.verify_source_tree_scan_evidence(
        payload,
        {"final_head": final_head},
    )

    assert "source_tree_scan_groups_unexpected" in failures
    assert "source_tree_scan_group_count_mismatch: active_test_literals" in failures
    assert "source_tree_scan_finding_count_mismatch" in failures


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake private markers used to test handoff safety rejection.
        "E:/repo/data/processed/a_share/factor/demo",
        "E:/repo/data/research/a_share/demo",
        "E:/repo/reports/private_research/a_share/run",
        "/workspace/qrp/data/processed/a_share/factor/demo",
        "/workspaces/qrp/data/research/a_share/demo",
        "/mnt/data/reports/private_research/a_share/run",
    ],
)
def test_source_tree_scan_private_path_rules_cover_handoff_bundle_text(
    text: str,
) -> None:
    builder = _load_handoff_module("build_handoff_package")

    rules = builder.source_tree_sensitive_negative_patterns()

    assert rules["sensitive_local_private_path"].search(text)


def test_handoff_safety_scans_external_json_sidecar(tmp_path: Path) -> None:
    safety = _load_handoff_module("handoff_safety")
    run_handoffs = ".run/handoffs"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "handoff_package_path": f"{run_handoffs}/handoff.zip",
                "handoff_sidecar_path": f"{run_handoffs}/handoff.json",
                # Intentional fake private markers for handoff safety rejection.
                "leaked_zip_path": "E:/quant-workspace/private/handoff.zip",
                "also_bad": "/mnt/data/private/handoff.zip",
                "home_bad": "/home/user/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = safety.scan_handoff_zip(package, json_path=sidecar)

    assert not result.ok
    assert any("sidecar_local_absolute_path_text" in item for item in result.failures)


def test_handoff_safety_rejects_data_research_members(tmp_path: Path) -> None:
    safety = _load_handoff_module("handoff_safety")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("data/research/a_share/demo/manifest.json", "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_safety_rejects_retired_private_report_paths(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    members = _retired_private_member_paths(safety.FORBIDDEN_PATH_SEGMENTS)

    assert members
    for index, member in enumerate(members):
        package = tmp_path / f"handoff-{index}.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(member, "{}\n")

        result = safety.scan_handoff_zip(package)

        assert not result.ok
        assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_safety_rejects_retired_private_report_path_text(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    retired_private_report_path = next(
        segment
        for segment in safety.FORBIDDEN_PATH_SEGMENTS
        if segment.startswith("reports/private_")
    )
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "handoff_evidence/leak.md",
            f"retired path {retired_private_report_path}/a_share/run\n",
        )

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("local_absolute_path_text" in item for item in result.failures)


def test_task_metadata_rejects_pr_url_in_evidence_mode() -> None:
    builder = _load_handoff_module("build_handoff_package")

    with pytest.raises(SystemExit, match="task_metadata_evidence_forbidden_field"):
        builder.task_metadata(
            {
                "schema_version": "handoff-task-metadata/0.1",
                "mode": "evidence",
                "slug": "demo",
                "branch": "feature/demo",
                "base": "0" * 40,
                "final_head": "1" * 40,
                "pr_url": "https://github.com/example/repo/pull/1",
            }
        )


def test_builder_requires_pr_url_for_pr_closure_even_without_verify(
    tmp_path: Path,
) -> None:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)

    with pytest.raises(SystemExit, match="task_pr_url_required_for_pr_closure"):
        builder.build_handoff_package(
            repo_root=repo,
            out_dir=tmp_path / "out",
            task={
                "schema_version": "handoff-task-metadata/0.1",
                "mode": "pr_closure",
                "slug": "demo",
                "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
            },
            verify=False,
        )


@pytest.mark.parametrize(
    "member",
    [
        ".run/private/s1/summary.json",
        ".run/smoke/s1/summary.json",
        ".run/s86/summary.json",
    ],
)
def test_handoff_safety_rejects_private_run_members(
    tmp_path: Path,
    member: str,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(member, "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


@pytest.mark.parametrize(
    "logical_path",
    [".run/private/s1", ".run\\\\private\\\\s1"],
)
def test_handoff_safety_rejects_private_run_path_in_sidecar(
    tmp_path: Path,
    logical_path: str,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps({"evidence_root": logical_path}),
        encoding="utf-8",
    )

    result = safety.scan_handoff_zip(package, json_path=sidecar)

    assert not result.ok
    assert any("private_run_path" in item for item in result.failures)


def test_handoff_safety_rejects_repo_local_codex_members(tmp_path: Path) -> None:
    safety = _load_handoff_module("handoff_safety")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(".codex/config.json", "{}\n")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("forbidden_path_segment" in item for item in result.failures)


def test_handoff_safety_allows_historical_codex_in_bundle_when_head_is_clean(
    tmp_path: Path,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    repo = tmp_path / "repo"
    _init_handoff_repo(repo)
    codex_config = repo / ".codex" / "config.json"
    codex_config.parent.mkdir()
    codex_config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".codex/config.json")
    _git(repo, "commit", "-m", "add historical repo-local codex fixture")
    _git(repo, "rm", ".codex/config.json")
    _git(repo, "commit", "-m", "remove historical repo-local codex fixture")
    bundle = tmp_path / "final-branch.bundle"
    _git(repo, "bundle", "create", str(bundle), TEST_BRANCH)
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")

    result = safety.scan_handoff_zip(package)

    assert result.ok


def test_handoff_safety_rejects_codex_in_bundle_head_tree(tmp_path: Path) -> None:
    safety = _load_handoff_module("handoff_safety")
    repo = tmp_path / "repo"
    _init_handoff_repo(repo)
    codex_config = repo / ".codex" / "config.json"
    codex_config.parent.mkdir()
    codex_config.write_text("{}\n", encoding="utf-8")
    _git(repo, "add", ".codex/config.json")
    _git(repo, "commit", "-m", "add current repo-local codex fixture")
    bundle = tmp_path / "final-branch.bundle"
    _git(repo, "bundle", "create", str(bundle), TEST_BRANCH)
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.write(bundle, "final-branch.bundle")

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("bundle_forbidden_path_segment" in item for item in result.failures)


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake private markers used to test handoff safety rejection.
        "/workspace/qrp/data/research/a_share/demo",
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/opt/qrp/data/processed/a_share/factor/demo",
    ],
)
def test_handoff_safety_rejects_absolute_paths_with_private_segments(
    tmp_path: Path,
    text: str,
) -> None:
    safety = _load_handoff_module("handoff_safety")
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("handoff_evidence/leak.md", text)

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("local_absolute_path_text" in item for item in result.failures)


def test_handoff_safety_rejects_local_paths_in_plain_text() -> None:
    safety = _load_handoff_module("handoff_safety")
    # Intentional fake private marker used to test sanitizer/package-safety rejection.
    text = "\n".join(
        [
            "old local example C:/Users/name/private.txt",
            "old workspace example /workspace/qrp/data/research/a_share/demo",
        ]
    )

    assert safety.unsafe_text_reason(text, name="handoff.txt") == "local_absolute_path"


def test_handoff_safety_allows_public_aggregate_evidence_counts() -> None:
    safety = _load_handoff_module("handoff_safety")
    text = "\n".join(
        [
            "| processed panel parity | row_count_bucket=large | verified aggregate |",
        ]
    )

    assert safety.unsafe_text_reason(text, name="handoff_evidence/summary.md") is None


def test_verify_fails_when_sidecar_safety_fails(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    run_handoffs = ".run/handoffs"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("artifact_manifest.json", "{}\n")
        archive.writestr("final-branch.bundle", b"not-a-real-bundle")
        archive.writestr("git_status.txt", "clean\n")
        archive.writestr("task_metadata.json", "{}\n")
        archive.writestr("validation_evidence.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "handoff_package_path": f"{run_handoffs}/handoff.zip",
                "handoff_sidecar_path": f"{run_handoffs}/handoff.json",
                "zip_sha256": verifier.sha256_file(package),
                "artifact_manifest_sha256": "0" * 64,
                "branch": TEST_BRANCH,
                "base": "0" * 40,
                "final_head": "1" * 40,
                "generated_at_utc": "2026-06-19T00:00:00+00:00",
                "status_summary": {
                    "validation": "NOT_RUN",
                    "handoff_verify": "NOT_RUN",
                    "handoff_safety": "NOT_RUN",
                    "blocked_count": 0,
                    "fail_count": 0,
                },
                # Intentional fake private marker for handoff safety rejection.
                "zip_path": "E:/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "sidecar_safety_failed" in failures


def test_verify_allows_evidence_task_metadata_without_pr_url() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_version": "handoff-task-metadata/0.1",
            "mode": "evidence",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
        }
    )

    assert "task_metadata_schema_missing: pr_url" not in failures


def test_verify_rejects_pr_url_in_evidence_task_metadata() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_version": "handoff-task-metadata/0.1",
            "mode": "evidence",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
            "final_head": "1" * 40,
            "pr_url": "https://github.com/example/repo/pull/1",
        }
    )

    assert failures == ["task_metadata_evidence_forbidden_field: pr_url"]


def test_verify_requires_pr_url_in_pr_closure_task_metadata() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_task_metadata_schema(
        {
            "schema_version": "handoff-task-metadata/0.1",
            "mode": "pr_closure",
            "slug": "demo",
            "branch": TEST_BRANCH,
            "base": "0" * 40,
        }
    )

    assert "task_metadata_schema_missing: pr_url" in failures


def test_verify_recursively_rejects_forbidden_handoff_evidence_payload_keys() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    builder = _load_handoff_module("build_handoff_package")
    forbidden_key = next(
        key
        for key in builder.HANDOFF_EVIDENCE_FORBIDDEN_KEY_COVERAGE_CASES
        if key == "check"
    )

    failures = verifier.handoff_evidence_forbidden_field_failures(
        "handoff_evidence/custom.json",
        {
            "schema_version": "demo/0.1",
            "checks": [{"nested": {forbidden_key: "old-row-key"}}],
        },
    )

    assert failures == [
        "handoff_evidence_forbidden_field: "
        "handoff_evidence/custom.json: checks.0.nested.check"
    ]


def test_verify_rejects_validation_row_extra_fields() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    validation_payload = {
        "schema_version": "handoff-validation-evidence/0.1",
        "checks": [
            {
                "name": "handoff_verify",
                "status": "PASS",
                "command": "python verify.py --zip <zip> --json <json>",
                "evidence": "PASS",
            },
        ],
    }

    failures = verifier.verify_validation_evidence_schema(validation_payload)

    assert "validation_evidence_row_forbidden_field: 0:command" in failures


def test_verify_baseline_final_compare_uses_metadata_groups() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _baseline_final_compare_payload(
        {
            "control_window_alpha": {
                "comparison_group": "old_baseline",
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
            "candidate_window_beta": {
                "comparison_group": "refreshed_prediction_baseline",
                "status": "PASS",
                "prediction_summary": {
                    "artifact_ids": [
                        "equal_weight_rank_ensemble_mom_20d_prediction_2020",
                        "ic_weighted_rank_ensemble_mom_20d_prediction_2020",
                    ],
                    "failed_artifacts": [],
                },
            },
        }
    )

    assert verifier.verify_baseline_final_compare(payload) == []


def test_verify_baseline_final_compare_rejects_flat_status_aliases() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _baseline_final_compare_payload(
        {
            "control_window_alpha": {
                "comparison_group": "old_baseline",
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
            "candidate_window_beta": {
                "comparison_group": "refreshed_prediction_baseline",
                "status": "PASS",
                "prediction_summary": {
                    "artifact_ids": [
                        "equal_weight_rank_ensemble_mom_20d_prediction_2020",
                        "ic_weighted_rank_ensemble_mom_20d_prediction_2020",
                    ],
                    "failed_artifacts": [],
                },
            },
        }
    )
    payload["overall_status"] = "PASS"

    failures = verifier.verify_baseline_final_compare(payload)

    assert "baseline_final_compare_flat_alias_present: overall_status" in failures


def test_verify_baseline_final_compare_requires_refreshed_group() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _baseline_final_compare_payload(
        {
            "control_window_alpha": {
                "comparison_group": "old_baseline",
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
        }
    )

    failures = verifier.verify_baseline_final_compare(payload)

    assert (
        "baseline_final_compare_missing_required_group: refreshed_prediction_baseline"
        in failures
    )


def test_verify_baseline_final_compare_rejects_loose_rank_artifact_prefix() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _baseline_final_compare_payload(
        {
            "control_window_alpha": {
                "comparison_group": "old_baseline",
                "status": "FAIL_EXPECTED",
                "reason_code": "stale_rank_ensemble_prediction_baseline",
                "failed_artifacts": [
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                ],
            },
            "candidate_window_beta": {
                "comparison_group": "refreshed_prediction_baseline",
                "status": "PASS",
                "prediction_summary": {
                    "artifact_ids": [
                        "equal_weight_rank_ensemble_smoke",
                        "ic_weighted_rank_ensemble_smoke",
                    ],
                    "failed_artifacts": [],
                },
            },
        }
    )

    failures = verifier.verify_baseline_final_compare(payload)

    assert (
        "baseline_final_compare_refreshed_missing_artifact: "
        "candidate_window_beta:equal_weight_rank_ensemble"
    ) in failures
    assert (
        "baseline_final_compare_refreshed_missing_artifact: "
        "candidate_window_beta:ic_weighted_rank_ensemble"
    ) in failures


def _baseline_final_compare_payload(
    comparisons: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "qrp-baseline-final-compare-summary/0.1",
        "status": {
            "raw": "FAIL",
            "policy_adjusted": "PASS",
            "validation_decision": "PASS",
        },
        "baseline_commit": "0" * 8,
        "baseline_commit_full": "0" * 40,
        "final_head": "1" * 40,
        "final_smoke_commit": "1" * 40,
        "git_status_before_smoke": "clean",
        "git_status_after_smoke": "clean_or_only_ignored_run_outputs",
        "comparison_groups": {
            "old_baseline": {"status": "FAIL_EXPECTED"},
            "refreshed_prediction_baseline": {"status": "PASS"},
        },
        "contract_failures": [],
        "comparisons": comparisons,
    }
