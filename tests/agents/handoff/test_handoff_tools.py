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


def _load_handoff_module(name: str):
    script = HANDOFF_TOOL_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    _git(repo, "checkout", "-b", "feature/qrp-clean-rewrite")
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
        "branch": "feature/qrp-clean-rewrite",
        "base": base,
        "final_head": final_head,
        "final_sha": final_head,
        "validation_evidence": {
            "schema_version": "handoff-validation-evidence/0.1",
            "checks": [
                {
                    "name": "github_review_threads",
                    "status": "BLOCKED",
                    "evidence": "blocked",
                }
            ],
        },
        "review_threads": {
            "schema_version": "handoff-review-threads/0.1",
            "status": "BLOCKED",
            "counts_available": False,
            "total_count": None,
            "unresolved_count": None,
            "threads": [],
            "evidence": (
                "Live GitHub review-thread access was unavailable; "
                "counts are unknown."
            ),
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


def test_builder_writes_raw_binary_diff_without_sanitizing(tmp_path: Path) -> None:
    repo, package, _sidecar, base, final_head = _build_test_package(tmp_path)
    raw_diff = _git(repo, "diff", "--binary", f"{base}..{final_head}").encode()

    with zipfile.ZipFile(package) as archive:
        packaged_diff = archive.read("diff.patch")
        manifest = json.loads(archive.read("artifact_manifest.json").decode("utf-8"))

    diff_entry = next(
        item for item in manifest["entries"] if item["path"] == "diff.patch"
    )
    assert packaged_diff == raw_diff
    assert diff_entry["sha256"] == hashlib.sha256(raw_diff).hexdigest()


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


def test_verifier_rejects_sanitized_diff_even_when_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    with zipfile.ZipFile(package) as archive:
        sanitized_diff = archive.read("diff.patch").replace(
            b"E:" + b"/negative-fixture/not-real.txt",
            b"<LOCAL_ABSOLUTE_PATH>",
        )
    _rewrite_package_member(package, "diff.patch", sanitized_diff)
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "diff_patch_mismatch" in failures


def test_verifier_rejects_non_applicable_diff_patch(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    _rewrite_package_member(package, "diff.patch", b"not a patch\n")
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "diff_patch_mismatch" in failures
    assert "diff_patch_apply_check_failed" in failures


def test_handoff_safety_scans_external_json_sidecar(tmp_path: Path) -> None:
    safety = _load_handoff_module("handoff_safety")
    run_reviews = "." + "run/reviews"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("task_metadata.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "zip_logical_path": f"{run_reviews}/handoff.zip",
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


@pytest.mark.parametrize(
    "text",
    [
        "/workspace/qrp/data/research/a_share/demo",
        "/workspaces/qrp/reports/private_real/a_share/run",
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
        archive.writestr("review_evidence/leak.md", text)

    result = safety.scan_handoff_zip(package)

    assert not result.ok
    assert any("local_absolute_path_text" in item for item in result.failures)


def test_verify_fails_when_external_sidecar_safety_fails(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    run_reviews = "." + "run/reviews"
    package = tmp_path / "handoff.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("artifact_manifest.json", "{}\n")
        archive.writestr("diff.patch", "")
        archive.writestr("final-branch.bundle", b"not-a-real-bundle")
        archive.writestr("git_status.txt", "clean\n")
        archive.writestr("pr_body.md", "")
        archive.writestr("review_threads.json", "{}\n")
        archive.writestr("task_metadata.json", "{}\n")
        archive.writestr("validation_evidence.json", "{}\n")
    sidecar = tmp_path / "handoff.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": "handoff-sidecar/0.1",
                "zip_filename": "handoff.zip",
                "zip_logical_path": f"{run_reviews}/handoff.zip",
                "zip_sha256": verifier.sha256_file(package),
                "artifact_manifest_sha256": "0" * 64,
                "branch": "feature/qrp-clean-rewrite",
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
                "zip_path": "E:/quant-workspace/private/handoff.zip",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "sidecar_safety_failed" in failures
