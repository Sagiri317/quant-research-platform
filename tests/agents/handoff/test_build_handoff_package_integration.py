from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path.cwd()
HANDOFF_ROOT = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
)
HANDOFF_SCRIPT = HANDOFF_ROOT / "scripts" / "handoff" / "build_handoff_package.py"
VERIFY_SCRIPT = HANDOFF_ROOT / "scripts" / "handoff" / "verify_handoff_package.py"
TEST_BRANCH = "feature/synthetic-handoff-review"


def test_build_handoff_package_evidence_only_zip_contract(tmp_path: Path) -> None:
    repo = tmp_path / "synthetic-repo"
    _init_synthetic_repo(repo)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "evidence.txt").write_text("final\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "final evidence change")
    final_head = _git(repo, "rev-parse", "HEAD").strip()

    out_dir = tmp_path / "out"
    task_path = tmp_path / "task-input.json"
    slug = "synthetic-handoff"
    run_handoffs = "." + "run/handoffs"
    task_path.write_text(
        json.dumps(
                {
                    "schema_version": "handoff-task-metadata/0.1",
                    "slug": slug,
                    "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
                "final_sha": final_head,
                    "pr_url": "https://github.com/Sagiri317/quant-research-platform/pull/1",
                    "zip_logical_path": f"{run_handoffs}/{slug}.zip",
                    "sidecar_logical_path": f"{run_handoffs}/{slug}.json",
                    "handoff_evidence": {
                        "generation": {"status": "FAIL", "final_head": "forged"},
                        "source-tree-scan": {
                            "status": "FAIL",
                            "scanned_head": "forged",
                        },
                        "diff" + ".patch": {"status": "PASS"},
                        "pr_body.md": {"status": "PASS"},
                        "review" + "_evidence": {"status": "PASS"},
                        "custom-note": {
                            "schema_version": "custom-note/0.1",
                            "status": "PASS",
                        },
                    },
                    "validation_evidence": {
                        "schema_version": "handoff-validation-evidence/0.1",
                        "checks": [
                        {
                            "name": "github_review_threads",
                            "status": "BLOCKED",
                            "evidence": "live GitHub access unavailable",
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(HANDOFF_SCRIPT),
            "--repo-root",
            str(repo),
            "--task",
            str(task_path),
            "--out-dir",
            str(out_dir),
            "--verify",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    package = out_dir / f"{slug}.zip"
    sidecar_path = out_dir / f"{slug}.json"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        f"{slug}.json",
        f"{slug}.zip",
    ]
    assert task_path.name not in [path.name for path in out_dir.iterdir()]

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        manifest_raw = archive.read("artifact_manifest.json")
        bundle_raw = archive.read("final-branch.bundle")
        manifest = json.loads(manifest_raw.decode("utf-8"))
        task_metadata = json.loads(archive.read("task_metadata.json").decode("utf-8"))
        validation_evidence = json.loads(
            archive.read("validation_evidence.json").decode("utf-8")
        )
        generation = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
        source_tree_scan = json.loads(
            archive.read("handoff_evidence/source-tree-scan.json").decode("utf-8")
        )

    required = {
        "final-branch.bundle",
        "git_status.txt",
        "task_metadata.json",
        "validation_evidence.json",
        "artifact_manifest.json",
    }
    assert required.issubset(names)
    assert "pr_body.md" not in names
    assert "review_threads.json" not in names
    assert "diff" + ".patch" not in names
    assert "handoff_evidence/diff" + ".patch.json" not in names
    assert "handoff_evidence/pr_body.md.json" not in names
    assert "handoff_evidence/review" + "_evidence.json" not in names
    assert "handoff_evidence/generation.json" in names
    assert "handoff_evidence/source-tree-scan.json" in names
    assert "handoff_evidence/custom-note.json" in names
    assert f"{slug}.json" not in names
    forbidden_prefixes = (
        "run_tools/",
        ".agents/",
        "src/",
        "tests/",
        "docs/",
        "templates/",
        "schemas/",
    )
    assert not any(
        name.startswith(forbidden_prefixes)
        for name in names
        if name != "final-branch.bundle"
    )
    assert {entry["path"] for entry in manifest["entries"]} | {
        "artifact_manifest.json"
    } == names
    assert "tool_source_repo_paths" in manifest
    assert sidecar["zip_sha256"] == hashlib.sha256(package.read_bytes()).hexdigest()
    assert sidecar["artifact_manifest_sha256"] == hashlib.sha256(
        manifest_raw
    ).hexdigest()
    assert task_metadata["final_head"] == final_head
    assert (
        task_metadata["pr_url"]
        == "https://github.com/Sagiri317/quant-research-platform/pull/1"
    )
    assert generation["mode"] == "evidence"
    assert generation["base"] == base
    assert generation["final_head"] == final_head
    assert generation["zip_filename"] == f"{slug}.zip"
    assert generation["zip_sha256"] == "external_sidecar"
    assert generation["bundle_sha256"] == hashlib.sha256(bundle_raw).hexdigest()
    assert generation["compare_evidence_path"] is None
    assert generation["source_tree_scan_path"] == (
        "handoff_evidence/source-tree-scan.json"
    )
    assert generation["validation_status"] == "BLOCKED"
    assert sidecar["zip_sha256"]
    assert source_tree_scan["status"] == "PASS"
    assert source_tree_scan["scanned_head"] == final_head
    assert source_tree_scan["expected_head"] == final_head
    assert set(source_tree_scan["scan_groups"]) == {
        "retired_terms",
        "active_test_literals",
    }
    assert source_tree_scan["scan_groups"]["retired_terms"]["finding_count"] == 0
    assert (
        source_tree_scan["scan_groups"]["active_test_literals"]["finding_count"]
        == 0
    )
    checks = {
        row["name"]: row
        for row in validation_evidence["checks"]
        if isinstance(row, dict) and "name" in row
    }
    assert checks["handoff_verify"]["status"] == "PASS"
    assert checks["handoff_verify"]["command"] == (
        "python .agents/skills/quant-research-platform-maintenance/scripts/"
        f"handoff/verify_handoff_package.py --zip {run_handoffs}/{slug}.zip "
        f"--json {run_handoffs}/{slug}.json"
    )
    assert checks["package_safety_check_zip"] == {
        "name": "package_safety_check_zip",
        "status": "PASS",
        "command": (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"package_safety_check.py {run_handoffs}/{slug}.zip"
        ),
        "evidence": "package_safety_check.py completed for final zip with PASS",
    }
    assert checks["package_safety_check_sidecar"] == {
        "name": "package_safety_check_sidecar",
        "status": "PASS",
        "command": (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"package_safety_check.py {run_handoffs}/{slug}.json"
        ),
        "evidence": "package_safety_check.py completed for sidecar JSON with PASS",
    }
    assert task_path.name not in names

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extract("final-branch.bundle", fresh)
    cloned = fresh / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            TEST_BRANCH,
            str(fresh / "final-branch.bundle"),
            str(cloned),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _git(cloned, "rev-parse", "HEAD").strip() == final_head
    assert _git(cloned, "show", "-s", "--format=%H", "HEAD").strip() == final_head
    assert (
        "synthetic"
        not in _git(cloned, "show", "-s", "--format=%s", "HEAD").strip().lower()
    )
    cloned_handoff = (
        cloned
        / ".agents"
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "handoff"
    )
    for command in (
        [sys.executable, str(cloned_handoff / "self_test.py")],
        [
            sys.executable,
            str(cloned_handoff / "verify_handoff_package.py"),
            "--zip",
            str(package),
            "--json",
            str(sidecar_path),
        ],
        [
            sys.executable,
            str(cloned_handoff / "handoff_safety.py"),
            str(package),
            "--json",
            str(sidecar_path),
        ],
    ):
        subprocess.run(
            command,
            cwd=cloned,
            check=True,
            capture_output=True,
            text=True,
        )

    verifier = _load_verify_module()
    assert verifier.verify_handoff_package(package, sidecar_path) == []


def _init_synthetic_repo(repo: Path) -> None:
    repo.mkdir()
    support_root = (
        repo / ".agents" / "skills" / "quant-research-platform-maintenance"
    )
    shutil.copytree(
        HANDOFF_ROOT / "scripts" / "handoff",
        support_root / "scripts" / "handoff",
    )
    shutil.copytree(
        HANDOFF_ROOT / "templates" / "handoff",
        support_root / "templates" / "handoff",
    )
    shutil.copytree(
        HANDOFF_ROOT / "schemas" / "handoff",
        support_root / "schemas" / "handoff",
    )
    runtime_root = repo / "src" / "quant_research_platform" / "runtime"
    runtime_root.mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "src" / "quant_research_platform" / "runtime" / "privacy.py",
        runtime_root / "privacy.py",
    )
    (repo / "evidence.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "handoff-integration@example.invalid")
    _git(repo, "config", "user.name", "Handoff Integration Test")
    _git(repo, "checkout", "-b", TEST_BRANCH)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _load_verify_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_handoff_package",
        VERIFY_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
