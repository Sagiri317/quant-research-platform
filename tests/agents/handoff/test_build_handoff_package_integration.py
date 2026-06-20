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
    run_reviews = "." + "run/reviews"
    task_path.write_text(
        json.dumps(
            {
                "schema_version": "handoff-task-metadata/0.1",
                "slug": slug,
                "branch": "feature/qrp-clean-rewrite",
                "base": base,
                "final_head": final_head,
                "final_sha": final_head,
                "zip_logical_path": f"{run_reviews}/{slug}.zip",
                "sidecar_logical_path": f"{run_reviews}/{slug}.json",
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
        manifest = json.loads(manifest_raw.decode("utf-8"))
        task_metadata = json.loads(archive.read("task_metadata.json").decode("utf-8"))
        review_threads = json.loads(
            archive.read("review_threads.json").decode("utf-8")
        )
        pr_body = archive.read("pr_body.md").decode("utf-8")
        packaged_diff = archive.read("diff.patch")

    required = {
        "final-branch.bundle",
        "diff.patch",
        "git_status.txt",
        "pr_body.md",
        "task_metadata.json",
        "validation_evidence.json",
        "review_threads.json",
        "artifact_manifest.json",
    }
    assert required.issubset(names)
    assert any(name.startswith("review_evidence/") for name in names)
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
    assert task_path.name not in names

    raw_diff = _git(repo, "diff", "--binary", f"{base}..{final_head}").encode()
    assert packaged_diff == raw_diff
    assert review_threads["status"] == "BLOCKED"
    assert review_threads["counts_available"] is False
    assert review_threads["total_count"] is None
    assert review_threads["unresolved_count"] is None
    assert "zip sidecar | `NOT_RUN`" not in pr_body
    assert f"{run_reviews}/{slug}.zip" in pr_body
    assert f"{run_reviews}/{slug}.json" in pr_body

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with zipfile.ZipFile(package) as archive:
        archive.extract("final-branch.bundle", fresh)
        (fresh / "diff.patch").write_bytes(archive.read("diff.patch"))
    cloned = fresh / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "feature/qrp-clean-rewrite",
            str(fresh / "final-branch.bundle"),
            str(cloned),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert _git(cloned, "rev-parse", "HEAD").strip() == final_head
    _git(cloned, "checkout", "--detach", base)
    subprocess.run(
        ["git", "apply", "--check", str(fresh / "diff.patch")],
        cwd=cloned,
        check=True,
        capture_output=True,
        text=True,
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
    (support_root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        HANDOFF_ROOT / "scripts" / "render_pr_body_from_validation_evidence.py",
        support_root / "scripts" / "render_pr_body_from_validation_evidence.py",
    )
    shutil.copytree(
        HANDOFF_ROOT / "templates" / "handoff",
        support_root / "templates" / "handoff",
    )
    shutil.copytree(
        HANDOFF_ROOT / "schemas" / "handoff",
        support_root / "schemas" / "handoff",
    )
    (repo / "evidence.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "handoff-integration@example.invalid")
    _git(repo, "config", "user.name", "Handoff Integration Test")
    _git(repo, "checkout", "-b", "feature/qrp-clean-rewrite")
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
