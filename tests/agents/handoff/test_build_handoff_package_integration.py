from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path.cwd()
HANDOFF_ROOT = REPO_ROOT / ".agents" / "skills" / "quant-research-platform-maintenance"
HANDOFF_SCRIPT = HANDOFF_ROOT / "scripts" / "handoff" / "build_handoff_package.py"
VERIFY_SCRIPT = HANDOFF_ROOT / "scripts" / "handoff" / "verify_handoff_package.py"
TEST_BRANCH = "feature/synthetic-handoff-review"
BUILDER_MODULE = None
GOVERNANCE_MODULE = None


def _load_builder_module():
    global BUILDER_MODULE
    if BUILDER_MODULE is not None:
        return BUILDER_MODULE
    spec = importlib.util.spec_from_file_location(
        "build_handoff_package",
        HANDOFF_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    BUILDER_MODULE = module
    return module


def _load_governance_module():
    global GOVERNANCE_MODULE
    if GOVERNANCE_MODULE is not None:
        return GOVERNANCE_MODULE
    path = HANDOFF_ROOT / "scripts" / "governance_contract.py"
    spec = importlib.util.spec_from_file_location("governance_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["governance_contract"] = module
    spec.loader.exec_module(module)
    GOVERNANCE_MODULE = module
    return module


def test_build_handoff_package_evidence_only_zip_contract(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    _init_synthetic_repo(repo)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "evidence.txt").write_text("final\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "final evidence change")
    final_head = _git(repo, "rev-parse", "HEAD").strip()

    out_dir = tmp_path / "o"
    task_path = tmp_path / "t.json"
    slug = "synthetic-handoff"
    run_handoffs = ".run/handoffs"
    task_path.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": slug,
                "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
                "handoff_package_path": f"{run_handoffs}/{slug}.zip",
                "handoff_sidecar_path": f"{run_handoffs}/{slug}.json",
                "handoff_evidence": {
                    **_formal_handoff_evidence(),
                    "custom-note": {
                        "schema_id": "qrp-custom-note",
                        "status": "PASS",
                    },
                },
                "handoff_markdown_evidence": {
                    "smoke-semantic-parity-compare": (
                        "# Smoke semantic parity compare\n\nstatus: PASS\n"
                    ),
                },
                "validation_evidence": {
                    "schema_id": "qrp-handoff-validation-evidence",
                    "checks": _validation_checks("PASS"),
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
        assert (
            validation_evidence["handoff_package_path"] == f"{run_handoffs}/{slug}.zip"
        )
        assert (
            validation_evidence["handoff_sidecar_path"] == f"{run_handoffs}/{slug}.json"
        )
        generation = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
        source_tree_scan = json.loads(
            archive.read("handoff_evidence/source-tree-scan.json").decode("utf-8")
        )
        bundle_reconstruction = json.loads(
            archive.read("handoff_evidence/bundle-reconstruction.json").decode("utf-8")
        )
        markdown_evidence = archive.read(
            "handoff_evidence/smoke-semantic-parity-compare.md"
        ).decode("utf-8")

    required = {
        "final-branch.bundle",
        "git_status.txt",
        "task_metadata.json",
        "validation_evidence.json",
        "artifact_manifest.json",
    }
    assert required.issubset(names)
    assert "handoff_evidence/generation.json" in names
    assert "handoff_evidence/source-tree-scan.json" in names
    assert "handoff_evidence/bundle-reconstruction.json" in names
    assert "handoff_evidence/package-smoke.json" in names
    assert "handoff_evidence/hard-scan-classification.json" in names
    assert "handoff_evidence/custom-note.json" in names
    assert "handoff_evidence/smoke-semantic-parity-compare.md" in names
    assert markdown_evidence == "# Smoke semantic parity compare\n\nstatus: PASS\n"
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
    assert (
        sidecar["artifact_manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    )
    assert task_metadata["final_head"] == final_head
    assert task_metadata["handoff_package_path"] == f"{run_handoffs}/{slug}.zip"
    assert task_metadata["handoff_sidecar_path"] == f"{run_handoffs}/{slug}.json"
    assert "pr_url" not in task_metadata
    assert task_metadata["mode"] == "evidence"
    assert generation["mode"] == "evidence"
    assert generation["base"] == base
    assert generation["final_head"] == final_head
    assert generation["zip_filename"] == f"{slug}.zip"
    assert generation["zip_sha256_source"] == "sidecar"
    assert generation["bundle_sha256"] == hashlib.sha256(bundle_raw).hexdigest()
    assert "compare_evidence_path" not in generation
    assert generation["source_tree_scan_path"] == (
        "handoff_evidence/source-tree-scan.json"
    )
    assert generation["validation_status"] == "PASS"
    assert sidecar["zip_sha256"]
    assert source_tree_scan["status"] == "PASS"
    assert source_tree_scan["scanned_head"] == final_head
    assert source_tree_scan["expected_head"] == final_head
    assert bundle_reconstruction["schema_id"] == "qrp-bundle-reconstruction-evidence"
    assert bundle_reconstruction["bundle_path"] == "final-branch.bundle"
    assert bundle_reconstruction["branch"] == TEST_BRANCH
    assert bundle_reconstruction["expected_head"] == final_head
    assert bundle_reconstruction["plain_clone_required"] is False
    assert bundle_reconstruction["bundle_verify"]["command"] == (
        "git bundle verify final-branch.bundle"
    )
    assert bundle_reconstruction["bundle_verify"]["status"] == "PASS"
    assert bundle_reconstruction["bundle_list_heads"]["command"] == (
        "git bundle list-heads final-branch.bundle"
    )
    assert bundle_reconstruction["bundle_list_heads"]["status"] == "PASS"
    assert bundle_reconstruction["clone_verification"]["command"] == (
        f"git clone -b {TEST_BRANCH} final-branch.bundle review-repo"
    )
    assert bundle_reconstruction["clone_verification"]["status"] == "PASS"
    assert bundle_reconstruction["clone_verification"]["cloned_head"] == final_head
    assert set(source_tree_scan["scan_groups"]) == {
        "retired_text",
        "retired_governance_paths",
        "retired_path_names",
        "python_symbols",
        "synthesized_strings",
        "active_test_literals",
        "structured_keys",
        "governance_truth",
        "sensitive_negative_literals",
        "allowance_contract",
    }
    assert source_tree_scan["scan_groups"]["retired_text"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["retired_text"]["finding_count"] == 0
    assert (
        source_tree_scan["scan_groups"]["retired_governance_paths"]["status"] == "PASS"
    )
    assert (
        source_tree_scan["scan_groups"]["retired_governance_paths"]["finding_count"]
        == 0
    )
    assert source_tree_scan["scan_groups"]["retired_path_names"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["retired_path_names"]["finding_count"] == 0
    assert source_tree_scan["scan_groups"]["python_symbols"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["python_symbols"]["finding_count"] == 0
    assert source_tree_scan["scan_groups"]["synthesized_strings"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["synthesized_strings"]["finding_count"] == 0
    assert source_tree_scan["scan_groups"]["active_test_literals"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["active_test_literals"]["finding_count"] == 0
    assert source_tree_scan["scan_groups"]["structured_keys"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["structured_keys"]["finding_count"] == 0
    assert source_tree_scan["scan_groups"]["governance_truth"]["status"] == "PASS"
    assert source_tree_scan["scan_groups"]["governance_truth"]["finding_count"] == 0
    assert (
        source_tree_scan["scan_groups"]["sensitive_negative_literals"]["status"]
        == "PASS"
    )
    assert (
        source_tree_scan["scan_groups"]["sensitive_negative_literals"]["finding_count"]
        == 0
    )
    checks = {
        row["name"]: row
        for row in validation_evidence["checks"]
        if isinstance(row, dict) and "name" in row
    }
    for check_name in _mandatory_validation_checks():
        assert checks[check_name]["status"] == "PASS"
    assert checks["handoff_verify"]["status"] == "PASS"
    assert checks["handoff_safety_scan"] == {
        "name": "handoff_safety_scan",
        "status": "PASS",
        "evidence": "verify_handoff_package.py safety scan completed with PASS",
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


def test_build_handoff_package_generates_missing_validation_evidence_mandatory_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "r"
    _init_synthetic_repo(repo)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "evidence.txt").write_text("final\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "final evidence change")
    final_head = _git(repo, "rev-parse", "HEAD").strip()

    out_dir = tmp_path / "o"
    task_path = tmp_path / "t.json"
    slug = "synthetic-missing-validation"
    run_handoffs = ".run/handoffs"
    task_path.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": slug,
                "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
                "handoff_package_path": f"{run_handoffs}/{slug}.zip",
                "handoff_sidecar_path": f"{run_handoffs}/{slug}.json",
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
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    package = out_dir / f"{slug}.zip"
    sidecar_path = out_dir / f"{slug}.json"
    with zipfile.ZipFile(package) as archive:
        validation_evidence = json.loads(
            archive.read("validation_evidence.json").decode("utf-8")
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checks = {row["name"]: row for row in validation_evidence["checks"]}

    assert set(checks) == set(_mandatory_validation_checks())
    assert "validation_evidence" not in checks
    assert {row["status"] for row in checks.values()} == {"NOT_RUN"}
    assert all(
        "task validation_evidence missing; generated mandatory NOT_RUN row"
        in row["evidence"]
        for row in checks.values()
    )
    assert sidecar["status_summary"]["validation"] == "NOT_RUN"
    failures = _load_verify_module().verify_handoff_package(package, sidecar_path)
    assert not any(
        failure.startswith("validation_evidence_mandatory_check_missing")
        for failure in failures
    )


def test_build_handoff_package_rejects_reserved_handoff_evidence_path(
    tmp_path: Path,
) -> None:
    governance = _load_governance_module()
    reserved_name = sorted(governance.RESERVED_HANDOFF_EVIDENCE_STEMS)[0]

    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {reserved_name: {"schema_id": "custom-note/0.1", "status": "PASS"}},
    )

    assert result.returncode != 0
    assert "handoff_evidence_forbidden_path" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_rejects_internal_handoff_evidence_path(
    tmp_path: Path,
) -> None:
    governance = _load_governance_module()
    internal_name = sorted(governance.BUILDER_OWNED_HANDOFF_EVIDENCE)[0]

    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {internal_name: {"schema_id": "custom-note/0.1", "status": "PASS"}},
    )

    assert result.returncode != 0
    assert "handoff_evidence_forbidden_path" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_rejects_non_object_handoff_evidence(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(tmp_path, "bad")

    assert result.returncode != 0
    assert "handoff_evidence_must_be_object" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_rejects_non_object_handoff_evidence_entry(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {"custom-note": "bad"},
    )

    assert result.returncode != 0
    assert "handoff_evidence_entry_must_be_object" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_rejects_one_forbidden_custom_evidence_key(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {
            "custom-note": {
                "schema_id": "qrp-custom-note",
                "status": "PASS",
                "details": {"check": "forged"},
            }
        },
    )

    assert result.returncode != 0
    assert "handoff_evidence_forbidden_field" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_allows_empty_advisory_without_review_evidence(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        _formal_handoff_evidence(),
        verify=True,
    )

    assert result.returncode == 0
    sidecar = json.loads((out_dir / "synthetic-handoff-negative.json").read_text())
    assert sidecar["advisory_reviews"] == {}


def test_build_handoff_package_allows_review_evidence_with_advisory(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {
            **_formal_handoff_evidence(),
            "subagent-review-reconciliation": _review_reconciliation_payload(),
        },
        validation_checks=_review_validation_checks("PASS"),
        verify=True,
    )

    assert result.returncode == 0
    sidecar = json.loads((out_dir / "synthetic-handoff-negative.json").read_text())
    assert sidecar["advisory_reviews"] == {"review_subagent": "PASS"}


def test_build_handoff_package_rejects_review_evidence_without_advisory(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {"subagent-review-reconciliation": _review_reconciliation_payload()},
    )

    assert result.returncode != 0
    assert "advisory_reviews_missing_for_review_evidence" in result.stderr
    assert not list(out_dir.glob("*"))


def test_build_handoff_package_rejects_advisory_without_review_evidence(
    tmp_path: Path,
) -> None:
    result, out_dir = _run_builder_with_handoff_evidence(
        tmp_path,
        {},
        validation_checks=_review_validation_checks("PASS"),
    )

    assert result.returncode != 0
    assert "advisory_reviews_without_review_evidence" in result.stderr
    assert not list(out_dir.glob("*"))


def _init_synthetic_repo(repo: Path) -> None:
    repo.mkdir()
    support_root = repo / ".agents" / "skills" / "quant-research-platform-maintenance"
    shutil.copytree(
        HANDOFF_ROOT / "scripts" / "handoff",
        support_root / "scripts" / "handoff",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    scripts_root = support_root / "scripts"
    shutil.copyfile(
        HANDOFF_ROOT / "scripts" / "governance_contract.py",
        scripts_root / "governance_contract.py",
    )
    shutil.copytree(
        HANDOFF_ROOT / "templates" / "handoff",
        support_root / "templates" / "handoff",
    )
    shutil.copytree(
        HANDOFF_ROOT / "schemas" / "handoff",
        support_root / "schemas" / "handoff",
    )
    package_root = repo / "src" / "quant_research_platform"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    policy_root = package_root / "policy"
    policy_root.mkdir(parents=True)
    (policy_root / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(
        REPO_ROOT / "src" / "quant_research_platform" / "policy" / "share_safety.py",
        policy_root / "share_safety.py",
    )
    shutil.copyfile(
        REPO_ROOT / "src" / "quant_research_platform" / "policy" / "check_status.py",
        policy_root / "check_status.py",
    )
    for relative in (
        "contracts/columns.py",
        "contracts/data.py",
        "contracts/ids.py",
        "markets/implementations/a_share/ids.py",
        "sources/csmar/ids.py",
    ):
        target = repo / "src" / "quant_research_platform" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            REPO_ROOT / "src" / "quant_research_platform" / relative,
            target,
        )
    (repo / "evidence.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "handoff-integration@example.invalid")
    _git(repo, "config", "user.name", "Handoff Integration Test")
    _git(repo, "checkout", "-b", TEST_BRANCH)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")


def _run_builder_with_handoff_evidence(
    tmp_path: Path,
    handoff_evidence: object,
    *,
    validation_checks: list[dict[str, str]] | None = None,
    verify: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    repo = tmp_path / "r"
    _init_synthetic_repo(repo)
    base = _git(repo, "rev-parse", "HEAD").strip()
    (repo / "evidence.txt").write_text("final\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "final evidence change")
    final_head = _git(repo, "rev-parse", "HEAD").strip()

    out_dir = tmp_path / "o"
    task_path = tmp_path / "t.json"
    slug = "synthetic-handoff-negative"
    run_handoffs = ".run/handoffs"
    task_path.write_text(
        json.dumps(
            {
                "schema_id": "qrp-handoff-task-metadata",
                "mode": "evidence",
                "slug": slug,
                "branch": TEST_BRANCH,
                "base": base,
                "final_head": final_head,
                "handoff_package_path": f"{run_handoffs}/{slug}.zip",
                "handoff_sidecar_path": f"{run_handoffs}/{slug}.json",
                "handoff_evidence": handoff_evidence,
                "validation_evidence": {
                    "schema_id": "qrp-handoff-validation-evidence",
                    "checks": (
                        validation_checks
                        if validation_checks is not None
                        else _validation_checks("PASS")
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(HANDOFF_SCRIPT),
        "--repo-root",
        str(repo),
        "--task",
        str(task_path),
        "--out-dir",
        str(out_dir),
    ]
    if verify:
        command.append("--verify")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result, out_dir


def _validation_checks(status: str) -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "status": status,
            "evidence": f"{name} {status}",
        }
        for name in _mandatory_validation_checks()
    ]


def _review_validation_checks(status: str) -> list[dict[str, str]]:
    return [
        *_validation_checks(status),
        {
            "name": "review_subagent",
            "status": status,
            "evidence": f"review_subagent {status}",
        },
    ]


def _review_reconciliation_payload() -> dict[str, object]:
    return {
        "schema_id": "qrp-subagent-review-reconciliation",
        "status": "PASS",
        "rows": [
            {
                "finding_id": "review-1",
                "classification": "NO_FINDINGS",
                "action": "none",
            }
        ],
    }


def _formal_handoff_evidence() -> dict[str, dict[str, object]]:
    return {
        "package-smoke": _package_smoke_evidence(),
        "hard-scan-classification": _hard_scan_classification_evidence(),
    }


def _package_smoke_evidence() -> dict[str, object]:
    return {
        "schema_id": "qrp-package-smoke-evidence",
        "status": "PASS",
        "build_command": "python -m build --sdist --wheel",
        "wheel_filename": "quant_research_platform-0.0.0-py3-none-any.whl",
        "wheel_sha256": "0" * 64,
        "sdist_filename": "quant_research_platform-0.0.0.tar.gz",
        "sdist_sha256": "1" * 64,
        "install_target": "temporary isolated target; not included in handoff",
        "import_command": "python -c import quant_research_platform",
        "import_status": "PASS",
        "content_scan": {
            "no_run_dir": True,
            "no_parquet_csv_pickle_feather_arrow": True,
            "no_row_level_data": True,
            "no_local_absolute_paths": True,
        },
    }


def _hard_scan_classification_evidence() -> dict[str, object]:
    return {
        "schema_id": "qrp-hard-scan-classification-evidence",
        "scan_name": "handoff-fixture-hard-scan",
        "command": "rg -n boundary-scan src tests",
        "status": "PASS",
        "raw_findings": [
            {
                "path": "tests/negative_fixture.py",
                "line": 1,
                "matched_text": "synthetic negative fixture",
                "classification": "negative_test_fixture",
                "reason": "synthetic fixture used only to exercise verifier shape",
            }
        ],
    }


def _mandatory_validation_checks() -> tuple[str, ...]:
    return tuple(sorted(_load_governance_module().MANDATORY_VALIDATION_CHECKS))


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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
