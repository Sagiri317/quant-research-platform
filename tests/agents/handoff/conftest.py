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

HANDOFF_TOOL_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)
HANDOFF_SKILL_ROOT = HANDOFF_TOOL_ROOT.parent.parent
TEST_BRANCH = "feature/synthetic-handoff-review"


def _load_handoff_module(name: str):
    script = HANDOFF_TOOL_ROOT / f"{name}.py"
    if name == "governance_contract":
        script = HANDOFF_TOOL_ROOT.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MANDATORY_VALIDATION_CHECKS = tuple(
    sorted(_load_handoff_module("governance_contract").MANDATORY_VALIDATION_CHECKS)
)


@pytest.fixture(scope="session")
def valid_handoff_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("valid-handoff-package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(root)
    return package, sidecar


def _copy_valid_package(source: tuple[Path, Path], target: Path) -> tuple[Path, Path]:
    source_package, source_sidecar = source
    package = target / source_package.name
    sidecar = target / source_sidecar.name
    shutil.copyfile(source_package, package)
    shutil.copyfile(source_sidecar, sidecar)
    return package, sidecar


def _build_test_package(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    out_dir = tmp_path / "out"
    receipts_dir = tmp_path / "receipts"
    _write_receipts(
        receipts_dir,
        repo=repo,
        final_head=final_head,
        include_advisory=True,
    )
    task = {
        "slug": "handoff-test",
        "mode": "evidence",
        "branch": TEST_BRANCH,
        "base": base,
        "final_head": final_head,
    }
    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=out_dir,
        receipts_dir=receipts_dir,
        task=task,
        verify=False,
    )
    assert result["zip_path"]
    return repo, Path(result["zip_path"]), Path(result["json_path"]), base, final_head


def _init_handoff_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir()
    support_root = repo / ".agents" / "skills" / "quant-research-platform-maintenance"
    shutil.copytree(
        HANDOFF_SKILL_ROOT / "scripts",
        support_root / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        HANDOFF_SKILL_ROOT / "templates",
        support_root / "templates",
    )
    shutil.copytree(
        HANDOFF_SKILL_ROOT / "schemas",
        support_root / "schemas",
    )
    package_root = repo / "src" / "quant_research_platform"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    policy_root = package_root / "policy"
    policy_root.mkdir()
    (policy_root / "__init__.py").write_text("", encoding="utf-8")
    for relative in ("check_status.py", "share_safety.py"):
        shutil.copyfile(
            Path("src") / "quant_research_platform" / "policy" / relative,
            policy_root / relative,
        )
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


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _validation_checks(status: str = "PASS") -> list[dict[str, str]]:
    return [
        {
            "name": name,
            "status": status,
            "evidence": f"{name} {status}",
        }
        for name in MANDATORY_VALIDATION_CHECKS
    ]


def _review_validation_checks(status: str = "PASS") -> list[dict[str, str]]:
    return [
        *_validation_checks(status),
        {
            "name": "review_subagent",
            "status": status,
            "evidence": f"review_subagent {status}",
        },
    ]


def _validation_payload(
    checks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-validation-summary",
        "checks": checks if checks is not None else _validation_checks("PASS"),
    }


def _package_smoke_evidence() -> dict[str, object]:
    return {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "wheel": {
            "filename": "quant_research_platform-0.0.0-py3-none-any.whl",
            "sha256": "0" * 64,
        },
        "sdist": {
            "filename": "quant_research_platform-0.0.0.tar.gz",
            "sha256": "1" * 64,
        },
        "install": {
            "target_kind": "temporary isolated target",
            "no_deps": True,
            "status": "PASS",
        },
        "import": {"status": "PASS"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
            "examples": [],
        },
    }


def _hard_scan_classification_evidence() -> dict[str, object]:
    return {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS",
        "scanned_head": "1" * 40,
        "scanner_source_sha256": "2" * 64,
        "scan_runs": [
            {
                "scan_id": "allowance_contract",
                "rule_set_id": "source-tree:allowance_contract",
                "match_count": 0,
                "actual_violation_count": 0,
                "deferred_count": 0,
                "negative_fixture_count": 0,
                "false_positive_count": 0,
                "result_sha256": "3" * 64,
                "replayable_from_bundle": True,
            }
        ],
        "classified_exceptions": [],
    }


def _test_checks_evidence() -> dict[str, object]:
    return {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [
            {
                "schema_id": "qrp-check-receipt",
                "command_id": "handoff-tests",
                "command": "pytest tests/agents/handoff -q",
                "scope": "handoff verifier tests",
                "status": "PASS",
                "exit_code": 0,
                "counts": {
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "deselected": 0,
                    "errors": 0,
                    "xfailed": 0,
                    "xpassed": 0,
                    "collected": 1,
                },
                "duration_seconds": 1.0,
                "stdout_sha256": "4" * 64,
                "stderr_sha256": "5" * 64,
                "output_summary": "1 passed",
                "failure_excerpt": "",
            }
        ],
    }


def _private_smoke_status() -> dict[str, object]:
    return {
        "schema_id": "qrp-private-smoke-status",
        "status": "NOT_RUN",
        "command_id": "private-smoke",
        "exit_code": None,
        "run_level_overall": "NOT_RUN",
        "private_smoke_run_level": "NOT_RUN",
        "private_smoke_child_checks": "NOT_RUN",
        "compare": "NOT_RUN",
        "full_horizon": "NOT_RUN",
        "child_statuses": {},
        "run_json_sha256": "",
        "row_level_data_included": False,
        "not_run_reason": "private smoke not run in test fixture",
    }


def _advisory_review_receipt() -> dict[str, object]:
    return {
        "schema_id": "qrp-advisory-review-receipt",
        "status": "NOT_RUN",
        "review_lanes": [],
        "finding_count": 0,
        "true_issue_count": 0,
        "false_positive_count": 0,
        "re_review_status": "NOT_RUN",
        "findings": [],
    }


def _write_receipts(
    receipts_dir: Path,
    *,
    repo: Path,
    final_head: str,
    include_advisory: bool = False,
) -> None:
    governance = _load_handoff_module("governance_contract")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "package-smoke-receipt.json": _package_smoke_evidence(),
        "hard-scan-receipt.json": governance.hard_scan_receipt_from_clone(
            repo,
            final_head=final_head,
        ),
        "check-receipts.json": _test_checks_evidence(),
        "private-smoke-status.json": _private_smoke_status(),
    }
    if include_advisory:
        payloads["advisory-review-receipt.json"] = _advisory_review_receipt()
    for name, payload in payloads.items():
        (receipts_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _rewrite_package_member(package: Path, member: str, data: bytes) -> None:
    temp = package.with_suffix(".rewrite.zip")
    with zipfile.ZipFile(package, "r") as src, zipfile.ZipFile(temp, "w") as dst:
        replaced = False
        for info in src.infolist():
            if info.filename == member:
                payload = data
                replaced = True
            else:
                payload = src.read(info.filename)
            dst.writestr(info, payload)
        if not replaced:
            dst.writestr(member, data)
    temp.replace(package)


def _refresh_manifest_and_sidecar(package: Path, sidecar: Path) -> None:
    with zipfile.ZipFile(package, "r") as archive:
        names = sorted(
            name for name in archive.namelist() if name != "artifact_manifest.json"
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
        "schema_id": "qrp-handoff-artifact-manifest",
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


def _replace_validation_summary(
    package: Path,
    sidecar: Path,
    payload: dict[str, object],
) -> None:
    _rewrite_package_member(
        package,
        "handoff_evidence/validation-summary.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    status_map = {}
    for row in payload.get("checks", []):
        if not isinstance(row, dict) or "name" not in row or "status" not in row:
            continue
        status = row["status"]
        if isinstance(status, str):
            status_map[str(row["name"])] = status
    mandatory_statuses = [
        status_map[name] for name in MANDATORY_VALIDATION_CHECKS if name in status_map
    ]
    if "FAIL" in mandatory_statuses:
        validation_status = "FAIL"
    elif "BLOCKED" in mandatory_statuses:
        validation_status = "BLOCKED"
    elif "NOT_RUN" in mandatory_statuses:
        validation_status = "NOT_RUN"
    elif len(mandatory_statuses) == len(MANDATORY_VALIDATION_CHECKS):
        validation_status = "PASS"
    else:
        validation_status = "FAIL"
    existing_summary = sidecar_payload.get("status_summary")
    if not isinstance(existing_summary, dict):
        existing_summary = {}
    sidecar_payload["status_summary"] = {
        "validation": validation_status,
        "tests": status_map.get("tests", "NOT_RUN"),
        "package_smoke": status_map.get("package_smoke", "NOT_RUN"),
        "profile_smoke": status_map.get("profile_smoke", "NOT_RUN"),
        "private_smoke_run_level": status_map.get(
            "private_smoke_run_level",
            "NOT_RUN",
        ),
        "private_smoke_child_checks": status_map.get(
            "private_smoke_child_checks",
            "NOT_RUN",
        ),
        "baseline_compare": status_map.get("baseline_compare", "NOT_RUN"),
        "governance": status_map.get("governance", "NOT_RUN"),
        "code_map": status_map.get("code_map", "NOT_RUN"),
        "handoff_verify": existing_summary.get("handoff_verify", "NOT_RUN"),
        "handoff_safety": existing_summary.get("handoff_safety", "NOT_RUN"),
        "blocked_count": sum(
            1 for status in mandatory_statuses if status == "BLOCKED"
        ),
        "fail_count": sum(1 for status in mandatory_statuses if status == "FAIL"),
    }
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _replace_generation_evidence(
    package: Path,
    sidecar: Path,
    payload: dict[str, object],
) -> None:
    _rewrite_package_member(
        package,
        "handoff_evidence/generation.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)


def _generation_evidence(package: Path) -> dict[str, object]:
    with zipfile.ZipFile(package) as archive:
        payload = json.loads(
            archive.read("handoff_evidence/generation.json").decode("utf-8")
        )
    assert isinstance(payload, dict)
    return payload
