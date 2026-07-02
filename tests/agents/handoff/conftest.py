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


MANDATORY_VALIDATION_ROWS = tuple(
    sorted(_load_handoff_module("handoff_contract").MANDATORY_VALIDATION_ROWS)
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
    _write_receipts(receipts_dir, repo=repo, final_head=final_head)
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
        verify=True,
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
    (repo / "tests" / "negative_fixtures.py").parent.mkdir()
    (repo / "tests" / "negative_fixtures.py").write_text(
        "NEGATIVE_FIXTURE = 'review-safe aggregate fixture'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add handoff fixture")
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
        {"name": name, "status": status, "evidence": f"{name} {status}"}
        for name in MANDATORY_VALIDATION_ROWS
    ]


def _validation_payload(
    checks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-validation-summary",
        "checks": checks if checks is not None else _validation_checks("PASS"),
    }


def _package_smoke_receipt() -> dict[str, object]:
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


def _check_receipt(
    command_id: str,
    *,
    status: str = "PASS",
    counts: dict[str, int] | None = None,
    not_run_reason: str | None = None,
    blocked_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": f"{command_id} scope",
        "command_argv": ["python", "-m", command_id],
        "command": f"python -m {command_id}",
        "status": status,
        "exit_code": 0 if status == "PASS" else (1 if status == "FAIL" else None),
        "duration_seconds": 1.0,
        "counts": counts,
        "stdout_sha256": "4" * 64,
        "stderr_sha256": "5" * 64,
        "output_summary": f"{command_id} {status}",
        "failure_excerpt": "" if status == "PASS" else f"{command_id} {status}",
    }
    if not_run_reason:
        payload["not_run_reason"] = not_run_reason
    if blocked_reason:
        payload["blocked_reason"] = blocked_reason
    return payload


def _check_receipts() -> dict[str, object]:
    return {
        "schema_id": "qrp-check-receipts",
        "status": "NOT_RUN",
        "checks": [
            _check_receipt(
                "profile-smoke",
                counts={
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "deselected": 0,
                    "errors": 0,
                    "xfailed": 0,
                    "xpassed": 0,
                    "collected": 1,
                },
            ),
            _check_receipt(
                "private-validation",
                status="NOT_RUN",
                not_run_reason="private validation not run in test fixture",
            ),
            _check_receipt(
                "baseline-compare",
                status="NOT_RUN",
                not_run_reason="baseline compare not run in test fixture",
            ),
            _check_receipt("code-map"),
        ],
    }


def _advisory_review_receipt() -> dict[str, object]:
    return {
        "schema_id": "qrp-advisory-review-receipt",
        "status": "PASS",
        "review_lanes": [],
        "finding_count": 0,
        "true_issue_count": 0,
        "false_positive_count": 0,
        "re_review_status": "PASS",
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
        "package-smoke-receipt.json": _package_smoke_receipt(),
        "hard-scan-receipt.json": governance.hard_scan_receipt_from_clone(
            repo,
            final_head=final_head,
        ),
        "check-receipts.json": _check_receipts(),
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
    payload["artifact_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    payload["zip_sha256"] = hashlib.sha256(package.read_bytes()).hexdigest()
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
    status_map = {
        str(row["name"]): str(row["status"])
        for row in payload.get("checks", [])
        if isinstance(row, dict) and "name" in row and "status" in row
    }
    handoff_contract = _load_handoff_module("handoff_contract")
    existing = sidecar_payload.get("status_summary")
    if not isinstance(existing, dict):
        existing = {}
    sidecar_payload["status_summary"] = handoff_contract.build_sidecar_status_summary(
        status_map,
        handoff_verify=str(existing.get("handoff_verify", "PASS")),
        share_safety_scan=str(existing.get("share_safety_scan", "NOT_RUN")),
    )
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_manifest_and_sidecar(package, sidecar)
