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
    task = {
        "slug": "handoff-test",
        "mode": "evidence",
        "branch": TEST_BRANCH,
        "base": base,
        "final_head": final_head,
        "validation_evidence": {
            "schema_id": "qrp-handoff-validation-evidence",
            "checks": _validation_checks("NOT_RUN"),
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


def _validation_payload(
    checks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-validation-evidence",
        "checks": checks if checks is not None else _validation_checks("PASS"),
    }


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


def _replace_validation_evidence(
    package: Path,
    sidecar: Path,
    payload: dict[str, object],
) -> None:
    _rewrite_package_member(
        package,
        "validation_evidence.json",
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
    sidecar_payload["status_summary"] = {
        "validation": validation_status,
        "handoff_verify": status_map.get("handoff_verify", "NOT_RUN"),
        "handoff_safety": status_map.get("handoff_safety_scan", "NOT_RUN"),
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
