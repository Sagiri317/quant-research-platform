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
    _load_handoff_module("handoff_contract").MANDATORY_VALIDATION_ROWS
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


def _build_test_package(tmp_path: Path, *, verify: bool = False) -> tuple[Path, Path, Path, str, str]:
    builder = _load_handoff_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = _init_handoff_repo(repo)
    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=tmp_path / "out",
        task_name="handoff-test",
        base_ref="origin/main",
        mode="evidence",
        verify=verify,
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
    _write_minimal_project_files(repo)
    _git(repo, "init")
    _git(repo, "config", "user.email", "handoff-test@example.invalid")
    _git(repo, "config", "user.name", "Handoff Test")
    _git(repo, "checkout", "-b", TEST_BRANCH)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    (repo / "tests" / "negative_fixtures.py").write_text(
        "NEGATIVE_FIXTURE = 'review-safe aggregate fixture'\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add handoff fixture")
    final_head = _git(repo, "rev-parse", "HEAD").strip()
    return base, final_head


def _write_minimal_project_files(repo: Path) -> None:
    (repo / ".gitignore").write_text(
        "\n".join(
            [
                ".pytest_cache/",
                "__pycache__/",
                "*.pyc",
                "build/",
                "dist/",
                "*.egg-info/",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["setuptools>=80"]',
                'build-backend = "setuptools.build_meta"',
                "",
                "[project]",
                'name = "quant-research-platform"',
                'version = "0.0.0"',
                'requires-python = ">=3.13"',
                "",
                "[tool.setuptools.packages.find]",
                'where = ["src"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    code_map = repo / "docs" / "code-map" / "platform-flow.md"
    code_map.parent.mkdir(parents=True)
    code_map.write_text(
        "\n".join(
            [
                "# Platform Flow",
                "local research platform",
                "data -> feature -> signal/model -> evaluate -> evidence",
                ".agents tooling",
                ".run artifacts",
                "private data safety",
                "current contract",
                "current-contract-only",
                "no trading execution",
                "no production trading boundary",
                "",
            ]
        ),
        encoding="utf-8",
    )
    smoke_test = repo / "tests" / "agents" / "handoff" / "test_smoke.py"
    smoke_test.parent.mkdir(parents=True)
    smoke_test.write_text("def test_smoke() -> None:\n    assert True\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _validation_rows(status: str = "PASS") -> list[dict[str, str]]:
    return [{"name": name, "status": status} for name in MANDATORY_VALIDATION_ROWS]


def _validation_payload(
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-validation-summary",
        "rows": rows if rows is not None else _validation_rows("PASS"),
    }


def _package_smoke_receipt() -> dict[str, object]:
    return {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "build": {"status": "PASS", "artifact_kinds": ["wheel", "sdist"]},
        "install": {
            "status": "PASS",
            "target_kind": "temporary_no_deps_install",
            "no_deps": True,
        },
        "import": {"status": "PASS", "module": "quant_research_platform"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
        },
    }


def _check_receipt(
    command_id: str,
    *,
    status: str = "PASS",
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    handoff_contract = _load_handoff_module("handoff_contract")
    spec = handoff_contract.replay_command_spec(command_id)
    return {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": list(spec.command_argv),
        "command": spec.command,
        "status": status,
        "exit_code": 0 if status == "PASS" else 1,
        "counts": counts,
    }


def _check_receipts() -> dict[str, object]:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 1,
    }
    return {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [
            _check_receipt("profile-smoke", counts=counts),
            _check_receipt("code-map"),
        ],
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
