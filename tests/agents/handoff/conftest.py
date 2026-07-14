from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.helpers.maintenance_script_loader import load_handoff_script_module
from tests.helpers.root_config_contract import (
    render_minimal_pyproject_from_root_contract,
)

HANDOFF_TOOL_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)
HANDOFF_SKILL_ROOT = HANDOFF_TOOL_ROOT.parent.parent
TEST_BRANCH = "feature/synthetic-handoff-review"


MANDATORY_VALIDATION_ROWS = tuple(
    load_handoff_script_module("handoff_contract").MANDATORY_VALIDATION_ROWS
)


@pytest.fixture(scope="session")
def full_valid_handoff_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("valid-handoff-package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(root)
    return package, sidecar


def copy_valid_handoff_package(
    source: tuple[Path, Path],
    target: Path,
) -> tuple[Path, Path]:
    source_package, source_sidecar = source
    package = target / source_package.name
    sidecar = target / source_sidecar.name
    shutil.copyfile(source_package, package)
    shutil.copyfile(source_sidecar, sidecar)
    return package, sidecar


def _build_test_package(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    builder = load_handoff_script_module("build_handoff_package")
    repo = tmp_path / "repo"
    base, final_head = init_handoff_test_repo(repo)
    result = builder.build_handoff_package(
        repo_root=repo,
        out_dir=tmp_path / "out",
        task_name="handoff-test",
        base_ref="origin/main",
    )
    assert result["zip_path"]
    with zipfile.ZipFile(result["zip_path"]) as archive:
        check_receipts = json.loads(
            archive.read("handoff_evidence/check-receipts.json").decode("utf-8")
        )
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
    assert check_receipts["status"] == "PASS"
    assert next(
        row for row in check_receipts["checks"] if row["command_id"] == "profile-smoke"
    )["status"] == "PASS"
    assert validation_summary["status"] == "PASS"
    assert {row["status"] for row in validation_summary["rows"]} == {"PASS"}
    return repo, Path(result["zip_path"]), Path(result["json_path"]), base, final_head


def init_handoff_test_repo(repo: Path) -> tuple[str, str]:
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
    contracts_root = package_root / "contracts"
    contracts_root.mkdir()
    (contracts_root / "__init__.py").write_text("", encoding="utf-8")
    shutil.copyfile(
        Path("src") / "quant_research_platform" / "contracts" / "_json.py",
        contracts_root / "_json.py",
    )
    policy_root = package_root / "policy"
    policy_root.mkdir()
    (policy_root / "__init__.py").write_text("", encoding="utf-8")
    for relative in ("check_status.py", "share_safety.py"):
        shutil.copyfile(
            Path("src") / "quant_research_platform" / "policy" / relative,
            policy_root / relative,
        )
    _write_minimal_project_files(repo)
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "handoff-test@example.invalid")
    run_git(repo, "config", "user.name", "Handoff Test")
    run_git(repo, "checkout", "-b", TEST_BRANCH)
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD").strip()
    run_git(repo, "update-ref", "refs/remotes/origin/main", base)
    (repo / "tests" / "negative_fixtures.py").write_text(
        "NEGATIVE_FIXTURE = 'review-safe aggregate fixture'\n",
        encoding="utf-8",
    )
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", "add handoff fixture")
    final_head = run_git(repo, "rev-parse", "HEAD").strip()
    return base, final_head


def _write_minimal_project_files(repo: Path) -> None:
    (repo / ".gitignore").write_text(
        "\n".join(
            [
                ".pytest_cache/",
                "__pycache__/",
                ".run/",
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
        render_minimal_pyproject_from_root_contract()
        + "\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n",
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
    smoke_test.write_text(
        "def test_smoke() -> None:\n    assert True\n", encoding="utf-8"
    )
    runner = repo / "tests" / "tools" / "run_profile.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import argparse",
                "import sys",
                "from pathlib import Path",
                "",
                "REPO_ROOT = Path(__file__).resolve().parents[2]",
                "REPO_SRC = REPO_ROOT / 'src'",
                "if str(REPO_SRC) not in sys.path:",
                "    sys.path.insert(0, str(REPO_SRC))",
                "",
                "from quant_research_platform.contracts._json import (  # noqa: E402",
                "    render_qrp_json_object,",
                ")",
                "",
                "",
                "def main() -> int:",
                "    parser = argparse.ArgumentParser()",
                "    parser.add_argument('--profile', required=True)",
                "    parser.add_argument('--run-dir', required=True)",
                "    args = parser.parse_args()",
                "    if args.profile != 'profile-smoke':",
                "        return 2",
                "    run_dir = Path(args.run_dir)",
                "    run_dir.mkdir(parents=True, exist_ok=True)",
                "    (run_dir / 'stdout.log').write_text(",
                "        '1 passed in 0.01s\\n', encoding='utf-8'",
                "    )",
                "    (run_dir / 'stderr.log').write_text('', encoding='utf-8')",
                "    summary = {",
                "        'schema_id': 'qrp-test-profile-run',",
                "        'profile': 'profile-smoke',",
                "        'status': 'PASS',",
                "        'return_code': 0,",
                "        'run_dir': args.run_dir,",
                "    }",
                "    summary_name = Path('summary').with_suffix('.json')",
                "    rendered_summary = render_qrp_json_object(summary)",
                "    (run_dir / summary_name).write_text(",
                "        rendered_summary + '\\n',",
                "        encoding='utf-8',",
                "    )",
                "    print(rendered_summary)",
                "    return 0",
                "",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_git(repo: Path, *args: str) -> str:
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
        "status": "PASS",
        "rows": rows if rows is not None else _validation_rows("PASS"),
    }


def rewrite_handoff_package_member(package: Path, member: str, data: bytes) -> None:
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


def refresh_handoff_manifest_and_sidecar(package: Path, sidecar: Path) -> None:
    with zipfile.ZipFile(package, "r") as archive:
        names = sorted(
            name for name in archive.namelist() if name != "artifact_manifest.json"
        )
        entries = []
        for name in names:
            raw = archive.read(name)
            entries.append(
                {
                    "path": name,
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    manifest = {
        "schema_id": "qrp-handoff-artifact-manifest",
        "entries": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    rewrite_handoff_package_member(package, "artifact_manifest.json", manifest_bytes)
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
    rewrite_handoff_package_member(
        package,
        "handoff_evidence/validation-summary.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    refresh_handoff_manifest_and_sidecar(package, sidecar)
