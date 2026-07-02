"""Replay public machine checks and produce canonical handoff receipts."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import hard_scan_receipt_from_clone  # noqa: E402
from handoff_contract import (  # noqa: E402
    CHECK_RECEIPTS_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    replay_command_spec,
)
from receipt_contract import (  # noqa: E402
    COUNT_FIELDS,
    PACKAGE_ARTIFACT_KINDS,
    PACKAGE_IMPORT_MODULE,
    PACKAGE_INSTALL_TARGET_KIND,
    PACKAGE_SCANNED_ARTIFACT_KINDS,
    normalize_hard_scan_receipt,
    normalize_package_smoke_receipt,
    validate_check_receipts,
)

FORBIDDEN_PACKAGE_SUFFIXES = {".parquet", ".csv", ".pkl", ".pickle", ".feather", ".arrow"}


def build_replayed_check_receipts(repo_root: Path) -> dict[str, object]:
    rows = [
        _run_replay_command(repo_root, "profile-smoke"),
        _run_replay_command(repo_root, "code-map"),
    ]
    status = _aggregate_status([str(row["status"]) for row in rows])
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": status,
        "checks": rows,
    }
    failures = validate_check_receipts(payload)
    if failures:
        raise SystemExit(";".join(failures))
    return payload


def build_replayed_package_smoke_receipt(repo_root: Path) -> dict[str, object]:
    return normalize_package_smoke_receipt(
        _build_replayed_package_smoke_receipt(repo_root)
    )


def build_replayed_hard_scan_receipt(
    repo_root: Path,
    final_head: str,
) -> dict[str, object]:
    return normalize_hard_scan_receipt(
        hard_scan_receipt_from_clone(repo_root, final_head=final_head)
    )


def build_machine_receipts(
    repo_root: Path,
    final_head: str,
) -> dict[str, dict[str, object]]:
    return {
        CHECK_RECEIPTS_ENTRY: build_replayed_check_receipts(repo_root),
        PACKAGE_SMOKE_RECEIPT_ENTRY: build_replayed_package_smoke_receipt(repo_root),
        HARD_SCAN_RECEIPT_ENTRY: build_replayed_hard_scan_receipt(repo_root, final_head),
    }


def _run_replay_command(repo_root: Path, command_id: str) -> dict[str, object]:
    spec = replay_command_spec(command_id)
    command_argv = list(spec.command_argv)
    run_argv = _runtime_command_argv(command_argv)
    result = subprocess.run(
        run_argv,
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    status = "PASS" if result.returncode == 0 else "FAIL"
    counts = parse_pytest_counts(
        "\n".join((result.stdout or "", result.stderr or ""))
    ) if command_id == "profile-smoke" else None
    return {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": command_argv,
        "command": spec.command,
        "status": status,
        "exit_code": result.returncode,
        "counts": counts,
    }


def parse_pytest_counts(text: str) -> dict[str, int]:
    counts = {field: 0 for field in COUNT_FIELDS}
    lowered = text.lower()
    for label, field in (
        ("passed", "passed"),
        ("failed", "failed"),
        ("skipped", "skipped"),
        ("deselected", "deselected"),
        ("error", "errors"),
        ("errors", "errors"),
        ("xfailed", "xfailed"),
        ("xpassed", "xpassed"),
    ):
        for match in re.finditer(rf"(\d+)\s+{label}\b", lowered):
            counts[field] += int(match.group(1))
    counts["collected"] = (
        counts["passed"]
        + counts["failed"]
        + counts["skipped"]
        + counts["errors"]
        + counts["xfailed"]
        + counts["xpassed"]
        + counts["deselected"]
    )
    return counts


def _runtime_command_argv(command_argv: list[str]) -> list[str]:
    if command_argv and command_argv[0] == "python":
        return [sys.executable, *command_argv[1:]]
    return command_argv


def _build_replayed_package_smoke_receipt(repo_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="qrp-package-smoke-") as temp:
        root = Path(temp)
        dist = root / "dist"
        target = root / "install-target"
        build_status, wheel_path, sdist_path = _build_package_artifacts(repo_root, dist)
        install_status = "NOT_RUN"
        import_status = "NOT_RUN"
        if build_status == "PASS" and wheel_path is not None:
            install_status, import_status = _install_package_smoke_target(
                repo_root,
                wheel_path,
                target,
            )
        forbidden = _scan_package_artifacts(wheel_path, sdist_path, target)
        content_status = "PASS" if not forbidden else "FAIL"
        status = _aggregate_status(
            [build_status, install_status, import_status, content_status]
        )
        shutil.rmtree(target, ignore_errors=True)
        return {
            "schema_id": "qrp-package-smoke-receipt",
            "status": status,
            "build": {
                "status": build_status,
                "artifact_kinds": list(PACKAGE_ARTIFACT_KINDS),
            },
            "install": {
                "target_kind": PACKAGE_INSTALL_TARGET_KIND,
                "no_deps": True,
                "status": install_status,
            },
            "import": {
                "status": import_status,
                "module": PACKAGE_IMPORT_MODULE,
            },
            "content_scan": {
                "status": content_status,
                "forbidden_member_count": len(forbidden),
                "scanned_artifact_kinds": list(PACKAGE_SCANNED_ARTIFACT_KINDS),
            },
        }


def _build_package_artifacts(
    repo_root: Path,
    dist: Path,
) -> tuple[str, Path | None, Path | None]:
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--outdir",
            str(dist),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return (
        "PASS" if build.returncode == 0 else "FAIL",
        _first_file(dist, "*.whl"),
        _first_file(dist, "*.tar.gz"),
    )


def _install_package_smoke_target(
    repo_root: Path,
    wheel_path: Path,
    target: Path,
) -> tuple[str, str]:
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    install_status = "PASS" if install.returncode == 0 else "FAIL"
    if install.returncode != 0:
        return install_status, "NOT_RUN"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {PACKAGE_IMPORT_MODULE}; print({PACKAGE_IMPORT_MODULE}.__name__)",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return install_status, "PASS" if probe.returncode == 0 else "FAIL"


def _scan_package_artifacts(
    wheel_path: Path | None,
    sdist_path: Path | None,
    target: Path,
) -> list[str]:
    return [
        *_archive_content_scan(wheel_path, "wheel"),
        *_archive_content_scan(sdist_path, "sdist"),
        *_install_target_content_scan(target),
    ]


def _first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _install_target_content_scan(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        f"install_target:{path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
    ]


def _archive_content_scan(path: Path | None, artifact_kind: str) -> list[str]:
    if path is None:
        return []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                f"{artifact_kind}:{name}"
                for name in archive.namelist()
                if Path(name).suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
            ]
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return [
                f"{artifact_kind}:{member.name}"
                for member in archive.getmembers()
                if member.isfile()
                and Path(member.name).suffix.lower() in FORBIDDEN_PACKAGE_SUFFIXES
            ]
    return []


def _aggregate_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    return "PASS"
