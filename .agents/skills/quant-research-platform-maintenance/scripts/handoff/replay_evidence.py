"""Replay public machine checks and produce canonical handoff receipts."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import governance_contract  # noqa: E402
from governance_contract import source_tree_scan_groups  # noqa: E402
from handoff_contract import (  # noqa: E402
    CHECK_RECEIPTS_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_BUILD_TIMEOUT_SECONDS,
    PACKAGE_IMPORT_TIMEOUT_SECONDS,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    aggregate_handoff_status,
    replay_command_spec,
)
from receipt_contract import (  # noqa: E402
    COUNT_FIELDS,
    PACKAGE_ARTIFACT_KINDS,
    PACKAGE_IMPORT_MODULE,
    PACKAGE_WHEEL_EXTRACT_METHOD,
    normalize_hard_scan_receipt,
    normalize_package_smoke_receipt,
    validate_check_receipts,
)

FORBIDDEN_PACKAGE_SUFFIXES = {
    ".parquet",
    ".csv",
    ".pkl",
    ".pickle",
    ".feather",
    ".arrow",
}
REPLAY_REQUIRED_DISTRIBUTIONS = (
    "build",
    "setuptools",
    "wheel",
    "pyarrow",
    "pytest",
)
REPLAY_DISTRIBUTION_CONTRACT_SOURCES = {
    "build": ("project", "optional-dependencies", "dev"),
    "pytest": ("project", "optional-dependencies", "dev"),
    "setuptools": ("build-system", "requires"),
    "wheel": ("build-system", "requires"),
    "pyarrow": ("project", "dependencies"),
}
REPLAY_DISTRIBUTION_SOURCE_LABELS = {
    "build": "pyproject.toml:[project.optional-dependencies].dev",
    "pytest": "pyproject.toml:[project.optional-dependencies].dev",
    "setuptools": "pyproject.toml:[build-system].requires",
    "wheel": "pyproject.toml:[build-system].requires",
    "pyarrow": "pyproject.toml:[project].dependencies",
}
PYTHON_CONTRACT_SOURCE_LABEL = "pyproject.toml:[project].requires-python"
PYPROJECT_PATH = "pyproject.toml"


class ReplayEnvironmentBlocked(Exception):
    def __init__(self, failures: tuple[str, ...]) -> None:
        self.failures = failures
        super().__init__(";".join(failures))


def collect_replay_environment_failures(repo_root: Path) -> tuple[str, ...]:
    failures: list[str] = []
    pyproject = _load_pyproject(repo_root)
    python_spec = str(pyproject.get("project", {}).get("requires-python", ""))
    if not python_spec:
        failures.append(
            f"dependency_contract_missing:python:source={PYTHON_CONTRACT_SOURCE_LABEL}"
        )
    else:
        installed_python = _python_version()
        if not _version_satisfies(installed_python, python_spec):
            failures.append(
                "python_version_mismatch:"
                f"expected={python_spec}:installed={installed_python}"
            )

    requirements = _replay_requirement_specs(pyproject)
    for distribution_name in REPLAY_REQUIRED_DISTRIBUTIONS:
        expected = requirements.get(distribution_name, "")
        if not expected:
            failures.append(
                "dependency_contract_missing:"
                f"{distribution_name}:source="
                f"{REPLAY_DISTRIBUTION_SOURCE_LABELS[distribution_name]}"
            )
            continue
        try:
            installed = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            failures.append(f"dependency_missing:{distribution_name}")
            continue
        expected_constraint = _version_constraint(expected)
        if not _version_satisfies(installed, expected_constraint):
            failures.append(
                "dependency_version_mismatch:"
                f"{distribution_name}:expected={expected_constraint}:"
                f"installed={installed}"
            )
    return tuple(failures)


def assert_replay_environment_ready(repo_root: Path) -> None:
    failures = collect_replay_environment_failures(repo_root)
    if failures:
        raise ReplayEnvironmentBlocked(failures)


def build_replayed_check_receipts(repo_root: Path) -> dict[str, object]:
    rows = [
        _run_replay_command(repo_root, "profile-smoke"),
        _run_replay_command(repo_root, "code-map"),
    ]
    status = aggregate_handoff_status(str(row["status"]) for row in rows)
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
    *,
    source_tree_scan_groups_payload: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    scanned_head = _git_text(repo_root, "rev-parse", "HEAD").strip()
    scan_runs: list[dict[str, object]] = []
    scan_groups = source_tree_scan_groups_payload
    if scan_groups is None:
        scan_groups = source_tree_scan_groups(repo_root)
    for group_name, group_payload in sorted(scan_groups.items()):
        findings = list(group_payload.get("findings") or [])
        result_payload = {
            "scan_id": group_name,
            "finding_count": len(findings),
            "findings": findings,
        }
        scan_runs.append(
            {
                "scan_id": group_name,
                "rule_set_id": f"source-tree:{group_name}",
                "match_count": len(findings),
                "actual_violation_count": len(findings),
                "result_sha256": _sha256_text(_stable_json(result_payload)),
                "replayable_from_bundle": True,
            }
        )
    payload = {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS"
        if scanned_head == final_head
        and all(run["actual_violation_count"] == 0 for run in scan_runs)
        else "FAIL",
        "scanned_head": scanned_head,
        "scanner_source_sha256": _sha256_file(
            Path(governance_contract.__file__).resolve()
        ),
        "scan_runs": scan_runs,
    }
    return normalize_hard_scan_receipt(payload)


def build_machine_receipts(
    repo_root: Path,
    final_head: str,
    *,
    source_tree_scan_groups_payload: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    assert_replay_environment_ready(repo_root)
    return {
        CHECK_RECEIPTS_ENTRY: build_replayed_check_receipts(repo_root),
        PACKAGE_SMOKE_RECEIPT_ENTRY: build_replayed_package_smoke_receipt(repo_root),
        HARD_SCAN_RECEIPT_ENTRY: build_replayed_hard_scan_receipt(
            repo_root,
            final_head,
            source_tree_scan_groups_payload=source_tree_scan_groups_payload,
        ),
    }


def _version_satisfies(installed: str, expected: str) -> bool:
    installed_version = _version_tuple(installed)
    for clause in expected.split(","):
        clause = clause.strip()
        if not clause:
            continue
        match = re.fullmatch(r"(>=|<=|<|>|==)\s*([0-9][0-9A-Za-z._+-]*)", clause)
        if match is None:
            return False
        operator, version = match.groups()
        target = _version_tuple(version)
        if operator == ">=" and not installed_version >= target:
            return False
        if operator == "<=" and not installed_version <= target:
            return False
        if operator == "<" and not installed_version < target:
            return False
        if operator == ">" and not installed_version > target:
            return False
        if operator == "==" and not installed_version == target:
            return False
    return True


def _version_constraint(spec: str) -> str:
    match = re.search(r"(>=|<=|<|>|==)\s*[0-9]", spec)
    if match is None:
        return ""
    return spec[match.start() :]


def _load_pyproject(repo_root: Path) -> dict[str, object]:
    path = repo_root / PYPROJECT_PATH
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _replay_requirement_specs(pyproject: Mapping[str, object]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for distribution_name, source_path in REPLAY_DISTRIBUTION_CONTRACT_SOURCES.items():
        items = _pyproject_list_at(pyproject, source_path)
        dependency_specs = _dependency_specs(items)
        if distribution_name in dependency_specs:
            specs[distribution_name] = dependency_specs[distribution_name]
    return specs


def _pyproject_list_at(
    pyproject: Mapping[str, object],
    source_path: tuple[str, ...],
) -> tuple[str, ...]:
    current: object = pyproject
    for key in source_path:
        if not isinstance(current, Mapping):
            return ()
        current = current.get(key)
    if not isinstance(current, list):
        return ()
    return tuple(str(item) for item in current)


def _dependency_specs(items: tuple[str, ...]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items:
        name = re.split(r"[<>=!~; ]", item, maxsplit=1)[0]
        if name:
            specs[name] = item
    return specs


def _python_version() -> str:
    version_info = sys.version_info
    return ".".join(str(part) for part in version_info[:3])


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\d+(?:\.\d+)*", value)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _run_replay_command(repo_root: Path, command_id: str) -> dict[str, object]:
    spec = replay_command_spec(command_id)
    command_argv = list(spec.command_argv)
    run_argv = _runtime_command_argv(command_argv)
    if command_id == "profile-smoke":
        shutil.rmtree(
            repo_root / ".run" / "handoff-replay" / "profile-smoke", ignore_errors=True
        )
    started = perf_counter()
    try:
        result = subprocess.run(
            run_argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=spec.timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        duration_seconds = perf_counter() - started
        return {
            "schema_id": "qrp-check-receipt",
            "command_id": command_id,
            "scope": spec.scope,
            "command_argv": command_argv,
            "command": spec.command,
            "status": "BLOCKED",
            "exit_code": None,
            "duration_seconds": round(duration_seconds, 6),
            "counts": None,
            "failure_class": "command_timeout",
            "blocked_reason": "timeout",
            "not_run_reason": None,
            "timeout_seconds": spec.timeout_seconds,
        }
    duration_seconds = perf_counter() - started
    status = _replay_status(repo_root, command_id, result)
    counts = (
        _profile_smoke_counts(repo_root, result)
        if command_id == "profile-smoke"
        else None
    )
    return {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": command_argv,
        "command": spec.command,
        "status": status,
        "exit_code": result.returncode,
        "duration_seconds": round(duration_seconds, 6),
        "counts": counts,
        "failure_class": None
        if status == "PASS"
        else _failure_class(command_id, result, counts),
        "blocked_reason": None,
        "not_run_reason": "no_tests_collected" if status == "NOT_RUN" else None,
        "timeout_seconds": None,
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


def _profile_smoke_counts(
    repo_root: Path,
    result: subprocess.CompletedProcess[str],
) -> dict[str, int]:
    stdout_path = repo_root / ".run" / "handoff-replay" / "profile-smoke" / "stdout.log"
    if stdout_path.exists():
        return parse_pytest_counts(
            stdout_path.read_text(encoding="utf-8", errors="replace")
        )
    return parse_pytest_counts("\n".join((result.stdout or "", result.stderr or "")))


def _replay_status(
    repo_root: Path,
    command_id: str,
    result: subprocess.CompletedProcess[str],
) -> str:
    if result.returncode == 0:
        return "PASS"
    if command_id == "profile-smoke":
        summary_path = (
            repo_root
            / ".run"
            / "handoff-replay"
            / "profile-smoke"
            / Path("summary").with_suffix(".json")
        )
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return "FAIL"
            status = payload.get("status")
            if status in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}:
                return str(status)
    return "FAIL"


def _failure_class(
    command_id: str,
    result: subprocess.CompletedProcess[str],
    counts: dict[str, int] | None,
) -> str:
    combined = "\n".join((result.stdout or "", result.stderr or "")).lower()
    if "modulenotfounderror" in combined or "no module named" in combined:
        return "dependency_missing"
    if command_id == "profile-smoke":
        if result.returncode == 5:
            return "runner_preflight_error"
        if counts and counts.get("errors", 0):
            return "pytest_collection_error"
        if counts and counts.get("failed", 0):
            return "pytest_failed"
    return "unknown_nonzero_exit"


def _build_replayed_package_smoke_receipt(repo_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="qrp-package-smoke-") as temp:
        root = Path(temp)
        dist = root / "dist"
        wheel_extract = root / "wheel-extract"
        build_status, wheel_path, sdist_path = _build_package_artifacts(repo_root, dist)
        extract_status = "NOT_RUN"
        import_status = "NOT_RUN"
        content_status = "NOT_RUN"
        forbidden: list[str] = []
        scanned_artifact_kinds: list[str] = []
        if build_status == "PASS" and wheel_path is not None:
            extract_status, import_status = _extract_wheel_smoke_target(
                repo_root,
                wheel_path,
                wheel_extract,
            )
        if build_status == "PASS" and extract_status == "PASS":
            forbidden, scanned_artifact_kinds = _scan_package_artifacts(
                wheel_path, sdist_path, wheel_extract
            )
            content_status = "PASS" if not forbidden else "FAIL"
        status = aggregate_handoff_status(
            [build_status, extract_status, import_status, content_status]
        )
        return {
            "schema_id": "qrp-package-smoke-receipt",
            "status": status,
            "build": {
                "status": build_status,
                "artifact_kinds": list(PACKAGE_ARTIFACT_KINDS)
                if build_status == "PASS"
                else [],
            },
            "wheel_extract": {
                "method": PACKAGE_WHEEL_EXTRACT_METHOD,
                "status": extract_status,
            },
            "import": {
                "status": import_status,
                "module": PACKAGE_IMPORT_MODULE,
            },
            "content_scan": {
                "status": content_status,
                "forbidden_member_count": len(forbidden),
                "scanned_artifact_kinds": scanned_artifact_kinds,
            },
        }


def _build_package_artifacts(
    repo_root: Path,
    dist: Path,
) -> tuple[str, Path | None, Path | None]:
    try:
        build = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
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
            timeout=PACKAGE_BUILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "BLOCKED", None, None
    wheel_path, wheel_ok = _select_single_artifact(dist, "*.whl")
    sdist_path, sdist_ok = _select_single_artifact(dist, "*.tar.gz")
    build_status = "PASS" if build.returncode == 0 and wheel_ok and sdist_ok else "FAIL"
    return (
        build_status,
        wheel_path if wheel_ok else None,
        sdist_path if sdist_ok else None,
    )


def _extract_wheel_smoke_target(
    repo_root: Path,
    wheel_path: Path,
    target: Path,
) -> tuple[str, str]:
    try:
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel_path) as archive:
            archive.extractall(target)
    except (OSError, zipfile.BadZipFile):
        return "FAIL", "NOT_RUN"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    try:
        probe_code = (
            f"import {PACKAGE_IMPORT_MODULE}; "
            f"print({PACKAGE_IMPORT_MODULE}.__name__)"
        )
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                probe_code,
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=PACKAGE_IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "PASS", "BLOCKED"
    return "PASS", "PASS" if probe.returncode == 0 else "FAIL"


def _scan_package_artifacts(
    wheel_path: Path | None,
    sdist_path: Path | None,
    wheel_extract: Path,
) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    scanned_artifact_kinds: list[str] = []
    if wheel_path is not None:
        findings.extend(_archive_content_scan(wheel_path, "wheel"))
        scanned_artifact_kinds.append("wheel")
    if sdist_path is not None:
        findings.extend(_archive_content_scan(sdist_path, "sdist"))
        scanned_artifact_kinds.append("sdist")
    if wheel_extract.exists():
        findings.extend(_wheel_extract_content_scan(wheel_extract))
        scanned_artifact_kinds.append("wheel_extract")
    return findings, scanned_artifact_kinds


def _select_single_artifact(root: Path, pattern: str) -> tuple[Path | None, bool]:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        return None, False
    return matches[0], True


def _wheel_extract_content_scan(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        f"wheel_extract:{path.relative_to(root).as_posix()}"
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


def _git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
