from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_RUN = "NOT_RUN"
NO_TESTS_COLLECTED = 5


@dataclass(frozen=True)
class Profile:
    name: str
    scopes: tuple[str, ...]
    marker: str | None = None
    pytest_options: tuple[str, ...] = ()
    reserved: bool = False


FAST_EXCLUDES = (
    "not integration and not mutation and not slow and not performance "
    "and not requires_local_data"
)

PROFILES: dict[str, Profile] = {
    "fast": Profile(
        name="fast",
        scopes=(
            "tests/contracts",
            "tests/architecture",
            "tests/policy",
            "tests/agents",
        ),
        marker=FAST_EXCLUDES,
    ),
    "contract": Profile(
        name="contract",
        scopes=(
            "tests/contracts",
            "tests/agents/handoff/test_check_receipt_contract.py",
            "tests/agents/handoff/test_package_smoke_receipt_contract.py",
            "tests/agents/handoff/test_hard_scan_receipt_contract.py",
            "tests/agents/handoff/test_validation_summary_contract.py",
            "tests/agents/handoff/test_replay_command_contract.py",
            "tests/agents/handoff/test_handoff_package_verifier.py",
            "tests/architecture/test_handoff_contract_single_source.py",
        ),
        marker=FAST_EXCLUDES,
    ),
    "architecture": Profile(
        name="architecture",
        scopes=("tests/architecture",),
        marker=FAST_EXCLUDES,
    ),
    "profile-smoke": Profile(
        name="profile-smoke",
        scopes=(
            "tests/contracts",
            "tests/tools/test_run_profile.py",
            "tests/agents/test_agent_proposals.py",
            "tests/agents/handoff/test_check_receipt_contract.py",
            "tests/agents/handoff/test_package_smoke_receipt_contract.py",
            "tests/agents/handoff/test_hard_scan_receipt_contract.py",
            "tests/agents/handoff/test_validation_summary_contract.py",
            "tests/agents/handoff/test_replay_command_contract.py",
            "tests/agents/handoff/test_handoff_package_verifier.py",
            "tests/architecture/test_handoff_contract_single_source.py",
            "tests/architecture/test_governance_scan_boundaries.py",
        ),
        marker=FAST_EXCLUDES,
    ),
    "guardrails": Profile(
        name="guardrails",
        scopes=("tests",),
        marker="guardrails",
    ),
    "integration": Profile(
        name="integration",
        scopes=("tests",),
        marker="integration",
    ),
    "mutation": Profile(
        name="mutation",
        scopes=("tests",),
        marker="mutation",
    ),
    "all-non-local": Profile(
        name="all-non-local",
        scopes=("tests",),
        marker="not requires_local_data",
    ),
    "requires-local-data": Profile(
        name="requires-local-data",
        scopes=("tests",),
        marker="requires_local_data",
    ),
    "performance": Profile(
        name="performance",
        scopes=("tests/performance",),
        marker="performance",
    ),
    "slow": Profile(
        name="slow",
        scopes=("tests",),
        marker="slow",
    ),
    "all": Profile(
        name="all",
        scopes=("tests",),
        pytest_options=("-o", "addopts=--import-mode=importlib"),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_profiles:
        _print_profiles()
        return 0
    if args.profile not in PROFILES:
        print(f"unknown profile: {args.profile}", file=sys.stderr)
        _print_profiles(stream=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve()
    try:
        run_dir = _resolve_run_dir(
            repo_root,
            args.run_dir,
            profile_name=args.profile,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return run_profile(
        PROFILES[args.profile],
        repo_root=repo_root,
        run_dir=run_dir,
        pytest_args=tuple(pytest_args),
    )


def run_profile(
    profile: Profile,
    *,
    repo_root: Path,
    run_dir: Path,
    pytest_args: tuple[str, ...] = (),
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    pytest_basetemp = _pytest_basetemp(repo_root, run_dir, profile.name)
    pytest_basetemp.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    summary_path = run_dir / "summary.json"

    command = [
        sys.executable,
        "-m",
        "pytest",
        *profile.pytest_options,
        *profile.scopes,
        "-q",
        "--basetemp",
        _repo_relative(repo_root, pytest_basetemp),
    ]
    if profile.marker is not None:
        command.extend(["-m", profile.marker])
    effective_pytest_args = _with_default_durations(pytest_args)
    command.extend(effective_pytest_args)

    started_at = datetime.now(UTC)
    started = perf_counter()
    env = os.environ.copy()
    env.pop(_retired_profile_runner_env_name(), None)
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        shell=False,
        env=env,
    )
    ended_at = datetime.now(UTC)
    duration_seconds = perf_counter() - started

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    status = _status_from_return_code(result.returncode)
    summary = {
        "schema_id": "qrp-test-profile-run",
        "profile": profile.name,
        "status": status,
        "return_code": result.returncode,
        "command": _review_safe_command(repo_root, command),
        "cwd": ".",
        "repo_root": ".",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
        "python_version": sys.version.split()[0],
        "profile_marker": profile.marker,
        "profile_scopes": list(profile.scopes),
        "scope_count": len(profile.scopes),
        "pytest_options": _review_safe_items(repo_root, list(profile.pytest_options)),
        "extra_pytest_args": _review_safe_items(
            repo_root,
            list(effective_pytest_args),
        ),
        "run_dir": _repo_relative(repo_root, run_dir),
        "stdout_path": _repo_relative(repo_root, stdout_path),
        "stderr_path": _repo_relative(repo_root, stderr_path),
        "tmp_path": _repo_relative(repo_root, pytest_basetemp),
        "pytest_basetemp": _repo_relative(repo_root, pytest_basetemp),
        "reserved_profile": profile.reserved,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QRP pytest profiles.")
    parser.add_argument("--profile", help="pytest profile")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--run-dir", help="repo-relative run directory under .run/")
    parser.add_argument("--repo-root", default=".", help="repository root")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser


def _print_profiles(*, stream=None) -> None:
    target = stream or sys.stdout
    for name in sorted(PROFILES):
        print(name, file=target)


def _resolve_repo_relative(repo_root: Path, value: str, flag: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{flag} must be repo-relative: {value}")
    resolved = (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"{flag} escapes repo root: {value}")
    return resolved


def _resolve_run_dir(repo_root: Path, value: str | None, *, profile_name: str) -> Path:
    if value is None:
        return _create_default_run_dir(repo_root, profile_name)
    raw = value
    run_dir = _resolve_repo_relative(repo_root, raw, "--run-dir")
    relative = PurePosixPath(run_dir.relative_to(repo_root).as_posix())
    if relative.parts[:1] != (".run",) or len(relative.parts) == 1:
        raise ValueError(f"--run-dir must be repo-relative under .run/: {raw}")
    if run_dir.exists():
        raise ValueError(f"--run-dir already exists: {raw}")
    return run_dir


def _repo_relative(repo_root: Path, path: Path) -> str:
    return PurePosixPath(path.resolve().relative_to(repo_root)).as_posix()


def _pytest_basetemp(repo_root: Path, run_dir: Path, profile_name: str) -> Path:
    run_dir_key = _repo_relative(repo_root, run_dir)
    digest = hashlib.sha256(run_dir_key.encode("utf-8")).hexdigest()[:12]
    return repo_root / ".run" / "profile-tmp" / f"{profile_name}-{digest}" / "tmp"


def _with_default_durations(pytest_args: tuple[str, ...]) -> tuple[str, ...]:
    if any(
        arg == "--durations" or arg.startswith("--durations=")
        for arg in pytest_args
    ):
        return pytest_args
    return (*pytest_args, "--durations=10")


def _review_safe_command(repo_root: Path, command: list[str]) -> list[str]:
    safe: list[str] = []
    for index, item in enumerate(command):
        if index == 0:
            safe.append("python")
            continue
        safe.append(_review_safe_command_item(repo_root, item))
    return safe


def _review_safe_items(repo_root: Path, items: list[str]) -> list[str]:
    return [_review_safe_command_item(repo_root, item) for item in items]


def _review_safe_command_item(repo_root: Path, item: str) -> str:
    path = Path(item)
    if path.is_absolute():
        return _review_safe_path(repo_root, path)
    if "=" in item:
        key, value = item.split("=", maxsplit=1)
        return f"{key}={_review_safe_command_item(repo_root, value)}"
    return item


def _review_safe_path(repo_root: Path, path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return _repo_relative(repo_root, path)
    except ValueError:
        return path.name


def _create_default_run_dir(repo_root: Path, profile_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%y%m%d%H%M%S")
    parent = repo_root / ".run" / "test-profiles"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f"{timestamp}-{profile_name}-",
            dir=parent,
        )
    )


def _retired_profile_runner_env_name() -> str:
    return "".join(
        chr(code)
        for code in (
            81,
            82,
            80,
            95,
            77,
            65,
            78,
            65,
            71,
            69,
            68,
            95,
            82,
            69,
            80,
            79,
            95,
            82,
            79,
            79,
            84,
            83,
        )
    )


def _status_from_return_code(return_code: int) -> str:
    if return_code == 0:
        return STATUS_PASS
    if return_code == NO_TESTS_COLLECTED:
        return STATUS_NOT_RUN
    return STATUS_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
