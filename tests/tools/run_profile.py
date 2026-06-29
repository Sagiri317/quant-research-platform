from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


FAST_EXCLUDES = (
    "not integration and not mutation and not slow and not performance "
    "and not requires_local_data and not private_slow and not data_dependency "
    "and not slow_integration"
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
            "tests/agents/handoff/test_validation_evidence_contract.py",
            "tests/agents/handoff/test_handoff_package_verifier.py",
            "tests/architecture/test_handoff_package_contract.py",
        ),
        marker=FAST_EXCLUDES,
    ),
    "architecture": Profile(
        name="architecture",
        scopes=("tests/architecture",),
        marker=FAST_EXCLUDES,
    ),
    "data-contract": Profile(
        name="data-contract",
        scopes=("tests/architecture/test_share_safe_evidence_data_contract.py",),
        marker="data_dependency or requires_local_data",
    ),
    "integration": Profile(
        name="integration",
        scopes=("tests",),
        marker="integration or slow_integration or data_dependency",
    ),
    "mutation": Profile(
        name="mutation",
        scopes=("tests",),
        marker="mutation",
    ),
    "all-non-slow": Profile(
        name="all-non-slow",
        scopes=("tests",),
        marker=FAST_EXCLUDES,
    ),
    "local-data": Profile(
        name="local-data",
        scopes=("tests",),
        marker="requires_local_data or data_dependency",
    ),
    "performance": Profile(
        name="performance",
        scopes=("tests",),
        marker="performance",
    ),
    "private-slow": Profile(
        name="private-slow",
        scopes=("tests",),
        marker="private_slow",
        pytest_options=("-o", "addopts=--import-mode=importlib"),
    ),
    "slow": Profile(
        name="slow",
        scopes=("tests",),
        marker="slow or slow_integration",
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
    if args.output_dir is None or args.tmp_dir is None:
        parser.error("--output-dir and --tmp-dir are required when --profile is used")

    repo_root = Path(args.repo_root).resolve()
    try:
        output_dir = _resolve_repo_relative(repo_root, args.output_dir, "--output-dir")
        tmp_dir = _resolve_repo_relative(repo_root, args.tmp_dir, "--tmp-dir")
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    return run_profile(
        PROFILES[args.profile],
        repo_root=repo_root,
        output_dir=output_dir,
        tmp_dir=tmp_dir,
        pytest_args=tuple(pytest_args),
    )


def run_profile(
    profile: Profile,
    *,
    repo_root: Path,
    output_dir: Path,
    tmp_dir: Path,
    pytest_args: tuple[str, ...] = (),
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_tmp = tmp_dir / _profile_tmp_name(profile.name)
    profile_tmp.mkdir(parents=True, exist_ok=True)
    pytest_tmp = profile_tmp
    stdout_path = output_dir / f"{profile.name}-stdout.log"
    stderr_path = output_dir / f"{profile.name}-stderr.log"
    summary_path = output_dir / f"{profile.name}-summary.json"

    command = [
        sys.executable,
        "-m",
        "pytest",
        *profile.pytest_options,
        *profile.scopes,
        "-q",
        "--basetemp",
        _repo_relative(repo_root, pytest_tmp),
    ]
    if profile.marker is not None:
        command.extend(["-m", profile.marker])
    command.extend(pytest_args)

    started_at = datetime.now(UTC)
    started = perf_counter()
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        shell=False,
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
        "command": command,
        "cwd": ".",
        "repo_root": ".",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
        "stdout_path": _repo_relative(repo_root, stdout_path),
        "stderr_path": _repo_relative(repo_root, stderr_path),
        "tmp_dir": _repo_relative(repo_root, profile_tmp),
        "pytest_tmp_dir": _repo_relative(repo_root, pytest_tmp),
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
    parser.add_argument("--output-dir", help="repo-relative output directory")
    parser.add_argument("--tmp-dir", help="repo-relative temporary directory")
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


def _repo_relative(repo_root: Path, path: Path) -> str:
    return PurePosixPath(path.resolve().relative_to(repo_root)).as_posix()


def _profile_tmp_name(profile_name: str) -> str:
    short_names = {
        "all-non-slow": "a",
        "architecture": "r",
        "contract": "c",
        "data-contract": "d",
        "fast": "f",
        "integration": "i",
        "local-data": "l",
        "mutation": "m",
        "performance": "p",
        "private-slow": "ps",
        "slow": "s",
    }
    return short_names.get(profile_name, profile_name[:1] or "p")


def _status_from_return_code(return_code: int) -> str:
    if return_code == 0:
        return STATUS_PASS
    if return_code == NO_TESTS_COLLECTED:
        return STATUS_NOT_RUN
    return STATUS_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
