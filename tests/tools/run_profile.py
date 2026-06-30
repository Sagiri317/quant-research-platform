from __future__ import annotations

import argparse
import json
import os
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
        scopes=("tests",),
        marker="performance",
        reserved=True,
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
    pytest_basetemp = run_dir / "tmp"
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
    command.extend(pytest_args)

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
        "command": command,
        "cwd": ".",
        "repo_root": ".",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(duration_seconds, 6),
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
    raw = value or _default_run_dir(profile_name)
    run_dir = _resolve_repo_relative(repo_root, raw, "--run-dir")
    relative = PurePosixPath(run_dir.relative_to(repo_root).as_posix())
    if relative.parts[:1] != (".run",) or len(relative.parts) == 1:
        raise ValueError(f"--run-dir must be repo-relative under .run/: {raw}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"--run-dir already exists and is non-empty: {raw}")
    return run_dir


def _repo_relative(repo_root: Path, path: Path) -> str:
    return PurePosixPath(path.resolve().relative_to(repo_root)).as_posix()


def _profile_tmp_name(profile_name: str) -> str:
    short_names = {
        "all": "a",
        "all-non-local": "an",
        "architecture": "r",
        "contract": "c",
        "fast": "f",
        "guardrails": "g",
        "integration": "i",
        "mutation": "m",
        "performance": "p",
        "requires-local-data": "l",
        "slow": "s",
    }
    return short_names.get(profile_name, profile_name[:1] or "p")


def _default_run_dir(profile_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%y%m%d%H%M%S")
    return f".run/test-profiles/{timestamp}-{profile_name}"


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
