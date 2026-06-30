from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

RUNNER_PATH = Path(__file__).with_name("run_profile.py")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_profile", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_profile"] = module
    spec.loader.exec_module(module)
    return module


def test_list_profiles_includes_required_profiles(capsys) -> None:
    runner = _load_runner()

    exit_code = runner.main(["--list-profiles"])

    assert exit_code == 0
    output = capsys.readouterr().out
    for profile in (
        "all",
        "all-non-local",
        "fast",
        "contract",
        "architecture",
        "guardrails",
        "integration",
        "mutation",
        "performance",
        "requires-local-data",
        "slow",
    ):
        assert profile in output


def test_profile_markers_are_declared_and_excluded_markers_have_profiles() -> None:
    runner = _load_runner()
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    marker_lines = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    declared_markers = {line.split(":", 1)[0] for line in marker_lines}

    profile_markers = {
        marker
        for profile in runner.PROFILES.values()
        if profile.marker is not None
        for marker in _marker_names(profile.marker)
    }
    assert profile_markers <= declared_markers
    assert "guardrails" in profile_markers

    fast_excluded_markers = _marker_names(runner.FAST_EXCLUDES)
    marker_owners = {
        marker: sorted(
            name
            for name, profile in runner.PROFILES.items()
            if name != "all-non-local"
            if profile.marker is not None
            and profile.marker != runner.FAST_EXCLUDES
            and marker in _marker_names(profile.marker)
        )
        for marker in fast_excluded_markers
    }

    assert marker_owners == {
        "integration": ["integration"],
        "mutation": ["mutation"],
        "performance": ["performance"],
        "requires_local_data": ["requires-local-data"],
        "slow": ["slow"],
    }


def test_default_pytest_excludes_non_default_profiles_and_local_data() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    marker_expression = " ".join(
        option
        for index, option in enumerate(addopts)
        if option != "-m" and (index == 0 or addopts[index - 1] == "-m")
    )

    assert "not integration" in marker_expression
    assert "not mutation" in marker_expression
    assert "not slow" in marker_expression
    assert "not performance" in marker_expression
    assert "not requires_local_data" in marker_expression


def test_default_run_dir_is_profile_scoped_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="pytest ok\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--",
            "--durations=3",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [runner.sys.executable, "-m", "pytest"]
    assert "tests/contracts" in command
    assert "tests/architecture" in command
    assert "tests/policy" in command
    assert "tests/agents" in command
    assert "--basetemp" in command
    basetemp = command[command.index("--basetemp") + 1]
    assert basetemp.startswith(".run/test-profiles/")
    basetemp_parts = PurePosixPath(basetemp).parts
    assert basetemp_parts[-1] == "tmp"
    assert basetemp_parts[-2].split("-")[1] == "fast"
    assert "--durations=3" in command
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["shell"] is False

    profile_dirs = list((tmp_path / ".run" / "test-profiles").iterdir())
    assert len(profile_dirs) == 1
    run_dir = profile_dirs[0]
    name_parts = run_dir.name.split("-")
    assert len(name_parts) >= 3
    assert name_parts[1] == "fast"
    assert len(name_parts[0]) == 12
    assert (run_dir / "stdout.log").read_text() == "pytest ok\n"
    assert (run_dir / "stderr.log").read_text() == ""
    assert (run_dir / "tmp").is_dir()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_id"] == "qrp-test-profile-run"
    assert summary["profile"] == "fast"
    assert summary["status"] == "PASS"
    assert summary["return_code"] == 0
    assert summary["run_dir"] == run_dir.relative_to(tmp_path).as_posix()
    assert summary["stdout_path"] == f"{summary['run_dir']}/stdout.log"
    assert summary["stderr_path"] == f"{summary['run_dir']}/stderr.log"
    assert summary["tmp_path"] == f"{summary['run_dir']}/tmp"
    assert summary["pytest_basetemp"] == f"{summary['run_dir']}/tmp"
    assert "reserved_profile" in summary

    assert set(summary) == {
        "schema_id",
        "profile",
        "status",
        "return_code",
        "command",
        "cwd",
        "repo_root",
        "started_at",
        "ended_at",
        "duration_seconds",
        "run_dir",
        "stdout_path",
        "stderr_path",
        "tmp_path",
        "pytest_basetemp",
        "reserved_profile",
    }


def test_default_run_dir_creations_are_unique_with_same_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    real_datetime = runner.datetime

    class FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            del tz
            return real_datetime(2026, 6, 30, 8, 9, 10, tzinfo=runner.UTC)

    monkeypatch.setattr(runner, "datetime", FixedDatetime)

    first = runner._resolve_run_dir(tmp_path, None, profile_name="fast")
    second = runner._resolve_run_dir(tmp_path, None, profile_name="fast")

    assert first != second
    assert first.parent == tmp_path / ".run" / "test-profiles"
    assert second.parent == first.parent
    assert first.name.startswith("260630080910-fast-")
    assert second.name.startswith("260630080910-fast-")
    assert first.is_dir()
    assert second.is_dir()


def test_default_run_dir_creation_is_concurrent_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    real_datetime = runner.datetime

    class FixedDatetime:
        @classmethod
        def now(cls, tz=None):
            del tz
            return real_datetime(2026, 6, 30, 8, 9, 10, tzinfo=runner.UTC)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="pytest ok\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "datetime", FixedDatetime)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    def run_once() -> int:
        return runner.main(["--profile", "fast", "--repo-root", str(tmp_path)])

    with ThreadPoolExecutor(max_workers=4) as pool:
        exit_codes = list(pool.map(lambda _index: run_once(), range(4)))

    assert exit_codes == [0, 0, 0, 0]
    profile_dirs = sorted((tmp_path / ".run" / "test-profiles").iterdir())
    assert len(profile_dirs) == 4
    assert len({path.name for path in profile_dirs}) == 4
    assert all(path.name.startswith("260630080910-fast-") for path in profile_dirs)
    assert all((path / "summary.json").is_file() for path in profile_dirs)


def test_explicit_run_dir_must_be_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    def fail_run(*_args, **_kwargs):
        raise AssertionError("pytest should not be called")

    monkeypatch.setattr(runner.subprocess, "run", fail_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            "profile-runs/fast",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "profile-runs").exists()


def test_explicit_run_dir_existing_non_empty_fails_before_pytest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    run_dir = tmp_path / ".run" / "manual" / "fast"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("occupied\n", encoding="utf-8")

    def fail_run(*_args, **_kwargs):
        raise AssertionError("pytest should not be called")

    monkeypatch.setattr(runner.subprocess, "run", fail_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/fast",
        ]
    )

    assert exit_code == 2


def test_explicit_run_dir_existing_empty_fails_before_pytest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    run_dir = tmp_path / ".run" / "manual" / "fast"
    run_dir.mkdir(parents=True)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("pytest should not be called")

    monkeypatch.setattr(runner.subprocess, "run", fail_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/fast",
        ]
    )

    assert exit_code == 2


def test_unknown_profile_fails_before_running_pytest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    def fail_run(*_args, **_kwargs):
        raise AssertionError("pytest should not be called")

    monkeypatch.setattr(runner.subprocess, "run", fail_run)

    exit_code = runner.main(
        [
            "--profile",
            "missing",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 2


def test_profile_with_no_collected_tests_returns_non_zero_not_run_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            args=command,
            returncode=runner.NO_TESTS_COLLECTED,
            stdout="no tests collected\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    exit_code = runner.main(
        [
            "--profile",
            "performance",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/performance",
        ]
    )

    summary = json.loads(
        (tmp_path / ".run" / "manual" / "performance" / "summary.json").read_text()
    )
    assert exit_code == runner.NO_TESTS_COLLECTED
    assert summary["status"] == "NOT_RUN"
    assert summary["reserved_profile"] is True


def _marker_names(expression: str) -> set[str]:
    operators = {"and", "or", "not"}
    return {
        token
        for token in expression.replace("(", " ").replace(")", " ").split()
        if token not in operators
    }
