from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

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


def test_default_pytest_excludes_slow_and_local_data() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    marker_expression = " ".join(
        option
        for index, option in enumerate(addopts)
        if option != "-m" and (index == 0 or addopts[index - 1] == "-m")
    )

    assert "not slow" in marker_expression
    assert "not requires_local_data" in marker_expression


def test_fast_profile_runs_pytest_with_repo_relative_paths_and_shell_false(
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
            "--output-dir",
            "out",
            "--tmp-dir",
            "tmp",
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
    assert "tmp/f" in command
    assert "--durations=3" in command
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["shell"] is False

    summary = json.loads((tmp_path / "out" / "fast-summary.json").read_text())
    assert summary["schema_id"] == "qrp-test-profile-run"
    assert summary["profile"] == "fast"
    assert summary["status"] == "PASS"
    assert summary["return_code"] == 0
    assert summary["stdout_path"] == "out/fast-stdout.log"
    assert summary["stderr_path"] == "out/fast-stderr.log"
    assert summary["tmp_dir"] == "tmp/f"
    assert summary["pytest_tmp_dir"] == "tmp/f"
    assert (tmp_path / "out" / "fast-stdout.log").read_text() == "pytest ok\n"


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
            "--output-dir",
            "out",
            "--tmp-dir",
            "tmp",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "out").exists()


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
            "--output-dir",
            "out",
            "--tmp-dir",
            "tmp",
        ]
    )

    summary = json.loads((tmp_path / "out" / "performance-summary.json").read_text())
    assert exit_code == runner.NO_TESTS_COLLECTED
    assert summary["status"] == "NOT_RUN"


def _marker_names(expression: str) -> set[str]:
    operators = {"and", "or", "not"}
    return {
        token
        for token in expression.replace("(", " ").replace(")", " ").split()
        if token not in operators
    }
