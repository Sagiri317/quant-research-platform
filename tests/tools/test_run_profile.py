from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

RUNNER_PATH = Path(__file__).with_name("run_profile.py")


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
        "fast",
        "contract",
        "architecture",
        "integration",
        "mutation",
        "all-non-slow",
    ):
        assert profile in output


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
