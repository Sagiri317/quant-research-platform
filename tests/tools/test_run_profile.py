from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from coverage.results import should_fail_under

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


def make_completed_run(
    command,
    *,
    returncode: int = 0,
    stdout: str = "pytest ok\n",
    stderr: str = "",
    **_kwargs,
):
    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def make_recording_run(
    calls: list,
    *,
    returncode: int = 0,
    stdout: str = "pytest ok\n",
    stderr: str = "",
    replace_basetemp: bool = False,
):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if replace_basetemp:
            option_index = len(command) - 1 - command[::-1].index("--basetemp")
            basetemp = Path(command[option_index + 1])
            if basetemp.exists():
                shutil.rmtree(basetemp)
            basetemp.mkdir()
        return make_completed_run(
            command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def reject_pytest_call(*_args, **_kwargs):
    raise AssertionError("pytest should not be called")


def fixed_profile_datetime(runner):
    real_datetime = runner.datetime
    return type(
        "FixedDatetime",
        (),
        {
            "now": classmethod(
                lambda cls, tz=None: real_datetime(
                    2026, 6, 30, 8, 9, 10, tzinfo=runner.UTC
                )
            )
        },
    )


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
        "profile-smoke",
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


def test_all_profile_scopes_exist() -> None:
    runner = _load_runner()
    missing = [
        f"{name}:{scope}"
        for name, profile in sorted(runner.PROFILES.items())
        for scope in profile.scopes
        if not (REPO_ROOT / scope).exists()
    ]

    assert missing == []
    assert runner.PROFILES["all"].pytest_options == (
        "-o",
        "addopts=--import-mode=importlib --strict-markers",
        "-p",
        "pytest_cov.plugin",
        "--cov=quant_research_platform",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
    )


def test_coverage_gate_rejects_real_total_below_threshold() -> None:
    report = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"][
        "coverage"
    ]["report"]

    assert should_fail_under(89.999999, report["fail_under"], report["precision"])
    assert not should_fail_under(90.0, report["fail_under"], report["precision"])


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


def test_default_run_dir_uses_short_external_basetemp_and_review_safe_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    calls = []
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing")
    monkeypatch.setenv("PYTEST_PLUGINS", "selection_plugin")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        make_recording_run(calls, replace_basetemp=True),
    )
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

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
    basetemp_path = Path(basetemp)
    assert basetemp_path.parent.parent == external_temp
    assert basetemp_path.parent.name.startswith(".p-q")
    allocation_id = basetemp_path.parent.name.rsplit("-", 1)[-1]
    assert len(allocation_id) == 12
    assert set(allocation_id) <= set("0123456789abcdef")
    assert basetemp_path.name == "t"
    assert "--durations=3" in command
    assert kwargs["cwd"] == tmp_path.resolve()
    assert kwargs["shell"] is False
    assert "PYTEST_ADDOPTS" not in kwargs["env"]
    assert "PYTEST_PLUGINS" not in kwargs["env"]

    profile_dirs = list((tmp_path / ".run" / "test-profiles").iterdir())
    assert len(profile_dirs) == 1
    run_dir = profile_dirs[0]
    name_parts = run_dir.name.split("-")
    assert len(name_parts) >= 3
    assert name_parts[1] == "fast"
    assert len(name_parts[0]) == 12
    assert (run_dir / "stdout.log").read_text() == "pytest ok\n"
    assert (run_dir / "stderr.log").read_text() == ""
    assert not basetemp_path.exists()
    assert not basetemp_path.parent.exists()

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_id"] == "qrp-test-profile-run"
    assert summary["profile"] == "fast"
    assert summary["status"] == "PASS"
    assert summary["return_code"] == 0
    assert summary["command"][:3] == ["python", "-m", "pytest"]
    assert not any(str(tmp_path) in item for item in summary["command"])
    assert summary["run_dir"] == run_dir.relative_to(tmp_path).as_posix()
    assert summary["stdout_path"] == f"{summary['run_dir']}/stdout.log"
    assert summary["stderr_path"] == f"{summary['run_dir']}/stderr.log"
    assert basetemp not in summary["command"]
    assert "<external>/t" in summary["command"]
    assert summary["tmp_path"] == "<external>/t"
    assert summary["pytest_basetemp"] == "<external>/t"
    assert "reserved_profile" in summary
    assert summary["profile_marker"] == runner.FAST_EXCLUDES
    assert summary["profile_scopes"] == [
        "tests/contracts",
        "tests/architecture",
        "tests/policy",
        "tests/agents",
    ]
    assert summary["scope_count"] == 4
    assert summary["pytest_options"] == []
    assert summary["extra_pytest_args"] == ["--durations=3"]
    assert summary["python_version"]

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
        "python_version",
        "profile_marker",
        "profile_scopes",
        "scope_count",
        "pytest_options",
        "extra_pytest_args",
        "run_dir",
        "stdout_path",
        "stderr_path",
        "tmp_path",
        "pytest_basetemp",
        "reserved_profile",
    }


def test_real_pytest_replaces_owned_child_but_not_private_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    caller_basetemp = tmp_path / "caller-basetemp"
    caller_basetemp.mkdir()
    caller_sentinel = caller_basetemp / "keep.txt"
    caller_sentinel.write_text("keep", encoding="utf-8")
    probe = tmp_path / "test_transfer_probe.py"
    probe.write_text(
        "def test_transfer_probe(tmp_path):\n    assert tmp_path.is_dir()\n",
        encoding="utf-8",
    )
    real_run = runner.subprocess.run
    observed: dict[str, object] = {}

    def run_real_pytest(command, **kwargs):
        option_index = len(command) - 1 - command[::-1].index("--basetemp")
        child = Path(command[option_index + 1])
        parent = child.parent
        child.mkdir(exist_ok=True)
        sentinel = child / "remove.txt"
        sentinel.write_text("remove", encoding="utf-8")
        parent_before = parent.stat()
        child_before = child.stat()
        result = real_run(command, **kwargs)
        observed.update(
            parent_same=os.path.samestat(parent_before, parent.stat()),
            child_same=os.path.samestat(child_before, child.stat()),
            child_sentinel=sentinel.exists(),
            child=child,
        )
        return result

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner.subprocess, "run", run_real_pytest)
    run_dir = tmp_path / ".run" / "transfer-smoke"
    profile = runner.Profile(name="transfer-smoke", scopes=(probe.name,))

    exit_code = runner.run_profile(
        profile,
        repo_root=tmp_path,
        run_dir=run_dir,
        pytest_args=(
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(caller_basetemp),
            "--durations=0",
        ),
    )

    assert exit_code == 0
    assert observed["parent_same"] is True
    assert observed["child_same"] is False
    assert observed["child_sentinel"] is False
    child = observed["child"]
    assert isinstance(child, Path)
    assert not child.exists()
    assert not child.parent.exists()
    assert caller_sentinel.read_text(encoding="utf-8") == "keep"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert Path(summary["pytest_basetemp"]).name == child.name
    assert not Path(summary["pytest_basetemp"]).is_absolute()
    assert child.parent.name.startswith(".p-q")
    allocation_id = child.parent.name.rsplit("-", 1)[-1]
    assert len(allocation_id) == 12
    assert set(allocation_id) <= set("0123456789abcdef")
    assert child.name == "t"
    assert str(tmp_path) not in json.dumps(summary)


@pytest.mark.parametrize(
    "caller_basetemp_args",
    (
        ("--basetemp", "caller-temp"),
        ("--basetemp=caller-temp",),
    ),
)
def test_runner_owned_basetemp_is_the_final_effective_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caller_basetemp_args: tuple[str, ...],
) -> None:
    runner = _load_runner()
    calls = []
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / "fast"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        make_recording_run(calls, replace_basetemp=True),
    )
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    assert (
        runner.main(
            [
                "--profile",
                "fast",
                "--repo-root",
                str(tmp_path),
                "--run-dir",
                run_dir.relative_to(tmp_path).as_posix(),
                "--",
                *caller_basetemp_args,
            ]
        )
        == 0
    )

    command, _kwargs = calls[0]
    owned = Path(command[-1])
    assert command[-2:] == ["--basetemp", str(owned)]
    assert "--durations=10" in command
    marker_option = command.index("-m", 3)
    assert command[marker_option + 1] == runner.FAST_EXCLUDES
    assert not owned.exists()
    assert not owned.parent.exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["command"][-2:] == ["--basetemp", "<external>/t"]
    assert summary["pytest_basetemp"] == "<external>/t"
    assert summary["extra_pytest_args"] == [*caller_basetemp_args, "--durations=10"]


@pytest.mark.parametrize(
    ("returncode", "expected_status"),
    [(1, "FAIL"), (5, "NOT_RUN")],
)
def test_pytest_basetemp_is_cleaned_after_nonzero_pytest_result(
    tmp_path: Path,
    monkeypatch,
    returncode: int,
    expected_status: str,
) -> None:
    runner = _load_runner()
    calls = []
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / f"fast-{returncode}"

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        make_recording_run(
            calls,
            returncode=returncode,
            replace_basetemp=True,
        ),
    )
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            run_dir.relative_to(tmp_path).as_posix(),
        ]
    )

    assert exit_code == returncode
    command, _kwargs = calls[0]
    basetemp = Path(command[command.index("--basetemp") + 1])
    assert not basetemp.exists()
    assert not basetemp.parent.exists()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == expected_status
    assert summary["return_code"] == returncode
    assert summary["pytest_basetemp"] == "<external>/t"


def test_pytest_basetemp_is_cleaned_when_subprocess_does_not_return(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    primary = KeyboardInterrupt()
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / "fast"
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    def raise_primary(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(runner.subprocess, "run", raise_primary)

    with pytest.raises(type(primary)) as caught:
        runner.main(
            [
                "--profile",
                "fast",
                "--repo-root",
                str(tmp_path),
                "--run-dir",
                run_dir.relative_to(tmp_path).as_posix(),
            ]
        )

    assert caught.value is primary
    assert not tuple(external_temp.iterdir())
    assert not (run_dir / "summary.json").exists()


@pytest.mark.parametrize(
    "preparation",
    ["command", "clock", "environment"],
)
def test_pytest_basetemp_is_not_allocated_before_fallible_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preparation: str,
) -> None:
    runner = _load_runner()
    primary = KeyboardInterrupt(f"{preparation}_primary")
    allocations: list[str] = []

    monkeypatch.setattr(
        runner,
        "_pytest_basetemp",
        lambda profile_name: allocations.append(profile_name) or tmp_path / "leak",
    )
    if preparation == "command":
        monkeypatch.setattr(
            runner,
            "_with_default_durations",
            lambda _args: (_ for _ in ()).throw(primary),
        )
    elif preparation == "clock":

        class InterruptedDatetime:
            @classmethod
            def now(cls, _timezone):
                raise primary

        monkeypatch.setattr(runner, "datetime", InterruptedDatetime)
    else:

        class InterruptedEnvironment(dict):
            def copy(self):
                raise primary

        monkeypatch.setattr(
            runner.os,
            "environ",
            InterruptedEnvironment(runner.os.environ),
        )

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_profile(
            runner.PROFILES["fast"],
            repo_root=tmp_path,
            run_dir=tmp_path / ".run" / "manual" / preparation,
        )

    assert caught.value is primary
    assert allocations == []


def test_pytest_basetemp_owner_cleans_when_path_binding_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    primary = KeyboardInterrupt("basetemp_path_binding")
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    real_lstat = Path.lstat
    calls = 0

    def interrupt_lstat(path: Path):
        nonlocal calls
        if path.parent == external_temp:
            calls += 1
            if calls == 1:
                raise primary
        return real_lstat(path)

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(Path, "lstat", interrupt_lstat)

    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_profile(
            runner.PROFILES["fast"],
            repo_root=tmp_path,
            run_dir=tmp_path / ".run" / "manual" / "fast",
        )

    assert caught.value is primary
    survivors = list(external_temp.iterdir())
    assert len(survivors) == 1
    orphan = survivors[0]
    monkeypatch.setattr(Path, "lstat", real_lstat)
    shared = __import__(
        "quant_research_platform.storage._scratch_ownership", fromlist=["*"]
    )
    monkeypatch.setattr(
        shared,
        "scratch_owner_is_stale",
        lambda path, *, prefix: Path(path) == orphan,
    )

    with runner._pytest_basetemp() as child:
        assert child.parent != orphan

    assert not orphan.exists()


def test_pytest_temp_parent_is_private_and_child_starts_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    mkdir_modes: list[int] = []
    real_mkdir = runner.os.mkdir

    def tracked_mkdir(path, mode=0o777, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if Path(path).parent == external_temp:
            mkdir_modes.append(mode)
        return real_mkdir(path, mode, *args, **kwargs)

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner.os, "mkdir", tracked_mkdir)

    with runner._pytest_basetemp() as child:
        parent = child.parent
        assert parent.parent == external_temp
        assert parent.is_dir()
        assert not child.exists()

    assert mkdir_modes == [0o700]
    assert not parent.exists()


def test_pytest_basetemp_rejects_aliased_os_temp_without_touching_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(alias))

    with pytest.raises(ValueError, match="^pytest_temp_root_invalid$"):
        with runner._pytest_basetemp():
            raise AssertionError("aliased temp root must fail before yield")

    assert tuple(target.iterdir()) == (sentinel,)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_pytest_temp_path_is_minimal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    allocation_id = "a1b2c3d4e5f6"

    class FixedUuid:
        hex = allocation_id * 2

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner, "uuid4", FixedUuid)

    with runner._pytest_basetemp() as child:
        assert child.parent.parent == external_temp
        assert child.parent.name.startswith(".p-q")
        assert child.parent.name.endswith(allocation_id)
        assert child.name == "t"
        assert len(child.parent.name) < 40


@pytest.mark.skipif(sys.platform != "win32", reason="Windows path budget")
def test_short_basetemp_runs_real_local_verification_sqlite_path() -> None:
    runner = _load_runner()
    all_profile = runner.PROFILES["all"]
    with runner._pytest_basetemp() as basetemp:
        result = subprocess.run(
            (
                sys.executable,
                "-m",
                "pytest",
                *all_profile.pytest_options[:2],
                (
                    "tests/workflows/"
                    "test_local_verification_processed_inputs.py::"
                    "test_local_verification_workflow_runs_real_tiny_"
                    "research_stack"
                ),
                "-q",
                "--durations=100",
                "--basetemp",
                str(basetemp),
            ),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    assert result.returncode == 0, "\n".join((result.stdout, result.stderr))


def test_pytest_temp_parent_collision_preserves_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    parent = external_temp / runner.owned_scratch_name(".p-", "a" * 12)
    parent.mkdir()
    sentinel = parent / "foreign.txt"
    sentinel.write_text("foreign", encoding="utf-8")

    class FixedUuid:
        hex = "a" * 32

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner, "uuid4", FixedUuid)

    with pytest.raises(FileExistsError):
        with runner._pytest_basetemp():
            raise AssertionError("collision must fail before yield")

    assert sentinel.read_text(encoding="utf-8") == "foreign"


def test_pytest_temp_parent_collision_is_global_across_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()

    class FixedUuid:
        hex = "a" * 32

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner, "uuid4", FixedUuid)

    with runner._pytest_basetemp() as first:
        sentinel = first.parent / "owned.txt"
        sentinel.write_text("owned", encoding="utf-8")
        with pytest.raises(FileExistsError):
            with runner._pytest_basetemp():
                raise AssertionError("collision must fail before yield")
        assert sentinel.read_text(encoding="utf-8") == "owned"


def test_pytest_temp_parent_does_not_delete_post_claim_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    with pytest.raises(ValueError, match="storage_scratch_identity_changed"):
        with runner._pytest_basetemp() as child:
            parent = child.parent
            assert parent != external_temp
            shutil.rmtree(parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")

    replacement = tuple(external_temp.iterdir())
    assert len(replacement) == 1
    assert (replacement[0] / "foreign.txt").is_file()


def test_pytest_temp_parent_preserves_primary_and_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    primary = SystemExit(9)
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    with pytest.raises(SystemExit) as caught:
        with runner._pytest_basetemp() as child:
            parent = child.parent
            assert parent != external_temp
            shutil.rmtree(parent)
            parent.mkdir()
            (parent / "foreign.txt").write_text("foreign", encoding="utf-8")
            raise primary

    assert caught.value is primary
    assert any("identity_changed" in note for note in primary.__notes__)
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "foreign"


def test_pytest_child_replacement_is_cleaned_with_exact_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))

    with runner._pytest_basetemp() as child:
        parent = child.parent
        assert not child.exists()
        child.mkdir()
        first_identity = child.stat()
        shutil.rmtree(child)
        child.mkdir()
        replacement_identity = child.stat()
        (child / "replacement.txt").write_text("replacement", encoding="utf-8")
        assert not os.path.samestat(first_identity, replacement_identity)

    assert not parent.exists()


def test_pytest_basetemp_is_cleaned_when_log_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    log_name = "stdout.log"
    primary = OSError(f"cannot write {log_name}")
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / "fast"
    real_write_text = runner.Path.write_text

    def write_text(path, *args, **kwargs):
        if path.name == log_name:
            raise primary
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", make_completed_run)
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner.Path, "write_text", write_text)

    with pytest.raises(OSError) as caught:
        runner.main(
            [
                "--profile",
                "fast",
                "--repo-root",
                str(tmp_path),
                "--run-dir",
                run_dir.relative_to(tmp_path).as_posix(),
            ]
        )

    assert caught.value is primary
    assert not tuple(external_temp.iterdir())
    assert not (run_dir / "summary.json").exists()


def test_cleanup_failure_without_primary_does_not_write_passing_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    cleanup_error = PermissionError("basetemp is locked")
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / "fast"

    def fail_cleanup(
        _path: Path,
        _identity: os.stat_result,
        **_kwargs: object,
    ) -> None:
        raise cleanup_error

    monkeypatch.setattr(runner.subprocess, "run", make_completed_run)
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner, "remove_owned_scratch", fail_cleanup)

    with pytest.raises(PermissionError) as caught:
        runner.main(
            [
                "--profile",
                "fast",
                "--repo-root",
                str(tmp_path),
                "--run-dir",
                run_dir.relative_to(tmp_path).as_posix(),
            ]
        )

    assert caught.value is cleanup_error
    assert (run_dir / "stdout.log").read_text() == "pytest ok\n"
    assert (run_dir / "stderr.log").read_text() == ""
    assert not (run_dir / "summary.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows read-only semantics")
def test_cleanup_removes_readonly_file_on_windows(tmp_path: Path) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    external_temp.mkdir()
    readonly_file: Path | None = None

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
            with runner._pytest_basetemp() as child:
                readonly_file = child / "repo" / ".git" / "objects" / "readonly-object"
                readonly_file.parent.mkdir(parents=True)
                readonly_file.write_bytes(b"object")
                readonly_file.chmod(stat.S_IREAD)
    finally:
        if readonly_file is not None and readonly_file.exists():
            readonly_file.chmod(stat.S_IWRITE)

    assert tuple(external_temp.iterdir()) == ()


def test_readonly_rmtree_retry_callback_is_permission_error_only(
    monkeypatch,
) -> None:
    runner = _load_runner()
    path = "readonly-entry"
    chmod_calls = []
    operation_calls = []

    def record_chmod(candidate: str, mode: int) -> None:
        chmod_calls.append((candidate, mode))

    def record_operation(candidate: str) -> None:
        operation_calls.append(candidate)

    monkeypatch.setattr(runner.os, "chmod", record_chmod)

    runner._retry_readonly_rmtree_entry(
        record_operation,
        path,
        PermissionError("read-only"),
    )

    assert chmod_calls == [(path, stat.S_IWRITE)]
    assert operation_calls == [path]

    other_error = OSError("not a permission error")
    with pytest.raises(OSError) as caught:
        runner._retry_readonly_rmtree_entry(record_operation, path, other_error)
    assert caught.value is other_error

    retry_error = OSError("retry failed")

    def fail_retry(_candidate: str) -> None:
        raise retry_error

    with pytest.raises(OSError) as caught:
        runner._retry_readonly_rmtree_entry(
            fail_retry,
            path,
            PermissionError("read-only"),
        )
    assert caught.value is retry_error


def test_primary_subprocess_failure_survives_cleanup_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    primary = KeyboardInterrupt()
    cleanup_calls = []
    external_temp = tmp_path.parent / f"{tmp_path.name}-os-temp"
    external_temp.mkdir()
    run_dir = tmp_path / ".run" / "manual" / "fast"

    def raise_primary(*_args, **_kwargs):
        raise primary

    def fail_cleanup(
        path: Path,
        _identity: os.stat_result,
        **_kwargs: object,
    ) -> None:
        cleanup_calls.append(path)
        raise PermissionError("basetemp is locked")

    monkeypatch.setattr(runner.subprocess, "run", raise_primary)
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    monkeypatch.setattr(runner, "remove_owned_scratch", fail_cleanup)

    with pytest.raises(type(primary)) as caught:
        runner.main(
            [
                "--profile",
                "fast",
                "--repo-root",
                str(tmp_path),
                "--run-dir",
                run_dir.relative_to(tmp_path).as_posix(),
            ]
        )

    assert caught.value is primary
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].parent == external_temp
    assert cleanup_calls[0].name.startswith(".p-q")
    assert len(cleanup_calls[0].name.rsplit("-", 1)[-1]) == 12
    assert not (run_dir / "summary.json").exists()


def test_pytest_basetemp_allocations_are_exclusive_and_preserve_unrelated_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    external_temp = tmp_path / "os-temp"
    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(external_temp))
    external_temp.mkdir()
    sentinel = external_temp / "foreign-sibling"
    sentinel.mkdir()
    (sentinel / "keep.txt").write_text("keep", encoding="utf-8")

    with runner._pytest_basetemp() as first:
        with runner._pytest_basetemp() as second:
            assert first.parent.parent == external_temp
            assert second.parent.parent == external_temp
            assert first.parent != second.parent
            assert first != second
            assert first.name == second.name == "t"
            assert (sentinel / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_default_run_dir_creations_are_unique_with_same_timestamp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    monkeypatch.setattr(runner, "datetime", fixed_profile_datetime(runner))

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

    monkeypatch.setattr(runner, "datetime", fixed_profile_datetime(runner))
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", make_recording_run(calls))

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
    summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in profile_dirs
    ]
    basetemps = [
        Path(command[command.index("--basetemp") + 1]) for command, _kwargs in calls
    ]
    assert len({path.parent for path in basetemps}) == 4
    assert len(set(basetemps)) == 4
    assert all(path.name == "t" for path in basetemps)
    allocation_ids = [path.parent.name.rsplit("-", 1)[-1] for path in basetemps]
    assert all(len(value) == 12 for value in allocation_ids)
    assert all(set(value) <= set("0123456789abcdef") for value in allocation_ids)
    basetemp_names = {summary["pytest_basetemp"] for summary in summaries}
    assert basetemp_names == {"<external>/t"}
    assert str(tmp_path) not in json.dumps(summaries)


def test_explicit_run_dir_must_be_under_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    monkeypatch.setattr(runner.subprocess, "run", reject_pytest_call)

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

    monkeypatch.setattr(runner.subprocess, "run", reject_pytest_call)

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

    monkeypatch.setattr(runner.subprocess, "run", reject_pytest_call)

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

    monkeypatch.setattr(runner.subprocess, "run", reject_pytest_call)

    exit_code = runner.main(
        [
            "--profile",
            "missing",
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 2


def test_performance_profile_runs_active_lightweight_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        make_recording_run(calls, stdout="performance ok\n"),
    )

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

    assert exit_code == 0
    command, _kwargs = calls[0]
    assert "tests/performance" in command
    marker_index = len(command) - 1 - list(reversed(command)).index("-m")
    assert command[marker_index + 1] == "performance"
    summary = json.loads(
        (tmp_path / ".run" / "manual" / "performance" / "summary.json").read_text()
    )
    assert summary["status"] == "PASS"
    assert summary["profile_scopes"] == ["tests/performance"]
    assert summary["scope_count"] == 1
    assert summary["reserved_profile"] is False


def test_profile_runner_adds_default_duration_summary_when_not_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(runner.subprocess, "run", make_recording_run(calls))

    exit_code = runner.main(
        [
            "--profile",
            "profile-smoke",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/profile-smoke",
        ]
    )

    assert exit_code == 0
    command, _kwargs = calls[0]
    assert "--durations=10" in command
    summary = json.loads(
        (tmp_path / ".run" / "manual" / "profile-smoke" / "summary.json").read_text()
    )
    assert summary["profile"] == "profile-smoke"
    assert summary["profile_marker"] == runner.FAST_EXCLUDES
    assert summary["scope_count"] == 9
    assert summary["profile_scopes"] == [
        "tests/contracts",
        "tests/tools/test_run_profile.py",
        "tests/agents/handoff/test_check_receipt_contract.py",
        "tests/agents/handoff/test_package_smoke_receipt_contract.py",
        "tests/agents/handoff/test_hard_scan_receipt_contract.py",
        "tests/agents/handoff/test_validation_summary_contract.py",
        "tests/agents/handoff/test_replay_command_contract.py",
        "tests/agents/handoff/test_handoff_package_verifier.py",
        "tests/architecture/test_handoff_contract_single_source.py",
    ]
    assert summary["extra_pytest_args"] == ["--durations=10"]


def test_profile_runner_passes_xdist_speed_gate_args_through(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    calls = []

    monkeypatch.setattr(runner.subprocess, "run", make_recording_run(calls))

    exit_code = runner.main(
        [
            "--profile",
            "all",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/all-xdist",
            "--",
            "-n",
            "4",
            "--dist",
            "loadfile",
            "--durations=100",
        ]
    )

    assert exit_code == 0
    command, kwargs = calls[0]
    assert command[:7] == [
        runner.sys.executable,
        "-m",
        "pytest",
        "-p",
        "xdist.plugin",
        "-o",
        "addopts=--import-mode=importlib --strict-markers",
    ]
    assert "pytest_cov.plugin" in command
    assert "--cov=quant_research_platform" in command
    assert "--cov-report=term-missing" in command
    assert "--cov-fail-under=90" in command
    assert kwargs["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert Path(kwargs["env"]["COVERAGE_FILE"]) == (
        tmp_path / ".run" / "manual" / "all-xdist" / ".coverage"
    )
    assert command.count("addopts=--import-mode=importlib --strict-markers") == 1
    assert command[-7:-2] == ["-n", "4", "--dist", "loadfile", "--durations=100"]
    assert command[-2] == "--basetemp"
    summary = json.loads(
        (tmp_path / ".run" / "manual" / "all-xdist" / "summary.json").read_text()
    )
    assert summary["profile"] == "all"
    assert summary["profile_scopes"] == ["tests"]
    assert summary["extra_pytest_args"] == [
        "-n",
        "4",
        "--dist",
        "loadfile",
        "--durations=100",
    ]


@pytest.mark.parametrize(
    "pytest_args",
    (
        ("--collect-only",),
        ("-k", "one_test"),
        ("--ignore=tests/tools",),
        ("--deselect=tests/tools/test_run_profile.py::test_all_profile_scopes_exist",),
    ),
    ids=("collect-only", "selection", "ignore", "deselect"),
)
def test_all_profile_rejects_execution_universe_reduction_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pytest_args: tuple[str, ...],
) -> None:
    runner = _load_runner()
    allocations: list[str] = []
    monkeypatch.setattr(
        runner,
        "_create_default_run_dir",
        lambda *_args: allocations.append("run-dir") or tmp_path / "unexpected",
    )
    monkeypatch.setattr(
        runner,
        "_pytest_basetemp",
        lambda: allocations.append("basetemp") or tmp_path / "unexpected",
    )
    monkeypatch.setattr(runner.subprocess, "run", reject_pytest_call)

    exit_code = runner.main(
        ["--profile", "all", "--repo-root", str(tmp_path), "--", *pytest_args]
    )

    assert exit_code == 2
    assert allocations == []


def test_profile_summary_sanitizes_absolute_paths_in_pytest_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner()
    local_report = tmp_path / "reports" / "profile.xml"
    local_log = tmp_path / "logs" / "pytest.log"
    local_equals_log = tmp_path / "logs=private" / "pytest.log"

    monkeypatch.setattr(runner.subprocess, "run", make_completed_run)

    exit_code = runner.main(
        [
            "--profile",
            "fast",
            "--repo-root",
            str(tmp_path),
            "--run-dir",
            ".run/manual/fast",
            "--",
            f"--junitxml={local_report}",
            "--log-file",
            str(local_log),
            str(local_equals_log),
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (tmp_path / ".run" / "manual" / "fast" / "summary.json").read_text()
    )
    encoded = json.dumps(summary, ensure_ascii=False)
    assert str(tmp_path) not in encoded
    assert (
        f"--junitxml={local_report.relative_to(tmp_path).as_posix()}"
        in (summary["extra_pytest_args"])
    )
    assert "--log-file" in summary["extra_pytest_args"]
    assert local_log.relative_to(tmp_path).as_posix() in summary["extra_pytest_args"]
    assert (
        local_equals_log.relative_to(tmp_path).as_posix()
        in (summary["extra_pytest_args"])
    )


def _marker_names(expression: str) -> set[str]:
    operators = {"and", "or", "not"}
    return {
        token
        for token in expression.replace("(", " ").replace(")", " ").split()
        if token not in operators
    }


def test_direct_all_profile_rejects_forged_definition_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    execution_hits: list[list[str]] = []
    forged = runner.Profile(name="all", scopes=("tests/tools",))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **_kwargs: (
            execution_hits.append(command) or make_completed_run(command)
        ),
    )
    caught: ValueError | None = None

    try:
        runner.run_profile(
            forged,
            repo_root=tmp_path,
            run_dir=tmp_path / ".run" / "forged-all",
            pytest_args=("--durations=100",),
        )
    except ValueError as error:
        caught = error

    assert execution_hits == []
    assert caught is not None
    assert str(caught) == "all_profile_definition_invalid"
    assert not (tmp_path / ".run" / "forged-all").exists()


def test_pytest_basetemp_stops_before_yielding_foreign_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    displaced = tmp_path / "displaced"
    real_create = runner.create_owned_scratch
    created_paths: list[Path] = []
    body_hits = 0

    def replace_after_create(candidate: Path, *, mode: int = 0o777) -> os.stat_result:
        result = real_create(candidate, mode=mode)
        path = Path(candidate)
        created_paths.append(path)
        temp_root.rename(displaced)
        temp_root.mkdir()
        path.mkdir()
        (path / "foreign.txt").write_text("foreign", encoding="utf-8")
        return result

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(runner, "create_owned_scratch", replace_after_create)
    caught: BaseException | None = None
    try:
        with runner._pytest_basetemp():
            body_hits += 1
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "create_hits": len(created_paths),
        "body_hits": body_hits,
        "error": None if caught is None else f"{type(caught).__name__}:{caught}",
        "foreign_preserved": len(created_paths) == 1
        and (created_paths[0] / "foreign.txt").read_text(encoding="utf-8") == "foreign",
    }
    expected = {
        "create_hits": 1,
        "body_hits": 0,
        "error": "ValueError:storage_scratch_identity_changed",
        "foreign_preserved": True,
    }
    assert actual == expected


def test_pytest_basetemp_binds_temp_root_before_stale_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    displaced = tmp_path / "displaced"
    real_cleanup = runner.cleanup_stale_owned_scratch
    candidate: Path | None = None
    body_hits = 0

    def replace_then_cleanup(parent: Path, **kwargs) -> None:  # noqa: ANN003
        nonlocal candidate
        Path(parent).rename(displaced)
        Path(parent).mkdir()
        candidate = Path(parent) / ".p-qzzzzzz-replacement"
        candidate.mkdir()
        return real_cleanup(parent, **kwargs)

    monkeypatch.setattr(runner.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(runner, "cleanup_stale_owned_scratch", replace_then_cleanup)
    monkeypatch.setitem(
        real_cleanup.__globals__,
        "scratch_owner_is_stale",
        lambda *_args, **_kwargs: True,
    )

    with pytest.raises(ValueError, match="^storage_scratch_identity_changed$"):
        with runner._pytest_basetemp():
            body_hits += 1

    assert body_hits == 0
    assert candidate is not None and candidate.is_dir()
