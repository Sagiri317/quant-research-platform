from __future__ import annotations

from pathlib import Path

import pytest

from tests.agents.handoff.conftest import init_handoff_test_repo
from tests.helpers.maintenance_script_loader import load_handoff_script_module
from tests.helpers.root_config_contract import (
    build_system_requirement,
    dependency_constraint,
    dev_dependency_requirement,
    project_dependency_requirement,
    project_requires_python,
    render_minimal_pyproject_from_root_contract,
)


def _write_minimal_pyproject(path: Path, *, include_pyarrow: bool = True) -> None:
    text = render_minimal_pyproject_from_root_contract(
        include_project_deps=include_pyarrow
    )
    path.write_text(text, encoding="utf-8")


def test_collect_replay_environment_failures_reports_missing_and_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    _write_minimal_pyproject(tmp_path / "pyproject.toml")
    package_missing = replay.importlib_metadata.PackageNotFoundError("build")
    versions: dict[str, object] = {
        "build": package_missing,
        "setuptools": "999.0.0",
        "wheel": _installed_inside(build_system_requirement("wheel")),
        "pyarrow": _installed_inside(project_dependency_requirement("pyarrow")),
        "pytest": _installed_inside(dev_dependency_requirement("pytest")),
    }

    def fake_version(distribution_name: str) -> str:
        value = versions[distribution_name]
        if isinstance(value, Exception):
            raise value
        return str(value)

    monkeypatch.setattr(replay.importlib_metadata, "version", fake_version)

    failures = replay.collect_replay_environment_failures(tmp_path)

    assert failures == (
        "dependency_missing:build",
        "dependency_version_mismatch:setuptools:"
        f"expected={dependency_constraint(build_system_requirement('setuptools'))}:"
        "installed=999.0.0",
    )
    assert all(":" in failure and "\\" not in failure for failure in failures)


def test_collect_replay_environment_failures_reports_python_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    _write_minimal_pyproject(tmp_path / "pyproject.toml")
    versions = {
        "build": _installed_inside(dev_dependency_requirement("build")),
        "setuptools": _installed_inside(build_system_requirement("setuptools")),
        "wheel": _installed_inside(build_system_requirement("wheel")),
        "pyarrow": _installed_inside(project_dependency_requirement("pyarrow")),
        "pytest": _installed_inside(dev_dependency_requirement("pytest")),
    }

    monkeypatch.setattr(replay.importlib_metadata, "version", versions.__getitem__)
    monkeypatch.setattr(replay.sys, "version_info", (9, 9, 9, "final", 0))

    failures = replay.collect_replay_environment_failures(tmp_path)

    assert failures == (
        f"python_version_mismatch:expected={project_requires_python()}:"
        "installed=9.9.9",
    )


def test_collect_replay_environment_failures_reports_contract_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    _write_minimal_pyproject(tmp_path / "pyproject.toml", include_pyarrow=False)
    versions = {
        "build": _installed_inside(dev_dependency_requirement("build")),
        "setuptools": _installed_inside(build_system_requirement("setuptools")),
        "wheel": _installed_inside(build_system_requirement("wheel")),
        "pyarrow": _installed_inside(project_dependency_requirement("pyarrow")),
        "pytest": _installed_inside(dev_dependency_requirement("pytest")),
    }

    monkeypatch.setattr(replay.importlib_metadata, "version", versions.__getitem__)

    failures = replay.collect_replay_environment_failures(tmp_path)

    assert failures == (
        "dependency_contract_missing:"
        "pyarrow:source=pyproject.toml:[project].dependencies",
    )


def test_replay_environment_preflight_does_not_read_environment_yml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    _write_minimal_pyproject(tmp_path / "pyproject.toml")
    (tmp_path / "environment.yml").write_text("not: valid: yaml", encoding="utf-8")
    versions = {
        "build": _installed_inside(dev_dependency_requirement("build")),
        "setuptools": _installed_inside(build_system_requirement("setuptools")),
        "wheel": _installed_inside(build_system_requirement("wheel")),
        "pyarrow": _installed_inside(project_dependency_requirement("pyarrow")),
        "pytest": _installed_inside(dev_dependency_requirement("pytest")),
    }

    monkeypatch.setattr(replay.importlib_metadata, "version", versions.__getitem__)

    assert replay.collect_replay_environment_failures(tmp_path) == ()


def test_build_machine_receipts_preflights_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    calls: list[str] = []

    def fake_ready(_repo_root: Path) -> None:
        calls.append("preflight")
        raise replay.ReplayEnvironmentBlocked(("dependency_missing:pyarrow",))

    def unexpected_replay(_repo_root: Path) -> dict[str, object]:
        calls.append("receipt")
        return {}

    monkeypatch.setattr(replay, "assert_replay_environment_ready", fake_ready)
    monkeypatch.setattr(replay, "build_replayed_check_receipts", unexpected_replay)
    monkeypatch.setattr(
        replay,
        "build_replayed_package_smoke_receipt",
        unexpected_replay,
    )

    with pytest.raises(replay.ReplayEnvironmentBlocked) as exc_info:
        replay.build_machine_receipts(tmp_path, "1" * 40)

    assert exc_info.value.failures == ("dependency_missing:pyarrow",)
    assert calls == ["preflight"]


@pytest.mark.integration
def test_verifier_reports_replay_environment_blocked_before_comparison(
    monkeypatch: pytest.MonkeyPatch,
    full_valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = load_handoff_script_module("verify_handoff_package")
    package, sidecar = full_valid_handoff_package
    package_payload = verifier.load_handoff_package_payload(package, sidecar)
    assert package_payload.raw_bundle is not None
    assert package_payload.task_metadata is not None

    def fake_build_machine_receipts(*_args: object, **_kwargs: object) -> None:
        raise verifier.ReplayEnvironmentBlocked(("dependency_missing:pyarrow",))

    monkeypatch.setattr(verifier, "build_machine_receipts", fake_build_machine_receipts)

    failures, share_safety_failures = verifier.verify_bundle_and_replay_once(
        package_payload.raw_bundle,
        package_payload.task_metadata,
        package_payload,
    )

    assert failures == ["replay_environment_blocked:dependency_missing:pyarrow"]
    assert share_safety_failures == []


def test_builder_reports_replay_environment_blocked_without_partial_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    init_handoff_test_repo(repo)
    builder = load_handoff_script_module("build_handoff_package")
    out_dir = tmp_path / "out"

    def fake_build_machine_receipts(*_args: object, **_kwargs: object) -> None:
        raise builder.ReplayEnvironmentBlocked(("dependency_missing:pyarrow",))

    monkeypatch.setattr(builder, "build_machine_receipts", fake_build_machine_receipts)

    result = builder.main(
        [
            "--repo-root",
            str(repo),
            "--out-dir",
            str(out_dir),
            "--task-name",
            "handoff-test",
            "--base-ref",
            "origin/main",
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert "replay_environment_blocked:dependency_missing:pyarrow" in captured.out
    assert not (out_dir / "handoff-test.zip").exists()
    assert not (out_dir / "handoff-test.json").exists()


def _installed_inside(spec: str) -> str:
    constraint = dependency_constraint(spec)
    match = constraint.split(",", 1)[0].removeprefix(">=")
    return match + ".0" if match.count(".") == 1 else match
