from __future__ import annotations

from pathlib import Path

import pytest

from tests.agents.handoff.receipt_contract_fixtures import (
    make_package_build_subprocess_run,
    make_package_smoke_receipt_payload,
    make_timeout_subprocess_run,
)
from tests.helpers.maintenance_script_loader import load_handoff_script_module


def test_package_smoke_rejects_child_fail_with_top_pass() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload(
        wheel_extract={
            "status": "FAIL",
            "method": "zipfile_extract",
        }
    )

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_status_mismatch" in failures

@pytest.mark.parametrize(
    "payload_overrides",
    (
        {
            "status": "FAIL",
            "build": {"status": "FAIL", "artifact_kinds": []},
            "wheel_extract": {"status": "NOT_RUN", "method": "zipfile_extract"},
            "content_scan": {
                "status": "NOT_RUN",
                "forbidden_member_count": 0,
                "scanned_artifact_kinds": [],
            },
        },
        {
            "status": "FAIL",
            "wheel_extract": {"status": "FAIL", "method": "zipfile_extract"},
            "content_scan": {
                "status": "NOT_RUN",
                "forbidden_member_count": 0,
                "scanned_artifact_kinds": [],
            },
        },
    ),
)
def test_package_smoke_accepts_failed_step_with_content_scan_not_run(
    payload_overrides: dict[str, object],
) -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload(**payload_overrides)
    payload["import"] = {"status": "NOT_RUN", "module": "quant_research_platform"}

    assert contract.validate_package_smoke_receipt(payload) == []

def test_package_smoke_rejects_build_fail_with_claimed_content_scan_pass() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload(
        status="FAIL",
        build={"status": "FAIL", "artifact_kinds": []},
        wheel_extract={"status": "NOT_RUN", "method": "zipfile_extract"},
    )
    payload["import"] = {"status": "NOT_RUN", "module": "quant_research_platform"}

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_content_scan_not_run_required" in failures

def test_package_smoke_rejects_old_artifact_and_example_fields() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload(
        wheel={"filename": "demo.whl", "sha256": "0" * 64},
        sdist={"filename": "demo.tar.gz", "sha256": "1" * 64},
    )
    payload["content_scan"] = {
        **payload["content_scan"],  # type: ignore[arg-type]
        "examples": [],
    }

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_unapproved_field: wheel" in failures
    assert "package_smoke_unapproved_field: sdist" in failures
    assert "package_smoke_content_scan_unapproved_field: examples" in failures

def test_package_smoke_rejects_fixed_value_drift() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload(
        build={"status": "PASS", "artifact_kinds": ["wheel"]},
        wheel_extract={
            "status": "PASS",
            "method": "copytree",
        },
    )
    payload["import"] = {"status": "PASS", "module": "other"}

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_artifact_kinds_invalid" in failures
    assert "package_smoke_wheel_extract_method_invalid" in failures
    assert "package_smoke_import_module_invalid" in failures


def test_package_smoke_requires_exact_lists_and_integer_count() -> None:
    contract = load_handoff_script_module("receipt_contract")
    payload = make_package_smoke_receipt_payload()
    payload["build"]["artifact_kinds"] = ("wheel", "sdist")
    payload["content_scan"]["scanned_artifact_kinds"] = (
        "wheel",
        "sdist",
        "wheel_extract",
    )
    payload["content_scan"]["forbidden_member_count"] = False

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_artifact_kinds_invalid" in failures
    assert "package_smoke_scanned_artifact_kinds_invalid" in failures
    assert "package_smoke_forbidden_member_count_invalid" in failures

@pytest.mark.parametrize(
    ("artifacts", "expected_status"),
    (
        (("demo-0.0.0.tar.gz",), "FAIL"),
        (
            (
                "demo-0.0.0-py3-none-any.whl",
                "demo-0.0.1-py3-none-any.whl",
                "demo-0.0.0.tar.gz",
            ),
            "FAIL",
        ),
        (("demo-0.0.0-py3-none-any.whl",), "FAIL"),
        (
            ("demo-0.0.0-py3-none-any.whl", "demo-0.0.0.tar.gz", "demo-0.0.1.tar.gz"),
            "FAIL",
        ),
        (("demo-0.0.0-py3-none-any.whl", "demo-0.0.0.tar.gz"), "PASS"),
    ),
)
def test_package_smoke_build_requires_exactly_one_wheel_and_sdist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts: tuple[str, ...],
    expected_status: str,
) -> None:
    replay = load_handoff_script_module("replay_evidence")

    monkeypatch.setattr(
        replay.subprocess,
        "run",
        make_package_build_subprocess_run(artifacts),
    )

    build_status, _wheel_path, _sdist_path = replay._build_package_artifacts(
        tmp_path,
        tmp_path / "dist",
    )

    assert build_status == expected_status

def test_package_smoke_build_uses_no_isolation_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    calls: list[tuple[list[str], object]] = []

    monkeypatch.setattr(
        replay.subprocess,
        "run",
        make_package_build_subprocess_run(
            ("demo-0.0.0-py3-none-any.whl", "demo-0.0.0.tar.gz"),
            calls,
        ),
    )

    build_status, _wheel_path, _sdist_path = replay._build_package_artifacts(
        tmp_path,
        tmp_path / "dist",
    )

    assert build_status == "PASS"
    assert calls
    argv, timeout = calls[0]
    assert "--no-isolation" in argv
    assert "--installer" not in argv
    assert "--skip-dependency-check" not in argv
    assert timeout == replay.PACKAGE_BUILD_TIMEOUT_SECONDS

def test_package_smoke_build_timeout_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")

    monkeypatch.setattr(replay.subprocess, "run", make_timeout_subprocess_run())

    build_status, wheel_path, sdist_path = replay._build_package_artifacts(
        tmp_path,
        tmp_path / "dist",
    )

    assert build_status == "BLOCKED"
    assert wheel_path is None
    assert sdist_path is None

def test_package_smoke_import_timeout_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")
    wheel_path = tmp_path / "demo-0.0.0-py3-none-any.whl"
    import zipfile

    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("quant_research_platform/__init__.py", "")

    monkeypatch.setattr(replay.subprocess, "run", make_timeout_subprocess_run())

    extract_status, import_status = replay._extract_wheel_smoke_target(
        tmp_path,
        wheel_path,
        tmp_path / "wheel-extract",
    )

    assert extract_status == "PASS"
    assert import_status == "BLOCKED"

def test_package_smoke_build_fail_keeps_content_scan_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = load_handoff_script_module("replay_evidence")

    def fake_build(*_: object) -> tuple[str, None, None]:
        return "FAIL", None, None

    monkeypatch.setattr(replay, "_build_package_artifacts", fake_build)

    receipt = replay._build_replayed_package_smoke_receipt(tmp_path)

    assert receipt["status"] == "FAIL"
    assert receipt["content_scan"] == {
        "status": "NOT_RUN",
        "forbidden_member_count": 0,
        "scanned_artifact_kinds": [],
    }
