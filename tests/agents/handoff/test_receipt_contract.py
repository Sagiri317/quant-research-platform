from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import _load_handoff_module
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import _load_handoff_module  # type: ignore[no-redef]


def _valid_counts(**overrides: int) -> dict[str, int]:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 1,
    }
    counts.update(overrides)
    return counts


def _valid_row(command_id: str = "profile-smoke", **overrides: object) -> dict[str, object]:
    handoff_contract = _load_handoff_module("handoff_contract")
    spec = handoff_contract.replay_command_spec(command_id)
    row: dict[str, object] = {
        "schema_id": "qrp-check-receipt",
        "command_id": command_id,
        "scope": spec.scope,
        "command_argv": list(spec.command_argv),
        "command": spec.command,
        "status": "PASS",
        "exit_code": 0,
        "counts": _valid_counts() if command_id == "profile-smoke" else None,
    }
    row.update(overrides)
    return row


def _check_receipts(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [_valid_row("profile-smoke"), _valid_row("code-map")],
    }
    payload.update(overrides)
    return payload


def _package_smoke_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "build": {"status": "PASS", "artifact_kinds": ["wheel", "sdist"]},
        "install": {
            "status": "PASS",
            "target_kind": "temporary_no_deps_install",
            "no_deps": True,
        },
        "import": {"status": "PASS", "module": "quant_research_platform"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
        },
    }
    payload.update(overrides)
    return payload


def _hard_scan_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-hard-scan-receipt",
        "status": "PASS",
        "scanned_head": "1" * 40,
        "scanner_source_sha256": "2" * 64,
        "scan_runs": [
            {
                "scan_id": "allowance_contract",
                "rule_set_id": "source-tree:allowance_contract",
                "match_count": 0,
                "actual_violation_count": 0,
                "result_sha256": "3" * 64,
                "replayable_from_bundle": True,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _failures(row: dict[str, object]) -> list[str]:
    contract = _load_handoff_module("receipt_contract")
    return contract.validate_check_receipt_row(row)


def test_check_receipt_rejects_missing_schema_id() -> None:
    row = _valid_row()
    row.pop("schema_id")

    assert "check_receipt_missing: 0:schema_id" in _failures(row)


def test_check_receipt_rejects_extra_field() -> None:
    assert "check_receipt_unapproved_field: 0:extra_field" in _failures(
        _valid_row(extra_field=True)
    )


def test_check_receipt_rejects_pass_with_nonzero_exit_code() -> None:
    assert "check_receipt_pass_exit_code_invalid: 0" in _failures(
        _valid_row(exit_code=1)
    )


def test_check_receipt_rejects_fail_with_zero_exit_code() -> None:
    assert "check_receipt_fail_exit_code_invalid: 0" in _failures(
        _valid_row(status="FAIL", exit_code=0)
    )


def test_check_receipt_rejects_pass_with_failed_or_error_count() -> None:
    assert "check_receipt_pass_counts_invalid: 0" in _failures(
        _valid_row(counts=_valid_counts(failed=1, collected=2))
    )
    assert "check_receipt_pass_counts_invalid: 0" in _failures(
        _valid_row(counts=_valid_counts(errors=1, collected=2))
    )


def test_check_receipt_rejects_not_run_or_blocked_rows() -> None:
    assert "check_receipt_status_invalid: 0" in _failures(
        _valid_row(status="NOT_RUN", exit_code=None)
    )
    assert "check_receipt_status_invalid: 0" in _failures(
        _valid_row(status="BLOCKED", exit_code=None)
    )


def test_check_receipt_rejects_old_trace_fields() -> None:
    failures = _failures(
        _valid_row(
            duration_seconds=1.0,
            stdout_sha256="0" * 64,
            stderr_sha256="1" * 64,
            output_summary="old",
            failure_excerpt="old",
            not_run_reason="old",
            blocked_reason="old",
        )
    )

    for field in (
        "duration_seconds",
        "stdout_sha256",
        "stderr_sha256",
        "output_summary",
        "failure_excerpt",
        "not_run_reason",
        "blocked_reason",
    ):
        assert f"check_receipt_unapproved_field: 0:{field}" in failures


def test_check_receipts_rejects_missing_or_extra_command() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _check_receipts(checks=[_valid_row("profile-smoke")])

    failures = contract.validate_check_receipts(payload)

    assert "check_receipts_missing_command: code-map" in failures
    extra = _valid_row("profile-smoke")
    extra["command_id"] = "extra"
    payload = _check_receipts(
        checks=[_valid_row("profile-smoke"), _valid_row("code-map"), extra]
    )
    failures = contract.validate_check_receipts(payload)
    assert "check_receipts_unapproved_command: extra" in failures


def test_check_receipt_rejects_command_display_or_argv_mismatch() -> None:
    assert "check_receipt_command_mismatch: 0" in _failures(
        _valid_row(command="python -m wrong")
    )
    row = _valid_row(command_argv=["python", "-c", "print('hello world')"])
    row["command"] = shlex.join(row["command_argv"])
    failures = _failures(row)
    assert "check_receipt_command_mismatch: 0" not in failures
    assert "check_receipt_command_argv_mismatch: profile-smoke" in failures


def test_package_smoke_rejects_child_fail_with_top_pass() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(
        install={
            "status": "FAIL",
            "target_kind": "temporary_no_deps_install",
            "no_deps": True,
        }
    )

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_status_mismatch" in failures


def test_package_smoke_rejects_old_artifact_and_example_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(
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
    contract = _load_handoff_module("receipt_contract")
    payload = _package_smoke_payload(
        build={"status": "PASS", "artifact_kinds": ["wheel"]},
        install={
            "status": "PASS",
            "target_kind": "temporary isolated target",
            "no_deps": True,
        },
    )
    payload["import"] = {"status": "PASS", "module": "other"}

    failures = contract.validate_package_smoke_receipt(payload)

    assert "package_smoke_artifact_kinds_invalid" in failures
    assert "package_smoke_install_target_kind_invalid" in failures
    assert "package_smoke_import_module_invalid" in failures


def test_hard_scan_rejects_old_classification_and_count_fields() -> None:
    contract = _load_handoff_module("receipt_contract")
    run = {
        **_hard_scan_payload()["scan_runs"][0],  # type: ignore[index]
        "deferred_count": 0,
        "negative_fixture_count": 0,
        "false_positive_count": 0,
    }
    payload = _hard_scan_payload(
        scan_runs=[run],
        classified_exceptions=[],
    )

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_unapproved_field: classified_exceptions" in failures
    assert "hard_scan_run_unapproved_field: 0:deferred_count" in failures
    assert "hard_scan_run_unapproved_field: 0:negative_fixture_count" in failures
    assert "hard_scan_run_unapproved_field: 0:false_positive_count" in failures


@pytest.mark.parametrize(
    ("payload_overrides", "run_overrides", "expected"),
    (
        (
            {"classified_exceptions": []},
            {},
            "hard_scan_unapproved_field: classified_exceptions",
        ),
        ({"extra": True}, {}, "hard_scan_unapproved_field: extra"),
        ({}, {"deferred_count": 0}, "hard_scan_run_unapproved_field: 0:deferred_count"),
        (
            {},
            {"negative_fixture_count": 0},
            "hard_scan_run_unapproved_field: 0:negative_fixture_count",
        ),
        (
            {},
            {"false_positive_count": 0},
            "hard_scan_run_unapproved_field: 0:false_positive_count",
        ),
        ({}, {"extra": True}, "hard_scan_run_unapproved_field: 0:extra"),
    ),
)
def test_normalize_hard_scan_rejects_raw_extra_fields_before_projection(
    payload_overrides: dict[str, object],
    run_overrides: dict[str, object],
    expected: str,
) -> None:
    contract = _load_handoff_module("receipt_contract")
    run = {
        **_hard_scan_payload()["scan_runs"][0],  # type: ignore[index]
        **run_overrides,
    }
    payload = _hard_scan_payload(scan_runs=[run], **payload_overrides)

    with pytest.raises(SystemExit) as excinfo:
        contract.normalize_hard_scan_receipt(payload)

    assert expected in str(excinfo.value)


def test_hard_scan_rejects_actual_violation_with_top_pass() -> None:
    contract = _load_handoff_module("receipt_contract")
    run = {
        **_hard_scan_payload()["scan_runs"][0],  # type: ignore[index]
        "actual_violation_count": 1,
    }
    payload = _hard_scan_payload(scan_runs=[run])

    failures = contract.validate_hard_scan_receipt(payload)

    assert "hard_scan_status_mismatch" in failures


def test_compare_functions_detect_replay_drift() -> None:
    contract = _load_handoff_module("receipt_contract")
    packaged = _check_receipts()
    replayed = _check_receipts()
    replayed["checks"][0]["exit_code"] = 1  # type: ignore[index]
    replayed["checks"][0]["status"] = "FAIL"  # type: ignore[index]
    replayed["status"] = "FAIL"

    assert contract.compare_check_receipts(packaged, replayed) == [
        "check_receipts_replay_mismatch"
    ]


@pytest.mark.parametrize(
    ("collected", "expected"),
    (
        (0, "check_receipt_counts_collected_mismatch: 0"),
        (2, "check_receipt_counts_collected_mismatch: 0"),
    ),
)
def test_profile_smoke_counts_collected_must_equal_outcome_total(
    collected: int,
    expected: str,
) -> None:
    failures = _failures(_valid_row(counts=_valid_counts(collected=collected)))

    assert expected in failures


def test_compare_functions_detect_pytest_count_drift() -> None:
    contract = _load_handoff_module("receipt_contract")
    packaged = _check_receipts()
    replayed = _check_receipts()
    replayed["checks"][0]["counts"]["passed"] += 1  # type: ignore[index]
    replayed["checks"][0]["counts"]["collected"] += 1  # type: ignore[index]

    assert contract.compare_check_receipts(packaged, replayed) == [
        "check_receipts_replay_mismatch"
    ]


def test_parse_pytest_counts_counts_deselected_as_collected() -> None:
    replay = _load_handoff_module("replay_evidence")

    counts = replay.parse_pytest_counts("35 passed, 33 deselected in 0.86s")

    assert counts["passed"] == 35
    assert counts["deselected"] == 33
    assert counts["collected"] == 68


def test_replay_command_executes_current_interpreter_but_records_canonical_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _load_handoff_module("replay_evidence")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "1 passed in 0.01s", "")

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    row = replay._run_replay_command(Path("."), "profile-smoke")

    assert calls
    assert calls[0][0] == sys.executable
    assert row["command_argv"][0] == "python"
    assert str(row["command"]).startswith("python ")


@pytest.mark.parametrize(
    ("artifacts", "expected_status"),
    (
        (("demo-0.0.0.tar.gz",), "FAIL"),
        (("demo-0.0.0-py3-none-any.whl", "demo-0.0.1-py3-none-any.whl", "demo-0.0.0.tar.gz"), "FAIL"),
        (("demo-0.0.0-py3-none-any.whl",), "FAIL"),
        (("demo-0.0.0-py3-none-any.whl", "demo-0.0.0.tar.gz", "demo-0.0.1.tar.gz"), "FAIL"),
        (("demo-0.0.0-py3-none-any.whl", "demo-0.0.0.tar.gz"), "PASS"),
    ),
)
def test_package_smoke_build_requires_exactly_one_wheel_and_sdist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifacts: tuple[str, ...],
    expected_status: str,
) -> None:
    replay = _load_handoff_module("replay_evidence")

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        dist = Path(argv[-1])
        dist.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            (dist / artifact).write_text("artifact", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(replay.subprocess, "run", fake_run)

    build_status, _wheel_path, _sdist_path = replay._build_package_artifacts(
        tmp_path,
        tmp_path / "dist",
    )

    assert build_status == expected_status
