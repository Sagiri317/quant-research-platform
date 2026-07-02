from __future__ import annotations

import shlex

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
