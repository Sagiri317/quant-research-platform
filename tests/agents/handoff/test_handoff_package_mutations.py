from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

try:
    from tests.agents.handoff.conftest import (
        _copy_valid_package,
        _load_handoff_module,
        _refresh_manifest_and_sidecar,
        _replace_validation_summary,
        _rewrite_package_member,
        _validation_checks,
        _validation_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _copy_valid_package,
        _load_handoff_module,
        _refresh_manifest_and_sidecar,
        _replace_validation_summary,
        _rewrite_package_member,
        _validation_checks,
        _validation_payload,
    )

pytestmark = pytest.mark.mutation


def _verify(package: Path, sidecar: Path) -> list[str]:
    verifier = _load_handoff_module("verify_handoff_package")
    failures = verifier.verify_handoff_package(package, sidecar)
    assert "sidecar_zip_sha256_mismatch" not in failures
    assert "sidecar_artifact_manifest_sha256_mismatch" not in failures
    assert "artifact_manifest_member_set_mismatch" not in failures
    return failures


def _read_member_json(package: Path, member: str) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(package) as archive:
        payload = json.loads(archive.read(member).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_member_json(
    package: Path,
    sidecar: Path,
    member: str,
    payload: dict[str, Any],
) -> None:
    _rewrite_package_member(
        package,
        member,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)


def _check_receipts(package: Path) -> dict[str, Any]:
    return _read_member_json(package, "handoff_evidence/check-receipts.json")


def _replace_check_receipts(
    package: Path,
    sidecar: Path,
    payload: dict[str, Any],
) -> None:
    _write_member_json(package, sidecar, "handoff_evidence/check-receipts.json", payload)


def _package_smoke(package: Path) -> dict[str, Any]:
    return _read_member_json(package, "handoff_evidence/package-smoke-receipt.json")


def _replace_package_smoke(
    package: Path,
    sidecar: Path,
    payload: dict[str, Any],
) -> None:
    _write_member_json(
        package,
        sidecar,
        "handoff_evidence/package-smoke-receipt.json",
        payload,
    )


def _validation_summary(package: Path) -> dict[str, Any]:
    return _read_member_json(package, "handoff_evidence/validation-summary.json")


def _with_validation_row(
    package: Path,
    sidecar: Path,
    row_name: str,
    **updates: object,
) -> None:
    payload = _validation_summary(package)
    checks = payload["checks"]
    assert isinstance(checks, list)
    for row in checks:
        if isinstance(row, dict) and row.get("name") == row_name:
            row.update(updates)
            break
    else:
        raise AssertionError(f"row not found: {row_name}")
    _replace_validation_summary(package, sidecar, payload)


def _remove_check_row(
    package: Path,
    sidecar: Path,
    command_id: str,
) -> None:
    payload = _check_receipts(package)
    checks = payload["checks"]
    assert isinstance(checks, list)
    payload["checks"] = [
        row
        for row in checks
        if not isinstance(row, dict) or row.get("command_id") != command_id
    ]
    _replace_check_receipts(package, sidecar, payload)


def _set_sidecar_advisory_reviews(sidecar: Path, payload: dict[str, str]) -> None:
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["advisory_reviews"] = payload
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_handoff_package_rejects_retired_entry_resurrection(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _rewrite_package_member(
        package,
        "handoff_evidence/generation.json",
        b'{"schema_id":"retired"}\n',
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = _verify(package, sidecar)

    assert "forbidden_handoff_entry: handoff_evidence/generation.json" in failures


def test_handoff_package_rejects_sidecar_old_fields(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["handoff_package_path"] = "handoffs/old.zip"
    payload["handoff_sidecar_path"] = "handoffs/old.json"
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = _verify(package, sidecar)

    assert "sidecar_unapproved_field: handoff_package_path" in failures
    assert "sidecar_unapproved_field: handoff_sidecar_path" in failures


def test_handoff_package_rejects_validation_summary_tests_row(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _validation_payload(
        [
            *_validation_checks("PASS"),
            {"name": "tests", "status": "PASS", "evidence": "old row"},
        ]
    )
    _replace_validation_summary(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert any(
        failure.startswith("validation_summary_name_invalid") for failure in failures
    )


def test_handoff_package_rejects_receipt_missing_schema_id(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = {
        "status": "PASS",
        "wheel": {"filename": "demo.whl", "sha256": "0" * 64},
        "sdist": {"filename": "demo.tar.gz", "sha256": "1" * 64},
        "install": {"status": "PASS"},
        "import": {"status": "PASS"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
            "scanned_artifact_kinds": ["wheel", "sdist", "install_target"],
        },
    }
    _rewrite_package_member(
        package,
        "handoff_evidence/package-smoke-receipt.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = _verify(package, sidecar)

    assert "package_smoke_schema_id_invalid" in failures


def test_handoff_package_rejects_pass_with_failed_count(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = {
        "schema_id": "qrp-check-receipts",
        "status": "PASS",
        "checks": [
            {
                "schema_id": "qrp-check-receipt",
                "command_id": "unit",
                "scope": "unit",
                "command_argv": ["python", "-m", "pytest"],
                "command": "python -m pytest",
                "status": "PASS",
                "exit_code": 0,
                "duration_seconds": 1.0,
                "counts": {
                    "passed": 1,
                    "failed": 1,
                    "skipped": 0,
                    "deselected": 0,
                    "errors": 0,
                    "xfailed": 0,
                    "xpassed": 0,
                    "collected": 2,
                },
                "stdout_sha256": "0" * 64,
                "stderr_sha256": "1" * 64,
                "output_summary": "bad pass",
                "failure_excerpt": "",
            }
        ],
    }
    _rewrite_package_member(
        package,
        "handoff_evidence/check-receipts.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = _verify(package, sidecar)

    assert "check_receipt_pass_counts_invalid: 0" in failures


def test_handoff_package_rejects_missing_profile_smoke_receipt_edge(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_check_row(package, sidecar, "profile-smoke")

    failures = _verify(package, sidecar)

    assert "validation_summary_receipt_edge_missing: profile_smoke:profile-smoke" in failures


def test_handoff_package_rejects_missing_code_map_receipt_edge(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_check_row(package, sidecar, "code-map")

    failures = _verify(package, sidecar)

    assert "validation_summary_receipt_edge_missing: code_map:code-map" in failures


def test_handoff_package_rejects_empty_pass_check_receipts(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _check_receipts(package)
    payload["status"] = "PASS"
    payload["checks"] = []
    _replace_check_receipts(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert "check_receipts_empty_pass_invalid" in failures


def test_handoff_package_rejects_unknown_profile_smoke_command_ref(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _with_validation_row(
        package,
        sidecar,
        "profile_smoke",
        evidence="handoff_evidence/check-receipts.json#missing",
    )

    failures = _verify(package, sidecar)

    assert "validation_summary_receipt_edge_missing: profile_smoke:missing" in failures


def test_handoff_package_rejects_code_map_manual_evidence(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _with_validation_row(package, sidecar, "code_map", evidence="manual claim only")

    failures = _verify(package, sidecar)

    assert "validation_summary_evidence_syntax_invalid: code_map" in failures


def test_handoff_package_rejects_profile_status_mismatch_with_receipt(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _with_validation_row(package, sidecar, "profile_smoke", status="FAIL")

    failures = _verify(package, sidecar)

    assert "validation_summary_status_mismatch: profile_smoke:FAIL!=PASS" in failures


def test_handoff_package_rejects_private_validation_pass_without_command(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_check_row(package, sidecar, "private-validation")
    _with_validation_row(
        package,
        sidecar,
        "private_validation",
        status="PASS",
        evidence="handoff_evidence/check-receipts.json#private-validation",
    )

    failures = _verify(package, sidecar)

    assert (
        "validation_summary_receipt_edge_missing: "
        "private_validation:private-validation"
    ) in failures


def test_handoff_package_rejects_baseline_compare_pass_without_command(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_check_row(package, sidecar, "baseline-compare")
    _with_validation_row(
        package,
        sidecar,
        "baseline_compare",
        status="PASS",
        evidence="handoff_evidence/check-receipts.json#baseline-compare",
    )

    failures = _verify(package, sidecar)

    assert (
        "validation_summary_receipt_edge_missing: "
        "baseline_compare:baseline-compare"
    ) in failures


def test_handoff_package_rejects_package_install_fail_with_top_pass(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _package_smoke(package)
    payload["install"]["status"] = "FAIL"
    _replace_package_smoke(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert "package_smoke_status_mismatch" in failures


def test_handoff_package_rejects_package_smoke_status_mismatch_with_summary(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _package_smoke(package)
    payload["status"] = "FAIL"
    payload["install"]["status"] = "FAIL"
    _replace_package_smoke(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert "validation_summary_status_mismatch: package_smoke:PASS!=FAIL" in failures


def test_handoff_package_rejects_command_display_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _check_receipts(package)
    checks = payload["checks"]
    assert isinstance(checks, list)
    checks[0]["command"] = "python -m wrong"
    _replace_check_receipts(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert "check_receipt_command_mismatch: 0" in failures


def test_handoff_package_rejects_advisory_receipt_without_sidecar_review(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = {
        "schema_id": "qrp-advisory-review-receipt",
        "status": "PASS",
        "review_lanes": [],
        "finding_count": 0,
        "true_issue_count": 0,
        "false_positive_count": 0,
        "re_review_status": "PASS",
        "findings": [],
    }
    _write_member_json(
        package,
        sidecar,
        "handoff_evidence/advisory-review-receipt.json",
        payload,
    )
    _set_sidecar_advisory_reviews(sidecar, {})

    failures = _verify(package, sidecar)

    assert "advisory_reviews_mismatch" in failures


def test_handoff_package_rejects_sidecar_advisory_review_without_receipt(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _set_sidecar_advisory_reviews(sidecar, {"review_subagent": "PASS"})

    failures = _verify(package, sidecar)

    assert "advisory_reviews_mismatch" in failures


def test_handoff_package_rejects_task_metadata_sidecar_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _read_member_json(package, "task_metadata.json")
    payload["branch"] = "feature/other"
    _write_member_json(package, sidecar, "task_metadata.json", payload)

    failures = _verify(package, sidecar)

    assert "task_metadata_sidecar_mismatch: branch" in failures


def test_handoff_package_rejects_hard_scan_head_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _read_member_json(package, "handoff_evidence/hard-scan-receipt.json")
    payload["scanned_head"] = "f" * 40
    _write_member_json(
        package,
        sidecar,
        "handoff_evidence/hard-scan-receipt.json",
        payload,
    )

    failures = _verify(package, sidecar)

    assert "hard_scan_scanned_head_sidecar_mismatch" in failures


def test_handoff_package_rejects_duplicate_validation_summary_row(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = _validation_summary(package)
    checks = payload["checks"]
    assert isinstance(checks, list)
    checks.append(dict(checks[0]))
    _replace_validation_summary(package, sidecar, payload)

    failures = _verify(package, sidecar)

    assert "validation_summary_duplicate_check: scripted_checks" in failures


def test_handoff_package_rejects_pass_row_manual_evidence(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _with_validation_row(package, sidecar, "profile_smoke", evidence="manual claim only")

    failures = _verify(package, sidecar)

    assert "validation_summary_evidence_syntax_invalid: profile_smoke" in failures


def test_handoff_package_rejects_not_run_row_without_not_run_reason(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _with_validation_row(
        package,
        sidecar,
        "private_validation",
        evidence="private validation was not run",
    )

    failures = _verify(package, sidecar)

    assert "validation_summary_evidence_syntax_invalid: private_validation" in failures


def test_handoff_package_rejects_blocked_row_without_blocked_reason_or_receipt(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_check_row(package, sidecar, "private-validation")
    _with_validation_row(
        package,
        sidecar,
        "private_validation",
        status="BLOCKED",
        evidence="private validation blocked",
    )

    failures = _verify(package, sidecar)

    assert "validation_summary_evidence_syntax_invalid: private_validation" in failures
