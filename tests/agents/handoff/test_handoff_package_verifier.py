from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import (
        _copy_valid_package,
        _load_handoff_module,
        _refresh_manifest_and_sidecar,
        _rewrite_package_member,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _copy_valid_package,
        _load_handoff_module,
        _refresh_manifest_and_sidecar,
        _rewrite_package_member,
    )

@pytest.mark.integration
def test_verifier_accepts_minimal_package(
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = valid_handoff_package

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert failures == ()


@pytest.mark.integration
def test_verifier_rejects_self_consistent_failed_validation_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        check_receipts = json.loads(
            archive.read("handoff_evidence/check-receipts.json").decode("utf-8")
        )
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
    profile_smoke = next(
        row for row in check_receipts["checks"] if row["command_id"] == "profile-smoke"
    )
    profile_smoke["status"] = "FAIL"
    profile_smoke["exit_code"] = 1
    profile_smoke["failure_class"] = "pytest_failed"
    check_receipts["status"] = "FAIL"
    for row in validation_summary["rows"]:
        if row["name"] in {"scripted_checks", "profile_smoke"}:
            row["status"] = "FAIL"
    validation_summary["status"] = "FAIL"
    _rewrite_package_member(
        package,
        "handoff_evidence/check-receipts.json",
        (json.dumps(check_receipts, indent=2, sort_keys=True) + "\n").encode(),
    )
    _rewrite_package_member(
        package,
        "handoff_evidence/validation-summary.json",
        (json.dumps(validation_summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)
    monkeypatch.setattr(verifier, "verify_bundle_and_replay_once", lambda *_: ([], []))

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert "validation_summary_row_not_pass:scripted_checks:FAIL" in failures
    assert "validation_summary_row_not_pass:profile_smoke:FAIL" in failures


@pytest.mark.integration
def test_verifier_rejects_self_consistent_blocked_package_smoke_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        package_smoke = json.loads(
            archive.read("handoff_evidence/package-smoke-receipt.json").decode("utf-8")
        )
        validation_summary = json.loads(
            archive.read("handoff_evidence/validation-summary.json").decode("utf-8")
        )
    package_smoke["status"] = "BLOCKED"
    package_smoke["build"] = {"status": "BLOCKED", "artifact_kinds": []}
    package_smoke["wheel_extract"] = {
        "status": "NOT_RUN",
        "method": "zipfile_extract",
    }
    package_smoke["import"] = {
        "status": "NOT_RUN",
        "module": "quant_research_platform",
    }
    package_smoke["content_scan"] = {
        "status": "NOT_RUN",
        "forbidden_member_count": 0,
        "scanned_artifact_kinds": [],
    }
    for row in validation_summary["rows"]:
        if row["name"] == "package_smoke":
            row["status"] = "BLOCKED"
    validation_summary["status"] = "BLOCKED"
    _rewrite_package_member(
        package,
        "handoff_evidence/package-smoke-receipt.json",
        (json.dumps(package_smoke, indent=2, sort_keys=True) + "\n").encode(),
    )
    _rewrite_package_member(
        package,
        "handoff_evidence/validation-summary.json",
        (json.dumps(validation_summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)
    monkeypatch.setattr(verifier, "verify_bundle_and_replay_once", lambda *_: ([], []))

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert "validation_summary_row_not_pass:package_smoke:BLOCKED" in failures


def test_receipt_contract_rejects_status_only_package_smoke_receipt() -> None:
    contract = _load_handoff_module("receipt_contract")

    failures = contract.validate_receipt_payload(
        "package-smoke-receipt.json",
        {"schema_id": "qrp-package-smoke-receipt", "status": "PASS"},
    )

    assert "package_smoke_missing: build" in failures
    assert "package_smoke_missing: content_scan" in failures
    assert "package_smoke_missing: wheel_extract" in failures


def test_receipt_contract_rejects_placeholder_check_receipt() -> None:
    contract = _load_handoff_module("receipt_contract")

    failures = contract.validate_receipt_payload(
        "check-receipts.json",
        {
            "schema_id": "qrp-check-receipts",
            "status": "PASS",
            "checks": [
                {
                    "schema_id": "qrp-check-receipt",
                    "command_id": "placeholder",
                    "scope": "tests",
                    "command": "pytest",
                    "command_argv": ["pytest"],
                    "status": "PASS",
                    "exit_code": 0,
                    "counts": None,
                }
            ],
        },
    )

    assert "check_receipts_missing_command: profile-smoke" in failures
    assert "check_receipts_missing_command: code-map" in failures
    assert "check_receipts_unapproved_command: placeholder" in failures


def test_receipt_contract_rejects_raw_output_in_hard_scan_receipt() -> None:
    contract = _load_handoff_module("receipt_contract")
    payload = _hard_scan_receipt()
    payload["raw_output"] = "raw output"

    failures = contract.validate_receipt_payload("hard-scan-receipt.json", payload)

    assert "receipt_forbidden_raw_field: hard-scan-receipt.json:raw_output" in failures


@pytest.mark.parametrize(
    "old_name",
    (
        "git_status.txt",
        "receipt-index.json",
        "handoff_evidence/generation.json",
        "handoff_evidence/bundle-reconstruction.json",
        "handoff_evidence/private-smoke-status.json",
        "handoff_evidence/private-validation-status.json",
        "handoff_evidence/advisory-review-receipt.json",
    ),
)
@pytest.mark.integration
def test_verifier_rejects_retired_handoff_entries(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
    old_name: str,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _rewrite_package_member(
        package,
        old_name,
        b'{"schema_id":"qrp-old-receipt","status":"PASS"}\n',
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert any(old_name in failure for failure in failures)


@pytest.mark.integration
def test_verifier_rejects_old_sidecar_status_surface(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status_summary"] = {"smoke": "PASS"}
    payload["advisory_reviews"] = {"review_subagent": "PASS"}
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert "sidecar_unapproved_field: status_summary" in failures
    assert "sidecar_unapproved_field: advisory_reviews" in failures


@pytest.mark.integration
def test_verifier_replays_hard_scan_from_bundle(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    with zipfile.ZipFile(package) as archive:
        receipt = json.loads(
            archive.read("handoff_evidence/hard-scan-receipt.json").decode("utf-8")
        )
    receipt["scan_runs"][0]["result_sha256"] = "f" * 64
    _rewrite_package_member(
        package,
        "handoff_evidence/hard-scan-receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    _refresh_manifest_and_sidecar(package, sidecar)

    failures = verifier.verify_handoff_package_result(package, sidecar).failures

    assert "hard_scan_replay_mismatch" in failures


@pytest.mark.integration
def test_verifier_reuses_clone_source_tree_scan_for_share_safety_and_hard_scan(
    monkeypatch: pytest.MonkeyPatch,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = valid_handoff_package
    inspection = verifier.inspect_handoff_zip_once(package, sidecar)
    assert inspection.raw_bundle is not None
    assert inspection.loaded.task_metadata is not None
    calls = 0
    scan_payload = {
        "allowance_contract": {
            "findings": [],
        }
    }
    received_payloads: list[object] = []

    def fake_source_tree_scan_groups(_repo_root: Path) -> dict[str, dict[str, object]]:
        nonlocal calls
        calls += 1
        return scan_payload

    def fake_build_machine_receipts(
        _repo_root: Path,
        _final_head: str,
        *,
        source_tree_scan_groups_payload: object | None = None,
    ) -> dict[str, dict[str, object]]:
        received_payloads.append(source_tree_scan_groups_payload)
        return inspection.loaded.receipts

    monkeypatch.setattr(
        verifier,
        "source_tree_scan_groups",
        fake_source_tree_scan_groups,
    )
    monkeypatch.setattr(verifier, "build_machine_receipts", fake_build_machine_receipts)

    failures, share_safety_failures = verifier.verify_bundle_and_replay_once(
        inspection.raw_bundle,
        inspection.loaded.task_metadata,
        inspection.loaded,
    )

    assert failures == []
    assert share_safety_failures == []
    assert calls == 1
    assert received_payloads == [scan_payload]


def _hard_scan_receipt() -> dict[str, object]:
    return {
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
