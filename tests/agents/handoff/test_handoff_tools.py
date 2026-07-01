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


def test_verifier_accepts_receipt_only_package(
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = valid_handoff_package

    failures = verifier.verify_handoff_package(package, sidecar)

    assert failures == []


def test_verifier_rejects_status_only_package_smoke_receipt() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_package_smoke_evidence(
        {"schema_id": "qrp-package-smoke-receipt", "status": "PASS"}
    )

    assert "package_smoke_missing: wheel" in failures
    assert "package_smoke_missing: content_scan" in failures


def test_verifier_rejects_placeholder_check_receipt() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_test_checks_evidence(
        {
            "schema_id": "qrp-check-receipts",
            "status": "PASS",
            "checks": [
                {
                    "schema_id": "qrp-check-receipt",
                    "command_id": "placeholder",
                    "scope": "tests",
                    "command": "pytest",
                    "status": "PASS",
                    "exit_code": 0,
                    "duration_seconds": 0.0,
                    "stdout_sha256": "0" * 64,
                    "stderr_sha256": "1" * 64,
                    "output_summary": "",
                    "failure_excerpt": "",
                    "counts": None,
                }
            ],
        }
    )

    assert "test_checks_row_output_summary_invalid: 0" in failures


def test_verifier_rejects_raw_output_in_hard_scan_receipt() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = _hard_scan_receipt()
    payload["raw_output"] = "raw rg output"

    failures = verifier.verify_hard_scan_classification_evidence(payload)

    assert "hard_scan_receipt_raw_output_present" in failures


@pytest.mark.parametrize(
    "old_name",
    (
        "handoff_evidence/package-smoke.json",
        "handoff_evidence/hard-scan-classification.json",
        "handoff_evidence/test-checks.json",
        "handoff_evidence/smoke-compare-status.json",
        "handoff_evidence/source-tree-scan.json",
        "handoff_evidence/verification-matrix.json",
    ),
)
def test_verifier_rejects_old_evidence_filenames(
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

    failures = verifier.verify_handoff_package(package, sidecar)

    assert f"old_handoff_evidence_entry: {old_name}" in failures


def test_verifier_rejects_generic_smoke_pass_in_sidecar(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status_summary"]["smoke"] = "PASS"
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "sidecar_status_summary_forbidden: smoke" in failures


def test_verifier_rejects_advisory_pass_without_receipt(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    _remove_member(package, "handoff_evidence/advisory-review-receipt.json")
    _refresh_manifest_and_sidecar(package, sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["advisory_reviews"] = {"review_subagent": "PASS"}
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "advisory_reviews_pass_without_advisory_receipt" in failures


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

    failures = verifier.verify_handoff_package(package, sidecar)

    assert any(
        failure.startswith("hard_scan_replay_result_sha256_mismatch")
        for failure in failures
    )


def test_builder_task_metadata_rejects_manual_evidence_keys() -> None:
    builder = _load_handoff_module("build_handoff_package")

    for manual_key in (
        "validation_evidence",
        "handoff_evidence",
        "handoff_markdown_evidence",
    ):
        with pytest.raises(SystemExit, match="task_metadata_forbidden_field"):
            builder.task_metadata(
                {
                    "schema_id": "qrp-handoff-task-metadata",
                    "mode": "evidence",
                    "slug": "demo",
                    "branch": "feature/demo",
                    "base": "0" * 40,
                    "final_head": "1" * 40,
                    manual_key: {},
                }
            )


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
                "deferred_count": 0,
                "negative_fixture_count": 0,
                "false_positive_count": 0,
                "result_sha256": "3" * 64,
                "replayable_from_bundle": True,
            }
        ],
        "classified_exceptions": [],
    }


def _remove_member(package: Path, member: str) -> None:
    temp = package.with_suffix(".remove.zip")
    with zipfile.ZipFile(package, "r") as src, zipfile.ZipFile(temp, "w") as dst:
        for info in src.infolist():
            if info.filename == member:
                continue
            dst.writestr(info, src.read(info.filename))
    temp.replace(package)
