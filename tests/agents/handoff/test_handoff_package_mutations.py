from __future__ import annotations

import json
from pathlib import Path

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
