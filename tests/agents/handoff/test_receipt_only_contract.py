from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

try:
    from tests.agents.handoff.conftest import (
        _build_test_package,
        _load_handoff_module,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        _build_test_package,
        _load_handoff_module,
    )


def test_run_check_writes_structured_pytest_counts(tmp_path: Path) -> None:
    receipt_out = tmp_path / "check-receipts.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/quant-research-platform-maintenance/scripts/run_check.py",
            "--id",
            "synthetic-pytest",
            "--scope",
            "unit",
            "--receipt-out",
            str(receipt_out),
            "--",
            sys.executable,
            "-c",
            "print('2 passed, 1 skipped, 3 deselected in 0.01s')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    receipt = json.loads(receipt_out.read_text(encoding="utf-8").splitlines()[0])
    assert result.returncode == 0
    assert "test_count" not in receipt
    assert receipt["counts"] == {
        "passed": 2,
        "failed": 0,
        "skipped": 1,
        "deselected": 3,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "collected": 0,
    }


def test_package_smoke_receipt_requires_all_scanned_artifact_kinds() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    payload = {
        "schema_id": "qrp-package-smoke-receipt",
        "status": "PASS",
        "wheel": {"filename": "demo.whl", "sha256": "0" * 64},
        "sdist": {"filename": "demo.tar.gz", "sha256": "1" * 64},
        "install": {
            "target_kind": "temporary isolated target",
            "no_deps": True,
            "status": "PASS",
        },
        "import": {"status": "PASS"},
        "content_scan": {
            "status": "PASS",
            "forbidden_member_count": 0,
        },
    }

    failures = verifier.verify_package_smoke_evidence(payload)

    assert "package_smoke_scanned_artifact_kinds_invalid" in failures


def test_package_smoke_scans_wheel_for_forbidden_members(tmp_path: Path) -> None:
    package_smoke = _load_handoff_module("run_package_smoke")
    wheel = tmp_path / "demo.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("quant_research_platform/private/raw.csv", "forbidden\n")

    findings = package_smoke.archive_content_scan(wheel, "wheel")

    assert findings == ["wheel:quant_research_platform/private/raw.csv"]


def test_package_smoke_scans_sdist_for_forbidden_members(tmp_path: Path) -> None:
    package_smoke = _load_handoff_module("run_package_smoke")
    sdist = tmp_path / "demo.tar.gz"
    payload = b"forbidden\n"
    info = tarfile.TarInfo("demo/private/raw.parquet")
    info.size = len(payload)
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    findings = package_smoke.archive_content_scan(sdist, "sdist")

    assert findings == ["sdist:demo/private/raw.parquet"]


def test_builder_outputs_receipt_only_handoff_evidence(tmp_path: Path) -> None:
    _repo, package, _sidecar, _base, _final_head = _build_test_package(tmp_path)

    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())

    assert "validation_evidence.json" not in names
    assert "handoff_evidence/source-tree-scan.json" not in names
    assert "handoff_evidence/verification-matrix.json" not in names
    assert "handoff_evidence/validation-summary.json" in names
    assert "handoff_evidence/package-smoke-receipt.json" in names
    assert "handoff_evidence/hard-scan-receipt.json" in names
    assert "handoff_evidence/check-receipts.json" in names
    assert "handoff_evidence/private-smoke-status.json" in names


def test_sidecar_rejects_generic_smoke_pass_status(tmp_path: Path) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    _repo, package, sidecar, _base, _final_head = _build_test_package(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status_summary"]["smoke"] = "PASS"
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    failures = verifier.verify_handoff_package(package, sidecar)

    assert "sidecar_status_summary_forbidden: smoke" in failures


def test_receipt_collector_exposes_dedicated_validators() -> None:
    collector = _load_handoff_module("collect_handoff_receipts")

    for name in (
        "validate_package_smoke_receipt",
        "validate_hard_scan_receipt",
        "validate_check_receipts",
        "validate_private_smoke_status",
        "validate_advisory_review_receipt",
    ):
        assert callable(getattr(collector, name))
