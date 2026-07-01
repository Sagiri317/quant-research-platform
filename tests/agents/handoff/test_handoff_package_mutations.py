from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

try:
    from tests.agents.handoff.conftest import (
        MANDATORY_VALIDATION_CHECKS,
        _copy_valid_package,
        _generation_evidence,
        _load_handoff_module,
        _replace_generation_evidence,
        _validation_checks,
        _validation_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - addopts='' import mode fallback
    from conftest import (  # type: ignore[no-redef]
        MANDATORY_VALIDATION_CHECKS,
        _copy_valid_package,
        _generation_evidence,
        _load_handoff_module,
        _replace_generation_evidence,
        _validation_checks,
        _validation_payload,
    )

pytestmark = pytest.mark.mutation


def test_handoff_package_rejects_missing_mandatory_validation_check() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    missing_check = MANDATORY_VALIDATION_CHECKS[0]
    checks = [
        row for row in _validation_checks("PASS") if row["name"] != missing_check
    ]

    failures = verifier.verify_validation_summary_schema(_validation_payload(checks))

    assert f"validation_summary_mandatory_check_missing: {missing_check}" in failures


def test_handoff_package_rejects_duplicate_validation_check() -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    checks = [
        *_validation_checks("PASS"),
        {"name": "tests", "status": "FAIL", "evidence": "first failure"},
        {"name": "tests", "status": "PASS", "evidence": "forged later pass"},
    ]

    failures = verifier.verify_validation_summary_schema(_validation_payload(checks))

    assert "validation_summary_duplicate_check: tests" in failures


def test_handoff_package_rejects_generation_validation_status_mismatch(
    tmp_path: Path,
    valid_handoff_package: tuple[Path, Path],
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")
    package, sidecar = _copy_valid_package(valid_handoff_package, tmp_path)
    generation = _generation_evidence(package)
    generation["validation_status"] = "PASS"
    _replace_generation_evidence(package, sidecar, generation)

    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    with zipfile.ZipFile(package) as archive:
        failures = verifier.verify_generation_evidence(
            generation,
            sidecar_payload,
            archive,
        )

    assert "generation_validation_status_sidecar_mismatch" in failures
