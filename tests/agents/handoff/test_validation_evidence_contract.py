from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HANDOFF_TOOL_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "handoff"
)


def _load_handoff_module(name: str):
    script = HANDOFF_TOOL_ROOT / f"{name}.py"
    if name == "governance_contract":
        script = HANDOFF_TOOL_ROOT.parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validation_evidence_payload(
    *,
    evidence: object = "command completed with PASS",
    name: object = "tests",
    status: object = "PASS",
) -> dict[str, object]:
    contract = _load_handoff_module("governance_contract")
    rows: list[dict[str, object]] = []
    check_names = [
        "tests",
        *(
            check_name
            for check_name in sorted(contract.MANDATORY_VALIDATION_CHECKS)
            if check_name != "tests"
        ),
    ]
    for check_name in check_names:
        rows.append(
            {
                "name": name if check_name == "tests" else check_name,
                "status": status if check_name == "tests" else "PASS",
                "evidence": evidence if check_name == "tests" else f"{check_name} PASS",
            }
        )
    return {
        "schema_id": "qrp-handoff-validation-evidence",
        "checks": rows,
    }


@pytest.mark.parametrize("evidence", ["", "   ", 123, [], {}])
def test_validation_evidence_rejects_invalid_evidence_values(
    evidence: object,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_evidence_schema(
        _validation_evidence_payload(evidence=evidence)
    )

    assert "validation_evidence_evidence_invalid: 0" in failures


def test_validation_evidence_accepts_non_empty_string_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_evidence_schema(
        _validation_evidence_payload(evidence="command completed with PASS")
    )

    assert not any(
        failure.startswith("validation_evidence_evidence_invalid")
        for failure in failures
    )


@pytest.mark.parametrize("name", [" tests ", "\ttests", "tests\n", "", 123])
def test_validation_evidence_rejects_invalid_check_names(name: object) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_evidence_schema(
        _validation_evidence_payload(name=name)
    )

    assert "validation_evidence_name_invalid: 0" in failures


def test_validation_evidence_accepts_exact_check_name() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_evidence_schema(
        _validation_evidence_payload(name="tests")
    )

    assert not any(
        failure.startswith("validation_evidence_name_invalid")
        for failure in failures
    )


@pytest.mark.parametrize("status", ["pass", "PASS ", "", None, 123])
def test_validation_evidence_rejects_invalid_status_values(status: object) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_evidence_schema(
        _validation_evidence_payload(status=status)
    )

    assert f"validation_evidence_status_invalid: 0:{status}" in failures


@pytest.mark.parametrize(
    ("validation_evidence", "expected_error"),
    [
        (
            {"schema_id": "qrp-handoff-validation-evidence"},
            "validation_evidence_checks_missing",
        ),
        (
            {"schema_id": "qrp-handoff-validation-evidence", "checks": None},
            "validation_evidence_checks_must_be_list",
        ),
        (
            {"schema_id": "qrp-handoff-validation-evidence", "checks": {}},
            "validation_evidence_checks_must_be_list",
        ),
        (
            {"schema_id": "qrp-handoff-validation-evidence", "checks": "bad"},
            "validation_evidence_checks_must_be_list",
        ),
        ("bad", "validation_evidence_must_be_object"),
        (None, "validation_evidence_must_be_object"),
    ],
)
def test_builder_task_validation_evidence_rejects_malformed_payloads(
    validation_evidence: object,
    expected_error: str,
) -> None:
    builder = _load_handoff_module("build_handoff_package")

    with pytest.raises(SystemExit, match=expected_error):
        builder.task_validation_evidence({"validation_evidence": validation_evidence})


def test_builder_task_validation_evidence_generates_missing_key_rows() -> None:
    builder = _load_handoff_module("build_handoff_package")
    contract = _load_handoff_module("governance_contract")

    payload = builder.task_validation_evidence({})
    checks = {row["name"]: row for row in payload["checks"]}

    assert set(checks) == set(contract.MANDATORY_VALIDATION_CHECKS)
    assert {row["status"] for row in checks.values()} == {"NOT_RUN"}
    assert all(
        "task validation_evidence missing; generated mandatory NOT_RUN row"
        in row["evidence"]
        for row in checks.values()
    )

    verifier = _load_handoff_module("verify_handoff_package")
    failures = verifier.verify_validation_evidence_schema(payload)
    assert not any(
        failure.startswith("validation_evidence_mandatory_check_missing")
        for failure in failures
    )
