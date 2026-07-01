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


def _validation_summary_payload(
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
        "schema_id": "qrp-handoff-validation-summary",
        "checks": rows,
    }


@pytest.mark.parametrize("evidence", ["", "   ", 123, [], {}])
def test_validation_summary_rejects_invalid_evidence_values(
    evidence: object,
) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_summary_schema(
        _validation_summary_payload(evidence=evidence)
    )

    assert "validation_summary_evidence_invalid: 0" in failures


def test_validation_summary_accepts_non_empty_string_evidence() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_summary_schema(
        _validation_summary_payload(evidence="command completed with PASS")
    )

    assert not any(
        failure.startswith("validation_summary_evidence_invalid")
        for failure in failures
    )


@pytest.mark.parametrize("name", [" tests ", "\ttests", "tests\n", "", 123])
def test_validation_summary_rejects_invalid_check_names(name: object) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_summary_schema(
        _validation_summary_payload(name=name)
    )

    assert "validation_summary_name_invalid: 0" in failures


def test_validation_summary_accepts_exact_check_name() -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_summary_schema(
        _validation_summary_payload(name="tests")
    )

    assert not any(
        failure.startswith("validation_summary_name_invalid") for failure in failures
    )


@pytest.mark.parametrize("status", ["pass", "PASS ", "", None, 123])
def test_validation_summary_rejects_invalid_status_values(status: object) -> None:
    verifier = _load_handoff_module("verify_handoff_package")

    failures = verifier.verify_validation_summary_schema(
        _validation_summary_payload(status=status)
    )

    assert f"validation_summary_status_invalid: 0:{status}" in failures


@pytest.mark.parametrize(
    "manual_key",
    ["validation_evidence", "handoff_evidence", "handoff_markdown_evidence"],
)
def test_builder_task_metadata_rejects_manual_evidence_keys(manual_key: str) -> None:
    builder = _load_handoff_module("build_handoff_package")
    payload = {
        "schema_id": "qrp-handoff-task-metadata",
        "mode": "evidence",
        "slug": "demo",
        "branch": "feature/demo",
        "base": "0" * 40,
        "final_head": "1" * 40,
        manual_key: {},
    }

    with pytest.raises(SystemExit, match="task_metadata_forbidden_field"):
        builder.task_metadata(payload)
