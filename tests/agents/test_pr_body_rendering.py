from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pr_body_module():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "render_pr_body_from_validation_evidence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_pr_body_from_validation_evidence",
        script,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pr_body_is_rendered_from_validation_evidence_statuses() -> None:
    module = _load_pr_body_module()
    task_metadata = {
        "branch": "feature/qrp-clean-rewrite",
        "base_commit_before_task": "9666c9d",
        "final_sha": "abc123",
        "pr_url": "https://github.com/Sagiri317/quant-research-platform/pull/1",
    }
    validation_evidence = {
        "validation_table": [
            {"check": "semantic_scan", "status": "PASS", "details": "ok"},
            {"check": "ruff", "status": "FAIL", "details": "E501"},
            {"check": "private_slow", "status": "NOT_RUN", "details": ""},
        ],
    }

    body = module.render_pr_body(task_metadata, validation_evidence)

    assert "| `semantic_scan` | `PASS` | ok |" in body
    assert "| `ruff` | `FAIL` | E501 |" in body
    assert "| `private_slow` | `NOT_RUN` |  |" in body
    module.validate_pr_body_against_evidence(body, validation_evidence)


def test_pr_body_validation_rejects_false_pass_claim() -> None:
    module = _load_pr_body_module()
    validation_evidence = {
        "validation_table": [
            {"check": "ruff", "status": "FAIL", "details": "E501"},
        ],
    }
    false_body = (
        "| Check | Status | Details |\n"
        "| --- | --- | --- |\n"
        "| `ruff` | `PASS` | E501 |\n"
    )

    result = module.validate_pr_body_against_evidence(
        false_body,
        validation_evidence,
    )

    assert result == ["validation_row_mismatch: ruff"]


def test_pr_body_reads_actual_checks_evidence_schema() -> None:
    module = _load_pr_body_module()
    task_metadata = {
        "branch": "feature/qrp-clean-rewrite",
        "base_commit_before_task": "9666c9d",
        "final_sha": "abc123",
        "pr_url": "https://github.com/Sagiri317/quant-research-platform/pull/1",
    }
    validation_evidence = {
        "checks": [
            {
                "name": "semantic_scan",
                "status": "PASS",
                "evidence": "architecture gate clean",
            },
            {
                "name": "private_slow",
                "status": "BLOCKED",
                "evidence": "private artifact unavailable",
            },
        ],
    }

    body = module.render_pr_body(task_metadata, validation_evidence)

    assert "| `semantic_scan` | `PASS` | architecture gate clean |" in body
    assert "| `private_slow` | `BLOCKED` | private artifact unavailable |" in body
    assert module.validate_pr_body_against_evidence(body, validation_evidence) == []


def test_pr_body_rejects_missing_or_empty_validation_rows() -> None:
    module = _load_pr_body_module()

    for validation_evidence in ({}, {"validation_table": []}, {"checks": []}):
        try:
            module.validation_rows(validation_evidence)
        except ValueError as exc:
            assert str(exc) == "validation_rows_required"
        else:  # pragma: no cover - failure branch for clear assertion message
            raise AssertionError("validation_rows_required")
