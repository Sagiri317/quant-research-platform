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
        "scope_summary": "Semantic closure without public API expansion.",
        "commit_list": ["abc123 Fix research foundation semantic closure"],
        "fixed_blockers": ["Removed prediction bundle vocabulary."],
        "layer_summary": {"contracts": "SignalSpec moved to contracts/signals.py"},
        "old_naming_cleanup_summary": "Retired names scanned clean.",
        "semantic_scan_summary": "Architecture semantic gate passed.",
        "smoke_compare": {"three_year_smoke": "PASS"},
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
    assert "Semantic closure without public API expansion." in body
    assert "abc123 Fix research foundation semantic closure" in body
    assert "`contracts`: SignalSpec moved to contracts/signals.py" in body
    assert "`three_year_smoke`: PASS" in body
    assert "`private_slow`: `NOT_RUN`" in body
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


def test_pr_body_validation_rejects_external_evidence_status_mismatch() -> None:
    module = _load_pr_body_module()
    validation_evidence = {
        "checks": [
            {
                "name": "handoff_verify",
                "status": "PASS",
                "evidence": "handoff_package_verify_status=PASS",
            },
            {
                "name": "pr_body_evidence_match",
                "status": "PASS",
                "evidence": "pr_body_validation_status=PASS",
            },
        ],
    }
    body = module.render_pr_body({}, validation_evidence)
    external_evidence = {
        "handoff_verify_status": "PASS",
        "package_verification_status": "NOT_RUN",
        "pr_body_evidence_match_status": "PASS",
        "validation_summary": {"status": "PASS"},
    }

    result = module.validate_pr_body_against_evidence(
        body,
        validation_evidence,
        external_evidence=external_evidence,
    )

    assert result == [
        "external_status_mismatch: package_verification_status=NOT_RUN "
        "!= handoff_verify=PASS"
    ]


def test_pr_body_evidence_consistency_uses_external_mismatch_status() -> None:
    module = _load_pr_body_module()
    validation_evidence = {
        "checks": [
            {
                "name": "handoff_verify",
                "status": "PASS",
                "evidence": "verified",
            },
        ],
    }
    external_evidence = {
        "package_verification_status": "FAIL",
    }

    body = module.render_pr_body(
        {},
        validation_evidence,
        external_evidence=external_evidence,
    )

    assert "| status cross-check | `FAIL` |" in body
    assert "| validation evidence | `PASS` |" in body
    assert "| PR body | `PASS` |" not in body
    assert "| zip sidecar | `PASS` |" not in body


def test_pr_body_missing_validation_evidence_renders_not_run() -> None:
    module = _load_pr_body_module()

    body = module.render_pr_body({}, {})

    assert "| `validation_evidence` | `NOT_RUN` | missing validation rows |" in body
    assert "| validation evidence | `NOT_RUN` |" in body
    assert "| validation evidence | `PASS` |" not in body


def test_pr_body_cli_verifies_external_evidence_statuses(tmp_path) -> None:
    module = _load_pr_body_module()
    task_metadata = tmp_path / "task_metadata.json"
    validation_path = tmp_path / "validation_evidence.json"
    external_path = tmp_path / "external_evidence.json"
    output = tmp_path / "pr_body.md"
    task_metadata.write_text("{}", encoding="utf-8")
    validation_path.write_text(
        """
{
  "checks": [
    {
      "name": "handoff_verify",
      "status": "PASS",
      "evidence": "handoff_package_verify_status=PASS"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    external_path.write_text(
        """
{
  "handoff_verify_status": "PASS",
  "package_verification_status": "NOT_RUN"
}
""".strip(),
        encoding="utf-8",
    )

    result = module.main(
        [
            "--task-metadata",
            str(task_metadata),
            "--validation-evidence",
            str(validation_path),
            "--external-evidence",
            str(external_path),
            "--output",
            str(output),
            "--verify",
        ]
    )

    assert result == 1


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


def test_pr_body_renders_optional_handoff_threads_and_closure_sections() -> None:
    module = _load_pr_body_module()
    task_metadata = {
        "closure_sections": [
            {
                "title": "A-share Date Default Cleanup",
                "items": [
                    "research_profile() remains API/CLI static catalog entry.",
                    "No public API/CLI was added.",
                ],
            }
        ],
    }
    validation_evidence = {
        "checks": [
            {"name": "handoff_verify", "status": "PASS", "evidence": "ok"},
        ],
    }
    review_threads = {
        "status": "PASS",
        "total_count": 1,
        "unresolved_count": 0,
        "threads": [
            {
                "thread_id": "thread-1",
                "path": "src/example.py",
                "classification": "outdated-fixed",
                "resolution_status": "resolved",
                "manual_action_required": False,
                "evidence": "fixed in abc123",
            }
        ],
    }
    handoff_results = {
        "results": [
            {
                "name": "handoff_verify",
                "status": "PASS",
                "duration_seconds": 1.2,
                "performance_status": "PASS_TARGET",
            }
        ],
    }

    body = module.render_pr_body(
        task_metadata,
        validation_evidence,
        review_threads=review_threads,
        handoff_results=handoff_results,
    )

    assert "## Handoff Fresh-Temp Verification" in body
    assert "| `handoff_verify` | `PASS` | 1.2s | `PASS_TARGET` |" in body
    assert "| review threads | `PASS` | total=1 unresolved=0 |" in body
    assert "| `thread-1` | `src/example.py` | `outdated-fixed` |" in body
    assert "full evidence: review_threads.json" in body
    assert "## A-share Date Default Cleanup" in body
    assert "- research_profile() remains API/CLI static catalog entry." in body


def test_pr_body_missing_or_empty_validation_rows_become_not_run() -> None:
    module = _load_pr_body_module()

    for validation_evidence in ({}, {"validation_table": []}, {"checks": []}):
        assert module.validation_rows(validation_evidence) == [
            {
                "check": "validation_evidence",
                "status": "NOT_RUN",
                "details": "missing validation rows",
            }
        ]
