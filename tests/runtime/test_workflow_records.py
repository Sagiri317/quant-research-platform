from __future__ import annotations

import pytest

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import WorkflowRunSummary
from quant_research_platform.runtime.workflow_records import (
    workflow_run_summary_to_public_record,
)


def _summary(*, workflow_id: str = "workflow") -> WorkflowRunSummary:
    return WorkflowRunSummary(
        workflow_id=workflow_id,
        workflow_version="0.1",
        execution_mode=ExecutionMode.DRY_RUN,
        run_id="workflow-record-test",
        status="validated",
        market_id="a_share",
        data_product_id="a_share_daily_bar",
        run_type="data_product_interim",
        input_spec_ids=("input_spec",),
        output_spec_ids=("output_spec",),
        node_count=1,
        completed_nodes=("validate",),
        private_boundary_required=True,
        artifact_writer_policy="spec_driven",
    )


def test_workflow_run_summary_to_record_returns_plain_mapping() -> None:
    record = _summary().to_record()

    assert record == {
        "workflow_id": "workflow",
        "workflow_version": "0.1",
        "execution_mode": "dry_run",
        "run_id": "workflow-record-test",
        "status": "validated",
        "market_id": "a_share",
        "data_product_id": "a_share_daily_bar",
        "run_type": "data_product_interim",
        "input_spec_ids": ["input_spec"],
        "output_spec_ids": ["output_spec"],
        "node_count": 1,
        "completed_nodes": ["validate"],
        "private_boundary_required": True,
        "artifact_writer_policy": "spec_driven",
    }


def test_workflow_run_summary_to_record_includes_failure_code_when_present() -> None:
    summary = _summary()
    failed_summary = WorkflowRunSummary(
        **{
            **summary.__dict__,
            "failure_code": "unknown_workflow_node",
        }
    )

    assert failed_summary.to_record()["failure_code"] == "unknown_workflow_node"


def test_workflow_run_summary_to_public_record_sanitizes_mapping(monkeypatch) -> None:
    import quant_research_platform.runtime.workflow_records as workflow_records

    calls: list[dict[str, object]] = []

    def fake_sanitizer(value: object) -> dict[str, object]:
        assert isinstance(value, dict)
        calls.append(value)
        return {"sanitized": True}

    summary = _summary(workflow_id="workflow-private-marker")
    monkeypatch.setattr(workflow_records, "sanitize_public_object", fake_sanitizer)

    assert workflow_run_summary_to_public_record(summary) == {"sanitized": True}
    assert calls == [summary.to_record()]


def test_workflow_run_summary_to_public_record_rejects_non_mapping(
    monkeypatch,
) -> None:
    import quant_research_platform.runtime.workflow_records as workflow_records

    monkeypatch.setattr(
        workflow_records,
        "sanitize_public_object",
        lambda _value: ["not-a-mapping"],
    )

    with pytest.raises(TypeError, match="workflow_public_record_must_be_mapping"):
        workflow_run_summary_to_public_record(_summary())
