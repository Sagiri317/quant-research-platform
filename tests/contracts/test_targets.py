from __future__ import annotations

from dataclasses import replace

import pytest

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.targets import (
    PlanDecision,
    ProducerBinding,
    ProducerKind,
    ResearchRunRecord,
    ResearchRunTargetResult,
    RunStatus,
    TargetPlan,
    TargetResult,
    TargetSpec,
    TargetStatus,
    unmeasured_operation_evidence,
)


def test_target_spec_and_plan_are_immutable_control_contracts() -> None:
    spec = TargetSpec(
        target_id="dataset",
        dependencies=("panel",),
        producer_id="a_share_dataset",
    )
    plan = TargetPlan(
        target_id=spec.target_id,
        upstream_target_ids=spec.dependencies,
        producer_id=spec.producer_id,
        decision=PlanDecision.BUILD,
        requested_identity="a" * 64,
    )

    assert plan.payload() == {
        "target_id": "dataset",
        "upstream_target_ids": ["panel"],
        "requested_identity": "a" * 64,
        "matching_instance_identity": None,
        "upstream_identities": {},
        "producer_binding": "a_share_dataset",
        "decision": "BUILD",
        "blocker": None,
        "rebuild": False,
    }


def test_target_result_allows_only_declared_states_and_reasons() -> None:
    result = TargetResult(
        target_id="dataset",
        status=TargetStatus.BUILT,
        instance_identity="a" * 64,
    )
    assert result.payload() == {
        "target_id": "dataset",
        "status": "BUILT",
        "reason": None,
        "instance_identity": "a" * 64,
    }
    with pytest.raises(ValueError, match="target_instance_identity_required"):
        TargetResult(target_id="dataset", status=TargetStatus.BUILT)
    with pytest.raises(ValueError, match="blocked_target_identity_not_allowed"):
        TargetResult(
            target_id="dataset",
            status=TargetStatus.BLOCKED,
            reason="blocked",
            instance_identity="a" * 64,
        )
    with pytest.raises(ValueError, match="target_result_reason_required"):
        TargetResult(target_id="dataset", status=TargetStatus.BLOCKED)
    with pytest.raises(ValueError, match="target_result_reason_required"):
        TargetResult(target_id="dataset", status=TargetStatus.FAILED)
    with pytest.raises(ValueError, match="failed_target_identity_not_allowed"):
        TargetResult(
            target_id="dataset",
            status=TargetStatus.FAILED,
            reason="failed",
            instance_identity="a" * 64,
        )
    with pytest.raises(ValueError, match="target_result_reason_invalid"):
        TargetResult(
            target_id="dataset",
            status=TargetStatus.FAILED,
            reason=123,  # type: ignore[arg-type]
        )


def test_build_plan_requires_requested_identity() -> None:
    with pytest.raises(ValueError, match="target_identity_required"):
        TargetPlan(
            target_id="dataset",
            upstream_target_ids=(),
            producer_id="producer",
            decision=PlanDecision.BUILD,
        )

    with pytest.raises(ValueError, match="build_matching_identity_requires_rebuild"):
        TargetPlan(
            target_id="dataset",
            upstream_target_ids=(),
            producer_id="producer",
            decision=PlanDecision.BUILD,
            requested_identity="a" * 64,
            matching_instance_identity="a" * 64,
        )
    rebuilt = TargetPlan(
        target_id="dataset",
        upstream_target_ids=(),
        producer_id="producer",
        decision=PlanDecision.BUILD,
        requested_identity="a" * 64,
        matching_instance_identity="a" * 64,
        rebuild=True,
    )
    assert rebuilt.payload()["rebuild"] is True


def test_producer_binding_is_typed_and_frozen() -> None:
    binding = ProducerBinding(ProducerKind.DATASET, lambda: None)
    assert binding.kind is ProducerKind.DATASET
    with pytest.raises(ValueError, match="producer_callable_required"):
        ProducerBinding(ProducerKind.DATASET, object())  # type: ignore[arg-type]


def test_status_vocabulary_has_no_readiness_or_persisted_stale_state() -> None:
    assert {status.value for status in RunStatus} == {"COMPLETE", "BLOCKED", "FAILED"}
    assert {status.value for status in TargetStatus} == {
        "REUSED",
        "BUILT",
        "BLOCKED",
        "FAILED",
    }


def test_research_run_record_requires_real_terminals_and_protocol_request_match() -> (
    None
):
    protocol = {
        "scenario_id": "scenario",
        "access_window": "development",
    }
    target = ResearchRunTargetResult(
        target_id="target",
        producer_kind=ProducerKind.MODEL,
        status=TargetStatus.BUILT,
        instance_identity="a" * 64,
        schema_identity="b" * 64,
        artifact_root="data/artifacts/market/target",
    )
    record = ResearchRunRecord(
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
        run_id="run_1",
        profile_id="profile",
        scenario_id="scenario",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window={
            "start": "2019-01-01",
            "end": "2019-12-31",
            "access": "development",
        },
        software_identity="c" * 64,
        terminal_targets=("target",),
        target_results=(target,),
        status=RunStatus.COMPLETE,
    )
    with pytest.raises(ValueError, match="research_run_terminal_targets_invalid"):
        replace(record, terminal_targets=(), target_results=())
    with pytest.raises(ValueError, match="research_run_protocol_request_mismatch"):
        replace(record, scenario_id="different")
