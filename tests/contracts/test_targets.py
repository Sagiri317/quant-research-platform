from __future__ import annotations

import pytest

from quant_research_platform.contracts.targets import (
    PlanDecision,
    ProducerBinding,
    ProducerKind,
    RunStatus,
    TargetPlan,
    TargetResult,
    TargetSpec,
    TargetStatus,
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


def test_producer_binding_is_typed_and_frozen() -> None:
    binding = ProducerBinding(ProducerKind.DATASET, lambda: None)
    assert binding.kind is ProducerKind.DATASET
    with pytest.raises(ValueError, match="producer_callable_required"):
        ProducerBinding(ProducerKind.DATASET, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source_root_resolver_callable_required"):
        ProducerBinding(
            ProducerKind.INTERIM,
            lambda: None,
            source_root_resolver=object(),  # type: ignore[arg-type]
        )


def test_status_vocabulary_has_no_readiness_or_persisted_stale_state() -> None:
    assert {status.value for status in RunStatus} == {"COMPLETE", "BLOCKED", "FAILED"}
    assert {status.value for status in TargetStatus} == {
        "REUSED",
        "BUILT",
        "BLOCKED",
        "FAILED",
    }
