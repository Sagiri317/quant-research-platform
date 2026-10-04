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


def test_run_record_deeply_isolates_inputs_payload_and_persisted_identity(tmp_path):
    import json

    from quant_research_platform.artifacts.research import (
        read_research_run_record,
        write_research_run_record,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    protocol = {
        "scenario_id": "scenario",
        "access_window": "holdout",
        "scope": {"assets": ["btc_usdt"]},
    }
    evidence = {
        "qualification": "REUSE",
        "prior_runs": [
            {"run_id": "prior_" + "a" * 64 + str(index), "identity": "d" * 64}
            for index in range(7500)
        ],
    }
    operation = unmeasured_operation_evidence("fixture_record")
    record = ResearchRunRecord(
        run_id="run_immutable",
        profile_id="profile",
        scenario_id="scenario",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window={
            "start": "2024-01-01",
            "end": "2025-12-31",
            "access": "holdout",
        },
        software_identity="c" * 64,
        terminal_targets=("target",),
        target_results=(
            ResearchRunTargetResult(
                target_id="target",
                producer_kind=ProducerKind.MODEL,
                status=TargetStatus.BUILT,
                instance_identity="a" * 64,
                schema_identity="b" * 64,
                artifact_root="data/artifacts/market/target",
            ),
        ),
        status=RunStatus.COMPLETE,
        operation_evidence=operation,
        holdout_evidence=evidence,
    )
    record = replace(
        record,
        terminal_targets=tuple(f"target_{i}" for i in range(4000)),
        target_results=tuple(
            replace(
                record.target_results[0],
                target_id=f"target_{i}",
                artifact_root=f"data/artifacts/market/target_{i}",
            )
            for i in range(4000)
        ),
    )
    original = record.payload()
    assert len(json.dumps(original["holdout_evidence"]).encode()) > 1024 * 1024
    protocol["scope"]["assets"].append("eth_usdt")
    evidence["prior_runs"].clear()
    operation["environment"]["reason"] = "changed"
    payload = record.payload()
    payload["protocol"]["scope"]["assets"].append("other")
    payload["holdout_evidence"]["prior_runs"].clear()
    payload["operation_evidence"]["environment"]["reason"] = "changed"
    assert record.payload() == original
    assert canonical_json_hash(record.payload()["protocol"]) == record.protocol_identity
    assert json.loads(json.dumps(record.payload())) == original
    with pytest.raises(TypeError):
        record.protocol["scope"]["other"] = "changed"
    with pytest.raises(AttributeError):
        record.protocol["scope"]["assets"].append("other")
    with pytest.raises(TypeError):
        record.holdout_evidence["prior_runs"][0]["identity"] = "e" * 64
    with pytest.raises(TypeError):
        record.operation_evidence["environment"]["reason"] = "changed"
    assert replace(record).payload() == original
    with runtime_resource_policy_scope() as policy:
        path = write_research_run_record(tmp_path / f"{record.run_id}.json", record)
        assert path.stat().st_size > 1024 * 1024
        assert read_research_run_record(path).payload() == original
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "field",
    (
        "terminal_targets",
        "target_results",
        "trial_ledgers",
        "protocol",
        "research_window",
        "operation_evidence",
        "holdout_evidence",
    ),
)
def test_run_record_reader_rejects_wrong_json_container(tmp_path, field):
    import json

    from quant_research_platform.artifacts.research import (
        read_research_run_record,
        write_research_run_record,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    protocol = {"scenario_id": "scenario", "access_window": "development"}
    record = ResearchRunRecord(
        run_id="shape-run",
        profile_id="profile",
        scenario_id="scenario",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window={
            "start": "2020-01-01",
            "end": "2020-01-02",
            "access": "development",
        },
        software_identity="a" * 64,
        terminal_targets=("target",),
        target_results=(
            ResearchRunTargetResult(
                target_id="target",
                producer_kind=ProducerKind.MODEL,
                status=TargetStatus.BLOCKED,
                reason="fixture_missing",
            ),
        ),
        status=RunStatus.BLOCKED,
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
    )
    path = tmp_path / "shape-run.json"
    with runtime_resource_policy_scope():
        write_research_run_record(path, record)
        assert read_research_run_record(path).payload() == record.payload()
        payload = record.payload()
        value = payload[field]
        payload[field] = (
            list(value.items())
            if isinstance(value, dict)
            else {str(item): "discarded" for item in value}
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="research_run_record_invalid"):
            read_research_run_record(path)
