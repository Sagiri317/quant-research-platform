"""Research operation entrypoints, request lifetimes, and run records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_research_platform.artifacts.research import (
    build_report_from_run,
    write_research_run_record,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.targets import (
    PlanDecision,
    ResearchRunRecord,
    ResearchRunTargetResult,
    RunStatus,
    TargetPlan,
    TargetResult,
    TargetStatus,
    unmeasured_operation_evidence,
)
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.sources.tardis.replay import (
    _REQUEST_ARCHIVES,
    _tardis_request_session,
)
from quant_research_platform.storage.experiments import research_run_trial_references
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.software_identity import ProductionSoftwareIdentity
from quant_research_platform.workflows import research


def plan_targets(
    *,
    profile: ResearchProfile,
    action: str,
    terminal_targets: Iterable[str],
    layout: StorageLayout,
    scenario_id: str | None = None,
    rebuild: bool = False,
    raw_root: Path | None = None,
    research_window: Mapping[str, object] | None = None,
    software_identity: ProductionSoftwareIdentity | None = None,
) -> dict[str, object]:
    request = research._prepare_plan(
        profile=profile,
        action=action,
        terminal_targets=terminal_targets,
        layout=layout,
        scenario_id=scenario_id,
        rebuild=rebuild,
        raw_root=raw_root,
        research_window=research_window,
        software_identity=software_identity,
    )
    return _plan_payload(request)


def _plan_payload(request):
    instances, _sessions = research._plan_targets(**request)
    profile = request["profile"]
    action = request["action"]
    terminals = request["terminal_targets"]
    protocol = request["protocol"]
    rebuild = request["rebuild"]
    software = request["software_identity"]
    window = request["research_window"]
    terminals = tuple(sorted(set(terminals)))
    plans = [plan for request, plan in instances if request.publish]
    blocked = [plan for _, plan in instances if plan.decision is PlanDecision.BLOCKED]
    return {
        "stage": action,
        "plan": True,
        "profile_id": profile.profile_id,
        "rebuild": rebuild,
        "terminal_targets": list(terminals),
        "transitive_dependencies": [
            plan.target_id for plan in plans if plan.target_id not in terminals
        ],
        "targets": [plan.payload() for plan in plans],
        "status": RunStatus.BLOCKED.value if blocked else RunStatus.COMPLETE.value,
        "blockers": [
            {"target_id": plan.target_id, "reason": plan.blocker} for plan in blocked
        ],
        "protocol": protocol.manifest(),
        "software_identity": software.identity_hash,
        "research_window": window,
    }


def data_materialize(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    raw_root: Path | None,
    start_date: str | None,
    end_date: str | None,
    data_ids: tuple[str, ...] | None = None,
    plan: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    request = research._prepare_materialization(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
        data_ids=data_ids,
        rebuild=rebuild,
    )
    return _plan_payload(request) if plan else _execute_action(**request)


def dataset_build(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    dataset_id: str,
    start_date: str,
    end_date: str,
    plan: bool = False,
    rebuild: bool = False,
    raw_root: Path | None = None,
) -> dict[str, object]:
    request = research._prepare_dataset(
        profile=profile,
        layout=layout,
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
        raw_root=raw_root,
    )
    return _plan_payload(request) if plan else _execute_action(**request)


def research_run(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    research_plan: Mapping[str, object],
    raw_root: Path | None = None,
    holdout: bool = False,
    development_run_id: str | None = None,
    plan: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    request, preflight_blocker = research._prepare_research(
        profile=profile,
        layout=layout,
        research_plan=research_plan,
        raw_root=raw_root,
        holdout=holdout,
        development_run_id=development_run_id,
        plan=plan,
        rebuild=rebuild,
    )
    profile = request["profile"]
    protocol = request["protocol"]
    terminals = request["terminal_targets"]
    window = request["research_window"]
    software = request["software_identity"]
    if preflight_blocker is not None:
        result = {
            "stage": "research run",
            "plan": plan,
            "status": "BLOCKED",
            "blockers": [
                {"target_id": target, "reason": preflight_blocker}
                for target in terminals
            ],
        }
        if plan:
            return result
        run_id = f"run_{uuid4().hex}"
        _write_research_run_record(
            layout=layout,
            run_id=run_id,
            profile=profile,
            protocol=protocol,
            research_window=window,
            software_identity=software,
            terminal_targets=terminals,
            target_results=tuple(
                TargetResult(
                    target_id=target,
                    status=TargetStatus.BLOCKED,
                    reason=preflight_blocker,
                )
                for target in terminals
            ),
            target_references={},
            status=RunStatus.BLOCKED,
            operation_evidence=unmeasured_operation_evidence(
                "holdout_preflight_blocked"
            ),
        )
        return {**result, "run_id": run_id}
    if plan:
        return _plan_payload(request)
    run_id = f"run_{uuid4().hex}"
    result_sink: dict[str, TargetResult] = {}
    reference_sink: dict[str, TargetPlan] = {}
    operation_evidence = unmeasured_operation_evidence("operation_not_started")
    try:
        result = _execute_action(
            **request,
            run_id=run_id,
            result_sink=result_sink,
            reference_sink=reference_sink,
            operation_evidence_sink=operation_evidence,
        )
    except BaseException as primary:
        try:
            _write_research_run_record(
                layout=layout,
                run_id=run_id,
                profile=profile,
                protocol=protocol,
                research_window=window,
                software_identity=software,
                terminal_targets=terminals,
                target_results=tuple(result_sink.values()),
                target_references=reference_sink,
                status=RunStatus.FAILED,
                failure_reason=f"research_run_failed:{type(primary).__name__.lower()}",
                operation_evidence=operation_evidence,
            )
        except BaseException as record_error:
            primary.add_note(
                f"research run record write also failed: {type(record_error).__name__}"
            )
        raise
    _write_research_run_record(
        layout=layout,
        run_id=run_id,
        profile=profile,
        protocol=protocol,
        research_window=window,
        software_identity=software,
        terminal_targets=terminals,
        target_results=tuple(result_sink.values()),
        target_references=reference_sink,
        status=RunStatus(result["status"]),
        operation_evidence=operation_evidence,
    )
    return {**result, "run_id": run_id}


def report_build(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    run_id: str,
    plan: bool = False,
) -> dict[str, object]:
    return build_report_from_run(
        **research._prepare_report(profile=profile, layout=layout, run_id=run_id),
        plan=plan,
    )


def _execute_action(
    *,
    profile: ResearchProfile,
    protocol: Any,
    action: str,
    terminal_targets: tuple[str, ...],
    layout: StorageLayout,
    raw_root: Path | None,
    rebuild: bool = False,
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity | None = None,
    result_sink: dict[str, TargetResult] | None = None,
    reference_sink: dict[str, TargetPlan] | None = None,
    run_id: str | None = None,
    operation_evidence_sink: dict[str, object] | None = None,
    reuse_sessions=None,
) -> dict[str, object]:
    with (
        (
            record_run_observations(operation_evidence_sink)
            if operation_evidence_sink is not None
            else nullcontext(None)
        ) as observe,
        ExitStack() as operation,
    ):
        if _REQUEST_ARCHIVES.get() is None:
            operation.enter_context(_tardis_request_session(operation))
        return research._execute_targets(
            run_id=run_id,
            profile=profile,
            protocol=protocol,
            action=action,
            terminal_targets=terminal_targets,
            layout=layout,
            raw_root=raw_root,
            rebuild=rebuild,
            research_window=research_window,
            software_identity=software_identity,
            result_sink=result_sink,
            reference_sink=reference_sink,
            observe=observe,
            reuse_sessions=reuse_sessions,
        )


def _write_research_run_record(
    *,
    run_id: str,
    layout: StorageLayout,
    profile: ResearchProfile,
    protocol: Any,
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity,
    terminal_targets: tuple[str, ...],
    target_results: tuple[TargetResult, ...],
    target_references: Mapping[str, TargetPlan],
    status: RunStatus,
    operation_evidence: Mapping[str, object],
    failure_reason: str | None = None,
) -> None:
    references: list[ResearchRunTargetResult] = []
    for result in target_results:
        planned = target_references.get(result.target_id)
        producer_kind = planned.producer_kind if planned is not None else None
        schema_identity = (
            planned.schema_identity
            if result.instance_identity is not None and planned is not None
            else None
        )
        artifact_root = (
            planned.artifact_root
            if result.instance_identity is not None and planned is not None
            else None
        )
        references.append(
            ResearchRunTargetResult(
                target_id=result.target_id,
                producer_kind=producer_kind,
                status=result.status,
                reason=result.reason,
                instance_identity=result.instance_identity,
                schema_identity=schema_identity,
                artifact_root=artifact_root,
            )
        )
    protocol_snapshot = protocol.manifest()
    trial_ledgers = research_run_trial_references(layout, run_id)
    values = dict(
        run_id=run_id,
        trial_ledgers=trial_ledgers,
        profile_id=profile.profile_id,
        scenario_id=protocol.scenario_id,
        protocol=protocol_snapshot,
        protocol_identity=canonical_json_hash(protocol_snapshot),
        research_window=research_window,
        software_identity=software_identity.identity_hash,
        terminal_targets=terminal_targets,
        target_results=tuple(references),
        status=status,
        operation_evidence=operation_evidence,
        failure_reason=failure_reason,
    )
    record = ResearchRunRecord(**values)
    write_research_run_record(layout.research_run_path(run_id), record)
