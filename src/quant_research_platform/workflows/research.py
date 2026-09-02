"""The single target-DAG resolver and public research workflow owner."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.artifacts.outputs import (
    read_prediction_output_bundle,
    validate_output_bundle,
    write_evaluation_outputs_from_chunks,
    write_prediction_outputs,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    validate_evaluation_output_metadata_shape,
)
from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_output_metadata_shape,
)
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.data import (
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import (
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.targets import (
    PlanDecision,
    ProducerKind,
    RunStatus,
    TargetPlan,
    TargetResult,
    TargetSpec,
    TargetStatus,
)
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.research.models.persisted import (
    read_model_artifact,
    write_model_artifact,
)
from quant_research_platform.research.models.production import (
    ModelSelectionResult,
)
from quant_research_platform.research.models.supervised import preflight_model_specs
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets.comparison import compare_datasets
from quant_research_platform.storage.datasets.inspection import (
    inspect_dataset as inspect_physical_dataset,
)
from quant_research_platform.storage.datasets.instances import (
    discover_dataset_instances,
)
from quant_research_platform.storage.datasets.schema import (
    COMMON_MANIFEST_METADATA_KEY,
    data_spec_payload,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import validate_instance_identity
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.storage.rebuild import (
    RebuildCandidateWorkspace,
    rebuild_candidate_workspace,
)
from quant_research_platform.storage.semantic_artifacts import (
    compare_semantic_artifacts,
    discover_semantic_artifact_instances,
    read_semantic_artifact,
    write_semantic_artifact,
)
from quant_research_platform.storage.software_identity import (
    ProductionSoftwareIdentity,
    production_software_identity,
)


def resolve_target_graph(
    target_specs: Iterable[TargetSpec],
    requested_targets: Iterable[str],
) -> tuple[TargetSpec, ...]:
    """Return one deterministic post-order expansion of requested targets."""

    specs: dict[str, TargetSpec] = {}
    for spec in target_specs:
        if spec.target_id in specs:
            raise ValueError(f"target_id_duplicate: {spec.target_id}")
        specs[spec.target_id] = spec
    requested = tuple(sorted(set(requested_targets)))
    if not requested:
        raise ValueError("target_request_empty")
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[TargetSpec] = []

    def visit(target_id: str) -> None:
        if target_id not in specs:
            raise ValueError(f"target_unknown: {target_id}")
        if target_id in visiting:
            raise ValueError(f"target_dependency_cycle: {target_id}")
        if target_id in visited:
            return
        visiting.add(target_id)
        for dependency_id in sorted(specs[target_id].dependencies):
            visit(dependency_id)
        visiting.remove(target_id)
        visited.add(target_id)
        ordered.append(specs[target_id])

    for target_id in requested:
        visit(target_id)
    return tuple(ordered)


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
    with runtime_resource_policy_scope(None):
        return _plan_targets(
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


def _plan_targets(
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
    terminals = tuple(sorted(set(terminal_targets)))
    scenario = scenario_id or (
        terminals[0]
        if action == "dataset build" and terminals
        else _default_scenario_id(profile)
    )
    protocol = _protocol(profile, scenario)
    protocol_terminals = tuple(protocol.terminal_targets(action))
    if action != "data materialize" and terminals != protocol_terminals:
        raise ValueError("protocol_terminal_targets_invalid")
    software = software_identity or production_software_identity()
    window = _research_window(protocol, research_window)
    source_root = raw_root or _default_raw_root(profile, layout)
    specs = _target_specs(
        profile,
        action=action,
        protocol=protocol,
    )
    ordered = resolve_target_graph(specs, terminals)
    requested = set(terminals)
    plans: list[TargetPlan] = []
    by_id: dict[str, TargetPlan] = {}
    for spec in ordered:
        data_spec = _catalog_data_spec(profile, spec.target_id)
        upstream_identities = {
            dependency: by_id[dependency].requested_identity
            for dependency in spec.dependencies
            if by_id[dependency].requested_identity is not None
        }
        blocker: str | None = None
        requested_identity: str | None = None
        matching_instance_identity: str | None = None
        decision = PlanDecision.BLOCKED
        blocked_dependency = next(
            (
                dependency
                for dependency in spec.dependencies
                if by_id[dependency].decision is PlanDecision.BLOCKED
            ),
            None,
        )
        if blocked_dependency is not None:
            blocker = f"dependency_blocked: {blocked_dependency}"
        elif spec.producer_id is None:
            blocker = "capability_not_implemented"
        else:
            binding = profile.producer_bindings.get(spec.producer_id)
            if binding is None:
                blocker = (
                    "capability_not_implemented"
                    if data_spec is None
                    else "producer_binding_missing"
                )
            elif data_spec is not None and binding.kind is not _producer_kind(
                data_spec
            ):
                blocker = "producer_binding_kind_mismatch"
            elif (
                isinstance(data_spec, InterimDataSpec)
                and binding.source_identity is None
            ):
                blocker = "source_identity_provider_missing"
            else:
                raw_identity = None
                if isinstance(data_spec, InterimDataSpec):
                    try:
                        raw_identity = binding.source_identity(
                            source_root,
                            spec.target_id,
                        )
                    except ValueError:
                        blocker = "source_identity_invalid"
                if (
                    blocker is None
                    and isinstance(data_spec, InterimDataSpec)
                    and raw_identity is None
                ):
                    blocker = "raw_content_identity_unavailable"
                elif blocker is None and raw_identity is not None:
                    try:
                        validate_instance_identity(raw_identity)
                    except ValueError:
                        blocker = "raw_content_identity_invalid"
                if blocker is None and any(
                    dependency not in upstream_identities
                    for dependency in spec.dependencies
                ):
                    blocker = "upstream_identity_unavailable"
                if blocker is None:
                    envelope = _target_envelope_for_spec(
                        profile=profile,
                        protocol=protocol,
                        target_spec=spec,
                        data_spec=data_spec,
                        research_window=window,
                        upstream_identities=upstream_identities,
                        software_identity=software,
                        raw_content_identity=raw_identity,
                    )
                    requested_identity = envelope.instance_identity
                    matching_instance_identity, blocker = _matching_instance(
                        profile=profile,
                        layout=layout,
                        target_id=spec.target_id,
                        requested_identity=requested_identity,
                    )
                    if (
                        blocker == "persisted_instance_invalid"
                        and rebuild
                        and _instance_path_exists(
                            profile=profile,
                            layout=layout,
                            target_id=spec.target_id,
                            instance_identity=requested_identity,
                        )
                    ):
                        matching_instance_identity = requested_identity
                        blocker = None
                    if blocker is None:
                        decision = (
                            PlanDecision.BUILD
                            if rebuild or matching_instance_identity is None
                            else PlanDecision.REUSE
                        )
        if blocker is not None:
            decision = PlanDecision.BLOCKED
        plan = TargetPlan(
            target_id=spec.target_id,
            upstream_target_ids=tuple(sorted(spec.dependencies)),
            producer_id=spec.producer_id,
            decision=decision,
            blocker=blocker,
            requested_identity=requested_identity,
            matching_instance_identity=matching_instance_identity,
            upstream_identities=upstream_identities,
            rebuild=rebuild,
        )
        plans.append(plan)
        by_id[spec.target_id] = plan
    blocked = [plan for plan in plans if plan.decision is PlanDecision.BLOCKED]
    return {
        "stage": action,
        "plan": True,
        "profile_id": profile.profile_id,
        "rebuild": rebuild,
        "terminal_targets": list(terminals),
        "transitive_dependencies": [
            plan.target_id for plan in plans if plan.target_id not in requested
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
    raw_root: Path,
    start_date: str | None,
    end_date: str | None,
    data_ids: tuple[str, ...] | None = None,
    plan: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    selected = tuple(data_ids or (spec.data_id for spec in profile.interim.list()))
    if plan:
        return plan_targets(
            profile=profile,
            action="data materialize",
            terminal_targets=selected,
            layout=layout,
            raw_root=raw_root,
            rebuild=rebuild,
            research_window=_research_window(
                _protocol(profile, _default_scenario_id(profile)),
                None,
                start_date=start_date,
                end_date=end_date,
            ),
        )
    return _execute_action(
        profile=profile,
        action="data materialize",
        terminal_targets=selected,
        layout=layout,
        time_window=None,
        raw_root=raw_root,
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
        research_window=_research_window(
            _protocol(profile, _default_scenario_id(profile)),
            None,
            start_date=start_date,
            end_date=end_date,
        ),
    )


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
    if plan:
        return plan_targets(
            profile=profile,
            action="dataset build",
            terminal_targets=(dataset_id,),
            layout=layout,
            scenario_id=dataset_id,
            rebuild=rebuild,
            raw_root=raw_root or _default_raw_root(profile, layout),
            research_window=_research_window(
                _protocol(profile, dataset_id),
                None,
                start_date=start_date,
                end_date=end_date,
            ),
        )
    return _execute_action(
        profile=profile,
        action="dataset build",
        terminal_targets=(dataset_id,),
        layout=layout,
        time_window=TimeWindow(start_date, end_date),
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
        raw_root=raw_root or _default_raw_root(profile, layout),
        research_window=_research_window(
            _protocol(profile, dataset_id),
            None,
            start_date=start_date,
            end_date=end_date,
        ),
    )


def research_run(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    scenario_id: str,
    start_date: str,
    end_date: str,
    holdout: bool = False,
    plan: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    if holdout:
        raise PermissionError("holdout_frozen_development_identity_unavailable")
    protocol = _protocol(profile, scenario_id)
    _require_development_window(protocol, start_date, end_date)
    terminal = _protocol_terminal_target(profile, scenario_id, "research run")
    if plan:
        return plan_targets(
            profile=profile,
            action="research run",
            terminal_targets=(terminal,),
            layout=layout,
            scenario_id=scenario_id,
            rebuild=rebuild,
            raw_root=_default_raw_root(profile, layout),
            research_window=_research_window(
                _protocol(profile, scenario_id),
                None,
                start_date=start_date,
                end_date=end_date,
            ),
        )
    return _execute_action(
        profile=profile,
        action="research run",
        terminal_targets=(terminal,),
        layout=layout,
        time_window=TimeWindow(start_date, end_date),
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
        raw_root=_default_raw_root(profile, layout),
        research_window=_research_window(
            _protocol(profile, scenario_id),
            None,
            start_date=start_date,
            end_date=end_date,
        ),
    )


def _require_development_window(protocol: Any, start_date: str, end_date: str) -> None:
    development = protocol.manifest().get("development_window")
    if type(development) is not dict:
        return
    start = development.get("start")
    end = development.get("end")
    if (
        type(start) is not str
        or type(end) is not str
        or pd.Timestamp(start) > pd.Timestamp(start_date)
        or pd.Timestamp(end) < pd.Timestamp(end_date)
    ):
        raise PermissionError("holdout_frozen_development_identity_unavailable")


def report_build(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    scenario_id: str,
    plan: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    terminal = _protocol_terminal_target(profile, scenario_id, "report build")
    if plan:
        return plan_targets(
            profile=profile,
            action="report build",
            terminal_targets=(terminal,),
            layout=layout,
            scenario_id=scenario_id,
            rebuild=rebuild,
            raw_root=_default_raw_root(profile, layout),
            research_window=_research_window(
                _protocol(profile, scenario_id),
                None,
            ),
        )
    return _execute_action(
        profile=profile,
        action="report build",
        terminal_targets=(terminal,),
        layout=layout,
        time_window=None,
        start_date=None,
        end_date=None,
        rebuild=rebuild,
        raw_root=_default_raw_root(profile, layout),
        research_window=_research_window(
            _protocol(profile, scenario_id),
            None,
        ),
    )


def status(*, profile: ResearchProfile, layout: StorageLayout) -> dict[str, object]:
    scenario_id = _default_scenario_id(profile)
    terminal = _protocol_terminal_target(profile, scenario_id, "research run")
    plan = plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=(terminal,),
        layout=layout,
        scenario_id=scenario_id,
        raw_root=_default_raw_root(profile, layout),
        research_window=_research_window(_protocol(profile, scenario_id), None),
    )
    return {
        "profile_id": profile.profile_id,
        "status": plan["status"],
        "terminal_targets": [terminal],
        "targets": plan["targets"],
        "blockers": plan["blockers"],
    }


def inspect(
    *, profile: ResearchProfile, layout: StorageLayout, dataset_id: str
) -> dict[str, object]:
    spec = profile.research_datasets.get(dataset_id)
    root = layout.research_dataset_root(spec.market_id, spec.data_id)
    result: dict[str, object] = {
        "profile_id": profile.profile_id,
        "dataset_id": dataset_id,
        "required_columns": list(spec.required_columns),
        "exists": root.exists(),
        "instances": [],
    }
    if not root.exists():
        result.update(
            status=RunStatus.BLOCKED.value,
            blocker="dataset_instances_missing",
        )
        return result
    try:
        instances = discover_dataset_instances(root)
    except ValueError:
        result.update(
            status=RunStatus.FAILED.value,
            blocker="dataset_instance_invalid",
        )
        return result
    if not instances:
        result.update(
            status=RunStatus.BLOCKED.value,
            blocker="dataset_instances_missing",
        )
        return result
    instance_summaries = []
    try:
        for instance in instances:
            manifest = read_dataset_manifest_evidence(instance)
            envelope = validate_business_manifest_envelope(
                manifest[COMMON_MANIFEST_METADATA_KEY]
            )
            instance_summaries.append(
                {
                    "instance_identity": instance.name,
                    "status": "COMPLETE",
                    "schema_identity": envelope["schema_identity"],
                    "row_count": manifest["row_count"],
                }
            )
    except (FileNotFoundError, OSError, ValueError):
        result.update(
            status=RunStatus.FAILED.value,
            blocker="dataset_instance_invalid",
        )
        return result
    result.update(
        status=RunStatus.BLOCKED.value,
        blocker="request_identity_required",
        instances=instance_summaries,
    )
    return result


def _execute_action(
    *,
    profile: ResearchProfile,
    action: str,
    terminal_targets: tuple[str, ...],
    layout: StorageLayout,
    time_window: TimeWindow | None,
    raw_root: Path | None,
    start_date: str | None,
    end_date: str | None,
    rebuild: bool = False,
    research_window: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if action == "research run":
        scenario_id = _scenario_from_terminal(profile, action, terminal_targets)
        protocol = _protocol(profile, scenario_id)
        model_ids = tuple(
            f"{scenario_id}_{algorithm}"
            for algorithm in protocol.manifest().get("selection_tie_break", ())
        )
        preflight_model_specs(profile.models.get(model_id) for model_id in model_ids)
    with runtime_resource_policy_scope(None) as policy:
        return _execute_action_with_policy(
            profile=profile,
            action=action,
            terminal_targets=terminal_targets,
            layout=layout,
            time_window=time_window,
            raw_root=raw_root,
            start_date=start_date,
            end_date=end_date,
            rebuild=rebuild,
            research_window=research_window,
            policy=policy,
        )


def _execute_action_with_policy(
    *,
    profile: ResearchProfile,
    action: str,
    terminal_targets: tuple[str, ...],
    layout: StorageLayout,
    time_window: TimeWindow | None,
    raw_root: Path | None,
    start_date: str | None,
    end_date: str | None,
    rebuild: bool = False,
    research_window: Mapping[str, object] | None = None,
    policy: Any,
) -> dict[str, object]:
    source_root = raw_root or _default_raw_root(profile, layout)
    scenario_id = _scenario_from_terminal(profile, action, terminal_targets)
    protocol = _protocol(profile, scenario_id)
    software = production_software_identity()
    window = _research_window(
        protocol,
        research_window,
        time_window=time_window,
        start_date=start_date,
        end_date=end_date,
    )
    plan = _plan_targets(
        profile=profile,
        action=action,
        terminal_targets=terminal_targets,
        layout=layout,
        scenario_id=scenario_id,
        rebuild=rebuild,
        raw_root=source_root,
        research_window=window,
        software_identity=software,
    )
    specs = _target_specs(
        profile,
        action=action,
        protocol=protocol,
    )
    ordered = resolve_target_graph(specs, terminal_targets)
    plans = {item["target_id"]: item for item in plan["targets"]}  # type: ignore[index]
    results: dict[str, TargetResult] = {}
    rebuild_evidence: list[dict[str, object]] = []
    candidate_workspace: RebuildCandidateWorkspace | None = None
    candidate_materialized = False
    with ExitStack() as operation:
        for spec in ordered:
            planned = plans[spec.target_id]
            if planned["decision"] == PlanDecision.BLOCKED.value:
                results[spec.target_id] = TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason=str(planned["blocker"]),
                )
                continue
            blocked_dependency = next(
                (
                    dependency
                    for dependency in spec.dependencies
                    if results[dependency].status
                    in {TargetStatus.BLOCKED, TargetStatus.FAILED}
                ),
                None,
            )
            if blocked_dependency is not None:
                results[spec.target_id] = TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason=f"dependency_blocked: {blocked_dependency}",
                )
                continue
            if planned["decision"] == PlanDecision.REUSE.value:
                results[spec.target_id] = _reuse_target(
                    profile=profile,
                    layout=layout,
                    spec=spec,
                    planned=planned,
                    policy=policy,
                    raw_root=source_root,
                )
                continue
            is_candidate = rebuild and planned["matching_instance_identity"] is not None
            if is_candidate and candidate_workspace is None:
                candidate_workspace = operation.enter_context(
                    rebuild_candidate_workspace(layout)
                )
            output_layout = layout
            output_root: Path | None = None
            if candidate_workspace is not None and is_candidate:
                output_layout = candidate_workspace.layout
                output_root = _target_instance_root(
                    profile,
                    output_layout,
                    spec.target_id,
                    planned["requested_identity"],
                )
            target_layout = _target_layout(
                profile=profile,
                layout=layout,
                spec=spec,
                plans=plans,
            )
            if output_root is not None:
                target_layout = target_layout.with_target_roots(
                    {spec.target_id: output_root}
                )
            result = _build_target(
                profile=profile,
                protocol=protocol,
                layout=output_layout,
                spec=spec,
                planned=planned,
                research_window=window,
                software_identity=software,
                target_layout=target_layout,
                policy=policy,
                time_window=time_window,
                raw_root=source_root,
                start_date=start_date,
                end_date=end_date,
            )
            if is_candidate:
                if result.status is TargetStatus.BUILT and output_root is not None:
                    candidate_materialized = True
                    old_good_root = _target_instance_root(
                        profile,
                        layout,
                        spec.target_id,
                        planned["requested_identity"],
                    )
                    comparison = (
                        compare_semantic_artifacts(old_good_root, output_root)
                        if _catalog_data_spec(profile, spec.target_id) is None
                        else compare_datasets(
                            old_good_root,
                            output_root,
                            resource_policy=policy,
                        )
                    )
                    rebuild_evidence.append(
                        {
                            "target_id": spec.target_id,
                            "candidate_token": candidate_workspace.token,
                            "comparison_status": comparison.status,
                        }
                    )
                    if comparison.status != "MATCH":
                        candidate_workspace.preserve()
                        result = TargetResult(
                            target_id=spec.target_id,
                            status=TargetStatus.FAILED,
                            reason=(f"rebuild_comparison_{comparison.status.lower()}"),
                        )
                elif candidate_workspace is not None and (
                    candidate_materialized
                    or output_root is not None
                    and output_root.exists()
                ):
                    assert candidate_workspace is not None
                    candidate_workspace.preserve()
                    rebuild_evidence.append(
                        {
                            "target_id": spec.target_id,
                            "candidate_token": candidate_workspace.token,
                            "comparison_status": "BUILD_BLOCKED",
                        }
                    )
            results[spec.target_id] = result
    status_value = _run_status(results.values())
    top_level_blockers = [
        {"target_id": result.target_id, "reason": result.reason}
        for result in results.values()
        if result.reason is not None
    ]
    return {
        "stage": action,
        "profile_id": profile.profile_id,
        "status": status_value.value,
        "terminal_targets": list(sorted(terminal_targets)),
        "target_results": [result.payload() for result in results.values()],
        "blockers": top_level_blockers,
        "rebuild_evidence": rebuild_evidence,
    }


def _build_target(
    *,
    profile: ResearchProfile,
    protocol: Any,
    layout: StorageLayout,
    spec: TargetSpec,
    planned: dict[str, object],
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity,
    target_layout: StorageLayout,
    policy: Any,
    time_window: TimeWindow | None,
    raw_root: Path,
    start_date: str | None,
    end_date: str | None,
) -> TargetResult:
    if spec.producer_id is None:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_missing",
        )
    requested_identity = planned["requested_identity"]
    if type(requested_identity) is not str:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="requested_identity_unavailable",
        )
    data_spec = _catalog_data_spec(profile, spec.target_id)
    if data_spec is None:
        return _build_semantic_target(
            profile=profile,
            protocol=protocol,
            layout=layout,
            spec=spec,
            planned=planned,
            target_layout=target_layout,
            policy=policy,
            time_window=time_window,
            research_window=research_window,
            software_identity=software_identity,
            start_date=start_date,
            end_date=end_date,
        )
    binding = profile.producer_bindings[spec.producer_id]
    if binding.kind is not _producer_kind(data_spec):
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_kind_mismatch",
        )
    producer = binding.producer
    raw_identity: str | None = None
    if isinstance(data_spec, InterimDataSpec):
        if binding.source_identity is None:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_provider_missing",
            )
        try:
            raw_identity = binding.source_identity(raw_root, spec.target_id)
        except ValueError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_invalid",
            )
        if raw_identity is None:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="raw_content_identity_unavailable",
            )
        try:
            validate_instance_identity(raw_identity)
        except ValueError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="raw_content_identity_invalid",
            )
    envelope = _target_envelope(
        profile=profile,
        data_spec=data_spec,
        research_window=research_window,
        upstream_identities=planned["upstream_identities"],
        software_identity=software_identity,
        raw_content_identity=raw_identity,
    )
    if envelope.instance_identity != requested_identity:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="source_identity_changed",
        )
    output_root = _target_instance_root(
        profile,
        layout,
        spec.target_id,
        requested_identity,
    )
    instance_metadata = {
        COMMON_MANIFEST_METADATA_KEY: envelope.payload(),
    }
    if binding.kind is ProducerKind.INTERIM:
        try:
            result = producer(
                layout=target_layout,
                raw_root=raw_root,
                data_ids=(spec.target_id,),
                start_date=start_date,
                end_date=end_date,
                resource_policy=policy,
                instance_metadata=instance_metadata,
            )
        except ValueError as exc:
            if (
                str(exc) == "csmar_raw_export_identity_changed"
                and not raw_root.exists()
            ):
                return TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason="external_data_unavailable",
                )
            raise
        roots = getattr(result, "artifact_roots", {})
        skipped = tuple(getattr(result, "skipped", ()))
        if spec.target_id in skipped or spec.target_id not in roots:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="external_data_unavailable",
            )
        if Path(roots[spec.target_id]) != output_root:
            raise ValueError("producer_output_root_mismatch")
        _validate_published_target(output_root, requested_identity)
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BUILT,
            instance_identity=requested_identity,
        )
    if binding.kind is ProducerKind.PROCESSED:
        try:
            processed_spec = profile.processed.get(spec.target_id)
        except KeyError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="producer_binding_kind_mismatch",
            )
        frame = producer(
            processed_spec,
            layout=target_layout,
            time_window=time_window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            resource_policy=policy,
        )
        if getattr(frame, "empty", False):
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="processed_target_empty",
            )
        write_dataset(
            output_root,
            frame,
            spec=processed_spec,
            instance_metadata=instance_metadata,
            resource_policy=policy,
            borrowed_input=True,
        )
        _validate_published_target(output_root, requested_identity)
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BUILT,
            instance_identity=requested_identity,
        )
    if binding.kind is ProducerKind.DATASET:
        try:
            profile.research_datasets.get(spec.target_id)
        except KeyError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="producer_binding_kind_mismatch",
            )
        producer(
            layout=target_layout,
            dataset_catalog=profile.research_datasets,
            processed_catalog=profile.processed,
            dataset_id=spec.target_id,
            time_window=time_window,
            resource_policy=policy,
            instance_metadata=instance_metadata,
        )
        _validate_published_target(output_root, requested_identity)
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BUILT,
            instance_identity=requested_identity,
        )
    return TargetResult(
        target_id=spec.target_id,
        status=TargetStatus.BLOCKED,
        reason="capability_not_implemented",
    )


def _build_semantic_target(
    *,
    profile: ResearchProfile,
    protocol: Any,
    layout: StorageLayout,
    spec: TargetSpec,
    planned: Mapping[str, object],
    target_layout: StorageLayout,
    policy: Any,
    time_window: TimeWindow | None,
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity,
    start_date: str | None,
    end_date: str | None,
) -> TargetResult:
    del start_date, end_date
    binding = profile.producer_bindings.get(spec.producer_id or "")
    if binding is None:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_missing",
        )
    if binding.kind not in {
        ProducerKind.EVALUATION,
        ProducerKind.MODEL,
        ProducerKind.PREDICTION,
    }:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_kind_mismatch",
        )
    if time_window is None:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="research_window_required",
        )
    try:
        envelope = _target_envelope_for_spec(
            profile=profile,
            protocol=protocol,
            target_spec=spec,
            data_spec=None,
            research_window=research_window,
            upstream_identities=planned["upstream_identities"],
            software_identity=software_identity,
            raw_content_identity=None,
        )
        logical_root = _target_logical_root(profile, layout, spec.target_id)
        if logical_root is None:
            raise ValueError("target_logical_root_missing")
        if binding.kind is ProducerKind.EVALUATION:
            evaluation_spec = profile.evaluations.get(spec.target_id)
            evaluation_result = binding.producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                evaluation_catalog=profile.evaluations,
                evaluation_id=spec.target_id,
                time_window=time_window,
                resource_policy=policy,
            )

            def write(staging: Path) -> None:
                with evaluation_result as result:
                    write_evaluation_outputs_from_chunks(
                        result.output_chunks,
                        summary_factory=result.summaries,
                        spec=evaluation_spec,
                        output_spec=_output_spec_for_target(
                            spec.target_id,
                            evaluation_spec.market_id,
                            "evaluation",
                        ),
                        layout=StorageLayout(staging),
                        n_groups=evaluation_spec.n_groups,
                        resource_policy=policy,
                    )

            try:
                write_semantic_artifact(logical_root, envelope, write)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"evaluation_write_failed:{_semantic_failure_code(exc)}"
                ) from exc
        elif binding.kind is ProducerKind.MODEL:
            scenario_id = protocol.scenario_id
            model_ids = tuple(
                f"{scenario_id}_{algorithm}"
                for algorithm in protocol.manifest()["selection_tie_break"]
            )
            model_result = binding.producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                model_catalog=profile.models,
                dataset_id=scenario_id,
                artifact_id=spec.target_id,
                model_ids=model_ids,
                tie_break_order=model_ids,
                time_window=time_window,
                resource_policy=policy,
                protocol_manifest=protocol.manifest(),
            )
            if not isinstance(model_result, ModelSelectionResult):
                raise ValueError("model_selection_result_invalid")
            write_model_artifact(
                logical_root,
                model_id=model_result.model_id,
                envelope=envelope,
                quarterly_results=model_result.quarterly_results,
            )
        else:
            model_dependencies = tuple(
                dependency
                for dependency in spec.dependencies
                if (
                    profile.producer_bindings.get(dependency) is not None
                    and profile.producer_bindings[dependency].kind is ProducerKind.MODEL
                )
            )
            if len(model_dependencies) != 1:
                raise ValueError("prediction_model_dependency_invalid")
            model_target = model_dependencies[0]
            model_identity = planned["upstream_identities"].get(model_target)
            if type(model_identity) is not str:
                raise ValueError("model_identity_missing")
            persisted = read_model_artifact(
                _target_instance_root(profile, layout, model_target, model_identity),
                expected_identity=model_identity,
            )
            prediction_spec = PredictionSpec(
                prediction_id=spec.target_id,
                model_id=persisted.model_id,
                research_dataset_id=protocol.scenario_id,
                train_end="2018-12-31",
                test_start=_date_text(time_window.start),
                test_end=_date_text(time_window.end),
                prediction_mode="a_share_oos",
                prediction_column=RAW_PREDICTION_COLUMN,
            )
            prediction_result = binding.producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                prediction_spec=prediction_spec,
                model_result=persisted.result,
                model_metadata=persisted.result.metadata,
                resource_policy=policy,
            )

            def write(staging: Path) -> None:
                with prediction_result as result:
                    write_prediction_outputs(
                        spec=prediction_spec,
                        output_spec=_output_spec_for_target(
                            spec.target_id,
                            profile.market_id,
                            "prediction",
                        ),
                        layout=StorageLayout(staging),
                        output_chunks=result.output_chunks,
                        model_metadata=result.model_metadata,
                        prediction_metadata_factory=result.prediction_metadata_factory,
                        resource_policy=policy,
                    )

            try:
                write_semantic_artifact(logical_root, envelope, write)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"prediction_write_failed:{_semantic_failure_code(exc)}"
                ) from exc
        output_root = _target_instance_root(
            profile,
            layout,
            spec.target_id,
            envelope.instance_identity,
        )
        _validate_semantic_target_root(
            profile,
            spec,
            output_root,
            envelope.instance_identity,
        )
    except (ImportError, KeyError, TypeError, ValueError) as exc:
        reason = {
            ProducerKind.EVALUATION: "evaluation_build_failed",
            ProducerKind.MODEL: "model_selection_failed",
            ProducerKind.PREDICTION: "prediction_build_failed",
        }[binding.kind]
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.FAILED,
            reason=f"{reason}:{_semantic_failure_code(exc)}",
        )
    return TargetResult(
        target_id=spec.target_id,
        status=TargetStatus.BUILT,
        instance_identity=envelope.instance_identity,
    )


def _output_spec_for_target(
    target_id: str,
    market_id: object,
    run_type: str,
) -> OutputSpec:
    return OutputSpec(
        output_id=target_id,
        output_definition_id=f"{run_type}_outputs",
        run_id=f"run_{target_id}",
        market_id=market_id,
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _semantic_failure_code(exc: BaseException) -> str:
    value = str(exc)
    if (
        value
        and len(value) <= 96
        and re.fullmatch(r"[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)*", value)
    ):
        return value
    return type(exc).__name__.lower()


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _reuse_target(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    spec: TargetSpec,
    planned: dict[str, object],
    policy: Any,
    raw_root: Path,
) -> TargetResult:
    requested_identity = planned["requested_identity"]
    matching_identity = planned["matching_instance_identity"]
    if type(requested_identity) is not str or requested_identity != matching_identity:
        raise ValueError("reuse_identity_invalid")
    root = _target_instance_root(
        profile,
        layout,
        spec.target_id,
        requested_identity,
    )
    data_spec = _catalog_data_spec(profile, spec.target_id)
    if data_spec is None:
        _validate_semantic_target_root(profile, spec, root, requested_identity)
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.REUSED,
            instance_identity=requested_identity,
        )
    envelope = _validate_published_target(root, requested_identity)
    if isinstance(data_spec, InterimDataSpec):
        binding = profile.producer_bindings.get(spec.producer_id or "")
        if binding is None or binding.source_identity is None:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_provider_missing",
            )
        try:
            raw_identity = binding.source_identity(raw_root, spec.target_id)
            if raw_identity is None:
                return TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason="raw_content_identity_unavailable",
                )
            validate_instance_identity(raw_identity)
            metadata = envelope["semantic_metadata"]
            expected_raw_identity = (
                metadata.get("raw_content_identity")
                if isinstance(metadata, Mapping)
                else None
            )
            if (
                type(expected_raw_identity) is not str
                or raw_identity != expected_raw_identity
            ):
                return TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason="source_identity_changed",
                )
        except ValueError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_invalid",
            )
    inspect_physical_dataset(
        root,
        time_window=None,
        resource_policy=policy,
    )
    return TargetResult(
        target_id=spec.target_id,
        status=TargetStatus.REUSED,
        instance_identity=requested_identity,
    )


def _validate_published_target(
    root: Path,
    expected_identity: str,
) -> dict[str, object]:
    manifest = read_dataset_manifest_evidence(root)
    envelope = validate_business_manifest_envelope(
        manifest[COMMON_MANIFEST_METADATA_KEY]
    )
    if (
        envelope["instance_identity"] != expected_identity
        or root.name != expected_identity
    ):
        raise ValueError("published_instance_identity_mismatch")
    return envelope


def _validate_semantic_target_root(
    profile: ResearchProfile,
    spec: TargetSpec,
    root: Path,
    expected_identity: str,
) -> None:
    binding = profile.producer_bindings.get(spec.producer_id or "")
    if binding is None:
        raise ValueError("producer_binding_missing")
    envelope = read_semantic_artifact(root, expected_identity=expected_identity)
    if envelope["logical_artifact_id"] != spec.target_id:
        raise ValueError("semantic_artifact_logical_id_mismatch")
    if binding.kind is ProducerKind.MODEL:
        read_model_artifact(root, expected_identity=expected_identity)
        return
    if binding.kind is ProducerKind.EVALUATION:
        try:
            manifest, _members = validate_output_bundle(
                root / "outputs" / spec.target_id,
                expected_output_id=spec.target_id,
                expected_run_id=f"run_{spec.target_id}",
            )
            validate_evaluation_output_metadata_shape(manifest["metadata"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation_inner_artifact_invalid:{_semantic_failure_code(exc)}"
            ) from exc
        return
    if binding.kind is ProducerKind.PREDICTION:
        try:
            with read_prediction_output_bundle(
                root / "outputs" / spec.target_id,
                expected_output_id=spec.target_id,
                expected_run_id=f"run_{spec.target_id}",
            ) as (manifest, _lookup):
                validate_prediction_output_metadata_shape(manifest["metadata"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prediction_inner_artifact_invalid:{_semantic_failure_code(exc)}"
            ) from exc
        return


def _target_specs(
    profile: ResearchProfile,
    *,
    action: str,
    protocol: Any,
) -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = [
        TargetSpec(
            target_id=spec.data_id,
            producer_id=spec.data_id,
        )
        for spec in profile.interim.list()
    ]
    processed_specs = tuple(
        TargetSpec(
            target_id=spec.data_id,
            dependencies=spec.source_input_data_ids + spec.processed_input_data_ids,
            producer_id=spec.data_id,
        )
        for spec in profile.processed.list()
    )
    specs.extend(processed_specs)
    known_ids = {spec.target_id for spec in specs}
    unbound_source_ids = sorted(
        {
            source_id
            for spec in processed_specs
            for source_id in spec.dependencies
            if source_id not in known_ids
        }
    )
    specs.extend(
        TargetSpec(
            target_id=source_id,
            producer_id=source_id,
        )
        for source_id in unbound_source_ids
    )
    specs.extend(
        TargetSpec(
            target_id=spec.data_id,
            dependencies=spec.input_data_ids,
            producer_id=spec.data_id,
        )
        for spec in profile.research_datasets.list()
    )
    specs.extend(protocol.target_specs(action))
    return tuple(specs)


def _catalog_data_spec(
    profile: ResearchProfile,
    target_id: str,
) -> DataSpec | None:
    for catalog in (profile.interim, profile.processed, profile.research_datasets):
        try:
            return catalog.get(target_id)
        except KeyError:
            continue
    return None


def _producer_kind(data_spec: DataSpec) -> ProducerKind:
    if isinstance(data_spec, InterimDataSpec):
        return ProducerKind.INTERIM
    if isinstance(data_spec, ProcessedDataSpec):
        return ProducerKind.PROCESSED
    if isinstance(data_spec, ResearchDatasetSpec):
        return ProducerKind.DATASET
    raise ValueError("unsupported_data_spec_type")


def _target_envelope(
    *,
    profile: ResearchProfile,
    data_spec: DataSpec,
    research_window: Mapping[str, object],
    upstream_identities: Mapping[str, str],
    software_identity: ProductionSoftwareIdentity,
    raw_content_identity: str | None,
) -> BusinessManifestEnvelope:
    binding = profile.producer_bindings.get(data_spec.data_id)
    if binding is None:
        raise ValueError("producer_binding_missing")
    spec_payload = data_spec_payload(data_spec)
    semantic_metadata: dict[str, object] = {}
    if raw_content_identity is not None:
        semantic_metadata["raw_content_identity"] = raw_content_identity
    artifact_window = dict(research_window)
    artifact_window["access"] = "not_applicable"
    return build_business_manifest_envelope(
        logical_artifact_id=data_spec.data_id,
        schema_identity=canonical_json_hash(spec_payload),
        protocol_or_spec_identity=canonical_json_hash(spec_payload),
        software_identity=software_identity.identity_hash,
        research_window=artifact_window,
        upstream_identities=upstream_identities,
        semantic_metadata=semantic_metadata,
    )


def _target_envelope_for_spec(
    *,
    profile: ResearchProfile,
    protocol: Any,
    target_spec: TargetSpec,
    data_spec: DataSpec | None,
    research_window: Mapping[str, object],
    upstream_identities: Mapping[str, str],
    software_identity: ProductionSoftwareIdentity,
    raw_content_identity: str | None,
) -> BusinessManifestEnvelope:
    if data_spec is not None:
        return _target_envelope(
            profile=profile,
            data_spec=data_spec,
            research_window=research_window,
            upstream_identities=upstream_identities,
            software_identity=software_identity,
            raw_content_identity=raw_content_identity,
        )
    if target_spec.producer_id is None:
        raise ValueError("producer_binding_missing")
    binding = profile.producer_bindings.get(target_spec.producer_id)
    if binding is None:
        raise ValueError("producer_binding_missing")
    semantic_payload = _semantic_target_payload(
        profile=profile,
        protocol=protocol,
        target_spec=target_spec,
        binding=binding,
    )
    spec_identity = canonical_json_hash(semantic_payload)
    return build_business_manifest_envelope(
        logical_artifact_id=target_spec.target_id,
        schema_identity=spec_identity,
        protocol_or_spec_identity=spec_identity,
        software_identity=software_identity.identity_hash,
        research_window=research_window,
        upstream_identities=upstream_identities,
        semantic_metadata={"target_contract": semantic_payload},
    )


def _semantic_target_payload(
    *,
    profile: ResearchProfile,
    protocol: Any,
    target_spec: TargetSpec,
    binding,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_id": target_spec.target_id,
        "dependencies": list(target_spec.dependencies),
        "producer_kind": binding.kind.value,
        "protocol": protocol.manifest(),
    }
    if binding.kind is ProducerKind.EVALUATION:
        spec = profile.evaluations.get(target_spec.target_id)
        payload["evaluation_spec"] = {
            "evaluation_id": spec.evaluation_id,
            "market_id": str(spec.market_id),
            "research_dataset_id": spec.research_dataset_id,
            "signals": [
                {
                    "signal_id": signal.signal_id,
                    "source_kind": signal.source_kind,
                    "source_column": signal.source_column,
                    "direction": signal.direction,
                    "transform": signal.transform,
                }
                for signal in spec.signals
            ],
            "label_columns": list(spec.label_columns),
            "evaluation_axis": spec.evaluation_axis,
            "filter_mode": spec.filter_mode,
            "n_groups": spec.n_groups,
            "run_type": spec.run_type,
        }
        payload["output_files"] = [
            "metrics.parquet",
            "grouped_label_values.parquet",
            "ic_summary.parquet",
            "group_spread_summary.parquet",
            "summary.md",
        ]
    elif binding.kind is ProducerKind.MODEL:
        model_ids = tuple(
            f"{protocol.scenario_id}_{algorithm}"
            for algorithm in protocol.manifest().get("selection_tie_break", ())
        )
        payload["model_specs"] = [
            _model_spec_payload(profile.models.get(model_id)) for model_id in model_ids
        ]
        payload["model_artifact_schema"] = "quarterly_route"
    elif binding.kind is ProducerKind.PREDICTION:
        model_dependencies = tuple(
            dependency
            for dependency in target_spec.dependencies
            if (
                profile.producer_bindings.get(dependency) is not None
                and profile.producer_bindings[dependency].kind is ProducerKind.MODEL
            )
        )
        if len(model_dependencies) != 1:
            raise ValueError("prediction_model_dependency_invalid")
        payload["prediction_schema"] = {
            "prediction_id": target_spec.target_id,
            "research_dataset_id": protocol.scenario_id,
            "model_target_id": model_dependencies[0],
            "prediction_mode": "a_share_oos",
            "columns": [
                "timestamp",
                "asset_id",
                "model_id",
                "raw_prediction",
                "alpha_score",
            ],
        }
    return payload


def _model_spec_payload(spec: object) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "model_algorithm": spec.model_algorithm,
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "label_column": spec.label_column,
        "sample_filter_mode": spec.sample_filter_mode,
        "hyperparameters": dict(spec.hyperparameters),
        "random_seed": spec.random_seed,
        "max_train_rows": spec.max_train_rows,
    }


def _research_window(
    protocol: Any,
    requested: Mapping[str, object] | None,
    *,
    time_window: TimeWindow | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    if requested is not None:
        return dict(requested)
    window = time_window
    if window is None and start_date is not None and end_date is not None:
        window = TimeWindow(start_date, end_date)
    if window is None:
        protocol_window = getattr(protocol, "time_window", None)
        if isinstance(protocol_window, TimeWindow):
            window = protocol_window
    if window is None:
        return {
            "start": "1970-01-01",
            "end": "1970-01-01",
            "access": "not_applicable",
        }
    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
        "access": "development",
    }


def _matching_instance(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    target_id: str,
    requested_identity: str,
) -> tuple[str | None, str | None]:
    logical_root = _target_logical_root(profile, layout, target_id)
    if logical_root is None:
        return None, "target_logical_root_missing"
    try:
        if _catalog_data_spec(profile, target_id) is None:
            instances = discover_semantic_artifact_instances(logical_root)
        else:
            instances = discover_dataset_instances(logical_root)
    except ValueError:
        return None, "persisted_instance_invalid"
    matching = tuple(
        instance.name for instance in instances if instance.name == requested_identity
    )
    if matching:
        return matching[0], None
    return None, None


def _target_instance_root(
    profile: ResearchProfile,
    layout: StorageLayout,
    target_id: str,
    instance_identity: object,
) -> Path:
    if type(instance_identity) is not str:
        raise ValueError("target_instance_identity_missing")
    logical_root = _target_logical_root(profile, layout, target_id)
    if logical_root is None:
        raise ValueError("target_logical_root_missing")
    return layout.instance_root(logical_root, instance_identity)


def _instance_path_exists(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    target_id: str,
    instance_identity: str,
) -> bool:
    logical_root = _target_logical_root(profile, layout, target_id)
    return (
        logical_root is not None
        and layout.instance_root(
            logical_root,
            instance_identity,
        ).exists()
    )


def _target_layout(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    spec: TargetSpec,
    plans: Mapping[str, object],
) -> StorageLayout:
    roots: dict[str, Path] = {}
    for target_id in (*spec.dependencies, spec.target_id):
        planned = plans.get(target_id)
        if not isinstance(planned, Mapping):
            raise ValueError("target_plan_missing")
        identity = planned.get("requested_identity")
        if type(identity) is not str:
            raise ValueError("target_instance_identity_missing")
        roots[target_id] = _target_instance_root(
            profile,
            layout,
            target_id,
            identity,
        )
    return layout.with_target_roots(roots)


def _target_logical_root(
    profile: ResearchProfile,
    layout: StorageLayout,
    target_id: str,
) -> Path | None:
    try:
        spec = profile.interim.get(target_id)
    except KeyError:
        pass
    else:
        return layout.interim_data_root(spec.source_id, spec.data_id)
    try:
        spec = profile.processed.get(target_id)
    except KeyError:
        pass
    else:
        return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    try:
        spec = profile.research_datasets.get(target_id)
    except KeyError:
        binding = profile.producer_bindings.get(target_id)
        if binding is None or binding.kind in {
            ProducerKind.INTERIM,
            ProducerKind.PROCESSED,
            ProducerKind.DATASET,
        }:
            return None
        return layout.semantic_artifact_root(profile.market_id, target_id)
    return layout.research_dataset_root(spec.market_id, spec.data_id)


def _default_raw_root(profile: ResearchProfile, layout: StorageLayout) -> Path:
    return layout.raw_source_root(profile.interim.list()[0].source_id)


def _protocol(profile: ResearchProfile, scenario_id: str) -> Any:
    return profile.protocol(scenario_id=scenario_id, access_window="development")


def _protocol_terminal_target(
    profile: ResearchProfile,
    scenario_id: str,
    action: str,
) -> str:
    terminals = _protocol(profile, scenario_id).terminal_targets(action)
    if len(terminals) != 1:
        raise ValueError("protocol_terminal_target_invalid")
    return terminals[0]


def _scenario_from_terminal(
    profile: ResearchProfile,
    action: str,
    terminal_targets: tuple[str, ...],
) -> str:
    if action == "data materialize":
        return _default_scenario_id(profile)
    if action == "dataset build":
        return terminal_targets[0]
    datasets = tuple(spec.data_id for spec in profile.research_datasets.list())
    for scenario_id in datasets:
        if terminal_targets[0] == _protocol_terminal_target(
            profile,
            scenario_id,
            action,
        ):
            return scenario_id
    raise ValueError("protocol_terminal_target_invalid")


def _default_scenario_id(profile: ResearchProfile) -> str:
    return profile.research_datasets.list()[0].data_id


def _run_status(results: Iterable[TargetResult]) -> RunStatus:
    statuses = tuple(result.status for result in results)
    if TargetStatus.FAILED in statuses:
        return RunStatus.FAILED
    if TargetStatus.BLOCKED in statuses:
        return RunStatus.BLOCKED
    return RunStatus.COMPLETE
