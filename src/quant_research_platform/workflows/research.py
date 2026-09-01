"""The single target-DAG resolver and public research workflow owner."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.targets import (
    PlanDecision,
    ProducerKind,
    RunStatus,
    TargetPlan,
    TargetResult,
    TargetSpec,
    TargetStatus,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence


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
        root = _physical_target_root(profile, layout, spec.target_id)
        blocker: str | None = None
        if root is not None and root.exists():
            blocker = "semantic_identity_not_available"
        elif rebuild:
            blocker = "semantic_identity_not_available"
        else:
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
            elif spec.producer_id not in profile.producer_bindings:
                blocker = "producer_binding_missing"
        decision = PlanDecision.BLOCKED if blocker else PlanDecision.BUILD
        plan = TargetPlan(
            target_id=spec.target_id,
            upstream_target_ids=tuple(sorted(spec.dependencies)),
            producer_id=spec.producer_id,
            decision=decision,
            blocker=blocker,
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
) -> dict[str, object]:
    selected = tuple(data_ids or (spec.data_id for spec in profile.interim.list()))
    if plan:
        return plan_targets(
            profile=profile,
            action="data materialize",
            terminal_targets=selected,
            layout=layout,
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
) -> dict[str, object]:
    if plan:
        return plan_targets(
            profile=profile,
            action="dataset build",
            terminal_targets=(dataset_id,),
            layout=layout,
            scenario_id=dataset_id,
            rebuild=rebuild,
        )
    return _execute_action(
        profile=profile,
        action="dataset build",
        terminal_targets=(dataset_id,),
        layout=layout,
        time_window=TimeWindow(start_date, end_date),
        raw_root=None,
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
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
    terminal = _protocol_terminal_target(profile, scenario_id, "research run")
    if plan:
        return plan_targets(
            profile=profile,
            action="research run",
            terminal_targets=(terminal,),
            layout=layout,
            scenario_id=scenario_id,
            rebuild=rebuild,
        )
    return _execute_action(
        profile=profile,
        action="research run",
        terminal_targets=(terminal,),
        layout=layout,
        time_window=TimeWindow(start_date, end_date),
        raw_root=None,
        start_date=start_date,
        end_date=end_date,
        rebuild=rebuild,
    )


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
        )
    return _execute_action(
        profile=profile,
        action="report build",
        terminal_targets=(terminal,),
        layout=layout,
        time_window=None,
        raw_root=None,
        start_date=None,
        end_date=None,
        rebuild=rebuild,
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
    )
    blockers = list(plan["blockers"])
    blockers.append(
        {"target_id": terminal, "reason": "semantic_identity_not_available"}
    )
    return {
        "profile_id": profile.profile_id,
        "status": RunStatus.BLOCKED.value,
        "terminal_targets": [terminal],
        "targets": plan["targets"],
        "blockers": blockers,
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
    }
    if not root.exists():
        result.update(
            status=RunStatus.BLOCKED.value,
            blocker="dataset_manifest_missing",
        )
        return result
    try:
        manifest = read_dataset_manifest_evidence(root)
    except FileNotFoundError:
        result.update(
            status=RunStatus.BLOCKED.value,
            blocker="dataset_manifest_missing",
        )
    except (OSError, ValueError):
        result.update(
            status=RunStatus.FAILED.value,
            blocker="dataset_manifest_invalid",
        )
    else:
        result.update(
            status=RunStatus.BLOCKED.value,
            physical_manifest_valid=True,
            blocker="semantic_identity_not_available",
            row_count=manifest["row_count"],
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
) -> dict[str, object]:
    plan = plan_targets(
        profile=profile,
        action=action,
        terminal_targets=terminal_targets,
        layout=layout,
        scenario_id=_scenario_from_terminal(profile, action, terminal_targets),
        rebuild=rebuild,
    )
    specs = _target_specs(
        profile,
        action=action,
        protocol=_protocol(
            profile,
            _scenario_from_terminal(profile, action, terminal_targets),
        ),
    )
    ordered = resolve_target_graph(specs, terminal_targets)
    plans = {item["target_id"]: item for item in plan["targets"]}  # type: ignore[index]
    results: dict[str, TargetResult] = {}
    source_root = raw_root or _default_raw_root(profile, layout)
    with runtime_resource_policy_scope(None) as policy:
        for spec in ordered:
            planned = plans[spec.target_id]
            if planned["decision"] == PlanDecision.BLOCKED.value:
                results[spec.target_id] = TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    blocker=str(planned["blocker"]),
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
                    blocker=f"dependency_blocked: {blocked_dependency}",
                )
                continue
            results[spec.target_id] = _build_target(
                profile=profile,
                layout=layout,
                spec=spec,
                policy=policy,
                time_window=time_window,
                raw_root=source_root,
                start_date=start_date,
                end_date=end_date,
            )
    status_value = _run_status(results.values())
    top_level_blockers = [
        {"target_id": result.target_id, "reason": result.blocker}
        for result in results.values()
        if result.blocker is not None
    ]
    if any(
        result.status in {TargetStatus.BUILT, TargetStatus.REUSED}
        for result in results.values()
    ):
        status_value = RunStatus.BLOCKED
        top_level_blockers.append(
            {
                "target_id": terminal_targets[0],
                "reason": "semantic_identity_not_available",
            }
        )
    return {
        "stage": action,
        "profile_id": profile.profile_id,
        "status": status_value.value,
        "terminal_targets": list(sorted(terminal_targets)),
        "target_results": [result.payload() for result in results.values()],
        "blockers": top_level_blockers,
    }


def _build_target(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    spec: TargetSpec,
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
            blocker="producer_binding_missing",
        )
    binding = profile.producer_bindings[spec.producer_id]
    producer = binding.producer
    if binding.kind is ProducerKind.INTERIM:
        try:
            result = producer(
                layout=layout,
                raw_root=raw_root,
                data_ids=(spec.target_id,),
                start_date=start_date,
                end_date=end_date,
                resource_policy=policy,
            )
        except ValueError as exc:
            if (
                str(exc) == "csmar_raw_export_identity_changed"
                and not raw_root.exists()
            ):
                return TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    blocker="external_data_unavailable",
                )
            raise
        roots = getattr(result, "artifact_roots", {})
        skipped = tuple(getattr(result, "skipped", ()))
        if spec.target_id in skipped or spec.target_id not in roots:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                blocker="external_data_unavailable",
            )
        return TargetResult(target_id=spec.target_id, status=TargetStatus.BUILT)
    if binding.kind is ProducerKind.PROCESSED:
        try:
            processed_spec = profile.processed.get(spec.target_id)
        except KeyError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                blocker="producer_binding_kind_mismatch",
            )
        frame = producer(
            processed_spec,
            layout=layout,
            time_window=time_window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            resource_policy=policy,
        )
        write_dataset(
            layout.processed_data_root(
                processed_spec.market_id,
                processed_spec.kind,
                processed_spec.data_id,
            ),
            frame,
            spec=processed_spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        return TargetResult(target_id=spec.target_id, status=TargetStatus.BUILT)
    if binding.kind is ProducerKind.DATASET:
        try:
            profile.research_datasets.get(spec.target_id)
        except KeyError:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                blocker="producer_binding_kind_mismatch",
            )
        producer(
            layout=layout,
            dataset_catalog=profile.research_datasets,
            processed_catalog=profile.processed,
            dataset_id=spec.target_id,
            time_window=time_window,
            resource_policy=policy,
        )
        return TargetResult(target_id=spec.target_id, status=TargetStatus.BUILT)
    return TargetResult(
        target_id=spec.target_id,
        status=TargetStatus.BLOCKED,
        blocker="capability_not_implemented",
    )


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


def _physical_target_root(
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
        return None
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
