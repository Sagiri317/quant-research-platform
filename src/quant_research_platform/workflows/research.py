"""Scientific request preparation, target-DAG execution, and qualification."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.artifacts.outputs import (
    read_prediction_output_bundle,
    validate_output_bundle,
    write_evaluation_outputs_from_chunks,
    write_prediction_outputs,
)
from quant_research_platform.artifacts.outputs.bundle_validation import (
    _PredictionOutputSession,
    _validate_prediction_output_bundle,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    validate_evaluation_output_metadata_shape,
)
from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_output_metadata_shape,
)
from quant_research_platform.artifacts.portfolio_ledger import (
    PORTFOLIO_LEDGER_TABLE_COLUMNS,
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.research import (
    _instance_root as _reference_instance_root,
)
from quant_research_platform.artifacts.research import (
    _validate_reference_envelope,
    read_research_run_record,
    read_validated_reference,
)
from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_COLUMNS,
    STATISTICS_TABLE_GROUPS,
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    model_spec_payload,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.portfolios import (
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.predictions import (
    A_SHARE_PREDICTION_PRIMARY_KEY,
    PREDICTION_PRIMARY_KEY,
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
    prediction_output_columns,
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
from quant_research_platform.contracts.time import (
    TimeWindow,
    is_datetime_window,
    time_value_isoformat,
)
from quant_research_platform.markets.implementations.crypto.portfolio import (
    validate_maker_calibration,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.research.datasets.io import _research_dataset_access_window
from quant_research_platform.research.models.persisted import (
    read_crypto_model_artifact,
    read_quarterly_model_artifact,
    write_crypto_model_artifact,
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.production import _crypto_model_chronology
from quant_research_platform.research.models.supervised import preflight_model_specs
from quant_research_platform.storage.datasets.comparison import compare_datasets
from quant_research_platform.storage.datasets.inspection import (
    inspect_dataset as inspect_physical_dataset,
)
from quant_research_platform.storage.datasets.inspection import (
    inspect_dataset_session,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import (
    COMMON_MANIFEST_METADATA_KEY,
    data_spec_payload,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.experiments import (
    verify_trial_ledger_snapshot,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
    validate_instance_identity,
    verified_file_handle,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.storage.rebuild import (
    RebuildCandidateWorkspace,
    rebuild_candidate_workspace,
)
from quant_research_platform.storage.semantic_artifacts import (
    _member_identity,
    _read_artifact_manifest,
    _visit_member_paths,
    compare_semantic_artifacts,
    read_semantic_artifact_record,
    verify_semantic_artifact_snapshot,
    write_semantic_artifact,
)
from quant_research_platform.storage.software_identity import (
    ProductionSoftwareIdentity,
    _artifact_software_identity,
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


@dataclass(frozen=True)
class _TargetRequest:
    spec: TargetSpec
    time_window: TimeWindow
    dependencies: tuple[tuple[str, TimeWindow], ...]
    publish: bool = True

    @property
    def key(self):
        return self.spec.target_id, self.time_window


def _target_requests(profile, action, protocol, ordered, terminals, window):
    specs = {spec.target_id: spec for spec in ordered}
    if protocol.protocol_id != "a_share_low_frequency" or action not in {
        "data materialize",
        "dataset build",
    }:
        windows = {
            spec.target_id: _semantic_target_time_window(
                action,
                protocol,
                spec,
                _catalog_data_spec(profile, spec.target_id),
                window,
            )
            for spec in ordered
        }
        return tuple(
            _TargetRequest(
                spec,
                windows[spec.target_id],
                tuple(
                    (dependency, windows[dependency])
                    for dependency in sorted(spec.dependencies)
                ),
            )
            for spec in ordered
        )

    from quant_research_platform.research.protocols.a_share import (
        processed_input_window,
    )

    requests = {}

    def visit(target_id, requested):
        spec = specs[target_id]
        data_spec = _catalog_data_spec(profile, target_id)
        target_window = _semantic_target_time_window(
            action, protocol, spec, data_spec, requested
        )
        key = target_id, target_window
        if key not in requests:
            dependency_window = _window_payload(
                processed_input_window(data_spec, target_window),
                access=str(window["access"]),
            )
            dependencies = tuple(
                visit(dependency, dependency_window)
                for dependency in sorted(spec.dependencies)
            )
            requests[key] = _TargetRequest(spec, target_window, dependencies)
        return key

    for target_id in terminals:
        visit(target_id, window)
    # Public target IDs stay unique; an explicit core request takes precedence
    # over the same target's internal history instance.
    preferred = {}
    for request in requests.values():
        target_id = request.spec.target_id
        previous = preferred.get(target_id)
        if previous is None or request.time_window.start > previous[1].start:
            preferred[target_id] = request.key
    return tuple(
        replace(request, publish=preferred[request.spec.target_id] == request.key)
        for request in requests.values()
    )


def _prepare_plan(
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
    if action == "report build":
        raise ValueError("report_run_id_required")
    terminals = tuple(sorted(set(terminal_targets)))
    scenario = scenario_id or (
        terminals[0]
        if action == "dataset build" and terminals
        else _default_scenario_id(profile)
    )
    protocol = profile.protocol(scenario_id=scenario, access_window="development")
    software = software_identity or production_software_identity()
    window = _research_window(protocol, research_window)
    return dict(
        profile=profile,
        action=action,
        terminal_targets=terminals,
        layout=layout,
        protocol=protocol,
        rebuild=rebuild,
        raw_root=raw_root,
        research_window=window,
        software_identity=software,
    )


def _plan_targets(
    *,
    profile: ResearchProfile,
    action: str,
    terminal_targets: Iterable[str],
    layout: StorageLayout,
    protocol: Any,
    rebuild: bool = False,
    raw_root: Path | None = None,
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity | None = None,
    reuse_sessions=None,
) -> tuple[
    tuple[tuple[_TargetRequest, TargetPlan], ...], dict[Path, DatasetSession | tuple]
]:
    if action == "report build":
        raise ValueError("report_run_id_required")
    terminals = tuple(sorted(set(terminal_targets)))
    protocol_terminals = tuple(sorted(set(protocol.terminal_targets(action))))
    if action != "data materialize" and terminals != protocol_terminals:
        raise ValueError("protocol_terminal_targets_invalid")
    software = software_identity or production_software_identity()
    window = research_window
    reuse_sessions = {} if reuse_sessions is None else reuse_sessions
    source_root = raw_root or _default_raw_root(layout)
    specs = _target_specs(
        profile,
        action=action,
        protocol=protocol,
    )
    ordered = resolve_target_graph(specs, terminals)
    requested = set(terminals)
    by_request: dict[tuple[str, TimeWindow], TargetPlan] = {}
    requests = _target_requests(profile, action, protocol, ordered, terminals, window)
    frozen = protocol.manifest().get("frozen_development")
    effective_requests = []
    for request in requests:
        spec = request.spec
        rebuild_target = rebuild and request.publish and spec.target_id in requested
        dependencies = {
            dependency: by_request[(dependency, dependency_window)]
            for dependency, dependency_window in request.dependencies
        }
        if frozen is not None and spec.target_id in frozen["targets"]:
            target = frozen["targets"][spec.target_id]
            reference = target["reference"]
            plan = TargetPlan(
                target_id=spec.target_id,
                upstream_target_ids=(),
                decision=PlanDecision.REUSE,
                requested_identity=reference["instance_identity"],
                matching_instance_identity=reference["instance_identity"],
                upstream_identities=target["upstream_identities"],
                rebuild=False,
                producer_kind=profile.producer_bindings[spec.target_id].kind,
                schema_identity=reference["schema_identity"],
                artifact_root=reference["path"],
            )
            by_request[request.key] = plan
            effective_requests.append(request)
            continue
        data_spec = _catalog_data_spec(profile, spec.target_id)
        absent = tuple(
            name
            for name in _optional_source_ids(profile, protocol, spec)
            if name in dependencies
            and dependencies[name].blocker == "raw_content_identity_unavailable"
        )
        if absent:
            spec = replace(
                spec,
                dependencies=tuple(
                    name for name in spec.dependencies if name not in absent
                ),
            )
            request = replace(
                request,
                spec=spec,
                dependencies=tuple(
                    item for item in request.dependencies if item[0] not in absent
                ),
            )
        effective_requests.append(request)
        target_window = request.time_window
        target_research_window = _window_payload(
            target_window,
            access=str(window["access"]),
        )
        upstream_identities = {
            dependency: dependencies[dependency].requested_identity
            for dependency in spec.dependencies
            if dependencies[dependency].requested_identity is not None
        }
        upstream_identities.update(
            {
                name: canonical_json_hash({"source_id": name, "presence": "ABSENT"})
                for name in absent
            }
        )
        blocker: str | None = None
        requested_identity: str | None = None
        matching_instance_identity: str | None = None
        producer_kind: ProducerKind | None = None
        schema_identity: str | None = None
        artifact_root: str | None = None
        decision = PlanDecision.BLOCKED
        blocked_dependency = next(
            (
                dependency
                for dependency in spec.dependencies
                if dependencies[dependency].decision is PlanDecision.BLOCKED
            ),
            None,
        )
        if blocked_dependency is not None:
            blocker = f"dependency_blocked: {blocked_dependency}"
        else:
            binding = profile.producer_bindings.get(spec.target_id)
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
            elif (
                protocol.protocol_id == "a_share_low_frequency"
                and binding.kind is ProducerKind.PREDICTION
                and protocol.access_window == "development"
                and protocol.selection_group_for_target(spec.target_id) == "global"
                and pd.Timestamp(target_window.end)
                < pd.Timestamp(protocol.manifest()["development_window"]["end"])
            ):
                blocker = "global_selection_requires_complete_development"
            else:
                producer_kind = binding.kind
                raw_identity = None
                if (
                    isinstance(data_spec, InterimDataSpec)
                    or binding.source_identity is not None
                ):
                    interim_raw_root = binding.raw_root or (
                        _interim_raw_root(source_root, data_spec)
                        if isinstance(data_spec, InterimDataSpec)
                        else source_root
                    )
                    try:
                        raw_identity = binding.source_identity(
                            interim_raw_root,
                            spec.target_id,
                        )
                    except PortfolioTargetBlocked as exc:
                        blocker = str(exc)
                    except ValueError:
                        blocker = "source_identity_invalid"
                if (
                    blocker is None
                    and (
                        isinstance(data_spec, InterimDataSpec)
                        or binding.source_identity is not None
                    )
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
                    try:
                        envelope = _target_envelope_for_spec(
                            profile=profile,
                            protocol=protocol,
                            target_spec=spec,
                            data_spec=data_spec,
                            research_window=target_research_window,
                            upstream_identities=upstream_identities,
                            software_identity=software,
                            raw_content_identity=raw_identity,
                        )
                    except PortfolioTargetBlocked as exc:
                        blocker = str(exc)
                if blocker is None:
                    requested_identity = envelope.instance_identity
                    schema_identity = envelope.schema_identity
                    logical_root = _target_logical_root(profile, layout, spec.target_id)
                    if logical_root is None:
                        raise ValueError("target_logical_root_missing")
                    artifact_root = logical_root.relative_to(
                        layout.project_root
                    ).as_posix()
                    matching_instance_identity, blocker = _matching_instance(
                        profile=profile,
                        layout=layout,
                        spec=spec,
                        protocol=protocol,
                        requested_identity=requested_identity,
                        reuse_sessions=reuse_sessions,
                    )
                    if (
                        blocker == "persisted_instance_invalid"
                        and rebuild_target
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
                            if rebuild_target or matching_instance_identity is None
                            else PlanDecision.REUSE
                        )
        if blocker is not None:
            decision = PlanDecision.BLOCKED
        plan = TargetPlan(
            target_id=spec.target_id,
            upstream_target_ids=tuple(sorted(spec.dependencies)),
            decision=decision,
            blocker=blocker,
            requested_identity=requested_identity,
            matching_instance_identity=matching_instance_identity,
            upstream_identities=upstream_identities,
            rebuild=rebuild_target,
            producer_kind=producer_kind,
            schema_identity=schema_identity,
            artifact_root=artifact_root,
        )
        by_request[request.key] = plan
    needed = set()
    effective_by_key = {request.key: request for request in effective_requests}

    def include(key):
        if key not in needed:
            needed.add(key)
            for dependency in effective_by_key[key].dependencies:
                include(dependency)

    for request in effective_requests:
        if request.publish and request.spec.target_id in terminals:
            include(request.key)
    return tuple(
        (request, by_request[request.key])
        for request in effective_requests
        if request.key in needed
    ), reuse_sessions


def _prepare_materialization(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    raw_root: Path | None,
    start_date: str | None,
    end_date: str | None,
    data_ids: tuple[str, ...] | None = None,
    rebuild: bool = False,
) -> dict[str, object]:
    protocol = profile.protocol(
        scenario_id=_default_scenario_id(profile), access_window="development"
    )
    selected = tuple(
        data_ids
        or protocol.terminal_targets("data materialize")
        or (spec.data_id for spec in profile.interim.list())
    )
    window = _research_window(protocol, None, start_date=start_date, end_date=end_date)
    return dict(
        profile=profile,
        action="data materialize",
        terminal_targets=selected,
        layout=layout,
        protocol=protocol,
        rebuild=rebuild,
        raw_root=raw_root,
        research_window=window,
        software_identity=production_software_identity(),
    )


def _prepare_dataset(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    dataset_id: str,
    start_date: str,
    end_date: str,
    rebuild: bool = False,
    raw_root: Path | None = None,
) -> dict[str, object]:
    protocol = profile.protocol(scenario_id=dataset_id, access_window="development")
    window = _research_window(protocol, None, start_date=start_date, end_date=end_date)
    source = raw_root or _default_raw_root(layout)
    return dict(
        profile=profile,
        action="dataset build",
        terminal_targets=(dataset_id,),
        layout=layout,
        protocol=protocol,
        rebuild=rebuild,
        raw_root=source,
        research_window=window,
        software_identity=production_software_identity(),
    )


def _validate_stage_selection(profile, layout, protocol):
    configuration = protocol.configuration()
    if "stage_selection" not in configuration:
        return
    stages = ("dataset", "evaluation", "models", "research")
    for source in configuration["stage_selection"]["sources"]:
        record = read_research_run_record(
            layout.research_run_path(source["run_id"]),
            expected_profile_id=profile.profile_id,
        )
        previous = record.payload()["protocol"]["research_configuration"]
        if (
            record.status is not RunStatus.COMPLETE
            or record.research_window["access"] != "development"
            or record.scenario_id != protocol.scenario_id
            or previous["windows"] != configuration["windows"]
            or stages.index(previous["stage"]) >= stages.index(configuration["stage"])
            or source["target_id"] not in record.terminal_targets
        ):
            raise ValueError("research_stage_source_ineligible")
        reference = next(
            (
                item
                for item in record.target_results
                if item.target_id == source["target_id"]
            ),
            None,
        )
        if (
            reference is None
            or reference.instance_identity != source["instance_identity"]
        ):
            raise ValueError("research_stage_source_identity_mismatch")
        read_validated_reference(layout, reference)


def _prepare_research(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    research_plan: Mapping[str, object],
    raw_root: Path | None = None,
    holdout: bool = False,
    development_run_id: str | None = None,
    plan: bool = False,
    rebuild: bool = False,
) -> tuple[dict[str, Any], str | None]:
    from quant_research_platform.profiles.research import bind_research_configuration

    profile = bind_research_configuration(
        profile,
        research_plan,
        tardis_root=(_default_raw_root(layout) if raw_root is None else Path(raw_root))
        / "tardis",
        access_window="holdout" if holdout else "development",
    )
    scenario_id = research_plan["scenario_id"]
    selected_window = research_plan["windows"]["holdout" if holdout else "development"]
    start_date, end_date = selected_window["start"], selected_window["end"]
    protocol = profile.protocol(scenario_id=scenario_id, access_window="development")
    if holdout and research_plan["stage"] != "research":
        raise ValueError("holdout_requires_research_stage")
    _validate_stage_selection(profile, layout, protocol)
    if not plan:
        model_ids = tuple(
            dict.fromkeys(
                model_id
                for target in resolve_target_graph(
                    _target_specs(profile, action="research run", protocol=protocol),
                    protocol.terminal_targets("research run"),
                )
                if profile.producer_bindings.get(target.target_id) is not None
                and profile.producer_bindings[target.target_id].kind
                is ProducerKind.MODEL
                for model_id in protocol.model_ids_for_target(target.target_id)
            )
        )
        specs = tuple(profile.models.get(model_id) for model_id in model_ids)
        if str(profile.market_id) == "a_share":
            from quant_research_platform.research.models.production import (
                _selection_model_specs,
            )

            configurations = {
                item["model_id"]: item for item in protocol.configuration()["models"]
            }
            specs = tuple(
                candidate
                for spec in specs
                for candidate in _selection_model_specs(
                    spec,
                    configurations[spec.model_id].get(
                        "parameter_candidates",
                        [configurations[spec.model_id]["hyperparameters"]],
                    ),
                    n_train=1,
                    root_seed=protocol.root_seed,
                    quarter_id="preflight",
                )
            )
        preflight_model_specs(specs)
    preflight_blocker = None
    reuse_sessions = {}
    if holdout:
        protocol, preflight_blocker = _frozen_holdout_protocol(
            profile,
            layout,
            protocol,
            development_run_id,
            start_date,
            end_date,
            reuse_sessions=reuse_sessions,
        )
        profile = replace(profile, protocol=lambda **kwargs: protocol)
    else:
        if development_run_id is not None:
            raise ValueError("development_run_id_requires_holdout")
        _require_development_window(protocol, start_date, end_date)
    terminals = tuple(protocol.terminal_targets("research run"))
    return dict(
        profile=profile,
        action="research run",
        terminal_targets=terminals,
        layout=layout,
        protocol=protocol,
        rebuild=rebuild,
        raw_root=raw_root,
        research_window=_research_window(
            protocol, None, start_date=start_date, end_date=end_date
        ),
        software_identity=production_software_identity(),
        reuse_sessions=reuse_sessions,
    ), preflight_blocker


def _frozen_holdout_protocol(
    profile: ResearchProfile,
    layout: StorageLayout,
    protocol: Any,
    development_run_id: str | None,
    start_date: str,
    end_date: str,
    *,
    reuse_sessions=None,
) -> tuple[Any, str | None]:
    if development_run_id is None:
        raise PermissionError("holdout_frozen_development_identity_unavailable")
    requested = TimeWindow(start_date, end_date)
    holdout_window = TimeWindow(**protocol.manifest()["holdout_window"])
    if requested.start < holdout_window.start or requested.end > holdout_window.end:
        raise PermissionError("holdout_window_invalid")
    record = read_research_run_record(
        layout.research_run_path(development_run_id),
        expected_profile_id=profile.profile_id,
    )
    recorded_protocol = record.payload()["protocol"]
    requested_protocol = protocol.manifest()
    if str(profile.market_id) == "a_share":
        from quant_research_platform.catalog.models import (
            _effective_model_configurations,
        )

        for manifest in (recorded_protocol, requested_protocol):
            configuration = manifest["research_configuration"]
            configuration["models"] = _effective_model_configurations(
                configuration["models"]
            )
    if (
        record.status is not RunStatus.COMPLETE
        or record.scenario_id != protocol.scenario_id
        or recorded_protocol != requested_protocol
        or record.research_window["access"] != "development"
        or pd.Timestamp(record.research_window["start"])
        != pd.Timestamp(protocol.manifest()["development_window"]["start"])
        or pd.Timestamp(record.research_window["end"])
        != pd.Timestamp(protocol.manifest()["development_window"]["end"])
        or set(record.terminal_targets)
        != set(protocol.terminal_targets("research run"))
    ):
        raise ValueError("holdout_development_run_invalid")
    references = {item.target_id: item for item in record.target_results}
    artifacts = {}
    reuse_sessions = {} if reuse_sessions is None else reuse_sessions
    frozen_targets = protocol.frozen_target_ids()
    specs = {item.target_id: item for item in protocol.target_specs("research run")}
    frozen = {
        "run_id": record.run_id,
        "run_identity": canonical_json_hash(record.payload()),
        "targets": {},
    }
    for target_id in dict.fromkeys((*record.terminal_targets, *frozen_targets)):
        reference = references.get(target_id)
        if reference is None:
            raise ValueError("holdout_development_reference_missing")
        if (
            target_id not in frozen_targets
            and reference.producer_kind is not ProducerKind.PORTFOLIO
        ):
            read_validated_reference(layout, reference)
            continue
        root = _reference_instance_root(layout, reference)
        envelope, _members, manifest_identity, _operational = _read_artifact_manifest(
            root, expected_identity=reference.instance_identity
        )
        _validate_reference_envelope(reference, envelope)
        identities = {"manifest.json": manifest_identity}
        directories = _visit_member_paths(
            root,
            lambda name, path, identities=identities: identities.__setitem__(
                name, _member_identity(path)
            ),
        )
        semantic_snapshots = {root: (identities, directories)}
        trial_snapshots = {}
        artifacts[target_id] = _validate_semantic_target_root(
            profile,
            protocol,
            specs[target_id],
            root,
            reference.instance_identity,
            trial_layout=layout,
            expected_member_identities=identities,
            semantic_snapshots=semantic_snapshots,
            trial_snapshots=trial_snapshots,
            reuse_sessions=reuse_sessions,
        )
        session = (envelope, semantic_snapshots, trial_snapshots, artifacts[target_id])
        _verify_semantic_reuse(session)
        reuse_sessions[root] = session
        if target_id in frozen_targets:
            frozen["targets"][target_id] = {
                "reference": reference.payload(),
                "upstream_identities": envelope["upstream_identities"],
            }
    result = protocol.freeze_development(record, references, frozen, artifacts)
    for target_id in artifacts:
        _verify_semantic_reuse(
            reuse_sessions[_reference_instance_root(layout, references[target_id])]
        )
    return result


def _require_development_window(protocol: Any, start_date: str, end_date: str) -> None:
    development = protocol.manifest().get("development_window")
    if type(development) is not dict:
        return
    start = development.get("start")
    end = development.get("end")
    if type(start) is not str or type(end) is not str:
        raise PermissionError("holdout_frozen_development_identity_unavailable")
    allowed = TimeWindow(start, end)
    requested = TimeWindow(start_date, end_date)
    if pd.Timestamp(allowed.start) > pd.Timestamp(requested.start) or pd.Timestamp(
        allowed.end
    ) < pd.Timestamp(requested.end):
        raise PermissionError("holdout_frozen_development_identity_unavailable")


def _prepare_report(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    run_id: str,
) -> dict[str, object]:
    record = read_research_run_record(
        layout.research_run_path(run_id), expected_profile_id=profile.profile_id
    )
    from quant_research_platform.profiles.research import bind_research_configuration

    profile = bind_research_configuration(
        profile,
        record.payload()["protocol"]["research_configuration"],
        tardis_root=layout.raw_source_root("tardis"),
        access_window=record.research_window["access"],
    )
    binding = profile.producer_bindings.get(f"{record.scenario_id}_report_artifact")
    if binding is None or binding.kind is not ProducerKind.REPORT:
        raise ValueError("report_producer_binding_missing")
    return dict(
        layout=layout,
        record=record,
        market_id=profile.market_id,
        produce_report=binding.producer,
        software_identity=_target_software_identity(
            binding, production_software_identity()
        ),
        execution_targets=profile.protocol(record.scenario_id).execution_targets()
        if str(profile.market_id) == "crypto"
        else (),
    )


def _execute_targets(
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
    observe=None,
    reuse_sessions=None,
) -> dict[str, object]:
    source_root = raw_root or _default_raw_root(layout)
    software = software_identity or production_software_identity()
    window = research_window
    instance_plans, reuse_sessions = _plan_targets(
        profile=profile,
        action=action,
        terminal_targets=terminal_targets,
        layout=layout,
        protocol=protocol,
        rebuild=rebuild,
        raw_root=source_root,
        research_window=window,
        software_identity=software,
        reuse_sessions=reuse_sessions,
    )
    plans = {request.key: plan for request, plan in instance_plans}
    instance_results: dict[tuple[str, TimeWindow], TargetResult] = {}
    if reference_sink is not None:
        reference_sink.update(
            {
                request.spec.target_id: plan
                for request, plan in instance_plans
                if request.publish
            }
        )
    results = result_sink if result_sink is not None else {}

    def record_result(request, result):
        instance_results[request.key] = result
        if request.publish:
            results[request.spec.target_id] = result

    rebuild_evidence: list[dict[str, object]] = []
    candidate_workspace: RebuildCandidateWorkspace | None = None
    candidate_materialized = False
    with ExitStack() as operation:
        for request, planned in instance_plans:
            spec = request.spec
            if observe is not None:
                observe()
            if planned.decision is PlanDecision.BLOCKED:
                record_result(
                    request,
                    TargetResult(
                        target_id=spec.target_id,
                        status=TargetStatus.BLOCKED,
                        reason=str(planned.blocker),
                    ),
                )
                continue
            blocked_dependency = next(
                (
                    dependency
                    for dependency, dependency_window in request.dependencies
                    if instance_results[(dependency, dependency_window)].status
                    in {TargetStatus.BLOCKED, TargetStatus.FAILED}
                ),
                None,
            )
            if blocked_dependency is not None:
                record_result(
                    request,
                    TargetResult(
                        target_id=spec.target_id,
                        status=TargetStatus.BLOCKED,
                        reason=f"dependency_blocked: {blocked_dependency}",
                    ),
                )
                continue
            is_candidate = (
                planned.rebuild and planned.matching_instance_identity is not None
            )
            try:
                if planned.decision is PlanDecision.REUSE:
                    result = _reuse_target(
                        profile=profile,
                        protocol=protocol,
                        layout=layout,
                        spec=spec,
                        planned=planned,
                        raw_root=source_root,
                        time_window=request.time_window,
                        session=reuse_sessions.get(
                            _target_instance_root(
                                profile,
                                layout,
                                spec.target_id,
                                planned.requested_identity,
                            )
                        ),
                    )
                else:
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
                            planned.requested_identity,
                        )
                    target_layout = _target_layout(
                        profile=profile,
                        protocol=protocol,
                        layout=layout,
                        spec=spec,
                        plans={
                            dependency: plans[(dependency, dependency_window)]
                            for dependency, dependency_window in request.dependencies
                        }
                        | {spec.target_id: planned},
                    )
                    if output_root is not None:
                        target_layout = target_layout.with_target_roots(
                            {spec.target_id: output_root}
                        )
                    result = _build_target(
                        run_id=run_id,
                        profile=profile,
                        protocol=protocol,
                        layout=output_layout,
                        spec=spec,
                        planned=planned,
                        time_window=request.time_window,
                        access=str(window["access"]),
                        software_identity=software,
                        target_layout=target_layout,
                        raw_root=source_root,
                        reuse_sessions=reuse_sessions,
                    )
            except PortfolioTargetBlocked as exc:
                result = TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason=str(exc),
                )
            except (ImportError, KeyError, TypeError, ValueError) as exc:
                kind = profile.producer_bindings[spec.target_id].kind
                reason = {
                    ProducerKind.INTERIM: "interim_build_failed",
                    ProducerKind.PROCESSED: "processed_build_failed",
                    ProducerKind.DATASET: "dataset_build_failed",
                    ProducerKind.EVALUATION: "evaluation_build_failed",
                    ProducerKind.MODEL: "model_selection_failed",
                    ProducerKind.PREDICTION: "prediction_build_failed",
                    ProducerKind.STATISTIC: "statistics_build_failed",
                    ProducerKind.PORTFOLIO: "portfolio_build_failed",
                }[kind]
                result = TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.FAILED,
                    reason=f"{reason}:{_semantic_failure_code(exc)}",
                )
            if is_candidate:
                if result.status is TargetStatus.BUILT and output_root is not None:
                    candidate_materialized = True
                    old_good_root = _target_instance_root(
                        profile,
                        layout,
                        spec.target_id,
                        planned.requested_identity,
                    )
                    comparison = (
                        compare_semantic_artifacts(
                            old_good_root, output_root, trial_layout=layout
                        )
                        if _catalog_data_spec(profile, spec.target_id) is None
                        else compare_datasets(old_good_root, output_root)
                    )
                    rebuild_evidence.append(
                        {
                            "target_id": spec.target_id,
                            "instance_identity": planned.requested_identity,
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
                    and filesystem_exists(output_root)
                ):
                    assert candidate_workspace is not None
                    candidate_workspace.preserve()
                    rebuild_evidence.append(
                        {
                            "target_id": spec.target_id,
                            "instance_identity": planned.requested_identity,
                            "candidate_token": candidate_workspace.token,
                            "comparison_status": "BUILD_BLOCKED",
                        }
                    )
            record_result(request, result)
            if observe is not None:
                observe()
    status_value = _run_status(instance_results.values())
    top_level_blockers = [
        {"target_id": result.target_id, "reason": result.reason}
        for result in instance_results.values()
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
    planned: TargetPlan,
    time_window: TimeWindow,
    access: str,
    software_identity: ProductionSoftwareIdentity,
    target_layout: StorageLayout,
    raw_root: Path,
    run_id: str | None = None,
    reuse_sessions=None,
) -> TargetResult:
    if spec.target_id not in profile.producer_bindings:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_missing",
        )
    requested_identity = planned.requested_identity
    if type(requested_identity) is not str:
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="requested_identity_unavailable",
        )
    data_spec = _catalog_data_spec(profile, spec.target_id)
    target_research_window = _window_payload(time_window, access=access)
    if data_spec is None:
        return _build_semantic_target(
            run_id=run_id,
            profile=profile,
            protocol=protocol,
            layout=layout,
            spec=spec,
            planned=planned,
            target_layout=target_layout,
            time_window=time_window,
            research_window=target_research_window,
            software_identity=software_identity,
            raw_root=raw_root,
            reuse_sessions=reuse_sessions,
        )
    binding = profile.producer_bindings[spec.target_id]
    if binding.kind is not _producer_kind(data_spec):
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BLOCKED,
            reason="producer_binding_kind_mismatch",
        )
    producer = binding.producer
    raw_identity: str | None = None
    if isinstance(data_spec, InterimDataSpec) or binding.source_identity is not None:
        interim_raw_root = binding.raw_root or (
            _interim_raw_root(raw_root, data_spec)
            if isinstance(data_spec, InterimDataSpec)
            else raw_root
        )
        if binding.source_identity is None:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_provider_missing",
            )
        try:
            raw_identity = binding.source_identity(
                interim_raw_root,
                spec.target_id,
            )
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
        research_window=target_research_window,
        upstream_identities=planned.upstream_identities,
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
                raw_root=interim_raw_root,
                data_ids=(spec.target_id,),
                start_date=time_value_isoformat(
                    time_window.start, preserve_datetime=True
                ),
                end_date=time_value_isoformat(time_window.end, preserve_datetime=True),
                instance_metadata=instance_metadata,
            )
        except ValueError as exc:
            if str(
                exc
            ) == "csmar_raw_export_identity_changed" and not filesystem_exists(
                interim_raw_root
            ):
                raise PortfolioTargetBlocked("external_data_unavailable") from exc
            raise
        roots = result.artifact_roots
        skipped = result.skipped
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
        _verify_absent_sources(
            profile,
            _optional_source_ids(profile, protocol, spec),
            planned.upstream_identities,
            raw_root,
        )
        processed_spec = data_spec
        produced = producer(
            processed_spec,
            layout=target_layout,
            time_window=time_window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
            **{
                "raw_root": binding.raw_root or raw_root,
                "expected_source_identity": raw_identity,
            }
            if binding.source_identity is not None
            else {},
        )
        with produced as (frame, producer_metadata, verify):
            if getattr(frame, "empty", False):
                return TargetResult(
                    target_id=spec.target_id,
                    status=TargetStatus.BLOCKED,
                    reason="processed_target_empty",
                )

            def verify_inputs(receipt):
                if verify is not None:
                    verify(receipt)
                _verify_absent_sources(
                    profile,
                    _optional_source_ids(profile, protocol, spec),
                    planned.upstream_identities,
                    raw_root,
                )

            # The producer may finish its diagnostics while the writer drains it.
            producer_metadata.update(instance_metadata)
            write_dataset(
                output_root,
                frame,
                spec=processed_spec,
                instance_metadata=producer_metadata,
                staged_validator=verify_inputs,
            )
        _validate_published_target(output_root, requested_identity)
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.BUILT,
            instance_identity=requested_identity,
        )
    if binding.kind is ProducerKind.DATASET:
        producer(
            layout=target_layout,
            dataset_catalog=profile.research_datasets,
            processed_catalog=profile.processed,
            dataset_id=spec.target_id,
            time_window=time_window,
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
    time_window: TimeWindow | None,
    research_window: Mapping[str, object],
    software_identity: ProductionSoftwareIdentity,
    raw_root: Path,
    run_id: str | None = None,
    reuse_sessions=None,
) -> TargetResult:
    binding = profile.producer_bindings.get(spec.target_id)
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
        ProducerKind.STATISTIC,
        ProducerKind.PORTFOLIO,
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
    _verify_absent_sources(
        profile,
        _optional_source_ids(profile, protocol, spec),
        planned.upstream_identities,
        raw_root,
    )
    raw_root = binding.raw_root or raw_root
    raw_identity = (
        binding.source_identity(raw_root, spec.target_id)
        if binding.source_identity is not None
        else None
    )
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=protocol,
        target_spec=spec,
        data_spec=None,
        research_window=research_window,
        upstream_identities=planned.upstream_identities,
        software_identity=software_identity,
        raw_content_identity=raw_identity,
    )
    if envelope.instance_identity != planned.requested_identity:
        raise PortfolioTargetBlocked("source_identity_changed")
    logical_root = _target_logical_root(profile, layout, spec.target_id)
    if logical_root is None:
        raise ValueError("target_logical_root_missing")
    target_protocol = envelope.payload()["semantic_metadata"]["target_contract"][
        "protocol"
    ]
    if protocol.protocol_id == "crypto_hft" and binding.kind is ProducerKind.PREDICTION:
        model_target = f"{protocol.scenario_id}_models"
        persisted = read_crypto_model_artifact(
            target_layout.target_root(model_target, Path("__missing_model__")),
            expected_identity=planned.upstream_identities[model_target],
            trial_layout=target_layout,
        )

        def write(staging):
            results = binding.producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                persisted_model=persisted,
                time_window=time_window,
                access_window=protocol.access_window,
            )
            with closing_iterator(iter(results)) as active:
                for result in active:
                    with result:
                        write_prediction_outputs(
                            output_chunks=result.output_chunks,
                            spec=result.spec,
                            output_spec=_output_spec_for_target(
                                result.spec.prediction_id, "crypto", "prediction"
                            ),
                            layout=StorageLayout(staging),
                            model_metadata=result.model_metadata,
                            prediction_metadata_factory=result.prediction_metadata_factory,
                        )

        write_semantic_artifact(logical_root, envelope, write)
    elif binding.kind is ProducerKind.EVALUATION:
        evaluation_spec = profile.evaluations.get(spec.target_id)
        evaluation_result = binding.producer(
            layout=target_layout,
            dataset_catalog=profile.research_datasets,
            evaluation_catalog=profile.evaluations,
            evaluation_id=spec.target_id,
            time_window=time_window,
            **{"protocol_manifest": target_protocol}
            if protocol.protocol_id == "crypto_hft"
            else {"access_window": protocol.access_window},
        )

        with evaluation_result as result:

            def write(staging: Path) -> None:
                write_evaluation_outputs_from_chunks(
                    result.output_chunks,
                    summary_factory=result.summaries,
                    spec=evaluation_spec,
                    output_spec=_output_spec_for_target(
                        spec.target_id, evaluation_spec.market_id, "evaluation"
                    ),
                    layout=StorageLayout(staging),
                )

            try:
                write_semantic_artifact(logical_root, envelope, write)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"evaluation_write_failed:{_semantic_failure_code(exc)}"
                ) from exc
    elif binding.kind is ProducerKind.MODEL:
        if run_id is None:
            raise ValueError("model_run_id_required")
        model_ids = protocol.model_ids_for_target(spec.target_id)
        dataset_spec = profile.research_datasets.get(protocol.scenario_id)
        dataset_root = target_layout.target_root(
            dataset_spec.data_id,
            target_layout.research_dataset_root(
                dataset_spec.market_id, dataset_spec.data_id
            ),
        )
        read_window = _research_dataset_access_window(dataset_spec.market_id, None)
        empty_scope = False
        if protocol.protocol_id == "crypto_hft":
            folds, _holdout = _crypto_model_chronology(target_protocol, time_window)
            empty_scope = not folds
            if folds:
                read_window = TimeWindow(
                    min(fold.train.start for fold in folds),
                    max(fold.validation.end for fold in folds),
                )
        dataset_session = (
            None
            if empty_scope
            else open_dataset_session(
                dataset_root, columns=None, time_window=read_window
            )
        )
        model_result = partial(
            binding.producer,
            layout=target_layout,
            dataset_session=dataset_session,
            dataset_catalog=profile.research_datasets,
            model_catalog=profile.models,
            dataset_id=protocol.scenario_id,
            model_ids=model_ids,
            time_window=time_window,
            protocol_manifest=target_protocol,
            run_id=run_id,
            research_protocol_identity=canonical_json_hash(target_protocol),
            model_target_id=spec.target_id,
            backend_versions=software_identity.payload["dependency_versions"],
        )
        if protocol.protocol_id == "crypto_hft":
            write_crypto_model_artifact(
                logical_root,
                envelope=envelope,
                model_results=model_result,
                trial_layout=target_layout,
                verify_source=(
                    (lambda: None) if empty_scope else dataset_session.verify
                ),
            )
        else:
            write_quarterly_model_artifact(
                logical_root,
                model_id=spec.target_id,
                envelope=envelope,
                model_results=model_result,
                trial_layout=target_layout,
                verify_source=dataset_session.verify,
            )
    elif binding.kind is ProducerKind.PREDICTION:
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
        model_identity = planned.upstream_identities.get(model_target)
        if type(model_identity) is not str:
            raise ValueError("model_identity_missing")
        persisted = read_quarterly_model_artifact(
            target_layout.target_root(model_target, Path("__missing_model_target__")),
            expected_identity=model_identity,
            trial_layout=target_layout,
        )
        prediction_spec = PredictionSpec(
            prediction_id=spec.target_id,
            model_id=persisted.model_id,
            research_dataset_id=protocol.scenario_id,
            train_end=None,
            test_start=time_value_isoformat(time_window.start, date_only=True),
            test_end=time_value_isoformat(time_window.end, date_only=True),
            prediction_mode="a_share_oos",
            prediction_column=RAW_PREDICTION_COLUMN,
        )
        prediction_result = binding.producer(
            layout=target_layout,
            dataset_catalog=profile.research_datasets,
            prediction_spec=prediction_spec,
            persisted_model=persisted,
            selection_group=protocol.selection_group_for_target(spec.target_id),
            access_window=protocol.access_window,
        )

        def write(staging: Path) -> None:
            with prediction_result as result:
                write_prediction_outputs(
                    spec=prediction_spec,
                    output_spec=_output_spec_for_target(
                        spec.target_id, profile.market_id, "prediction"
                    ),
                    layout=StorageLayout(staging),
                    output_chunks=result.output_chunks,
                    model_metadata=result.model_metadata,
                    prediction_metadata_factory=result.prediction_metadata_factory,
                )

        try:
            write_semantic_artifact(logical_root, envelope, write)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prediction_write_failed:{_semantic_failure_code(exc)}"
            ) from exc
    elif (
        binding.kind is ProducerKind.STATISTIC and protocol.protocol_id == "crypto_hft"
    ):
        group = protocol.statistics_table_group(spec.target_id)
        producer = partial(
            binding.producer,
            layout=target_layout,
            protocol_manifest=target_protocol,
            time_window=time_window,
            upstream_identities=planned.upstream_identities,
        )
        if group == "crypto_maker_calibration":
            write_statistics_artifact(
                logical_root,
                envelope=envelope,
                tables=partial(producer, raw_root=raw_root),
                table_group=group,
                validate_tables=validate_maker_calibration,
            )
        else:
            tables = producer()
            write_statistics_artifact(
                logical_root,
                envelope=envelope,
                tables=tables,
                table_group=group,
            )

    elif (
        binding.kind is ProducerKind.STATISTIC
        and spec.target_id == f"{protocol.scenario_id}_portfolio_statistics"
    ):
        candidates = protocol.portfolio_candidates()
        ledgers = {
            candidate.candidate_id: read_portfolio_ledger_artifact(
                target_layout.target_root(
                    f"{protocol.scenario_id}_{candidate.candidate_id}_ledger",
                    Path("__missing_ledger__"),
                ),
                expected_identity=planned.upstream_identities[
                    f"{protocol.scenario_id}_{candidate.candidate_id}_ledger"
                ],
            )
            for candidate in candidates
        }
        tables = binding.producer(
            ledgers=ledgers,
            candidates=candidates,
            protocol_manifest=target_protocol,
            time_window=time_window,
            layout=target_layout,
        )
        write_statistics_artifact(
            logical_root,
            envelope=envelope,
            tables=tables,
            table_group="portfolio",
        )
    elif (
        binding.kind is ProducerKind.STATISTIC
        and spec.target_id == protocol.factor_statistics_target_id
    ):
        tables = binding.producer(
            layout=target_layout,
            dataset_catalog=profile.research_datasets,
            dataset_id=protocol.scenario_id,
            protocol_manifest=target_protocol,
            time_window=time_window,
        )
        write_statistics_artifact(
            logical_root,
            envelope=envelope,
            tables=tables,
            table_group="factor",
        )
    elif binding.kind is ProducerKind.STATISTIC:
        dependency_kinds = {
            dependency: profile.producer_bindings.get(dependency)
            for dependency in spec.dependencies
        }
        prediction_dependencies = tuple(
            dependency
            for dependency, dependency_binding in dependency_kinds.items()
            if dependency_binding is not None
            and dependency_binding.kind is ProducerKind.PREDICTION
        )
        model_dependencies = tuple(
            dependency
            for dependency, dependency_binding in dependency_kinds.items()
            if dependency_binding is not None
            and dependency_binding.kind is ProducerKind.MODEL
        )
        dataset_dependencies = tuple(
            dependency
            for dependency, dependency_binding in dependency_kinds.items()
            if dependency_binding is not None
            and dependency_binding.kind is ProducerKind.DATASET
        )
        if (
            len(prediction_dependencies) != 1
            or len(model_dependencies) != 1
            or len(dataset_dependencies) != 1
        ):
            raise ValueError("statistics_dependencies_invalid")
        prediction_target = prediction_dependencies[0]
        model_target = model_dependencies[0]
        dataset_target = dataset_dependencies[0]
        model_identity = planned.upstream_identities.get(model_target)
        if type(model_identity) is not str:
            raise ValueError("model_identity_missing")
        persisted = read_quarterly_model_artifact(
            target_layout.target_root(model_target, Path("__missing_model_target__")),
            expected_identity=model_identity,
            trial_layout=target_layout,
        )
        prediction_root = target_layout.target_root(
            prediction_target,
            Path("__missing_prediction_target__"),
        )
        if prediction_root.name.startswith("__missing_"):
            raise ValueError("prediction_target_root_missing")
        global_statistics = None
        if target_protocol["selection_group"] == "quarterly":
            global_target = protocol.statistics_target_ids[0]
            global_statistics = read_statistics_artifact(
                target_layout.target_root(
                    global_target, Path("__missing_global_statistics__")
                ),
                expected_identity=planned.upstream_identities[global_target],
            )
        with _prediction_consumer(
            prediction_root,
            reuse_sessions=reuse_sessions,
            expected_output_id=prediction_target,
            expected_run_id=f"run_{prediction_target}",
            time_window=time_window,
        ) as (_manifest, prediction_lookup):
            tables = binding.producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                dataset_id=dataset_target,
                prediction_id=prediction_target,
                persisted_model=persisted,
                protocol_manifest=target_protocol,
                time_window=time_window,
                prediction_lookup=prediction_lookup,
                global_statistics=global_statistics,
            )
        write_statistics_artifact(
            logical_root,
            envelope=envelope,
            tables=tables,
        )
    elif (
        binding.kind is ProducerKind.PORTFOLIO and protocol.protocol_id == "crypto_hft"
    ):
        target = protocol.execution_target_for_target(spec.target_id)
        candidate = target.candidate

        def produce_ledger(emit):
            callback = (
                {"on_ledger": emit}
                if target.execution_mode == "continuous"
                else {"on_episode": emit}
            )
            binding.producer(
                layout=target_layout,
                candidate=candidate,
                protocol_manifest=target_protocol,
                dataset_catalog=profile.research_datasets,
                output_window=time_window,
                upstream_identities=planned.upstream_identities,
                expected_funding_identity=raw_identity,
                raw_root=raw_root,
                **callback,
            )

        if target.execution_mode == "continuous":
            writer = write_portfolio_ledger_artifact
        else:
            from quant_research_platform.artifacts.crypto_hft import (
                write_crypto_episode_artifact,
            )

            writer = write_crypto_episode_artifact
        writer(logical_root, envelope=envelope, produce=produce_ledger)
    elif binding.kind is ProducerKind.PORTFOLIO:
        candidate = protocol.portfolio_candidate_for_target(spec.target_id)
        prediction_target = candidate.prediction_target_id
        prediction_root = target_layout.target_root(
            prediction_target,
            Path("__missing_prediction_target__"),
        )
        if prediction_root.name.startswith("__missing_"):
            raise ValueError("prediction_target_root_missing")
        prediction_window = time_window
        if candidate.construction == "benchmark_enhanced":
            prediction_window = protocol.target_window(
                "research run", None, time_window, candidate.prediction_target_id
            )
        with _prediction_consumer(
            prediction_root,
            reuse_sessions=reuse_sessions,
            expected_output_id=prediction_target,
            expected_run_id=f"run_{prediction_target}",
            time_window=prediction_window,
            family_selection=candidate.prediction_selector,
        ) as (prediction_manifest, prediction_lookup):
            lineage = prediction_manifest["metadata"]["model_metadata"]
            persisted = read_quarterly_model_artifact(
                _target_instance_root(
                    profile,
                    target_layout,
                    lineage["model_artifact_id"],
                    lineage["model_artifact_identity"],
                ),
                expected_identity=lineage["model_artifact_identity"],
                trial_layout=target_layout,
            )
            final_calibrations = []
            portfolio_chunks = binding.producer(
                candidate=candidate,
                layout=target_layout,
                upstream_identities=planned.upstream_identities,
                output_window=time_window,
                input_window=protocol.data_window,
                formal_dataset_window=protocol.formal_dataset_window,
                prediction_window=prediction_window,
                frozen_calibration=dict(protocol.frozen_alpha).get(
                    (
                        candidate.selection_group,
                        candidate.prediction_selector,
                        candidate.rebalance_schedule,
                    )
                ),
                on_final_calibration=final_calibrations.append,
                prediction_lookup=prediction_lookup,
                persisted_model=persisted,
                access_window=target_protocol["access_window"],
                assumptions=protocol.execution_assumptions,
                risk_config=protocol.risk_config,
                optimizer_config=protocol.optimizer_config,
            )

            def produce_ledger(emit):
                with closing_iterator(iter(portfolio_chunks)) as chunks:
                    for chunk in chunks:
                        emit(chunk)
                    _verify_absent_sources(
                        profile,
                        _optional_source_ids(profile, protocol, spec),
                        planned.upstream_identities,
                        raw_root,
                    )
                    if reuse_sessions is not None and prediction_root in reuse_sessions:
                        _verify_semantic_reuse(reuse_sessions[prediction_root])

            def final_calibration():
                if len(final_calibrations) != 1:
                    raise ValueError("portfolio_final_calibration_missing")
                return final_calibrations[0]

            write_portfolio_ledger_artifact(
                logical_root,
                envelope=envelope,
                produce=produce_ledger,
                final_calibration=final_calibration
                if candidate.construction == "benchmark_enhanced"
                and target_protocol["access_window"] == "development"
                else None,
            )
    else:
        raise ValueError("producer_binding_kind_invalid")
    output_root = _target_instance_root(
        profile,
        layout,
        spec.target_id,
        envelope.instance_identity,
    )
    if binding.kind is ProducerKind.PREDICTION and reuse_sessions is not None:
        reuse_sessions[output_root] = _semantic_reuse_session(
            profile,
            protocol,
            spec,
            output_root,
            envelope.instance_identity,
            layout=target_layout,
            reuse_sessions=reuse_sessions,
        )
    else:
        _validate_semantic_target_root(
            profile,
            protocol,
            spec,
            output_root,
            envelope.instance_identity,
            trial_layout=target_layout,
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
    )


def _semantic_failure_code(exc: BaseException) -> str:
    value = str(exc)
    code, separator, context = value.partition(": ")
    if (
        isinstance(exc, ValueError)
        and separator
        and code
        in {
            "missing_filter_columns",
            "null_filter_column_values",
            "invalid_filter_column_dtype",
        }
    ):
        from quant_research_platform.contracts.filtering import FILTER_COLUMNS

        if context in FILTER_COLUMNS:
            return f"{code}:{context}"
    if (
        isinstance(exc, ValueError)
        and value
        and len(value) <= 96
        and re.fullmatch(r"[a-zA-Z0-9_.-]+(?::[a-zA-Z0-9_.-]+)*", value)
    ):
        return value
    reason = type(exc).__name__.lower()
    trace = exc.__traceback__
    origin = None
    while trace is not None:
        module = trace.tb_frame.f_globals.get("__name__", "")
        if module.startswith("quant_research_platform."):
            origin = f"{module}.{trace.tb_frame.f_code.co_name}:{trace.tb_lineno}"
        trace = trace.tb_next
    return reason if origin is None else f"{reason}:{origin}"


def _optional_source_ids(profile, protocol, spec):
    data_spec = _catalog_data_spec(profile, spec.target_id)
    if isinstance(data_spec, ProcessedDataSpec):
        return data_spec.optional_source_input_data_ids
    selector = getattr(protocol, "optional_source_input_data_ids", None)
    return () if data_spec is not None or selector is None else selector(spec.target_id)


def _verify_absent_sources(profile, sources, upstream_identities, raw_root):
    for name in sources:
        if upstream_identities.get(name) != canonical_json_hash(
            {"source_id": name, "presence": "ABSENT"}
        ):
            continue
        source_spec = profile.interim.get(name)
        binding = profile.producer_bindings[name]
        if (
            binding.source_identity(
                binding.raw_root or _interim_raw_root(raw_root, source_spec), name
            )
            is not None
        ):
            raise PortfolioTargetBlocked("optional_source_presence_changed")


def _reuse_target(
    *,
    profile: ResearchProfile,
    protocol: Any,
    layout: StorageLayout,
    spec: TargetSpec,
    planned: TargetPlan,
    raw_root: Path,
    time_window: TimeWindow,
    session: DatasetSession | tuple | None = None,
) -> TargetResult:
    requested_identity = planned.requested_identity
    matching_identity = planned.matching_instance_identity
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
        _verify_absent_sources(
            profile,
            _optional_source_ids(profile, protocol, spec),
            planned.upstream_identities,
            raw_root,
        )
        binding = profile.producer_bindings[spec.target_id]
        with ExitStack() as operation:
            if binding.source_identity is not None:
                envelope, _members, manifest_identity, _operational = (
                    _read_artifact_manifest(root, expected_identity=requested_identity)
                )
                manifest_path = root / "manifest.json"
                handle = operation.enter_context(
                    closing_iterator(open_filesystem_file(manifest_path, "rb"))
                )
                operation.enter_context(
                    verified_file_handle(
                        manifest_path,
                        handle,
                        expected_identity=manifest_identity,
                        error="semantic_artifact_member_identity_changed",
                    )
                )
                try:
                    raw_identity = binding.source_identity(
                        binding.raw_root or raw_root, spec.target_id
                    )
                    if (
                        raw_identity
                        != envelope["semantic_metadata"]["raw_content_identity"]
                    ):
                        raise PortfolioTargetBlocked("source_identity_changed")
                except PortfolioTargetBlocked as exc:
                    return TargetResult(
                        spec.target_id, TargetStatus.BLOCKED, reason=str(exc)
                    )
                except ValueError:
                    return TargetResult(
                        spec.target_id,
                        TargetStatus.BLOCKED,
                        reason="source_identity_invalid",
                    )
            if session is None:
                _validate_semantic_target_root(
                    profile,
                    protocol,
                    spec,
                    root,
                    requested_identity,
                    trial_layout=layout,
                )
            else:
                _verify_semantic_reuse(session)
            _verify_absent_sources(
                profile,
                _optional_source_ids(profile, protocol, spec),
                planned.upstream_identities,
                raw_root,
            )
        return TargetResult(
            target_id=spec.target_id,
            status=TargetStatus.REUSED,
            instance_identity=requested_identity,
        )
    _verify_absent_sources(
        profile,
        _optional_source_ids(profile, protocol, spec),
        planned.upstream_identities,
        raw_root,
    )
    envelope = _validate_published_target(root, requested_identity, session=session)
    binding = profile.producer_bindings.get(spec.target_id)
    if isinstance(data_spec, InterimDataSpec) or (
        binding is not None and binding.source_identity is not None
    ):
        interim_raw_root = (
            binding.raw_root
            if binding is not None and binding.raw_root is not None
            else (
                _interim_raw_root(raw_root, data_spec)
                if isinstance(data_spec, InterimDataSpec)
                else raw_root
            )
        )
        if binding is None or binding.source_identity is None:
            return TargetResult(
                target_id=spec.target_id,
                status=TargetStatus.BLOCKED,
                reason="source_identity_provider_missing",
            )
        try:
            raw_identity = binding.source_identity(
                interim_raw_root,
                spec.target_id,
            )
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
    if session is None:
        inspect_physical_dataset(
            root,
            time_window=time_window
            if any(
                getattr(role, "value", role) == "timestamp"
                for role in data_spec.column_roles.values()
            )
            else None,
        )
    else:
        session.verify()
    _verify_absent_sources(
        profile,
        _optional_source_ids(profile, protocol, spec),
        planned.upstream_identities,
        raw_root,
    )
    return TargetResult(
        target_id=spec.target_id,
        status=TargetStatus.REUSED,
        instance_identity=requested_identity,
    )


def _validate_published_target(
    root: Path,
    expected_identity: str,
    *,
    session: DatasetSession | None = None,
) -> dict[str, object]:
    manifest = (
        session.manifest
        if session is not None
        else read_dataset_manifest_evidence(root)
    )
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
    protocol: Any,
    spec: TargetSpec,
    root: Path,
    expected_identity: str,
    *,
    trial_layout: StorageLayout | None = None,
    expected_member_identities=None,
    semantic_snapshots=None,
    trial_snapshots=None,
    reuse_sessions=None,
) -> None:
    def read_model(model_root, identity, *, member_identities=None):
        saved = None if reuse_sessions is None else reuse_sessions.get(model_root)
        if saved is not None and saved[3] is not None:
            _verify_semantic_reuse(saved)
            semantic_snapshots.update(saved[1])
            trial_snapshots.update(saved[2])
            return saved[3]
        snapshot = None
        if semantic_snapshots is not None and member_identities is None:
            _header, _members, manifest_identity, _operational = (
                _read_artifact_manifest(model_root, expected_identity=identity)
            )
            member_identities = {"manifest.json": manifest_identity}
            directories = _visit_member_paths(
                model_root,
                lambda name, path: member_identities.__setitem__(
                    name, _member_identity(path)
                ),
            )
            snapshot = member_identities, directories
        persisted = (
            read_crypto_model_artifact
            if protocol.protocol_id == "crypto_hft"
            else read_quarterly_model_artifact
        )(
            model_root,
            expected_identity=identity,
            trial_layout=trial_layout,
            expected_member_identities=member_identities,
            trial_snapshots=trial_snapshots,
        )
        if snapshot is not None:
            verify_semantic_artifact_snapshot(model_root, *snapshot)
            semantic_snapshots[model_root] = snapshot
        return persisted

    binding = profile.producer_bindings.get(spec.target_id)
    if binding is None:
        raise ValueError("producer_binding_missing")
    if binding.kind is ProducerKind.STATISTIC:
        artifact = read_statistics_artifact(
            root,
            expected_identity=expected_identity,
            expected_member_identities=expected_member_identities,
            validate_tables=validate_maker_calibration
            if protocol.protocol_id == "crypto_hft"
            and protocol.statistics_table_group(spec.target_id)
            == "crypto_maker_calibration"
            else None,
        )
        envelope = artifact.envelope
    elif binding.kind is ProducerKind.PORTFOLIO:
        if (
            protocol.protocol_id == "crypto_hft"
            and protocol.execution_target_for_target(spec.target_id).execution_mode
            == "independent_segments"
        ):
            from quant_research_platform.artifacts.crypto_hft import (
                read_crypto_episode_artifact,
            )

            artifact = read_crypto_episode_artifact(
                root,
                expected_identity=expected_identity,
                expected_member_identities=expected_member_identities,
            )
            envelope = artifact._envelope
        else:
            artifact = read_portfolio_ledger_artifact(
                root,
                expected_identity=expected_identity,
                expected_member_identities=expected_member_identities,
            )
            envelope = artifact.envelope
    elif binding.kind is ProducerKind.MODEL:
        persisted = read_model(
            root, expected_identity, member_identities=expected_member_identities
        )
        envelope = persisted.envelope
    else:
        identities = {}
        envelope, _members = read_semantic_artifact_record(
            root,
            expected_identity=expected_identity,
            expected_member_identities=expected_member_identities,
            member_identities=identities,
        )
    if envelope["logical_artifact_id"] != spec.target_id:
        raise ValueError("semantic_artifact_logical_id_mismatch")
    if binding.kind in {ProducerKind.STATISTIC, ProducerKind.PORTFOLIO}:
        return artifact

    def output_identities(output_id):
        prefix = f"outputs/{output_id}/"
        return {
            path[len(prefix) :]: identity
            for path, identity in identities.items()
            if path.startswith(prefix)
        }

    if protocol.protocol_id == "crypto_hft" and binding.kind in {
        ProducerKind.MODEL,
        ProducerKind.PREDICTION,
    }:
        if binding.kind is ProducerKind.MODEL:
            return persisted
        else:
            model_target = f"{protocol.scenario_id}_models"
            model_identity = envelope["upstream_identities"][model_target]
            persisted = read_model(
                _target_instance_root(
                    profile,
                    trial_layout or StorageLayout(root),
                    model_target,
                    model_identity,
                ),
                model_identity,
            )
            for model_id in protocol.model_ids_for_target(model_target):
                output_id = f"{model_id}_prediction"
                (manifest, _lookup) = _validate_prediction_output_bundle(
                    root / "outputs" / output_id,
                    expected_output_id=output_id,
                    expected_run_id=f"run_{output_id}",
                    expected_member_identities=output_identities(output_id),
                )
                if manifest["metadata"][
                    "model_metadata"
                ] != persisted.prediction_lineage(
                    model_id,
                    TimeWindow(
                        envelope["research_window"]["start"],
                        envelope["research_window"]["end"],
                    ),
                    access_window=protocol.access_window,
                ):
                    raise ValueError("crypto_prediction_model_lineage_invalid")
        return
    if binding.kind is ProducerKind.MODEL:
        expected_model_ids = protocol.model_ids_for_target(spec.target_id)
        for decisions in persisted.selections.values():
            for selection in decisions.values():
                if (
                    tuple(selection["candidate_family_ids"]) != expected_model_ids
                    or tuple(selection["tie_break_order"]) != expected_model_ids
                ):
                    raise ValueError("model_selection_protocol_mismatch")
        for entry in (*persisted.candidate_models, *persisted.final_models):
            metadata = entry["model_metadata"]
            expected_spec_id = entry["family_id"]
            if expected_spec_id not in expected_model_ids:
                raise ValueError("model_selection_family_mismatch")
            expected_spec = profile.models.get(expected_spec_id)
            if expected_spec.model_algorithm != metadata["model_algorithm"]:
                raise ValueError("model_selection_algorithm_mismatch")
            if list(expected_spec.input_columns) != metadata["input_columns"]:
                raise ValueError("model_selection_input_columns_mismatch")
        return persisted
    if binding.kind is ProducerKind.EVALUATION:
        try:
            manifest, _members = validate_output_bundle(
                root / "outputs" / spec.target_id,
                expected_output_id=spec.target_id,
                expected_run_id=f"run_{spec.target_id}",
                expected_member_identities=output_identities(spec.target_id),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation_inner_artifact_invalid:{_semantic_failure_code(exc)}"
            ) from exc
        return
    if binding.kind is ProducerKind.PREDICTION:
        try:
            prediction_session = _PredictionOutputSession(
                root / "outputs" / spec.target_id,
                expected_output_id=spec.target_id,
                expected_run_id=f"run_{spec.target_id}",
                expected_member_identities=output_identities(spec.target_id),
            )
            manifest = prediction_session.manifest
            prediction_metadata = manifest["metadata"]
            lineage = prediction_metadata["model_metadata"]
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
            if (
                lineage["model_artifact_id"] != model_target
                or lineage["model_artifact_identity"]
                != envelope["upstream_identities"].get(model_target)
                or lineage["model_artifact_schema"]
                != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
            ):
                raise ValueError("prediction_model_lineage_mismatch")
            if trial_layout is None:
                raise ValueError("prediction_model_layout_required")
            model_artifact = read_model(
                _target_instance_root(
                    profile,
                    trial_layout,
                    model_target,
                    lineage["model_artifact_identity"],
                ),
                lineage["model_artifact_identity"],
            )
            if lineage != model_artifact.prediction_lineage(
                TimeWindow(
                    prediction_metadata["test_start"],
                    prediction_metadata["test_end"],
                ),
                selection_group=protocol.selection_group_for_target(spec.target_id),
                access_window=protocol.access_window,
            ):
                raise ValueError("prediction_model_lineage_mismatch")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"prediction_inner_artifact_invalid:{_semantic_failure_code(exc)}"
            ) from exc
        return prediction_session


def _target_specs(
    profile: ResearchProfile,
    *,
    action: str,
    protocol: Any,
) -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = [
        TargetSpec(
            target_id=spec.data_id,
        )
        for spec in profile.interim.list()
    ]
    processed_specs = tuple(
        TargetSpec(
            target_id=spec.data_id,
            dependencies=spec.source_input_data_ids + spec.processed_input_data_ids,
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
        )
        for source_id in unbound_source_ids
    )
    specs.extend(
        TargetSpec(
            target_id=spec.data_id,
            dependencies=spec.input_data_ids,
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


def _target_software_identity(binding, software):
    from quant_research_platform.artifacts import (
        outputs,
        portfolio_ledger,
        statistics,
    )
    from quant_research_platform.artifacts import (
        research as research_artifacts,
    )
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
        write_crypto_episode_artifact,
    )
    from quant_research_platform.research.models import persisted
    from quant_research_platform.storage import semantic_artifacts
    from quant_research_platform.storage.datasets import inspection, writing

    # Bind canonical persistence owners, independently of workflow instrumentation.
    writers = {
        ProducerKind.INTERIM: (writing.write_dataset, inspection.inspect_dataset),
        ProducerKind.PROCESSED: (writing.write_dataset, inspection.inspect_dataset),
        ProducerKind.DATASET: (writing.write_dataset, inspection.inspect_dataset),
        ProducerKind.EVALUATION: (
            outputs.write_evaluation_outputs_from_chunks,
            validate_evaluation_output_metadata_shape,
            semantic_artifacts.write_semantic_artifact,
        ),
        ProducerKind.MODEL: (
            persisted.write_quarterly_model_artifact,
            persisted.write_crypto_model_artifact,
        ),
        ProducerKind.PREDICTION: (
            outputs.write_prediction_outputs,
            persisted.read_quarterly_model_artifact,
            persisted.read_crypto_model_artifact,
            validate_prediction_output_metadata_shape,
            semantic_artifacts.write_semantic_artifact,
        ),
        ProducerKind.STATISTIC: (
            statistics.write_statistics_artifact,
            semantic_artifacts.write_semantic_artifact,
        ),
        ProducerKind.PORTFOLIO: (
            portfolio_ledger.write_portfolio_ledger_artifact,
            portfolio_ledger.read_portfolio_ledger_artifact,
            outputs.read_prediction_output_bundle,
            persisted.read_quarterly_model_artifact,
            persisted.read_crypto_model_artifact,
        ),
        ProducerKind.REPORT: (
            research_artifacts.write_research_report_artifact,
            research_artifacts.read_research_report_artifact,
            persisted.read_quarterly_model_artifact,
            persisted.read_crypto_model_artifact,
        ),
    }
    owners = writers[binding.kind]
    if (
        binding.kind is ProducerKind.PORTFOLIO
        and isinstance(binding.producer, partial)
        and binding.producer.keywords.get("execution_mode") == "independent_segments"
    ):
        owners += (read_crypto_episode_artifact, write_crypto_episode_artifact)
    return _artifact_software_identity(
        software,
        binding.producer,
        binding.source_identity,
        owners,
        composition_modules=(__name__, ResearchProfile.__module__),
    )


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
        software_identity=_target_software_identity(binding, software_identity),
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
    binding = profile.producer_bindings.get(target_spec.target_id)
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
        software_identity=_target_software_identity(binding, software_identity),
        research_window=research_window,
        upstream_identities=upstream_identities,
        semantic_metadata={
            "target_contract": semantic_payload,
            **(
                {"raw_content_identity": raw_content_identity}
                if raw_content_identity is not None
                else {}
            ),
            **(
                {"portfolio_capital": semantic_payload["portfolio_capital"]}
                if binding.kind is ProducerKind.PORTFOLIO
                else {}
            ),
        },
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
        "protocol": protocol.target_manifest(target_spec.target_id, binding.kind.value),
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
            "label_availability_columns": dict(spec.label_availability_columns),
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
    elif binding.kind is ProducerKind.STATISTIC:
        group = (
            protocol.statistics_table_group(target_spec.target_id)
            if protocol.protocol_id == "crypto_hft"
            else "portfolio"
            if target_spec.target_id == f"{protocol.scenario_id}_portfolio_statistics"
            else "factor"
            if target_spec.target_id == protocol.factor_statistics_target_id
            else "research"
        )
        payload["statistics_tables"] = {
            member.removesuffix(".parquet"): list(STATISTICS_TABLE_COLUMNS[member])
            for member in STATISTICS_TABLE_GROUPS[group]
        }
    elif binding.kind is ProducerKind.PORTFOLIO:
        candidate = protocol.portfolio_candidate_for_target(target_spec.target_id)
        if protocol.protocol_id == "crypto_hft":
            payload["execution_contract"] = protocol.portfolio_contract(
                target_spec.target_id
            )
            payload["ledger_schema"] = {
                name: list(columns)
                for name, columns in PORTFOLIO_LEDGER_TABLE_COLUMNS.items()
            }
            if (
                protocol.execution_target_for_target(
                    target_spec.target_id
                ).execution_mode
                == "independent_segments"
            ):
                from quant_research_platform.artifacts.crypto_hft import (
                    crypto_episode_schema_payload,
                )

                payload["episode_index_schema"] = crypto_episode_schema_payload()
            payload["portfolio_capital"] = {
                "candidate_id": candidate.candidate_id,
                "initial_capital": protocol.research_plan.initial_capital,
                "valuation_currency": protocol.research_plan.valuation_currency,
                "valuation_method": "linear_perpetual",
            }
            return payload
        payload["portfolio_candidate"] = {
            "candidate_id": candidate.candidate_id,
            "selection_group": candidate.selection_group,
            "construction": candidate.construction,
            "prediction_target_id": candidate.prediction_target_id,
            "prediction_selector": candidate.prediction_selector,
            "top_quantile": candidate.top_quantile,
            "rebalance_schedule": candidate.rebalance_schedule,
            "benchmark_id": candidate.benchmark_id,
            "benchmark_weights_target_id": candidate.benchmark_weights_target_id,
            "research_dataset_id": candidate.research_dataset_id,
            "market_panel_target_id": candidate.market_panel_target_id,
            "execution_mask_target_id": candidate.execution_mask_target_id,
            "trading_calendar_target_id": candidate.trading_calendar_target_id,
            "fee_schedule_target_id": candidate.fee_schedule_target_id,
            "risk_style_columns": list(candidate.risk_style_columns),
            "risk_context_target_ids": list(candidate.risk_context_target_ids),
        }
        payload["final_alpha_calibration_schema"] = "qrp-final-alpha-calibration"
        payload["ledger_schema"] = {
            name: list(columns)
            for name, columns in PORTFOLIO_LEDGER_TABLE_COLUMNS.items()
        }
        payload["portfolio_capital"] = {
            "candidate_id": candidate.candidate_id,
            "initial_capital": protocol.execution_assumptions.initial_nav,
            "valuation_currency": protocol.manifest()["execution_assumptions"][
                "valuation_currency"
            ],
            "valuation_method": "adjusted_notional",
        }
    elif binding.kind is ProducerKind.MODEL:
        model_ids = protocol.model_ids_for_target(target_spec.target_id)
        payload["model_specs"] = [
            model_spec_payload(profile.models.get(model_id)) for model_id in model_ids
        ]
        payload["model_artifact_schema"] = (
            CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            if protocol.protocol_id == "crypto_hft"
            else A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
        )
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
            "columns": list(
                prediction_output_columns("a_share_oos", RAW_PREDICTION_COLUMN)
            ),
            "primary_key": list(A_SHARE_PREDICTION_PRIMARY_KEY),
        }
        if protocol.protocol_id == "crypto_hft":
            payload["prediction_schema"] = {
                "prediction_mode": "crypto_oos",
                "models": list(
                    protocol.model_ids_for_target(f"{protocol.scenario_id}_models")
                ),
                "primary_key": list(PREDICTION_PRIMARY_KEY),
                "prediction_column": "prediction_score",
            }
    return payload


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
        if start_date is not None or end_date is not None:
            if window is None:
                raise ValueError("research_window_default_missing")
            window = TimeWindow(
                window.start if start_date is None else start_date,
                window.end if end_date is None else end_date,
            )
    if window is None:
        return {
            "start": "1970-01-01",
            "end": "1970-01-01",
            "access": "not_applicable",
        }
    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
        "access": protocol.access_window,
    }


def _target_time_window(
    action: str,
    protocol: Any,
    data_spec: DataSpec | None,
    requested: Mapping[str, object],
) -> TimeWindow:
    try:
        requested_window = TimeWindow(requested["start"], requested["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research_window_invalid") from exc
    if (
        data_spec is not None
        and data_spec.frequency is DataFrequency.TICK
        and not is_datetime_window(requested_window)
        and (
            action in {"data materialize", "dataset build"}
            or requested["access"] != "not_applicable"
        )
    ):
        raise ValueError("tick_data_requires_intraday_window")
    return protocol.target_window(action, data_spec, requested_window)


def _window_payload(window: TimeWindow, *, access: str) -> dict[str, object]:
    if access not in {"development", "holdout", "not_applicable"}:
        raise ValueError("research_window_invalid")
    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
        "access": access,
    }


def _semantic_target_time_window(
    action: str,
    protocol: Any,
    target_spec: TargetSpec,
    data_spec: Any,
    requested: Mapping[str, object],
) -> TimeWindow:
    window = _target_time_window(action, protocol, data_spec, requested)
    if action != "research run" or data_spec is not None:
        return window
    return protocol.target_window(action, None, window, target_spec.target_id)


@contextmanager
def _prediction_consumer(root, *, reuse_sessions=None, **selection):
    """Borrow admission only; every consumer owns independent matching state."""
    saved = None if reuse_sessions is None else reuse_sessions.get(root)
    if saved is not None:
        _verify_semantic_reuse(saved)
    primary = None
    try:
        manifest, lookup = read_prediction_output_bundle(
            root / "outputs" / selection["expected_output_id"],
            validation_session=None if saved is None else saved[3],
            **selection,
        )
        with closing_iterator(lookup):
            yield manifest, lookup
    except BaseException as error:
        primary = error
        raise
    finally:
        if saved is not None:
            try:
                _verify_semantic_reuse(saved)
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    "prediction snapshot verification also failed: "
                    f"{type(error).__name__}"
                )


def _verify_semantic_reuse(session):
    """Keep one request's validated native result bound to all of its files."""
    _envelope, semantic_snapshots, trial_snapshots, _native = session
    for saved_root, snapshot in semantic_snapshots.items():
        verify_semantic_artifact_snapshot(saved_root, *snapshot)
    for saved_root, snapshot in trial_snapshots.items():
        verify_trial_ledger_snapshot(saved_root, snapshot)


def _semantic_reuse_session(
    profile,
    protocol,
    spec,
    instance_root,
    requested_identity,
    *,
    layout,
    reuse_sessions,
):
    """Admit built and reused semantic outputs through the same native boundary."""
    envelope, _members, manifest_identity, _operational = _read_artifact_manifest(
        instance_root, expected_identity=requested_identity
    )
    identities = {"manifest.json": manifest_identity}
    directories = _visit_member_paths(
        instance_root,
        lambda name, path, identities=identities: identities.__setitem__(
            name, _member_identity(path)
        ),
    )
    semantic_snapshots = {instance_root: (identities, directories)}
    trial_snapshots = {}
    native = _validate_semantic_target_root(
        profile,
        protocol,
        spec,
        instance_root,
        requested_identity,
        trial_layout=layout,
        expected_member_identities=identities,
        semantic_snapshots=semantic_snapshots,
        trial_snapshots=trial_snapshots,
        reuse_sessions=reuse_sessions,
    )
    session = (envelope, semantic_snapshots, trial_snapshots, native)
    _verify_semantic_reuse(session)
    return session


def _matching_instance(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    spec: TargetSpec,
    protocol: Any,
    requested_identity: str,
    reuse_sessions: dict | None = None,
) -> tuple[str | None, str | None]:
    target_id = spec.target_id
    logical_root = _target_logical_root(profile, layout, target_id)
    if logical_root is None:
        return None, "target_logical_root_missing"
    try:
        instance_root = require_unaliased_path(
            layout.instance_root(logical_root, requested_identity),
            error="persisted_instance_invalid",
        )
        try:
            filesystem_stat(instance_root)
        except FileNotFoundError:
            return None, None
        data_spec = _catalog_data_spec(profile, target_id)
        if (
            data_spec is None
            and reuse_sessions is not None
            and instance_root in reuse_sessions
        ):
            session = reuse_sessions[instance_root]
            _verify_semantic_reuse(session)
            envelope = session[0]
        elif data_spec is None:
            session = _semantic_reuse_session(
                profile,
                protocol,
                spec,
                instance_root,
                requested_identity,
                layout=layout,
                reuse_sessions=reuse_sessions,
            )
            envelope = session[0]
        else:
            session = (
                None if reuse_sessions is None else reuse_sessions.get(instance_root)
            )
            if session is None:
                session = open_dataset_session(
                    instance_root, columns=None, time_window=None
                )
                inspect_dataset_session(session)
            else:
                session.verify()
            envelope = _validate_published_target(
                instance_root, requested_identity, session=session
            )
        if envelope["logical_artifact_id"] != target_id:
            raise ValueError("persisted_instance_logical_id_mismatch")
        if reuse_sessions is not None:
            reuse_sessions[instance_root] = session
    except (ValueError, KeyError, FileNotFoundError, NotADirectoryError):
        return None, "persisted_instance_invalid"
    return requested_identity, None


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
    return logical_root is not None and filesystem_exists(
        layout.instance_root(
            logical_root,
            instance_identity,
        )
    )


def _target_layout(
    *,
    profile: ResearchProfile,
    protocol: Any,
    layout: StorageLayout,
    spec: TargetSpec,
    plans: Mapping[str, TargetPlan],
) -> StorageLayout:
    roots = {
        name: None
        for name in _optional_source_ids(profile, protocol, spec)
        if name not in spec.dependencies
    }
    for target_id in (*spec.dependencies, spec.target_id):
        planned = plans.get(target_id)
        if planned is None:
            raise ValueError("target_plan_missing")
        identity = planned.requested_identity
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


def _default_raw_root(layout: StorageLayout) -> Path:
    return layout.project_root / "data/raw"


def _interim_raw_root(raw_root: Path, spec: InterimDataSpec) -> Path:
    return Path(raw_root) / str(spec.source_id)


def _default_scenario_id(profile: ResearchProfile) -> str:
    return profile.research_datasets.list()[0].data_id


def _run_status(results: Iterable[TargetResult]) -> RunStatus:
    statuses = tuple(result.status for result in results)
    if TargetStatus.FAILED in statuses:
        return RunStatus.FAILED
    if TargetStatus.BLOCKED in statuses:
        return RunStatus.BLOCKED
    return RunStatus.COMPLETE
