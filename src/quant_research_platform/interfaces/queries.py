"""Read-only catalog, run, report, and aggregate portfolio projections."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts import research as report_artifact
from quant_research_platform.artifacts.crypto_hft import read_crypto_episode_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.targets import (
    ProducerKind,
    RunStatus,
    TargetStatus,
)
from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
    ResearchDatasetRequirementsResponse,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import (
    ResearchProfile,
    bind_research_configuration,
)
from quant_research_platform.storage.datasets.instances import (
    discover_dataset_instances,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.storage.semantic_artifacts import (
    discover_semantic_artifact_instances,
)


def list_factors(profile: ResearchProfile) -> FactorCatalogResponse:
    return FactorCatalogResponse(
        items=[item.to_manifest() for item in profile.factors.list()],
    )


def list_labels(profile: ResearchProfile) -> LabelCatalogResponse:
    return LabelCatalogResponse(
        items=[item.to_manifest() for item in profile.labels.list()],
    )


def describe_research_dataset_requirements(
    dataset_id: str,
    profile: ResearchProfile,
) -> ResearchDatasetRequirementsResponse:
    spec = profile.research_datasets.get(dataset_id)
    return ResearchDatasetRequirementsResponse(
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.data_id,
        market_id=str(spec.market_id),
        input_data_ids=list(spec.input_data_ids),
        factor_count=len(spec.factor_columns),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
    )


def read_report(
    *, profile: ResearchProfile, layout: StorageLayout, run_id, report_identity
):
    """Read one already published report and its immutable Markdown bytes."""
    record = report_artifact.read_research_run_record(
        layout.research_run_path(run_id), expected_profile_id=profile.profile_id
    )
    root = layout.instance_root(
        layout.semantic_artifact_root(
            profile.market_id, f"{record.scenario_id}_report_artifact"
        ),
        report_identity,
    )
    root = Path(native_filesystem_path(root))
    document = report_artifact.read_research_report_artifact(
        root,
        expected_identity=report_identity,
        source_run=record,
        include_markdown=True,
    )
    return {
        "profile_id": profile.profile_id,
        "run_id": run_id,
        "report_identity": report_identity,
        **document,
    }


def portfolio_curve(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    run_id,
    target_id,
    stride=1,
):
    """Project aggregate NAV and recorded OOS markers; never expose asset rows."""
    if type(stride) is not int or stride < 1:
        raise ValueError("portfolio_curve_stride_invalid")
    record = report_artifact.read_research_run_record(
        layout.research_run_path(run_id), expected_profile_id=profile.profile_id
    )
    result = next(
        (item for item in record.target_results if item.target_id == target_id),
        None,
    )
    if result is None:
        raise KeyError(target_id)
    if result.producer_kind is not ProducerKind.PORTFOLIO:
        raise ValueError("portfolio_curve_target_invalid")
    if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
        return {
            "target_id": target_id,
            "status": result.status.value,
            "reason": result.reason,
        }
    root = report_artifact._instance_root(layout, result)
    mode = "continuous"
    if str(profile.market_id) == "crypto":
        frozen_profile = bind_research_configuration(
            profile,
            record.payload()["protocol"]["research_configuration"],
            tardis_root=layout.raw_source_root("tardis"),
            access_window=record.research_window["access"],
        )
        mode = (
            frozen_profile.protocol(
                record.scenario_id, access_window=record.research_window["access"]
            )
            .execution_target_for_target(target_id)
            .execution_mode
        )
    if mode == "independent_segments":
        episodes = []

        def consume_episode(index, _child, batches):
            episodes.append(
                {
                    **{
                        key: index[key]
                        for key in (
                            "episode_id",
                            "start_ns",
                            "stop_ns",
                            "terminal_valuation_ns",
                            "termination",
                            "stop_reasons",
                        )
                    },
                    "instance_identity": _child.identity,
                    **_ledger_curve(batches, stride=stride),
                }
            )

        artifact = read_crypto_episode_artifact(
            root,
            expected_identity=result.instance_identity,
            consume_episode=consume_episode,
        )
        envelope = artifact._envelope
        curve = {
            "episodes": episodes,
            "episode_count": len(episodes),
            "source_point_count": sum(item["source_point_count"] for item in episodes),
            "displayed_point_count": sum(
                item["displayed_point_count"] for item in episodes
            ),
            "stride": stride,
        }
    elif mode == "continuous":
        curve = {}
        artifact = read_portfolio_ledger_artifact(
            root,
            expected_identity=result.instance_identity,
            consume_batches=lambda _artifact, batches: curve.update(
                _ledger_curve(batches, stride=stride)
            ),
        )
        envelope = artifact.envelope
        curve = {
            "episode_id": "continuous",
            **curve,
        }
    else:
        raise ValueError("portfolio_curve_execution_mode_invalid")
    report_artifact._validate_reference_envelope(result, envelope)
    if str(profile.market_id) == "crypto" and (
        envelope["semantic_metadata"]["target_contract"]["execution_contract"][
            "execution_mode"
        ]
        != mode
    ):
        raise ValueError("portfolio_curve_execution_mode_invalid")
    return {
        "profile_id": profile.profile_id,
        "run_id": run_id,
        "target_id": target_id,
        "instance_identity": result.instance_identity,
        "status": "DEFINED",
        "candidate_id": artifact.capital.candidate_id,
        "valuation_currency": artifact.capital.valuation_currency,
        "initial_capital": artifact.capital.initial_capital,
        "execution_mode": mode,
        **curve,
    }


def _ledger_curve(batches, *, stride):
    points, boundaries = [], []
    total_points = 0
    last_points = []
    last_retained_sequence = -1
    with closing_iterator(iter(batches)) as batches:
        for batch in batches:
            last_points = [
                {
                    "sequence": batch.sequence,
                    "timestamp_ns": max(
                        row.timestamp_ns
                        for row in batch.valuations
                        if row.phase == phase
                    ),
                    "phase": phase,
                    "nav": sum(
                        row.value for row in batch.valuations if row.phase == phase
                    ),
                }
                for phase in ("before", "after", "end")
            ]
            total_points += len(last_points)
            has_marker = any(
                row.name in {"oos_window_start", "oos_window_end"}
                for row in batch.diagnostics
            )
            if batch.sequence % stride == 0 or has_marker:
                points.extend(last_points)
                last_retained_sequence = batch.sequence
            for marker in batch.diagnostics:
                if marker.name not in {"oos_window_start", "oos_window_end"}:
                    continue
                boundaries.append(
                    {
                        "timestamp_ns": last_points[-1]["timestamp_ns"],
                        "kind": marker.name,
                        "window_index": int(marker.number),
                    }
                )
    if last_points and last_retained_sequence != last_points[-1]["sequence"]:
        points.extend(last_points)
    return {
        "points": points,
        "boundaries": boundaries,
        "source_point_count": total_points,
        "displayed_point_count": len(points),
        "stride": stride,
        "selection": "every_nth_ledger_transition_plus_boundaries_and_last",
    }


def status(
    *, profile: ResearchProfile, layout: StorageLayout, run_id: str | None = None
) -> dict[str, object]:
    """Read exact run evidence or list records without planning any new work."""
    if run_id is not None:
        return report_artifact.read_research_run_record(
            layout.research_run_path(run_id), expected_profile_id=profile.profile_id
        ).payload()
    root = layout.research_run_path("index").parent
    records = []
    for path in filesystem_glob(root, "*.json"):
        record = report_artifact.read_research_run_record(path)
        if record.profile_id != profile.profile_id:
            continue
        records.append(
            {
                "run_id": record.run_id,
                "scenario_id": record.scenario_id,
                "status": record.status.value,
                "research_window": dict(record.research_window),
            }
        )
    records.sort(key=lambda item: item["run_id"])
    return {"profile_id": profile.profile_id, "runs": records}


def inspect(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    dataset_id: str,
    run_id: str | None = None,
) -> dict[str, object]:
    if run_id is not None:
        return _inspect_run_target(
            profile=profile, layout=layout, target_id=dataset_id, run_id=run_id
        )
    spec = profile.research_datasets.get(dataset_id)
    root = layout.research_dataset_root(spec.market_id, spec.data_id)
    result: dict[str, object] = {
        "profile_id": profile.profile_id,
        "dataset_id": dataset_id,
        "required_columns": list(spec.required_columns),
        "exists": filesystem_exists(root),
        "instances": [],
    }
    if not filesystem_exists(root):
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


def _inspect_run_target(*, profile, layout, target_id, run_id):
    record = report_artifact.read_research_run_record(
        layout.research_run_path(run_id), expected_profile_id=profile.profile_id
    )
    if target_id == f"{record.scenario_id}_report_artifact":
        logical = layout.semantic_artifact_root(profile.market_id, target_id)
        instances = []
        selected = {}

        def belongs_to_run(envelope):
            metadata = report_artifact._report_manifest_metadata(envelope)
            if envelope["logical_artifact_id"] != target_id:
                raise ValueError("research_report_semantics_invalid")
            if metadata["source_run_id"] == run_id:
                selected[envelope["instance_identity"]] = envelope
                return True
            return False

        for instance in discover_semantic_artifact_instances(
            logical, manifest_scope=belongs_to_run
        ):
            report = report_artifact.read_research_report_artifact(
                instance, expected_identity=instance.name, source_run=record
            )
            instances.append(
                {
                    "instance_identity": instance.name,
                    "schema_identity": selected[instance.name]["schema_identity"],
                    "report_status": report["report_status"],
                }
            )
        return {
            "profile_id": profile.profile_id,
            "run_id": run_id,
            "target_id": target_id,
            "status": "BLOCKED",
            "instances": instances,
            "blocker": "report_identity_required"
            if instances
            else "report_instances_missing",
        }
    result = next(
        (item for item in record.target_results if item.target_id == target_id),
        None,
    )
    if result is None:
        raise KeyError(target_id)
    if result.status in {TargetStatus.BUILT, TargetStatus.REUSED}:
        report_artifact.read_validated_reference(layout, result)
    return {"profile_id": profile.profile_id, "run_id": run_id, **result.payload()}
