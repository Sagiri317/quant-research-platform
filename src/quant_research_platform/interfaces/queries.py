"""Read-only catalog, run, report, and aggregate portfolio projections."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts import research as report_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import json_record_working_set_bytes
from quant_research_platform.contracts.targets import ProducerKind, TargetStatus
from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
    ResearchDatasetRequirementsResponse,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research

API_VERSION = "0.1"


def list_factors(profile: ResearchProfile) -> FactorCatalogResponse:
    return FactorCatalogResponse(
        api_version=API_VERSION,
        items=[item.to_manifest() for item in profile.factors.list()],
    )


def list_labels(profile: ResearchProfile) -> LabelCatalogResponse:
    return LabelCatalogResponse(
        api_version=API_VERSION,
        items=[item.to_manifest() for item in profile.labels.list()],
    )


def describe_research_dataset_requirements(
    dataset_id: str,
    profile: ResearchProfile,
) -> ResearchDatasetRequirementsResponse:
    spec = profile.research_datasets.get(dataset_id)
    return ResearchDatasetRequirementsResponse(
        api_version=API_VERSION,
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.data_id,
        market_id=str(spec.market_id),
        input_data_ids=list(spec.input_data_ids),
        factor_count=len(spec.factor_columns),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
    )


def run_status(*, profile: ResearchProfile, layout: StorageLayout, run_id=None):
    return research.status(profile=profile, layout=layout, run_id=run_id)


def inspect_target(
    *, profile: ResearchProfile, layout: StorageLayout, target_id, run_id=None
):
    return research.inspect(
        profile=profile, layout=layout, dataset_id=target_id, run_id=run_id
    )


def read_report(
    *, profile: ResearchProfile, layout: StorageLayout, run_id, report_identity
):
    """Read one already published report and its immutable Markdown bytes."""
    with runtime_resource_policy_scope(current_runtime_resource_policy()) as policy:
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
        _envelope, members = report_artifact.read_semantic_artifact_record(
            root, expected_identity=report_identity
        )
        by_path = report_artifact._require_members(
            members, ("report.json", "report.md")
        )
        with policy.lease(
            "queries.report",
            report_artifact.report_payload_working_set(record.payload())
            + json_record_working_set_bytes(by_path["report.json"]["size"])
            + 8 * by_path["report.md"]["size"],
        ):
            payload = report_artifact.read_research_report_artifact(
                root, expected_identity=report_identity, source_run=record
            )
            markdown = report_artifact._read_utf8(
                root / "report.md", expected_member=by_path["report.md"]
            )
            return {
                "profile_id": profile.profile_id,
                "run_id": run_id,
                "report_identity": report_identity,
                "report": payload,
                "markdown": markdown,
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
    with runtime_resource_policy_scope(current_runtime_resource_policy()) as policy:
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
        with policy.lease(
            "queries.portfolio_curve",
            report_artifact.report_payload_working_set(record.payload()) + 8192,
        ) as retained:
            report_artifact._validate_reference(layout, result)
            artifact = read_portfolio_ledger_artifact(
                report_artifact._instance_root(layout, result),
                expected_identity=result.instance_identity,
            )
            points, boundaries = [], []
            total_points = 0
            last_points = []
            last_retained_sequence = -1
            with closing_iterator(artifact.iter_chunks()) as batches:
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
                                row.value
                                for row in batch.valuations
                                if row.phase == phase
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
                        retained.grow(1024 * len(last_points))
                        points.extend(last_points)
                        last_retained_sequence = batch.sequence
                    for marker in batch.diagnostics:
                        if marker.name not in {"oos_window_start", "oos_window_end"}:
                            continue
                        retained.grow(1024)
                        boundaries.append(
                            {
                                "timestamp_ns": last_points[-1]["timestamp_ns"],
                                "kind": marker.name,
                                "window_index": int(marker.number),
                            }
                        )
            if last_points and last_retained_sequence != last_points[-1]["sequence"]:
                retained.grow(1024 * len(last_points))
                points.extend(last_points)
            return {
                "profile_id": profile.profile_id,
                "run_id": run_id,
                "target_id": target_id,
                "instance_identity": result.instance_identity,
                "status": "DEFINED",
                "candidate_id": artifact.capital.candidate_id,
                "valuation_currency": artifact.capital.valuation_currency,
                "initial_capital": artifact.capital.initial_capital,
                "points": points,
                "boundaries": boundaries,
                "source_point_count": total_points,
                "displayed_point_count": len(points),
                "stride": stride,
                "selection": "every_nth_ledger_transition_plus_boundaries_and_last",
            }
