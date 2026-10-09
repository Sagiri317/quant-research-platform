"""Persisted research-run records and rendered reports."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from html import escape
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import pandas as pd

from quant_research_platform.artifacts.crypto_hft import (
    read_crypto_episode_artifact,
)
from quant_research_platform.artifacts.evidence import (
    read_json_evidence,
    write_json_evidence,
    write_markdown_evidence,
)
from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_GROUPS,
    read_statistics_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.targets import (
    ProducerKind,
    ResearchRunRecord,
    ResearchRunTargetResult,
    RunStatus,
    TargetStatus,
    research_run_record_from_payload,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.reports import (
    a_share,
    crypto_hft,
    diagnostic_status,
)
from quant_research_platform.reports.a_share import build_a_share_report
from quant_research_platform.reports.crypto_hft import build_crypto_research_report
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    semantic_artifact_read_session,
    verify_semantic_artifact_snapshot,
    write_semantic_artifact,
)

_REPORT_CONTRACTS = {
    a_share.REPORT_SCHEMA_ID: (
        a_share.REQUIRED_REPORT_SECTIONS,
        "A-share Research Report",
    ),
    crypto_hft.REPORT_SCHEMA_ID: (
        crypto_hft.REQUIRED_REPORT_SECTIONS,
        "Crypto HFT Research Report",
    ),
}


def _report_contract(schema_id):
    if type(schema_id) is not str or schema_id not in _REPORT_CONTRACTS:
        raise ValueError("research_report_schema_invalid")
    return _REPORT_CONTRACTS[schema_id]


def write_research_run_record(path: Path, record: ResearchRunRecord) -> Path:
    return write_json_evidence(Path(path), record.payload())


def read_research_run_record(
    path: Path, *, expected_profile_id: str | None = None
) -> ResearchRunRecord:
    candidate = Path(path)
    payload = read_json_evidence(
        candidate,
        expected_schema_id="qrp-research-run",
        error="research_run_record_invalid",
    )
    record = research_run_record_from_payload(payload)
    if candidate.stem != record.run_id:
        raise ValueError("research_run_id_mismatch")
    if expected_profile_id is not None and record.profile_id != expected_profile_id:
        raise ValueError("research_run_profile_mismatch")
    return record


def _report_rows(frame):
    return [
        {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in row.items()
            if not pd.isna(value)
        }
        for row in frame.to_dict("records")
    ]


def read_evaluation_report_evidence(
    root, *, evaluation_id, instance_identity, expected_member_identities=None
):
    """Read validated aggregate evaluation evidence for both market reports."""
    tables = {}

    def observe(_manifest, members):
        for path, identity in members:
            if path.name not in {"ic_summary.parquet", "group_spread_summary.parquet"}:
                continue
            rows = []
            with closing_iterator(
                iter_declared_parquet_member_batches(path, expected_identity=identity)
            ) as chunks:
                for frame in chunks:
                    rows.extend(_report_rows(frame))
            tables[path.stem] = rows

    manifest, _members = validate_output_bundle(
        root / "outputs" / evaluation_id,
        expected_output_id=evaluation_id,
        expected_run_id=f"run_{evaluation_id}",
        expected_member_identities=None
        if expected_member_identities is None
        else {
            key.removeprefix(f"outputs/{evaluation_id}/"): value
            for key, value in expected_member_identities.items()
            if key.startswith(f"outputs/{evaluation_id}/")
        },
        observer=observe,
    )
    undefined = not any(tables.values()) or any(
        (
            value
            for rows in tables.values()
            for row in rows
            for key, value in row.items()
            if key.endswith("_reason")
        )
    )
    return {
        "evaluation_id": evaluation_id,
        "identity": instance_identity,
        "status": "UNDEFINED" if undefined else "DEFINED",
        "tables": tables,
        "metadata": manifest["metadata"],
    }


def load_a_share_run_evidence(
    *,
    layout: StorageLayout,
    record: ResearchRunRecord,
    selections,
    reference_member_identities=None,
    model_artifacts=None,
) -> dict[str, object]:
    references = {result.target_id: result for result in record.target_results}
    issues: list[dict[str, str]] = [
        _issue(
            result.target_id, result.status.value, result.reason or "target_unavailable"
        )
        for result in record.target_results
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}
    ]
    loaded: dict[str, ResearchRunTargetResult] = {}
    captured = {}
    operations = {}

    def load(target_id: str, *, native=False) -> ResearchRunTargetResult | None:
        if target_id in loaded:
            return loaded[target_id]
        result = references.get(target_id)
        if result is None:
            issues.append(_issue(target_id, "BLOCKED", "run_target_result_missing"))
            return None
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
            return None
        expected = (
            None
            if reference_member_identities is None
            else reference_member_identities.get(target_id)
        )
        if native:
            captured[target_id] = expected
            loaded[target_id] = result
            return result
        if model_artifacts is not None and target_id in model_artifacts:
            artifact = model_artifacts[target_id]
            _validate_reference_envelope(result, artifact.envelope)
            captured[target_id] = artifact.member_identities
            operations[target_id] = artifact.operational_metadata
            loaded[target_id] = result
            return result
        try:
            identities = {}
            operational = {}
            read_validated_reference(
                layout,
                result,
                member_identities=identities,
                expected_member_identities=expected,
                operational_metadata=operational,
            )
            captured[target_id] = identities
            operations[target_id] = operational
        except (FileNotFoundError, OSError, ValueError):
            issues.append(_issue(target_id, "FAILED", "target_reference_invalid"))
            return None
        loaded[target_id] = result
        return result

    def table_records(artifact, member):
        with artifact.table(member) as frame:
            return frame.astype(object).where(frame.notna(), None).to_dict("records")

    scenario = record.scenario_id
    configuration = record.payload()["protocol"]["research_configuration"]
    full_research = configuration["stage"] in {"models", "research"}
    factor_requested = configuration["stage"] != "dataset"
    for target_id in (
        scenario,
        *((f"{scenario}_evaluation",) if factor_requested else ()),
        *(
            (
                f"{scenario}_model_selection_refit",
                *(
                    f"{scenario}_{group}_persisted_predictions"
                    for group in ("global", "quarterly")
                ),
            )
            if full_research
            else ()
        ),
    ):
        load(target_id)
    trial_refs = {
        (item["run_id"], item["experiment_id"]): dict(item)
        for item in record.trial_ledgers
    }
    coverage = _unavailable(issues, scenario)
    dataset_ref = loaded.get(scenario)
    if dataset_ref is not None:
        manifest = read_dataset_manifest_evidence(
            _instance_root(layout, dataset_ref),
            expected_member_identities=captured[dataset_ref.target_id],
        )
        coverage = {
            "status": "DEFINED",
            "dataset_identity": dataset_ref.instance_identity,
            **{
                key: manifest[key]
                for key in (
                    "row_count",
                    "date_min",
                    "date_max",
                    "null_count_summary",
                    "non_finite_count_summary",
                    "duplicate_primary_key_count",
                )
            },
            **{
                f"{role}_nan_profile": {
                    column: manifest[f"{role}_nan_profile"].get(column, 0)
                    for column, declared_role in manifest["column_roles"].items()
                    if declared_role == role and column in manifest["columns"]
                }
                for role in ("factor", "label")
            },
        }
    pit_targets = []
    for target_id in ("fundamental_pit", "fundamental_factors"):
        if target_id not in references:
            pit_targets.append(
                {
                    "target_id": target_id,
                    "status": "UNDEFINED",
                    "reason": "target_not_in_run",
                }
            )
            continue
        reference = load(target_id)
        if reference is None:
            pit_targets.append(
                {"target_id": target_id, **_unavailable(issues, target_id)}
            )
            continue
        manifest = read_dataset_manifest_evidence(
            _instance_root(layout, reference),
            expected_member_identities=captured[reference.target_id],
        )
        pit_targets.append(
            {
                "target_id": target_id,
                "status": reference.status.value,
                "identity": reference.instance_identity,
                "upstream_identities": manifest[COMMON_MANIFEST_METADATA_KEY][
                    "upstream_identities"
                ],
            }
        )
    coverage["pit"] = {
        "scope": "published_run_references",
        "raw_completeness": "UNDEFINED",
        "reason": "raw_completeness_not_established_by_published_run",
        "targets": pit_targets,
    }
    evaluation_ref = loaded.get(f"{scenario}_evaluation")
    factor_evaluation = (
        _unavailable(issues, f"{scenario}_evaluation")
        if factor_requested
        else {"status": "UNDEFINED", "reason": "not_requested"}
    )
    if evaluation_ref is not None:
        factor_evaluation = read_evaluation_report_evidence(
            _instance_root(layout, evaluation_ref),
            evaluation_id=evaluation_ref.target_id,
            instance_identity=evaluation_ref.instance_identity,
            expected_member_identities=captured[evaluation_ref.target_id],
        )
        coverage["evaluation"] = factor_evaluation["metadata"]["evaluation_diagnostics"]
    model_costs = []
    for target in record.target_results:
        if target.producer_kind is not ProducerKind.MODEL:
            continue
        model_ref = loaded.get(target.target_id) or load(target.target_id)
        if model_ref is None:
            continue
        operational = operations[model_ref.target_id]
        if "model_phase_observations" in operational:
            model_costs.append(
                {
                    "model_target_id": model_ref.target_id,
                    **operational["model_phase_observations"],
                }
            )
        for item in operational.get("model_trial_ledgers", []):
            trial_refs[item["run_id"], item["experiment_id"]] = item
    trial_counts = Counter()
    trial_status = "DEFINED"
    for reference in trial_refs.values():
        try:
            _plan, records = read_model_trial_ledger_reference(layout, reference)
            trial_counts.update(item.status for item in records)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            trial_status = "FAILED"
            issues.append(
                _issue("model_trials", "FAILED", "trial_ledger_reference_invalid")
            )
    factor_target = f"{scenario}_factor_statistics"
    factor_ref = load(factor_target, native=True) if factor_requested else None
    factor_statistics = (
        _unavailable(issues, factor_target)
        if factor_requested
        else {"status": "UNDEFINED", "reason": "not_requested"}
    )
    if factor_ref is not None:
        try:
            artifact = read_statistics_artifact(
                _instance_root(layout, factor_ref),
                expected_identity=str(factor_ref.instance_identity),
                expected_member_identities=captured[factor_ref.target_id],
            )
            _validate_reference_envelope(factor_ref, artifact.envelope)
            factor_statistics = {
                "status": "DEFINED",
                "identity": factor_ref.instance_identity,
                **{
                    name: table_records(artifact, name + ".parquet")
                    for name in (
                        "factor_diagnostics",
                        "fama_macbeth",
                    )
                },
            }
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(factor_target, "FAILED", "statistics_reference_invalid")
            )
            factor_statistics = _unavailable(issues, factor_target)
    statistics = {}
    for group in ("global", "quarterly"):
        statistics_target = f"{scenario}_{group}_statistics"
        statistics_ref = load(statistics_target, native=True) if full_research else None
        if statistics_ref is None:
            group_statistics = (
                _unavailable(issues, statistics_target)
                if full_research
                else {"status": "UNDEFINED", "reason": "not_requested"}
            )
        else:
            try:
                artifact = read_statistics_artifact(
                    _instance_root(layout, statistics_ref),
                    expected_identity=str(statistics_ref.instance_identity),
                    expected_member_identities=captured[statistics_ref.target_id],
                )
                _validate_reference_envelope(statistics_ref, artifact.envelope)
                tables = {
                    name: table_records(artifact, name + ".parquet")
                    for name in (
                        "model_trials",
                        "prediction_diagnostics",
                        "model_comparison",
                        "model_group_comparison",
                        "model_reality_checks",
                        "robustness",
                    )
                }
                group_statistics = {
                    "status": "DEFINED",
                    "identity": statistics_ref.instance_identity,
                    **tables,
                }
            except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
                issues.append(
                    _issue(statistics_target, "FAILED", "statistics_reference_invalid")
                )
                group_statistics = _unavailable(issues, statistics_target)
        statistics[group] = group_statistics
    candidates = record.protocol.get("portfolio_candidates")
    if type(candidates) is not tuple or any(
        type(item) is not str for item in candidates
    ):
        raise ValueError("research_run_protocol_portfolios_invalid")
    portfolios: list[dict[str, object]] = []
    portfolio_target = f"{scenario}_portfolio_statistics"
    portfolio_ref = load(portfolio_target, native=True) if candidates else None
    portfolio_statistics = (
        (
            _unavailable(issues, portfolio_target)
            if candidates
            else {"status": "UNDEFINED", "reason": "not_requested"}
        )
        if portfolio_ref is None
        else None
    )
    if portfolio_ref is not None:
        try:
            artifact = read_statistics_artifact(
                _instance_root(layout, portfolio_ref),
                expected_identity=portfolio_ref.instance_identity,
                expected_member_identities=captured[portfolio_ref.target_id],
            )
            _validate_reference_envelope(portfolio_ref, artifact.envelope)
            summaries = {
                name: table_records(artifact, "portfolio_" + name + ".parquet")
                for name in (
                    "performance",
                    "capacity_summary",
                    "robustness",
                    "group_comparison",
                )
            }
            with artifact.table("portfolio_attribution.parquet") as attribution:
                with artifact.table("portfolio_daily.parquet") as daily:
                    latest_cash = daily.loc[
                        daily.entry_at.eq(
                            daily.groupby("candidate_id").entry_at.transform("max")
                        ),
                        [
                            "candidate_id",
                            "entry_at",
                            "cash_post",
                            "nav_post",
                            "cash_weight_post",
                        ],
                    ]
                    merged = attribution.merge(
                        daily[["candidate_id", "entry_at", "end_at", "nav_before"]],
                        on=["candidate_id", "entry_at", "end_at"],
                        how="left",
                        validate="many_to_one",
                    )
                    attribution_summary = []
                    for (candidate, component, component_type), rows in merged.groupby(
                        ["candidate_id", "component_id", "component_type"], sort=True
                    ):
                        defined = rows.status.eq("DEFINED").all()
                        attribution_summary.append(
                            dict(
                                candidate_id=candidate,
                                component_id=component,
                                component_type=component_type,
                                pnl=float((rows.value * rows.nav_before).sum())
                                if defined
                                else None,
                                status="DEFINED"
                                if defined
                                else "BLOCKED"
                                if rows.status.eq("BLOCKED").any()
                                else "UNDEFINED",
                                reason=""
                                if defined
                                else str(
                                    rows.loc[rows.status.ne("DEFINED"), "reason"].iloc[
                                        0
                                    ]
                                ),
                                observation_count=len(rows),
                            )
                        )
            with artifact.table("portfolio_exposures.parquet") as exposures:
                latest = exposures.loc[
                    exposures.entry_at.eq(
                        exposures.groupby("candidate_id").entry_at.transform("max")
                    )
                ]
                exposure_evidence = {
                    "status": "DEFINED" if not latest.empty else "UNDEFINED",
                    "reason": "" if not latest.empty else "no_exposure_observations",
                    "artifact_identity": portfolio_ref.instance_identity,
                    "table": "portfolio_exposures.parquet",
                    "key_columns": [
                        "candidate_id",
                        "decision_at",
                        "entry_at",
                        "factor_id",
                    ],
                    "value_column": "value",
                    "latest": _report_rows(latest),
                }
            portfolio_statistics = {
                "status": diagnostic_status(
                    [
                        *summaries["performance"],
                        *summaries["capacity_summary"],
                        *summaries["robustness"],
                        *attribution_summary,
                    ]
                ),
                "artifact_status": "DEFINED",
                "identity": portfolio_ref.instance_identity,
                "performance": summaries["performance"],
                "group_comparison": summaries["group_comparison"],
                "capacity": summaries["capacity_summary"],
                "capacity_detail": {
                    "artifact_identity": portfolio_ref.instance_identity,
                    "table": "portfolio_capacity.parquet",
                    "key_columns": ["candidate_id", "entry_at", "participation"],
                    "value_column": "bottleneck_asset_id",
                },
                "robustness": summaries["robustness"],
                "attribution": attribution_summary,
                "exposures": exposure_evidence,
                "cash_exposure": {
                    "artifact_identity": portfolio_ref.instance_identity,
                    "table": "portfolio_daily.parquet",
                    "key_columns": ["candidate_id", "entry_at"],
                    "cash_column": "cash_post",
                    "weight_column": "cash_weight_post",
                    "definition": (
                        "Cash / NAV at the execution open "
                        "after completed trades and fees."
                    ),
                    "latest": _report_rows(latest_cash),
                    "summary": [
                        row
                        for row in summaries["performance"]
                        if row["metric_id"]
                        in {
                            "mean_cash_weight_post",
                            "terminal_cash_weight_post",
                            "terminal_cash_post",
                        }
                    ],
                },
                "nav_path": {
                    "table": "portfolio_daily.parquet",
                    "post_nav": "nav_post",
                    "post_timestamp": "entry_at",
                    "terminal_nav": "nav_end",
                    "terminal_timestamp": "end_at",
                },
            }
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    portfolio_target, "FAILED", "portfolio_statistics_reference_invalid"
                )
            )
            portfolio_statistics = _unavailable(issues, portfolio_target)
    for candidate in candidates:
        target_id = f"{scenario}_{candidate}_ledger"
        ledger_ref = load(target_id, native=True)
        metrics = [
            row
            for row in portfolio_statistics.get("performance", [])
            if row["candidate_id"] == candidate
        ]
        if ledger_ref is None:
            if metrics:
                portfolios.append(
                    {
                        "candidate_id": candidate,
                        "statistics_identity": portfolio_ref.instance_identity,
                        "metrics": metrics,
                    }
                )
            continue
        try:
            economics = {}
            ledger = read_portfolio_ledger_artifact(
                _instance_root(layout, ledger_ref),
                expected_identity=ledger_ref.instance_identity,
                expected_member_identities=captured[ledger_ref.target_id],
                consume_batches=lambda artifact, batches, result=economics: (
                    result.update(summarize_portfolio_ledger(artifact.capital, batches))
                ),
            )
            _validate_reference_envelope(ledger_ref, ledger.envelope)
            portfolios.append(
                {
                    "candidate_id": candidate,
                    "status": "DEFINED",
                    "ledger_identity": ledger_ref.instance_identity,
                    "final_calibration": ledger.final_alpha_calibration,
                    "calibration_observations": ledger.operational_metadata.get(
                        "final_alpha_observations"
                    ),
                    "economics": economics,
                    "metrics": metrics,
                }
            )
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(target_id, "FAILED", "portfolio_ledger_reference_invalid")
            )
    return {
        "references": {
            target_id: result.instance_identity for target_id, result in loaded.items()
        },
        "coverage": coverage,
        "factor_evaluation": factor_evaluation,
        "factor_statistics": factor_statistics,
        "selections": selections,
        "statistics": statistics,
        "portfolio_statistics": portfolio_statistics,
        "model_costs": model_costs,
        "model_trials": {
            "status": trial_status,
            "attempt_count": sum(trial_counts.values()),
            "status_counts": dict(trial_counts),
            "ledger_count": len(trial_refs),
        },
        "portfolios": portfolios,
        "issues": issues,
    }


def produce_a_share_report(
    *,
    layout: StorageLayout,
    record: ResearchRunRecord,
    read_model_artifact,
) -> dict[str, object]:
    selections = {}
    models = {}
    primary = None
    try:
        for result in record.target_results:
            if result.producer_kind is not ProducerKind.MODEL or result.status in {
                TargetStatus.BLOCKED,
                TargetStatus.FAILED,
            }:
                continue
            persisted = read_model_artifact(
                _instance_root(layout, result),
                expected_identity=result.instance_identity,
                trial_layout=layout,
            )
            models[result.target_id] = persisted
            _validate_reference_envelope(result, persisted.envelope)
            selections = persisted.selections
        return build_a_share_report(
            record=record,
            evidence=load_a_share_run_evidence(
                layout=layout,
                record=record,
                selections=selections,
                model_artifacts=models,
            ),
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        for artifact in models.values():
            try:
                verify_semantic_artifact_snapshot(
                    artifact.instance_root,
                    artifact.member_identities,
                    artifact.directory_identities,
                )
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(
                    "semantic artifact verification also failed: "
                    + type(error).__name__
                )


def read_validated_reference(
    layout: StorageLayout,
    result: ResearchRunTargetResult,
    *,
    member_identities=None,
    expected_member_identities=None,
    operational_metadata=None,
):
    if result.instance_identity is None or result.schema_identity is None:
        raise ValueError("research_run_target_reference_invalid")
    root = _instance_root(layout, result)
    if result.producer_kind in {
        ProducerKind.INTERIM,
        ProducerKind.PROCESSED,
        ProducerKind.DATASET,
    }:
        envelope = validate_business_manifest_envelope(
            read_dataset_manifest_evidence(
                root,
                member_identities=member_identities,
                expected_member_identities=expected_member_identities,
            )[COMMON_MANIFEST_METADATA_KEY]
        )
    else:
        envelope, _members = read_semantic_artifact_record(
            root,
            expected_identity=result.instance_identity,
            member_identities=member_identities,
            expected_member_identities=expected_member_identities,
            operational_metadata=operational_metadata,
        )
    _validate_reference_envelope(result, envelope)
    return (root, envelope)


def _validate_reference_envelope(result, envelope):
    if (
        envelope["logical_artifact_id"] != result.target_id
        or envelope["instance_identity"] != result.instance_identity
        or envelope["schema_identity"] != result.schema_identity
    ):
        raise ValueError("research_run_target_reference_invalid")


def _instance_root(layout: StorageLayout, result: ResearchRunTargetResult) -> Path:
    if result.instance_identity is None or result.artifact_root is None:
        raise ValueError("research_run_target_reference_invalid")
    relative = PurePosixPath(result.artifact_root)
    return layout.instance_root(
        layout.project_root / Path(*relative.parts), result.instance_identity
    )


def _issue(target_id: str, status: str, reason: str) -> dict[str, str]:
    return {"target_id": target_id, "status": status, "reason": reason}


def _unavailable(issues: list[dict[str, str]], target_id: str) -> dict[str, object]:
    issue = next(
        (item for item in reversed(issues) if item.get("target_id") == target_id), None
    )
    return {
        "status": issue["status"] if issue is not None else "BLOCKED",
        "reason": issue["reason"] if issue is not None else "target_unavailable",
    }


def write_research_report_artifact(
    logical_root: Path,
    *,
    envelope: BusinessManifestEnvelope,
    payload: dict[str, object],
) -> Path:
    _validate_report_payload(payload)
    _validate_report_semantics(envelope.payload(), payload)

    def write(staging: Path) -> None:
        write_json_evidence(staging / "report.json", payload)
        write_markdown_evidence(
            staging / "report.md",
            render_research_report(
                payload,
                sections=_report_contract(payload["schema_id"])[0],
                title=_report_contract(payload["schema_id"])[1],
            ),
        )

    return write_semantic_artifact(Path(logical_root), envelope, write)


def read_research_report_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
    source_run: ResearchRunRecord | None = None,
    include_markdown: bool = False,
) -> dict[str, object]:
    with semantic_artifact_read_session(
        Path(instance_root), expected_identity=expected_identity, member_identities={}
    ) as (envelope, members):
        by_path = _require_members(members, ("report.json", "report.md"))
        contract = envelope["semantic_metadata"].get("target_contract")
        if type(contract) is not dict:
            raise ValueError("research_report_semantics_invalid")
        schema_id = contract.get("schema_id")
        _report_contract(schema_id)
        payload = read_json_evidence(
            Path(instance_root) / "report.json",
            expected_schema_id=schema_id,
            error="research_report_payload_invalid",
            expected_member=by_path["report.json"],
        )
        _validate_report_payload(payload)
        _validate_report_semantics(envelope, payload)
        if source_run is not None and (
            payload["run_id"] != source_run.run_id
            or payload["scenario_id"] != source_run.scenario_id
            or payload["source_run_status"] != source_run.status.value
            or (envelope["research_window"] != dict(source_run.research_window))
            or (envelope["protocol_or_spec_identity"] != source_run.protocol_identity)
            or (
                envelope["upstream_identities"]
                != {"source_run": canonical_json_hash(source_run.payload())}
            )
        ):
            raise ValueError("research_report_run_lineage_mismatch")
        if payload["scenario_id"] != envelope["logical_artifact_id"].removesuffix(
            "_report_artifact"
        ):
            raise ValueError("research_report_payload_invalid")
        if include_markdown:
            markdown = _read_utf8(
                Path(instance_root) / "report.md", expected_member=by_path["report.md"]
            )
            return {"report": payload, "markdown": markdown}
        return payload


def _validate_report_payload(payload: object) -> None:
    if type(payload) is not dict or set(payload) != {
        "schema_id",
        "run_id",
        "scenario_id",
        "access_window",
        "source_run_status",
        "report_status",
        "sections",
    }:
        raise ValueError("research_report_payload_invalid")
    if (
        type(payload["source_run_status"]) is not str
        or payload["source_run_status"] not in {"COMPLETE", "BLOCKED", "FAILED"}
        or type(payload["report_status"]) is not str
        or payload["report_status"] not in {"COMPLETE", "BLOCKED", "FAILED"}
        or type(payload["sections"]) is not dict
        or (set(payload["sections"]) != set(_report_contract(payload["schema_id"])[0]))
    ):
        raise ValueError("research_report_payload_invalid")
    try:
        validate_run_id(payload["run_id"])
        validate_run_id(payload["scenario_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError("research_report_payload_invalid") from exc


def _report_manifest_metadata(envelope: Mapping[str, object]) -> dict[str, object]:
    metadata = envelope.get("semantic_metadata")
    window = envelope.get("research_window")
    upstream = envelope.get("upstream_identities")
    if (
        type(metadata) is not dict
        or set(metadata)
        != {"target_contract", "source_run_id", "source_run_status", "report_status"}
        or type(window) is not dict
        or type(upstream) is not dict
    ):
        raise ValueError("research_report_semantics_invalid")
    try:
        validate_run_id(metadata["source_run_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError("research_report_semantics_invalid") from exc
    for field in ("source_run_status", "report_status"):
        if type(metadata[field]) is not str or metadata[field] not in {
            "COMPLETE",
            "BLOCKED",
            "FAILED",
        }:
            raise ValueError("research_report_semantics_invalid")
    contract = metadata.get("target_contract")
    if type(contract) is not dict or set(contract) != {
        "schema_id",
        "required_sections",
        "source_run_identity",
    }:
        raise ValueError("research_report_semantics_invalid")
    if (
        contract["required_sections"]
        != list(_report_contract(contract["schema_id"])[0])
        or upstream != {"source_run": contract["source_run_identity"]}
        or envelope["schema_identity"] != canonical_json_hash(contract)
    ):
        raise ValueError("research_report_semantics_invalid")
    return metadata


def _validate_report_semantics(
    envelope: Mapping[str, object], payload: dict[str, object]
) -> None:
    metadata = _report_manifest_metadata(envelope)
    contract = metadata["target_contract"]
    if (
        envelope.get("logical_artifact_id")
        != f"{payload['scenario_id']}_report_artifact"
        or envelope["research_window"].get("access") != payload["access_window"]
        or metadata.get("source_run_id") != payload["run_id"]
        or (metadata.get("source_run_status") != payload["source_run_status"])
        or (metadata.get("report_status") != payload["report_status"])
        or (contract.get("schema_id") != payload["schema_id"])
    ):
        raise ValueError("research_report_semantics_invalid")


def _require_members(
    members: tuple[dict[str, object], ...], expected: tuple[str, ...]
) -> MappingProxyType[str, object]:
    by_path = {str(member.get("path")): member for member in members}
    if len(members) != len(expected) or tuple(sorted(by_path)) != expected:
        raise ValueError("research_artifact_members_invalid")
    return MappingProxyType(by_path)


def _read_utf8(path: Path, *, expected_member: Mapping[str, object]) -> str:
    with closing_iterator(open_filesystem_file(path)) as handle:
        raw = handle.read(expected_member["size"] + 1)
    if (
        len(raw) != expected_member["size"]
        or hashlib.sha256(raw).hexdigest() != expected_member["sha256"]
    ):
        raise ValueError("research_report_markdown_identity_changed")
    return raw.decode("utf-8")


def produce_crypto_report(
    *,
    layout,
    record,
    panel_id,
    read_model_artifact,
    expected_symbols,
    read_reconstruction,
    validate_maker_calibration,
    calibration_candidates,
    comparison_ids,
    execution_targets,
):
    protocol = record.payload()["protocol"]
    final_test = record.research_window["access"] == "holdout"
    issues, references, captured = ([], {}, {})
    native_targets = {
        f"{record.scenario_id}_models",
        *(item.calibration_target_id for item in calibration_candidates),
        *(
            target
            for item in execution_targets
            if final_test
            for target in (item.target_id, item.statistics_target_id)
        ),
    }
    for result in record.target_results:
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
            issues.append(_issue(result.target_id, result.status.value, result.reason))
            continue
        if result.target_id in native_targets:
            references[result.target_id] = result
            captured[result.target_id] = None
            continue
        try:
            identities = {}
            read_validated_reference(layout, result, member_identities=identities)
            captured[result.target_id] = identities
        except (OSError, ValueError):
            issues.append(
                _issue(result.target_id, "FAILED", "target_reference_invalid")
            )
        else:
            references[result.target_id] = result
    data_rows, quality = ([], {})
    recording_counts = {}
    reconstruction_defined = False
    data_targets = {panel_id, record.scenario_id} | {
        result.target_id
        for result in record.target_results
        if result.producer_kind in {ProducerKind.PROCESSED, ProducerKind.DATASET}
    }
    for target in (panel_id, record.scenario_id):
        if target not in references and (
            not any(issue["target_id"] == target for issue in issues)
        ):
            issues.append(_issue(target, "BLOCKED", "run_target_result_missing"))
    for target, result in references.items():
        if result.producer_kind in {ProducerKind.PROCESSED, ProducerKind.DATASET}:
            try:
                manifest = read_dataset_manifest_evidence(
                    _instance_root(layout, result),
                    expected_member_identities=captured[result.target_id],
                )
                if target == panel_id:
                    recording_counts = {
                        key: manifest["producer_diagnostics"][key]
                        for key in ("raw_record_count", "source_file_count")
                    }
                    reconstruction = manifest["producer_diagnostics"]["reconstruction"]
                    if type(reconstruction) is not dict or any(
                        type(stats) is not dict for stats in reconstruction.values()
                    ):
                        raise ValueError("crypto_panel_diagnostics_invalid")
                    if set(reconstruction) != set(expected_symbols):
                        raise ValueError("crypto_panel_diagnostics_symbols_invalid")
                    states = [
                        read_reconstruction(stats) for stats in reconstruction.values()
                    ]
                    reconstruction_defined = all(
                        state.invalid_segments == 0
                        and state.valid_segments > 0
                        and bool(state.coverage_segments)
                        for state in states
                    )
                    quality = {
                        symbol: {
                            key: value
                            for key, value in stats.items()
                            if key != "coverage_segments"
                        }
                        for symbol, stats in reconstruction.items()
                    }
            except (OSError, KeyError, TypeError, ValueError):
                issues.append(
                    _issue(target, "FAILED", "crypto_dataset_evidence_invalid")
                )
                continue
            data_rows.append(
                {
                    "target_id": target,
                    "identity": result.instance_identity,
                    "row_count": manifest["row_count"],
                    "file_count": manifest["file_count"],
                }
            )
    data_issues = [issue for issue in issues if issue["target_id"] in data_targets]
    data = {
        "status": diagnostic_status(data_issues)
        if data_issues
        else "DEFINED"
        if reconstruction_defined
        else "UNDEFINED",
        "datasets": data_rows,
        "reconstruction": quality,
        "recording_counts": recording_counts,
        "issues": data_issues,
        "reconstruction_status": "DEFINED" if reconstruction_defined else "UNDEFINED",
    }
    model_target = f"{record.scenario_id}_models"
    models = _unavailable(issues, model_target)
    if model_target in references:
        reference = references[model_target]
        try:
            artifact = read_model_artifact(
                _instance_root(layout, reference),
                expected_identity=reference.instance_identity,
                expected_member_identities=captured[reference.target_id],
                trial_layout=layout,
            )
            _validate_reference_envelope(reference, artifact.envelope)
            models = {
                "status": "DEFINED",
                "identity": reference.instance_identity,
                "fold_models": list(artifact.models),
                "final_models": list(artifact.final_models),
                "validation_prediction_role": (
                    "train_model_predictions_of_mature_validation_rows"
                ),
            }
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(_issue(model_target, "FAILED", "model_artifact_invalid"))
            models = _unavailable(issues, model_target)
    prediction_target = f"{record.scenario_id}_predictions"
    predictions = (
        _unavailable(issues, prediction_target)
        if final_test
        else {"status": "UNDEFINED", "reason": "not_requested"}
    )
    if prediction_target in references:
        reference = references[prediction_target]
        try:
            rows = []
            for model_id in protocol["model_scopes"]:
                output_id = f"{model_id}_prediction"
                manifest, _members = validate_output_bundle(
                    _instance_root(layout, reference) / "outputs" / output_id,
                    expected_output_id=output_id,
                    expected_run_id=f"run_{output_id}",
                    expected_member_identities={
                        key.removeprefix(f"outputs/{output_id}/"): value
                        for key, value in captured[reference.target_id].items()
                        if key.startswith(f"outputs/{output_id}/")
                    },
                )
                metadata = manifest["metadata"]
                if (
                    metadata["model_metadata"]["model_artifact_identity"]
                    != references[model_target].instance_identity
                ):
                    raise ValueError("prediction_model_identity_mismatch")
                rows.append(
                    {
                        "model_id": model_id,
                        "prediction_id": output_id,
                        "primary_key": metadata["primary_key"],
                        "training_asset_ids": metadata["training_asset_ids"],
                        "evaluation_asset_ids": metadata["evaluation_asset_ids"],
                        "prediction_metadata": metadata["prediction_metadata"],
                    }
                )
            predictions = {
                "status": "DEFINED",
                "identity": reference.instance_identity,
                "outputs": rows,
            }
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(prediction_target, "FAILED", "prediction_artifact_invalid")
            )
            predictions = _unavailable(issues, prediction_target)
    evaluations = []
    for result in record.target_results:
        if result.producer_kind is not ProducerKind.EVALUATION:
            continue
        if result.target_id not in references:
            evaluations.append(
                {
                    "evaluation_id": result.target_id,
                    **_unavailable(issues, result.target_id),
                }
            )
            continue
        try:
            evaluations.append(
                read_evaluation_report_evidence(
                    _instance_root(layout, result),
                    evaluation_id=result.target_id,
                    instance_identity=result.instance_identity,
                    expected_member_identities=captured[result.target_id],
                )
            )
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(result.target_id, "FAILED", "evaluation_artifact_invalid")
            )
            evaluations.append(
                {
                    "evaluation_id": result.target_id,
                    **_unavailable(issues, result.target_id),
                }
            )
    calibrations = []
    for item in calibration_candidates:
        calibration_target = item.calibration_target_id
        calibration = _unavailable(issues, calibration_target)
        if calibration_target in references:
            reference = references[calibration_target]
            try:
                artifact = read_statistics_artifact(
                    _instance_root(layout, reference),
                    expected_identity=reference.instance_identity,
                    expected_member_identities=captured[reference.target_id],
                    validate_tables=validate_maker_calibration,
                )
                _validate_reference_envelope(reference, artifact.envelope)
                with artifact.table("crypto_maker_calibration.parquet") as frame:
                    calibration = {
                        "status": "DEFINED"
                        if frame.status.eq("DEFINED").all()
                        else "UNDEFINED",
                        "identity": reference.instance_identity,
                        "summary": _report_rows(frame),
                        "method": (
                            "paired_validation_event_paths_and_forecast_error_by_side"
                        ),
                        "quote_order_type": "post_only_limit",
                        "outcome_clock": "original_decision_plus_500ms",
                        "sample_tables": [
                            "crypto_maker_samples.parquet",
                            "crypto_maker_events.parquet",
                        ],
                    }
            except (OSError, KeyError, TypeError, ValueError):
                issues.append(
                    _issue(
                        calibration_target,
                        "FAILED",
                        "maker_calibration_artifact_invalid",
                    )
                )
                calibration = _unavailable(issues, calibration_target)
        calibrations.append(
            {
                "candidate_id": item.candidate_id,
                "asset_id": item.asset_id,
                **calibration,
            }
        )
    portfolios = []
    statistics = []
    for execution_target in execution_targets if final_test else ():
        candidate = execution_target.candidate
        target = execution_target.target_id
        key = {
            "candidate_id": candidate.candidate_id,
            "asset_id": candidate.asset_id,
            "execution_mode": execution_target.execution_mode,
            "target_id": target,
        }
        if target not in references:
            portfolios.append({**key, **_unavailable(issues, target)})
        else:
            reference = references[target]
            try:
                root = _instance_root(layout, reference)
                if execution_target.execution_mode == "continuous":
                    economics = {}
                    ledger = read_portfolio_ledger_artifact(
                        root,
                        expected_identity=reference.instance_identity,
                        expected_member_identities=captured[reference.target_id],
                        consume_batches=lambda ledger, batches, result=economics: (
                            result.update(
                                summarize_portfolio_ledger(ledger.capital, batches)
                            )
                        ),
                    )
                    _validate_reference_envelope(reference, ledger.envelope)
                    result = {
                        "episode_id": "continuous",
                        "economics": economics,
                        "execution_curve": {
                            "artifact_root": reference.artifact_root,
                            "identity": reference.instance_identity,
                            "member": "valuations.parquet",
                            "aggregation": "sum_end_values_by_sequence",
                            "clock": "persisted_event_time",
                            "use": "execution_only_not_annualized",
                        },
                    }
                else:
                    episode_rows = []

                    def consume_episode(
                        index,
                        ledger,
                        batches,
                        reference=reference,
                        episode_rows=episode_rows,
                    ):
                        row = {
                            "episode_id": index["episode_id"],
                            "start_at": pd.Timestamp(index["start_ns"]).isoformat(),
                            "stop_at": pd.Timestamp(index["stop_ns"]).isoformat(),
                            "terminal_valuation_at": pd.Timestamp(
                                index["terminal_valuation_ns"]
                            ).isoformat(),
                            "termination": index["termination"],
                            "stop_reasons": index["stop_reasons"],
                            "economics": summarize_portfolio_ledger(
                                ledger.capital, batches
                            ),
                            "execution_curve": {
                                "artifact_root": reference.artifact_root,
                                "identity": index["ledger_identity"],
                                "member": index["ledger_root"] + "/valuations.parquet",
                                "aggregation": "sum_end_values_by_sequence",
                                "clock": "persisted_event_time",
                                "use": "independent_episode_only",
                            },
                        }
                        episode_rows.append(row)

                    artifact = read_crypto_episode_artifact(
                        root,
                        expected_identity=reference.instance_identity,
                        expected_member_identities=captured[reference.target_id],
                        consume_episode=consume_episode,
                    )
                    _validate_reference_envelope(reference, artifact._envelope)
                    result = {
                        "episodes": episode_rows,
                        "return_policy": "per_episode_only",
                    }
                portfolios.append(
                    {
                        **key,
                        "status": "DEFINED",
                        "identity": reference.instance_identity,
                        **result,
                    }
                )
            except (OSError, KeyError, TypeError, ValueError):
                issues.append(
                    _issue(target, "FAILED", "portfolio_execution_artifact_invalid")
                )
                portfolios.append({**key, **_unavailable(issues, target)})
        statistics_target = execution_target.statistics_target_id
        key = {
            "candidate_id": candidate.candidate_id,
            "execution_mode": execution_target.execution_mode,
            "target_id": statistics_target,
        }
        if statistics_target not in references:
            statistics.append({**key, **_unavailable(issues, statistics_target)})
            continue
        reference = references[statistics_target]
        try:
            artifact = read_statistics_artifact(
                _instance_root(layout, reference),
                expected_identity=reference.instance_identity,
                expected_member_identities=captured[reference.target_id],
            )
            _validate_reference_envelope(reference, artifact.envelope)
            tables = {}
            for member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]:
                with artifact.table(member) as frame:
                    tables[member.removesuffix(".parquet")] = _report_rows(frame)
            statistics.append(
                {
                    **key,
                    "status": "DEFINED",
                    "identity": reference.instance_identity,
                    "tables": tables,
                }
            )
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(statistics_target, "FAILED", "portfolio_statistics_invalid")
            )
            statistics.append({**key, **_unavailable(issues, statistics_target)})
    return build_crypto_research_report(
        record=record,
        evidence={
            "data": data,
            "models": models,
            "predictions": predictions,
            "evaluations": [
                item
                for item in evaluations
                if item["evaluation_id"] not in comparison_ids
            ],
            "model_comparisons": [
                item for item in evaluations if item["evaluation_id"] in comparison_ids
            ],
            "calibrations": calibrations,
            "portfolios": portfolios,
            "statistics": statistics,
            "issues": issues,
        },
    )


def build_report_from_run(
    *,
    market_id,
    produce_report,
    software_identity: str,
    execution_targets=(),
    layout: StorageLayout,
    record: ResearchRunRecord,
    plan: bool,
) -> dict[str, object]:
    report_target = f"{record.scenario_id}_report_artifact"
    report_inputs = (
        {"execution_targets": execution_targets} if str(market_id) == "crypto" else {}
    )
    payload = produce_report(layout=layout, record=record, **report_inputs)
    report_status = RunStatus(payload["report_status"])
    source_run_identity = canonical_json_hash(record.payload())
    contract = {
        "schema_id": payload["schema_id"],
        "required_sections": list(payload["sections"]),
        "source_run_identity": source_run_identity,
    }
    envelope = build_business_manifest_envelope(
        logical_artifact_id=report_target,
        schema_identity=canonical_json_hash(contract),
        protocol_or_spec_identity=record.protocol_identity,
        software_identity=software_identity,
        research_window=record.research_window,
        upstream_identities={"source_run": source_run_identity},
        semantic_metadata={
            "target_contract": contract,
            "source_run_id": record.run_id,
            "source_run_status": record.status.value,
            "report_status": report_status.value,
        },
    )
    logical_root = layout.semantic_artifact_root(market_id, report_target)
    instance_root = layout.instance_root(logical_root, envelope.instance_identity)
    decision = "REUSE" if filesystem_exists(instance_root) else "BUILD"
    if plan:
        blocker = None
        if decision == "REUSE":
            try:
                if (
                    read_research_report_artifact(
                        instance_root, expected_identity=envelope.instance_identity
                    )
                    != payload
                ):
                    raise ValueError("research_report_reuse_mismatch")
            except (FileNotFoundError, OSError, ValueError):
                decision = "BLOCKED"
                blocker = "persisted_report_invalid"
                report_status = RunStatus.FAILED
        return {
            "stage": "report build",
            "profile_id": record.profile_id,
            "run_id": record.run_id,
            "source_run_status": record.status.value,
            "status": report_status.value,
            "target": {
                "target_id": report_target,
                "decision": decision,
                "requested_identity": envelope.instance_identity,
                "blocker": blocker,
            },
        }
    target_status = TargetStatus.BUILT
    if decision == "REUSE":
        existing = read_research_report_artifact(
            instance_root, expected_identity=envelope.instance_identity
        )
        if existing != payload:
            raise ValueError("research_report_reuse_mismatch")
        target_status = TargetStatus.REUSED
    else:
        write_research_report_artifact(
            logical_root,
            envelope=envelope,
            payload=payload,
        )
    return {
        "stage": "report build",
        "profile_id": record.profile_id,
        "run_id": record.run_id,
        "source_run_status": record.status.value,
        "status": report_status.value,
        "target_result": {
            "target_id": report_target,
            "status": target_status.value,
            "instance_identity": envelope.instance_identity,
        },
        "report": payload,
    }


def render_research_report(payload: dict[str, object], *, sections, title) -> str:
    lines = [
        f"# {title}",
        "",
        f"Run: {payload['run_id']}",
        f"Scenario: {payload['scenario_id']}",
        f"Access: {payload['access_window']}",
        f"Source run status: {payload['source_run_status']}",
        f"Report status: {payload['report_status']}",
        "",
        "完整机器证据：[report.json](report.json)。"
        "表中的 — 表示未提供值；状态与原因见对应列。",
        "",
    ]
    content = payload["sections"]
    if type(content) is not dict:
        raise ValueError("research_report_payload_invalid")
    for name in sections:
        lines.extend((f"## {_markdown_cell(name)}", ""))
        _markdown_content(lines, content[name], depth=3)
    return "\n".join(lines)


def _markdown_cell(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{value:,}"
    return escape(str(value), quote=False).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(lines, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        return
    lines.extend(
        (
            "| " + " | ".join((_markdown_cell(key) for key in columns)) + " |",
            "| " + " | ".join("---" for _key in columns) + " |",
        )
    )
    lines.extend(
        "| " + " | ".join(_markdown_cell(row.get(key)) for key in columns) + " |"
        for row in rows
    )
    lines.append("")


def _markdown_content(lines, value, *, depth):
    if isinstance(value, Mapping):
        scalars = [
            {"指标": key, "值": item}
            for key, item in value.items()
            if not isinstance(item, (Mapping, list, tuple))
        ]
        _markdown_table(lines, scalars)
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                lines.extend((f"{'#' * min(depth, 6)} {_markdown_cell(key)}", ""))
                _markdown_content(lines, item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        if not value:
            lines.extend(("无记录。", ""))
        elif all(isinstance(item, Mapping) for item in value):
            _markdown_table(
                lines,
                [
                    {
                        key: item
                        for key, item in row.items()
                        if not isinstance(item, (Mapping, list, tuple))
                    }
                    for row in value
                ],
            )
            for index, row in enumerate(value, start=1):
                label = next(
                    (
                        row[key]
                        for key in ("candidate_id", "model_id", "target_id", "fold_id")
                        if key in row
                    ),
                    str(index),
                )
                for key, item in row.items():
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.extend(
                            (
                                f"{'#' * min(depth, 6)} {_markdown_cell(label)} / "
                                f"{_markdown_cell(key)}",
                                "",
                            )
                        )
                        _markdown_content(lines, item, depth=depth + 1)
        else:
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    _markdown_content(lines, item, depth=depth + 1)
                else:
                    lines.append(f"- {_markdown_cell(item)}")
            lines.append("")
    else:
        lines.extend((_markdown_cell(value), ""))
