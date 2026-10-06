"""Persisted research-run records and rendered reports."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from html import escape
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import pandas as pd

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
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.targets import (
    ProducerKind,
    ResearchRunRecord,
    ResearchRunTargetResult,
    TargetStatus,
    research_run_record_from_payload,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.reports import a_share, crypto_hft, diagnostic_status
from quant_research_platform.reports.a_share import build_a_share_report
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
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

    def load(target_id: str) -> ResearchRunTargetResult | None:
        if target_id in loaded:
            return loaded[target_id]
        result = references.get(target_id)
        if result is None:
            issues.append(_issue(target_id, "BLOCKED", "run_target_result_missing"))
            return None
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
            return None
        try:
            identities = {}
            read_validated_reference(
                layout,
                result,
                member_identities=identities,
                expected_member_identities=None
                if reference_member_identities is None
                else reference_member_identities.get(target_id),
            )
            captured[target_id] = identities
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
    full_research = configuration["stage"] == "research"
    for target_id in (
        scenario,
        f"{scenario}_evaluation",
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
    factor_evaluation = _unavailable(issues, f"{scenario}_evaluation")
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
        operational = {}
        read_semantic_artifact_record(
            _instance_root(layout, model_ref),
            expected_identity=model_ref.instance_identity,
            expected_member_identities=captured[model_ref.target_id],
            operational_metadata=operational,
        )
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
    statistics = {}
    for group in ("global", "quarterly"):
        statistics_target = f"{scenario}_{group}_statistics"
        statistics_ref = load(statistics_target) if full_research else None
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
                tables = {
                    name: table_records(artifact, name + ".parquet")
                    for name in (
                        "factor_diagnostics",
                        "model_trials",
                        "model_comparison",
                        "model_group_comparison",
                        "model_reality_checks",
                        "robustness",
                    )
                }
                factors = tables["factor_diagnostics"]
                group_statistics = {
                    "status": "DEFINED",
                    "identity": statistics_ref.instance_identity,
                    "factor_row_count": len(factors),
                    "factor_defined_rows": sum(
                        row["status"] == "DEFINED" for row in factors
                    ),
                    "factor_undefined_rows": sum(
                        row["status"] == "UNDEFINED" for row in factors
                    ),
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
    portfolio_ref = load(portfolio_target) if candidates else None
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
        ledger_ref = load(target_id)
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
            ledger_operations = {}
            read_semantic_artifact_record(
                _instance_root(layout, ledger_ref),
                expected_identity=ledger_ref.instance_identity,
                expected_member_identities=captured[ledger_ref.target_id],
                operational_metadata=ledger_operations,
            )
            portfolios.append(
                {
                    "candidate_id": candidate,
                    "status": "DEFINED",
                    "ledger_identity": ledger_ref.instance_identity,
                    "final_calibration": ledger.final_alpha_calibration,
                    "calibration_observations": ledger_operations.get(
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
    selections,
    reference_member_identities=None,
) -> dict[str, object]:
    return build_a_share_report(
        record=record,
        evidence=load_a_share_run_evidence(
            layout=layout,
            record=record,
            selections=selections,
            reference_member_identities=reference_member_identities,
        ),
    )


def read_validated_reference(
    layout: StorageLayout,
    result: ResearchRunTargetResult,
    *,
    member_identities=None,
    expected_member_identities=None,
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
        )
    if (
        envelope["logical_artifact_id"] != result.target_id
        or envelope["instance_identity"] != result.instance_identity
        or envelope["schema_identity"] != result.schema_identity
    ):
        raise ValueError("research_run_target_reference_invalid")
    return (root, envelope)


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
        write_markdown_evidence(staging / "report.md", _markdown(payload))

    return write_semantic_artifact(Path(logical_root), envelope, write)


def read_research_report_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
    source_run: ResearchRunRecord | None = None,
    include_markdown: bool = False,
) -> dict[str, object]:
    envelope, members = read_semantic_artifact_record(
        Path(instance_root), expected_identity=expected_identity
    )
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
        payload["source_run_status"] not in {"COMPLETE", "BLOCKED", "FAILED"}
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


def _validate_report_semantics(
    envelope: Mapping[str, object], payload: dict[str, object]
) -> None:
    metadata = envelope.get("semantic_metadata")
    window = envelope.get("research_window")
    upstream = envelope.get("upstream_identities")
    if (
        type(metadata) is not dict
        or type(window) is not dict
        or type(upstream) is not dict
    ):
        raise ValueError("research_report_semantics_invalid")
    contract = metadata.get("target_contract")
    if type(contract) is not dict or set(contract) != {
        "schema_id",
        "required_sections",
        "source_run_identity",
    }:
        raise ValueError("research_report_semantics_invalid")
    if (
        envelope.get("logical_artifact_id")
        != f"{payload['scenario_id']}_report_artifact"
        or window.get("access") != payload["access_window"]
        or metadata.get("source_run_id") != payload["run_id"]
        or (metadata.get("source_run_status") != payload["source_run_status"])
        or (metadata.get("report_status") != payload["report_status"])
        or (contract.get("schema_id") != payload["schema_id"])
        or (
            contract.get("required_sections")
            != list(_report_contract(payload["schema_id"])[0])
        )
        or (upstream != {"source_run": contract.get("source_run_identity")})
        or (envelope.get("schema_identity") != canonical_json_hash(contract))
    ):
        raise ValueError("research_report_semantics_invalid")


def _require_members(
    members: tuple[dict[str, object], ...], expected: tuple[str, ...]
) -> MappingProxyType[str, object]:
    by_path = {str(member.get("path")): member for member in members}
    if len(members) != len(expected) or tuple(sorted(by_path)) != expected:
        raise ValueError("research_artifact_members_invalid")
    return MappingProxyType(by_path)


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        f"# {_report_contract(payload['schema_id'])[1]}",
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
    sections = payload["sections"]
    if type(sections) is not dict:
        raise ValueError("research_report_payload_invalid")
    for name in _report_contract(payload["schema_id"])[0]:
        lines.extend((f"## {_markdown_cell(name)}", ""))
        _markdown_content(lines, sections[name], depth=3)
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


def _read_utf8(path: Path, *, expected_member: Mapping[str, object]) -> str:
    with open_filesystem_file(path) as handle:
        raw = handle.read(expected_member["size"] + 1)
    if (
        len(raw) != expected_member["size"]
        or hashlib.sha256(raw).hexdigest() != expected_member["sha256"]
    ):
        raise ValueError("research_report_markdown_identity_changed")
    return raw.decode("utf-8")
