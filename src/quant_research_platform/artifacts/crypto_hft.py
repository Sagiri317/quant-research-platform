"""Crypto HFT execution evidence owner."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.artifacts.portfolio_ledger import (
    PORTFOLIO_LEDGER_TABLE_COLUMNS,
    PortfolioLedgerArtifact,
    _capital,
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.research import (
    _instance_root,
    _issue,
    _report_rows,
    _unavailable,
    read_evaluation_report_evidence,
    read_validated_reference,
    report_payload_working_set,
)
from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_GROUPS,
    read_statistics_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.targets import ProducerKind, TargetStatus
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.reports import diagnostic_status
from quant_research_platform.reports.crypto_hft import build_crypto_research_report
from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    native_filesystem_path,
)
from quant_research_platform.storage.manifests import (
    read_dataset_manifest_evidence,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    _member_inventory,
    _visit_member_paths,
    read_semantic_artifact_record,
    write_semantic_artifact,
)

CRYPTO_EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("stop_ns", pa.int64(), nullable=False),
        pa.field("terminal_valuation_ns", pa.int64(), nullable=False),
        pa.field("termination", pa.string(), nullable=False),
        pa.field("stop_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("ledger_root", pa.string(), nullable=False),
        pa.field("ledger_identity", pa.string(), nullable=False),
    ]
)
_EPISODE_STOP_REASONS = frozenset(
    {
        "planned_oos_end",
        "book_coverage_end",
        "trade_coverage_end",
        "funding_coverage_end",
        "archive_end",
    }
)


def crypto_episode_schema_payload():
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in CRYPTO_EPISODE_SCHEMA
    ]


def _episode_identity(parent_identity, start_ns, stop_ns):
    return canonical_json_hash(
        {
            "execution_instance_identity": parent_identity,
            "start_ns": start_ns,
            "stop_ns": stop_ns,
        }
    )


def _episode_envelope(parent, start, stop):
    episode_id = _episode_identity(parent["instance_identity"], start, stop)
    return build_business_manifest_envelope(
        logical_artifact_id="episode_" + episode_id,
        schema_identity=canonical_json_hash(PORTFOLIO_LEDGER_TABLE_COLUMNS),
        protocol_or_spec_identity=parent["protocol_or_spec_identity"],
        software_identity=parent["software_identity"],
        research_window={
            **parent["research_window"],
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(stop - 1).isoformat(),
        },
        upstream_identities={"episode_parent": parent["instance_identity"]},
        semantic_metadata={
            "portfolio_capital": parent["semantic_metadata"]["portfolio_capital"]
        },
    )


def _episode_rows(root, envelope, members, identities, manifest_identity=None):
    """Keep the parent's captured physical member barrier through child consumption."""
    previous_stop = None
    expected_paths = {"episodes.parquet"}
    window = envelope["research_window"]
    first, last = pd.Timestamp(window["start"]).value, pd.Timestamp(window["end"]).value
    try:
        with _opened_parquet(
            root / "episodes.parquet", expected_identity=identities["episodes.parquet"]
        ) as (parquet, _identity, _handle):
            if not parquet.schema_arrow.equals(
                CRYPTO_EPISODE_SCHEMA, check_metadata=False
            ):
                raise ValueError("crypto_episode_schema_invalid")
        policy = current_runtime_resource_policy()
        chunks = iter_declared_parquet_member_batches(
            root / "episodes.parquet",
            expected_identity=identities["episodes.parquet"],
            target_chunk_bytes=max(1, policy.planning_headroom_bytes // 8)
            if policy is not None
            else 1024 * 1024,
        )
        count = 0
        with closing_iterator(chunks):
            for frame in chunks:
                for row in frame.to_dict("records"):
                    start, stop, terminal = (
                        int(row["start_ns"]),
                        int(row["stop_ns"]),
                        int(row["terminal_valuation_ns"]),
                    )
                    reasons = list(row["stop_reasons"])
                    episode_id = _episode_identity(
                        envelope["instance_identity"], start, stop
                    )
                    child_root = (
                        f"episodes/{episode_id}/instances/{row['ledger_identity']}"
                    )
                    if (
                        row["episode_id"] != episode_id
                        or row["ledger_root"] != child_root
                        or not first <= start <= terminal < stop <= last + 1
                        or previous_stop is not None
                        and start < previous_stop
                        or not reasons
                        or reasons != sorted(set(reasons))
                        or not set(reasons) <= _EPISODE_STOP_REASONS
                        or row["termination"]
                        != ("PLANNED_END" if stop == last + 1 else "TRUNCATED")
                        or ("planned_oos_end" in reasons) != (stop == last + 1)
                    ):
                        raise ValueError("crypto_episode_index_invalid")
                    previous_stop = stop
                    row["stop_reasons"] = reasons
                    child_paths = {
                        child_root + "/" + member
                        for member in (
                            "manifest.json",
                            *(
                                name + ".parquet"
                                for name in PORTFOLIO_LEDGER_TABLE_COLUMNS
                            ),
                        )
                    }
                    if not child_paths <= identities.keys():
                        raise ValueError("crypto_episode_child_members_invalid")
                    expected_paths.update(child_paths)
                    terminal_observed = None

                    def consume(batches):
                        nonlocal terminal_observed
                        for batch in batches:
                            try:
                                for value in batch.valuations:
                                    if value.phase == "end":
                                        terminal_observed = (
                                            value.timestamp_ns
                                            if terminal_observed is None
                                            else max(
                                                terminal_observed, value.timestamp_ns
                                            )
                                        )

                            finally:
                                batch = value = None

                    expected_child = _episode_envelope(envelope, start, stop)
                    if row["ledger_identity"] != expected_child.instance_identity:
                        raise ValueError("crypto_episode_child_contract_invalid")
                    child_identities = {
                        path.removeprefix(child_root + "/"): identities[path]
                        for path in child_paths
                    }
                    child = read_portfolio_ledger_artifact(
                        root / child_root,
                        expected_identity=expected_child.instance_identity,
                        expected_member_identities=child_identities,
                        consume_batches=consume,
                    )
                    if terminal_observed != terminal:
                        raise ValueError("crypto_episode_child_contract_invalid")
                    count += 1
                    yield row, child
        if not count or expected_paths != {member["path"] for member in members}:
            raise ValueError("crypto_episode_child_members_invalid")
    finally:
        primary = sys.exception()
        try:
            observed_paths = set()
            _visit_member_paths(
                root, lambda relative, path: observed_paths.add(relative)
            )
            if observed_paths != set(identities) or any(
                dataset_file_identity(root / path) != identity
                for path, identity in identities.items()
            ):
                raise ValueError("crypto_episode_member_identity_changed")
            if (
                manifest_identity is not None
                and dataset_file_identity(root / "manifest.json") != manifest_identity
            ):
                raise ValueError("crypto_episode_manifest_identity_changed")
        except BaseException as error:
            if primary is None or isinstance(primary, GeneratorExit):
                raise
            primary.add_note(
                "episode member verification also failed: " + type(error).__name__
            )


@dataclass(frozen=True, slots=True)
class CryptoEpisodeArtifact:
    root: Path
    identity: str
    capital: PortfolioCapital
    _envelope: dict
    _members: tuple
    _identities: dict
    _manifest_identity: object

    def iter_episodes(self):
        yield from _episode_rows(
            self.root,
            self._envelope,
            self._members,
            self._identities,
            self._manifest_identity,
        )


def read_crypto_episode_artifact(
    root, *, expected_identity, expected_member_identities=None
):
    root = Path(root)
    identities = {}
    envelope, members = read_semantic_artifact_record(
        root,
        expected_identity=expected_identity,
        member_identities=identities,
        expected_member_identities=expected_member_identities,
    )
    manifest_identity = identities.pop("manifest.json")
    artifact = CryptoEpisodeArtifact(
        root,
        expected_identity,
        _capital(envelope),
        envelope,
        members,
        identities,
        manifest_identity,
    )
    with closing_iterator(artifact.iter_episodes()) as episodes:
        for _row, _ledger in episodes:
            pass
    return artifact


def write_crypto_episode_artifact(logical_root, *, envelope, produce):
    """Publish the complete episode collection only after every child validates."""

    def write(staging):
        count = 0
        with parquet_table_writer(
            native_filesystem_path(staging / "episodes.parquet"),
            parent=staging,
            compression="zstd",
            schema=CRYPTO_EPISODE_SCHEMA,
        ) as write_index:

            def emit_episode(*, start_ns, stop_ns, termination, stop_reasons, produce):
                nonlocal count
                episode_id = _episode_identity(
                    envelope.instance_identity, start_ns, stop_ns
                )
                child = _episode_envelope(envelope.payload(), start_ns, stop_ns)
                terminal = None

                def produce_child(emit):
                    def capture(batch):
                        nonlocal terminal
                        for value in batch.valuations:
                            if value.phase == "end":
                                terminal = (
                                    value.timestamp_ns
                                    if terminal is None
                                    else max(terminal, value.timestamp_ns)
                                )
                        emit(batch)

                    return produce(capture)

                root = write_portfolio_ledger_artifact(
                    staging / "episodes" / episode_id,
                    envelope=child,
                    produce=produce_child,
                )
                row = {
                    "episode_id": episode_id,
                    "start_ns": start_ns,
                    "stop_ns": stop_ns,
                    "terminal_valuation_ns": terminal,
                    "termination": termination,
                    "stop_reasons": list(stop_reasons),
                    "ledger_root": root.relative_to(staging).as_posix(),
                    "ledger_identity": child.instance_identity,
                }
                write_index(pa.Table.from_pylist([row], schema=CRYPTO_EPISODE_SCHEMA))
                count += 1

            produce(emit_episode)
            if not count:
                raise PortfolioTargetBlocked(
                    "crypto_execution_qualified_episode_missing"
                )

    def validate(staging, members):
        identities = {}
        if (
            _member_inventory(
                staging,
                expected_paths={member["path"] for member in members},
                verified_identities=identities,
            )
            != members
        ):
            raise ValueError("crypto_episode_member_integrity_invalid")
        with closing_iterator(
            _episode_rows(staging, envelope.payload(), members, identities)
        ) as rows:
            for _row, _ledger in rows:
                pass

    return write_semantic_artifact(
        Path(logical_root), envelope, write, staged_validator=validate
    )


def produce_crypto_report(
    *,
    layout,
    record,
    panel_id,
    read_model_artifact,
    report_lease,
    expected_symbols,
    read_reconstruction,
    validate_maker_calibration,
    calibration_candidates,
    comparison_ids,
    execution_targets,
):
    protocol = record.payload()["protocol"]
    final_test = record.research_window["access"] == "holdout"
    issues, references, captured = [], {}, {}
    for result in record.target_results:
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
            issues.append(_issue(result.target_id, result.status.value, result.reason))
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

    data_rows, quality = [], {}
    recording_counts = {}
    reconstruction_defined = False
    data_targets = {panel_id, record.scenario_id} | {
        result.target_id
        for result in record.target_results
        if result.producer_kind in {ProducerKind.PROCESSED, ProducerKind.DATASET}
    }
    for target in (panel_id, record.scenario_id):
        if target not in references and not any(
            issue["target_id"] == target for issue in issues
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
            report_lease.grow(
                report_payload_working_set(
                    {
                        "fold_models": list(artifact.models),
                        "final_models": list(artifact.final_models),
                    }
                )
            )
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
                    report_lease=report_lease,
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
                with artifact.table("crypto_maker_calibration.parquet") as frame:
                    calibration = {
                        "status": "DEFINED"
                        if frame.status.eq("DEFINED").all()
                        else "UNDEFINED",
                        "identity": reference.instance_identity,
                        "summary": _report_rows(frame, report_lease),
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
                    ledger = read_portfolio_ledger_artifact(
                        root,
                        expected_identity=reference.instance_identity,
                        expected_member_identities=captured[reference.target_id],
                    )
                    result = {
                        "episode_id": "continuous",
                        "economics": summarize_portfolio_ledger(ledger),
                        "execution_curve": {
                            "artifact_root": reference.artifact_root,
                            "identity": reference.instance_identity,
                            "member": "valuations.parquet",
                            "aggregation": "sum_end_values_by_sequence",
                            "clock": "persisted_event_time",
                            "use": "execution_only_not_annualized",
                        },
                    }
                    report_lease.grow(report_payload_working_set(result))
                else:
                    collection = read_crypto_episode_artifact(
                        root,
                        expected_identity=reference.instance_identity,
                        expected_member_identities=captured[reference.target_id],
                    )
                    episode_rows = []
                    with closing_iterator(collection.iter_episodes()) as episodes:
                        for index, ledger in episodes:
                            row = {
                                "episode_id": index["episode_id"],
                                "start_at": pd.Timestamp(index["start_ns"]).isoformat(),
                                "stop_at": pd.Timestamp(index["stop_ns"]).isoformat(),
                                "terminal_valuation_at": pd.Timestamp(
                                    index["terminal_valuation_ns"]
                                ).isoformat(),
                                "termination": index["termination"],
                                "stop_reasons": index["stop_reasons"],
                                "economics": summarize_portfolio_ledger(ledger),
                                "execution_curve": {
                                    "artifact_root": reference.artifact_root,
                                    "identity": index["ledger_identity"],
                                    "member": index["ledger_root"]
                                    + "/valuations.parquet",
                                    "aggregation": "sum_end_values_by_sequence",
                                    "clock": "persisted_event_time",
                                    "use": "independent_episode_only",
                                },
                            }
                            report_lease.grow(report_payload_working_set(row))
                            episode_rows.append(row)
                    result = {
                        "episodes": episode_rows,
                        "return_policy": ("per_episode_only"),
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
            tables = {}
            for member in STATISTICS_TABLE_GROUPS["crypto_portfolio"]:
                with artifact.table(member) as frame:
                    tables[member.removesuffix(".parquet")] = _report_rows(
                        frame, report_lease
                    )
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


def summarize_crypto_execution(artifact):
    result = None
    with closing_iterator(artifact.iter_chunks()) as chunks:
        for batch in chunks:
            try:
                for diagnostic in batch.diagnostics:
                    if diagnostic.name != "execution_summary":
                        continue
                    if (
                        result is not None
                        or type(diagnostic.text) is not str
                        or len(diagnostic.text) > 65536
                    ):
                        raise ValueError("crypto_execution_summary_invalid")
                    result = parse_qrp_json_object(diagnostic.text.encode("utf-8"))
                    if PORTFOLIO_ECONOMIC_SUMMARY_FIELDS.intersection(result) or any(
                        type(value) not in {int, float, str, bool, type(None)}
                        for value in result.values()
                    ):
                        raise ValueError("crypto_execution_duplicate_economic_fields")
            finally:
                batch = diagnostic = None
    if result is None:
        raise ValueError("crypto_execution_summary_missing")
    return result


def summarize_crypto_ledger_windows(
    artifact: PortfolioLedgerArtifact,
    *,
    windows_ns: tuple[tuple[int, int], ...],
) -> list[dict[str, object]]:
    """Project OOS and gap economics from ordered persisted boundary markers."""
    expected = [
        (name, float(index), timestamp)
        for index, bounds in enumerate(windows_ns)
        for name, timestamp in zip(
            ("oos_window_start", "oos_window_end"), bounds, strict=True
        )
    ]
    if not expected:
        raise ValueError("crypto_ledger_windows_missing")
    fees = funding = turnover = 0.0
    fills = marker_index = 0
    previous = None
    intervals = []
    nav = artifact.capital.initial_capital
    chunks = artifact.iter_chunks()
    with closing_iterator(chunks):
        for batch in chunks:
            try:
                fees -= sum(
                    row.amount for row in batch.cashflows if row.kind.endswith("_cost")
                )
                funding += sum(
                    row.amount for row in batch.cashflows if row.kind == "funding"
                )
                turnover += sum(abs(row.notional) for row in batch.trades)
                fills += len(batch.trades)
                ends = [row for row in batch.valuations if row.phase == "end"]
                nav = sum(row.value for row in ends)
                for diagnostic in batch.diagnostics:
                    if diagnostic.name not in {"oos_window_start", "oos_window_end"}:
                        continue
                    if (
                        marker_index >= len(expected)
                        or (
                            diagnostic.name,
                            diagnostic.number,
                            ends[0].timestamp_ns,
                        )
                        != expected[marker_index]
                    ):
                        raise ValueError("crypto_ledger_window_marker_invalid")
                    current = (nav, fees, funding, turnover, fills)
                    if previous is None:
                        if current != (
                            artifact.capital.initial_capital,
                            0.0,
                            0.0,
                            0.0,
                            0,
                        ):
                            raise ValueError("crypto_ledger_before_window_economics")
                    else:
                        net, cost, cash, traded, count = (
                            value - before
                            for value, before in zip(current, previous, strict=True)
                        )
                        intervals.append(
                            {
                                "kind": "oos" if marker_index % 2 else "gap",
                                "start_ns": expected[marker_index - 1][2],
                                "end_ns": expected[marker_index][2],
                                "start_nav": previous[0],
                                "end_nav": nav,
                                "net_pnl": net,
                                "gross_pnl": net + cost - cash,
                                "fees": cost,
                                "funding": cash,
                                "turnover": traded,
                                "fill_events": count,
                            }
                        )
                    previous = current
                    marker_index += 1
            finally:
                batch = ends = diagnostic = None
    if marker_index != len(expected):
        raise ValueError("crypto_ledger_window_marker_missing")
    if previous != (nav, fees, funding, turnover, fills):
        raise ValueError("crypto_ledger_after_window_economics")
    return intervals
