"""Crypto HFT execution evidence owner."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.artifacts.portfolio_ledger import (
    PortfolioLedgerArtifact,
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.research import (
    _instance_root,
    _issue,
    _unavailable,
    _validate_reference,
    report_payload_working_set,
    reserve_report_frames,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
)
from quant_research_platform.contracts.targets import ProducerKind, TargetStatus
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.reports import diagnostic_status
from quant_research_platform.reports.crypto_hft import build_crypto_research_report
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.manifests import (
    read_dataset_manifest_evidence,
)


def _report_rows(frame, report_lease):
    reserve_report_frames(report_lease, (frame,))
    return [
        {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in row.items()
            if not pd.isna(value)
        }
        for row in frame.to_dict("records")
    ]


def produce_crypto_report(
    *,
    layout,
    record,
    panel_id,
    read_model_artifact,
    report_lease,
    expected_symbols,
    read_reconstruction,
):
    protocol = record.payload()["protocol"]
    issues, references = [], {}
    for result in record.target_results:
        if result.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}:
            issues.append(_issue(result.target_id, result.status.value, result.reason))
            continue
        try:
            _validate_reference(layout, result)
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
                    _instance_root(layout, result)
                )
                if target == panel_id:
                    recording_counts = {
                        key: manifest["producer_diagnostics"][key]
                        for key in ("raw_record_count", "recorder_file_count")
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
                trial_layout=layout,
            )
            report_lease.grow(
                report_payload_working_set({"fold_models": list(artifact.models)})
            )
            models = {
                "status": "DEFINED",
                "identity": reference.instance_identity,
                "fold_models": [
                    {
                        key: entry[key]
                        for key in (
                            "fold_id",
                            "model_id",
                            "training_window",
                            "validation_window",
                            "prediction_window",
                            "training_asset_ids",
                            "validation_asset_ids",
                            "evaluation_asset_ids",
                            "model_metadata",
                            "trial_semantic_identity",
                        )
                    }
                    for entry in artifact.models
                ],
                "validation_prediction_role": (
                    "train_model_predictions_of_mature_validation_rows"
                ),
            }
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(_issue(model_target, "FAILED", "model_artifact_invalid"))
            models = _unavailable(issues, model_target)

    prediction_target = f"{record.scenario_id}_predictions"
    predictions = _unavailable(issues, prediction_target)
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
            tables = {}

            def observe(_manifest, members, tables=tables):
                for path, identity in members:
                    if path.name not in {
                        "ic_summary.parquet",
                        "group_spread_summary.parquet",
                    }:
                        continue
                    rows = []
                    with closing_iterator(
                        iter_declared_parquet_member_batches(
                            path,
                            expected_identity=identity,
                            target_chunk_bytes=256 * 1024,
                        )
                    ) as chunks:
                        for frame in chunks:
                            rows.extend(_report_rows(frame, report_lease))
                    tables[path.stem] = rows

            manifest, _members = validate_output_bundle(
                _instance_root(layout, result) / "outputs" / result.target_id,
                expected_output_id=result.target_id,
                expected_run_id=f"run_{result.target_id}",
                observer=observe,
            )
            undefined = any(
                value
                for rows in tables.values()
                for row in rows
                for key, value in row.items()
                if key.endswith("_reason")
            )
            evaluations.append(
                {
                    "evaluation_id": result.target_id,
                    "identity": result.instance_identity,
                    "status": "UNDEFINED" if undefined else "DEFINED",
                    "tables": tables,
                    "metadata": manifest["metadata"],
                }
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

    calibration_target = f"{record.scenario_id}_maker_calibration"
    calibration = _unavailable(issues, calibration_target)
    if calibration_target in references:
        reference = references[calibration_target]
        try:
            artifact = read_statistics_artifact(
                _instance_root(layout, reference),
                expected_identity=reference.instance_identity,
            )
            with artifact.table("crypto_maker_calibration.parquet") as frame:
                calibration = {
                    "status": "DEFINED"
                    if frame.status.eq("DEFINED").all()
                    else "UNDEFINED",
                    "identity": reference.instance_identity,
                    "summary": _report_rows(frame, report_lease),
                    "method": "paired_fill_quantity_and_forecast_error_by_side",
                    "quote_order_type": "post_only_limit",
                    "outcome_clock": "original_decision_plus_500ms",
                    "sample_tables": [
                        "crypto_maker_samples.parquet",
                        "crypto_maker_fills.parquet",
                    ],
                }
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(
                    calibration_target, "FAILED", "maker_calibration_artifact_invalid"
                )
            )
            calibration = _unavailable(issues, calibration_target)

    portfolios = []
    for candidate in protocol["portfolio_candidates"]:
        target = f"{candidate['candidate_id']}_ledger"
        if target not in references:
            portfolios.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    **_unavailable(issues, target),
                }
            )
            continue
        reference = references[target]
        try:
            ledger = read_portfolio_ledger_artifact(
                _instance_root(layout, reference),
                expected_identity=reference.instance_identity,
            )
            portfolios.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": "DEFINED",
                    "identity": reference.instance_identity,
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
            )
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(_issue(target, "FAILED", "portfolio_ledger_invalid"))
            portfolios.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    **_unavailable(issues, target),
                }
            )

    statistics_target = f"{record.scenario_id}_portfolio_statistics"
    statistics = _unavailable(issues, statistics_target)
    if statistics_target in references:
        reference = references[statistics_target]
        try:
            artifact = read_statistics_artifact(
                _instance_root(layout, reference),
                expected_identity=reference.instance_identity,
            )
            tables = {}
            for name in (
                "crypto_portfolio_intervals",
                "crypto_portfolio_daily",
                "crypto_portfolio_statistics",
                "crypto_execution_diagnostics",
                "crypto_cost_sensitivity",
            ):
                with artifact.table(f"{name}.parquet") as frame:
                    tables[name] = _report_rows(frame, report_lease)
            statistics = {
                "status": "DEFINED",
                "identity": reference.instance_identity,
                "tables": tables,
            }
        except (OSError, KeyError, TypeError, ValueError):
            issues.append(
                _issue(statistics_target, "FAILED", "portfolio_statistics_invalid")
            )
            statistics = _unavailable(issues, statistics_target)
    return build_crypto_research_report(
        record=record,
        evidence={
            "data": data,
            "models": models,
            "predictions": predictions,
            "evaluations": evaluations,
            "calibration": calibration,
            "portfolios": portfolios,
            "statistics": statistics,
            "issues": issues,
        },
    )


def summarize_crypto_execution(artifact):
    result = None
    with closing_iterator(artifact.iter_chunks()) as chunks:
        for batch in chunks:
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
                    if current != (artifact.capital.initial_capital, 0.0, 0.0, 0.0, 0):
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
    if marker_index != len(expected):
        raise ValueError("crypto_ledger_window_marker_missing")
    if previous != (nav, fees, funding, turnover, fills):
        raise ValueError("crypto_ledger_after_window_economics")
    return intervals
