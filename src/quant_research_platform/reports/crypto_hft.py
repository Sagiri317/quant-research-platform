"""Crypto HFT report content."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quant_research_platform.contracts.json_codec import (
    replace_nonfinite_json_numbers,
)

_MODEL_REPORT_KEYS = (
    "model_algorithm",
    "input_columns",
    "label_column",
    "random_seed",
    "max_train_rows",
    "purge_gap_steps",
    "rows_input",
    "rows_after_filter_mode",
    "rows_after_input_label_dropna",
    "train_start",
    "train_end",
    "resolved_hyperparameters",
    "dependency_status",
)


def build_crypto_hft_report(
    *,
    run_id: str,
    market_data_status: str,
    research_status: str,
    evaluation_status: str,
    evaluation_valid_ic_count: int,
    reconstructions: Mapping[str, Mapping[str, Any]],
    raw_record_count: int,
    row_counts: Mapping[str, int],
    algorithms: tuple[str, ...],
    prediction_outputs: Mapping[str, str],
    prediction_evaluations: Mapping[str, Mapping[str, Any]],
    model_metadata: Mapping[str, Mapping[str, Any]],
    evaluation_id: str,
    evaluation_dataset_id: str,
    evaluation_axis: str,
    evaluation_primary_key: tuple[str, ...],
    evaluation_requested_window: Mapping[str, str],
    evaluation_observed_window: Mapping[str, str | None],
    evaluation_path: str,
    execution_path: str,
    execution_payload: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction_counts = {
        key: sum(int(item[key]) for item in reconstructions.values())
        for key in next(iter(reconstructions.values()))
        if key != "coverage_segments"
    }
    return replace_nonfinite_json_numbers(
        {
            "schema_id": "qrp-crypto-hft-report",
            "run_id": run_id,
            "status": "PASS"
            if research_status == evaluation_status == "PASS"
            else "PARTIAL",
            "market_data_status": market_data_status,
            "research_status": research_status,
            "evaluation_status": evaluation_status,
            "model_status": research_status,
            "execution_status": research_status,
            "lineage": dict(execution_payload["lineage"]),
            "data_quality": {
                **reconstruction_counts,
                "reconstruction_by_symbol": {
                    symbol: dict(item) for symbol, item in reconstructions.items()
                },
                "raw_record_count": raw_record_count,
                "panel_rows": row_counts["panel"],
                "factor_rows": row_counts["factor"],
                "label_rows": row_counts["label"],
            },
            "evaluation": {
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "research_dataset_id": evaluation_dataset_id,
                "evaluation_axis": evaluation_axis,
                "primary_key": list(evaluation_primary_key),
                "requested_window": dict(evaluation_requested_window),
                "observed_window": dict(evaluation_observed_window),
                "path": evaluation_path,
                "status": evaluation_status,
                "valid_ic_count": evaluation_valid_ic_count,
            },
            "models": {
                "algorithms": list(algorithms),
                "prediction_outputs": dict(prediction_outputs),
                "oos_evaluation": {
                    model_id: dict(evaluation)
                    for model_id, evaluation in prediction_evaluations.items()
                },
                "metadata": {
                    model_id: {key: metadata[key] for key in _MODEL_REPORT_KEYS}
                    for model_id, metadata in model_metadata.items()
                },
            },
            "execution": {
                "taker_model_id": execution_payload["taker_model_id"],
                "maker_policy_id": execution_payload["maker_policy_id"],
                "maker_prediction_used": execution_payload["maker_prediction_used"],
                "maker_model_id": execution_payload["maker_model_id"],
                "path": execution_path,
                "taker": execution_payload["taker"],
                "taker_models": execution_payload["taker_models"],
                "maker": execution_payload["maker"],
                "maker_baseline": execution_payload["maker_baseline"],
                "latency_sweep": execution_payload["latency_sweep"],
                "cost_sensitivity": execution_payload["cost_sensitivity"],
                "multi_asset_execution": execution_payload["multi_asset_execution"],
            },
            "resource": dict(resource),
        }
    )
