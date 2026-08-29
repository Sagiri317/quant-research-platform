"""Crypto HFT execution evidence owner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_research_platform.artifacts.evidence import write_json_evidence
from quant_research_platform.contracts.json_codec import (
    replace_nonfinite_json_numbers,
)


def write_crypto_hft_execution_artifact(
    path: Path,
    *,
    taker_model_id: str,
    maker_policy_id: str,
    maker_prediction_used: bool,
    maker_model_id: str,
    source_identity: Mapping[str, str],
    panel_id: str,
    factor_id: str,
    label_id: str,
    research_dataset_id: str,
    evaluation_id: str,
    taker: Mapping[str, Any],
    taker_models: Mapping[str, Mapping[str, Any]],
    maker: Mapping[str, Any],
    maker_baseline: Mapping[str, Any],
    latency_sweep: list[dict[str, Any]],
) -> dict[str, Any]:
    lineage = {
        "source_artifact_identity": dict(source_identity),
        "panel": panel_id,
        "factor": factor_id,
        "label": label_id,
        "research_dataset": research_dataset_id,
        "evaluation": evaluation_id,
    }
    payload = replace_nonfinite_json_numbers(
        {
            "schema_id": "qrp-crypto-hft-execution-diagnostics",
            "taker_model_id": taker_model_id,
            "maker_policy_id": maker_policy_id,
            "maker_prediction_used": maker_prediction_used,
            "maker_model_id": maker_model_id,
            "lineage": lineage,
            "taker": dict(taker),
            "taker_models": {
                model_id: dict(result) for model_id, result in taker_models.items()
            },
            "maker": dict(maker),
            "maker_baseline": dict(maker_baseline),
            "latency_sweep": latency_sweep,
        }
    )
    write_json_evidence(path, payload)
    return payload
