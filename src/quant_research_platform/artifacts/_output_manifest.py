"""Output manifest payload builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict

import pandas as pd

from quant_research_platform.artifacts._portfolio_output_validation import (
    PORTFOLIO_ARTIFACTS,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.policy.share_safety import sanitize_shared_object


def stage_profile_output_manifest_payload(
    *,
    spec: OutputSpec,
) -> dict[str, object]:
    return output_manifest_payload(
        output_spec=spec,
        metadata={
            "market_id": str(spec.market_id),
            "run_type": spec.run_type,
            "scope": spec.scope,
        },
        files=("stage_profile.csv", "stage_profile.json", "stage_profile.md"),
    )


def output_manifest_payload(
    *,
    output_spec: OutputSpec,
    metadata: Mapping[str, object],
    files: Iterable[str],
) -> dict[str, object]:
    return {
        "schema_id": "qrp-output-manifest",
        "output_id": output_spec.output_id,
        "output_definition_id": output_spec.output_definition_id,
        "run_id": output_spec.run_id,
        "metadata": _without_json_version_fields(dict(metadata)),
        "files": list(files),
    }


def _without_json_version_fields(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_json_version_fields(item)
            for key, item in value.items()
            if key not in {"schema_version", "contract_version"}
        }
    if isinstance(value, list | tuple):
        return [_without_json_version_fields(item) for item in value]
    return value


def prediction_output_metadata(
    *,
    spec: PredictionSpec,
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "prediction_id": spec.prediction_id,
        "model_id": spec.model_id,
        "research_dataset_id": spec.research_dataset_id,
        "prediction_mode": spec.prediction_mode,
        "is_oos": spec.is_oos,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "test_end": spec.test_end,
        "is_final_oos_window": spec.is_final_oos_window,
        "prediction_column": spec.prediction_column,
        "primary_key": list(spec.primary_key),
        "sample_filter_mode": spec.sample_filter_mode,
        "model_metadata": model_metadata,
        "prediction_metadata": prediction_metadata,
    }


def evaluation_output_metadata(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "run_type": spec.run_type,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "signals": [asdict(signal) for signal in spec.signals],
        "label_columns": list(spec.label_columns),
        "filter_mode": spec.filter_mode,
        "evaluation_diagnostics": sanitize_shared_object(
            asdict(evaluation_diagnostics)
        ),
    }


def portfolio_output_metadata(
    *,
    spec: PortfolioSpec,
    artifacts: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    return {
        "portfolio_id": spec.portfolio_id,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "run_type": spec.run_type,
        "signals": [asdict(signal) for signal in spec.signals],
        "return_source_column": spec.return_source_column,
        "rebalance_frequency": spec.rebalance_frequency,
        "execution_lag_steps": spec.execution_lag_steps,
        "weight_config": asdict(spec.weight_config),
        "tradability_config": asdict(spec.tradability_config),
        "cost_config": asdict(spec.cost_config),
        "diagnostics_config": asdict(spec.diagnostics_config),
        "row_counts": {
            artifact_name: int(len(frame)) for artifact_name, frame in artifacts.items()
        },
    }


def portfolio_output_metadata_from_row_counts(
    *,
    spec: PortfolioSpec,
    row_counts: Mapping[str, int],
) -> dict[str, object]:
    payload = portfolio_output_metadata(
        spec=spec,
        artifacts={
            artifact_name: pd.DataFrame(index=range(int(row_counts[artifact_name])))
            for artifact_name in PORTFOLIO_ARTIFACTS
        },
    )
    payload["row_counts"] = {
        artifact_name: int(row_counts[artifact_name])
        for artifact_name in PORTFOLIO_ARTIFACTS
    }
    return payload
