"""Save and load explicitly selected trained models, without discovery or refitting."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.models import validate_model_metadata_core
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
    validate_model_state,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.experiments import read_trial_ledger


def _validate_facts(facts, metadata):
    from quant_research_platform.contracts.models import _fit_windows, _model_spec

    spec = _model_spec(facts["model"])
    from quant_research_platform.contracts.models import _resolve_candidate_parameters

    expected_parameters = _resolve_candidate_parameters(
        spec.model_algorithm,
        facts["candidate_parameters"],
        metadata["rows_after_input_label_dropna"],
    )
    if canonical_identity_bytes(expected_parameters) != canonical_identity_bytes(
        metadata["hyperparameters"]
    ):
        raise ValueError("model_sample_scaled_parameters_mismatch")
    if not metadata["training_asset_ids"]:
        raise ValueError("native_model_training_asset_ids_required")
    if facts.get("asset_ids") is not None and not set(
        metadata["training_asset_ids"]
    ) <= set(facts["asset_ids"]):
        raise ValueError("model_training_asset_scope_mismatch")
    for key in (
        "model_algorithm",
        "target_column",
        "sample_filter_mode",
        "random_seed",
    ):
        if getattr(spec, key) != metadata[key]:
            raise ValueError("model_facts_metadata_mismatch")
    if list(spec.input_columns) != metadata["input_columns"]:
        raise ValueError("model_facts_metadata_mismatch")

    cutoff = strict_timestamp_series(
        pd.Series([facts["fit_cutoff"]]), error="invalid_fit_cutoff"
    ).iloc[0]
    windows = _fit_windows(facts["fit_windows"], cutoff)
    bounds = strict_timestamp_series(
        pd.Series([metadata["train_start"], metadata["train_end"]]),
        error="invalid_model_training_window",
    )
    if bounds.iloc[0] > bounds.iloc[1] or bounds.iloc[1] > cutoff:
        raise ValueError("model_training_window_invalid")
    from quant_research_platform.contracts.time import time_window_timestamp_interval

    def contains(window, timestamp):
        start, end, closed = time_window_timestamp_interval(window)
        return pd.Timestamp(start) <= timestamp and (
            timestamp <= pd.Timestamp(end) if closed else timestamp < pd.Timestamp(end)
        )

    if not all(any(contains(window, bound) for window in windows) for bound in bounds):
        raise ValueError("model_training_window_outside_declared_fit")
    if facts.get("test_window") is not None:
        if not facts.get("selection") or cutoff >= pd.Timestamp(
            facts["test_window"]["start"]
        ):
            raise ValueError("model_test_freeze_missing")
    if facts.get("label_unit") in (None, ""):
        raise ValueError("model_label_unit_required")


def save_model(path: Path, result: ModelFitResult, *, facts: dict, trial: dict) -> Path:
    """Publish one already-fitted predictor and its scientific facts atomically."""
    metadata = validate_model_metadata_core(
        result.metadata,
        rank_calibration="label_calibration"
        in result.metadata.get("rank_parameters", {}),
    )
    validate_model_state(result.prediction_model, metadata)
    _validate_facts(facts, metadata)
    if (
        trial.get("status") != "completed"
        or trial.get("result", {}).get("model_metadata") != metadata
    ):
        raise ValueError("completed_model_trial_required")
    path = Path(path)

    def write(staging):
        with (staging / "predictor.pkl").open("xb") as handle:
            pickle.dump(
                result.prediction_model, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        atomic_write_json(
            staging / "model.json",
            {"model_metadata": metadata, "facts": facts, "trial_id": trial["trial_id"]},
        )

    atomic_write_directory(path, write)
    return path


def load_model(path: Path) -> tuple[ModelFitResult, dict]:
    """Load a trusted local model once; never use untrusted pickle files."""
    path = Path(path)
    payload = parse_qrp_json_object((path / "model.json").read_bytes())
    if set(payload) != {"model_metadata", "facts", "trial_id"}:
        raise ValueError("model_manifest_invalid")
    metadata = payload["model_metadata"]
    metadata = validate_model_metadata_core(
        metadata,
        rank_calibration="label_calibration" in metadata.get("rank_parameters", {}),
    )
    _validate_facts(payload["facts"], metadata)
    records = read_trial_ledger(Path(payload["facts"]["trial"]))
    terminal = records[-1]
    if (
        terminal.get("status") != "completed"
        or terminal.get("trial_id") != payload["trial_id"]
        or terminal.get("result", {}).get("model_metadata") != metadata
    ):
        raise ValueError("completed_model_trial_required")
    for key, value in payload["facts"].items():
        if key != "trial" and terminal.get(key) != value:
            raise ValueError("model_trial_facts_mismatch")
    with (path / "predictor.pkl").open("rb") as handle:
        predictor = pickle.load(handle)
        if handle.read(1):
            raise ValueError("model_member_trailing_data")
    validate_model_state(predictor, metadata)
    return ModelFitResult(metadata=metadata, prediction_model=predictor), payload[
        "facts"
    ]
