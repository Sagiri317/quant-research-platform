"""Save and load explicitly selected trained models, without discovery or refitting."""

from __future__ import annotations

import pickle
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
)
from quant_research_platform.contracts.models import (
    validate_model_facts,
    validate_model_metadata_core,
)
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
    validate_model_state,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)


def save_model(path: Path, result: ModelFitResult, *, facts: dict, trial: dict) -> Path:
    """Publish one already-fitted predictor and its scientific facts atomically."""
    metadata = validate_model_metadata_core(
        result.metadata,
        rank_calibration="label_calibration"
        in result.metadata.get("rank_parameters", {}),
    )
    validate_model_state(result.prediction_model, metadata)
    validate_model_facts(facts, metadata)
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
    validate_model_facts(payload["facts"], metadata)
    with (path / "predictor.pkl").open("rb") as handle:
        predictor = pickle.load(handle)
        if handle.read(1):
            raise ValueError("model_member_trailing_data")
    validate_model_state(predictor, metadata)
    return ModelFitResult(metadata=metadata, prediction_model=predictor), payload[
        "facts"
    ]
