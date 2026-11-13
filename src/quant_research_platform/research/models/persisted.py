"""Save and load explicitly selected trained models, without discovery or refitting."""

from __future__ import annotations

import pickle
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
)
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)


def save_model(
    path: Path, result: ModelFitResult, *, facts: dict, trial_id: str
) -> Path:
    """Publish one already-fitted predictor and its scientific facts atomically."""
    metadata = result.metadata
    path = Path(path)

    def write(staging):
        with (staging / "predictor.pkl").open("xb") as handle:
            pickle.dump(
                result.prediction_model, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        atomic_write_json(
            staging / "model.json",
            {"model_metadata": metadata, "facts": facts, "trial_id": trial_id},
        )

    atomic_write_directory(path, write)
    return path


def load_model(path: Path) -> tuple[ModelFitResult, dict]:
    """Load a trusted local model once; never use untrusted pickle files."""
    path = Path(path)
    payload = parse_qrp_json_object((path / "model.json").read_bytes())
    metadata = payload["model_metadata"]
    with (path / "predictor.pkl").open("rb") as handle:
        predictor = pickle.load(handle)
    return ModelFitResult(metadata=metadata, prediction_model=predictor), payload[
        "facts"
    ]
