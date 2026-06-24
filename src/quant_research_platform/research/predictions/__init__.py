"""Prediction output builders."""

from quant_research_platform.research.predictions.outputs import (
    PredictionResult,
    build_prediction_result_from_frames,
    validate_prediction_outputs,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
    build_prediction_results_from_research_dataset,
)

__all__ = [
    "PredictionResult",
    "build_prediction_result_from_frames",
    "build_prediction_result_from_research_dataset",
    "build_prediction_results_from_research_dataset",
    "validate_prediction_outputs",
]
