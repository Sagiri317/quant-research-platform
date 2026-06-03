"""Evaluation helpers for QRP 0.1."""

from quant_research_platform.research.evaluations.engine import (
    EvaluationResult,
    evaluate_factor_frame,
)
from quant_research_platform.research.evaluations.parity import (
    DeltaClassification,
    ParityDeltaResult,
    classify_parity_delta,
    ensure_no_unexplained_regression,
)
from quant_research_platform.research.evaluations.runner import (
    run_evaluation_from_research_dataset,
    write_evaluation_result,
)

__all__ = [
    "DeltaClassification",
    "EvaluationResult",
    "ParityDeltaResult",
    "classify_parity_delta",
    "ensure_no_unexplained_regression",
    "evaluate_factor_frame",
    "run_evaluation_from_research_dataset",
    "write_evaluation_result",
]
