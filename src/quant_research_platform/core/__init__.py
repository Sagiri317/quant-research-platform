"""Core contracts and evaluation orchestration."""

from typing import TYPE_CHECKING

from quant_research_platform.core.contracts import (
    PORTFOLIO_MODES,
    CostConfig,
    EvaluationSpec,
    ExecutionConfig,
    NeutralizationConfig,
    TensorBatchConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.panel import PanelSpec, validate_panel
from quant_research_platform.core.profiling import (
    ProfileConfig,
    StageProfiler,
    StageRecord,
)
from quant_research_platform.core.results import EvaluationSetResult

if TYPE_CHECKING:
    from quant_research_platform.core.evaluation import run_evaluation

__all__ = [
    "CostConfig",
    "EvaluationSetResult",
    "EvaluationSpec",
    "ExecutionConfig",
    "NeutralizationConfig",
    "PORTFOLIO_MODES",
    "PanelSpec",
    "ProfileConfig",
    "StageProfiler",
    "StageRecord",
    "TensorBatchConfig",
    "TradabilityConfig",
    "WeightConfig",
    "run_evaluation",
    "validate_panel",
]


def __getattr__(name: str) -> object:
    if name == "run_evaluation":
        from quant_research_platform.core.evaluation import run_evaluation

        return run_evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
