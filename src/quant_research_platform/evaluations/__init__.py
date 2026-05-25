"""提供 evaluation 计算边界的公开入口。"""

from quant_research_platform.evaluations.evaluation_payload import EvaluationRunPayload
from quant_research_platform.evaluations.evaluation_result import EvaluationSetResult
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    PORTFOLIO_MODES,
    CostConfig,
    EvaluationSpec,
    ExecutionConfig,
    NeutralizationConfig,
    TensorBatchConfig,
    TimeSemanticsConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.evaluations.panel_validation import (
    PanelSpec,
    validate_evaluation_set_return_source_column,
    validate_evaluation_spec_columns,
    validate_panel,
)
from quant_research_platform.runtime.profiling import (
    ProfileConfig,
    StageProfiler,
    StageRecord,
)

__all__ = [
    "CostConfig",
    "EvaluationRunPayload",
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
    "TimeSemanticsConfig",
    "TradabilityConfig",
    "WeightConfig",
    "run_evaluation",
    "validate_evaluation_set_return_source_column",
    "validate_evaluation_spec_columns",
    "validate_panel",
]
