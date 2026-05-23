"""本地多市场量化研究平台。"""

from quant_research_platform._version import __version__
from quant_research_platform.datasets.parquet_panel import make_parquet_evaluation_spec
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    CostConfig,
    EvaluationSpec,
    WeightConfig,
)

__all__ = [
    "EvaluationSpec",
    "CostConfig",
    "WeightConfig",
    "__version__",
    "make_parquet_evaluation_spec",
    "run_evaluation",
]
