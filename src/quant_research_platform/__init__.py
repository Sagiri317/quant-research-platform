"""Local multi-market quantitative research platform."""

from quant_research_platform._version import __version__
from quant_research_platform.core.contracts import EvaluationSpec, WeightConfig
from quant_research_platform.core.evaluation import run_evaluation
from quant_research_platform.io.parquet_panel import make_parquet_evaluation_spec

__all__ = [
    "EvaluationSpec",
    "WeightConfig",
    "__version__",
    "make_parquet_evaluation_spec",
    "run_evaluation",
]
