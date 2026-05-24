"""第一版 Factor Factory 入口。"""

from quant_research_platform.research.factor_factory.batch_runner import (
    evaluate_existing_factor_candidates,
)
from quant_research_platform.research.factor_factory.candidate import (
    CandidateEvaluationRecord,
    FactorCandidate,
)
from quant_research_platform.research.factor_factory.factor_registry import (
    FactorTemplateRegistry,
    RegisteredFactorTemplate,
)

__all__ = [
    "CandidateEvaluationRecord",
    "FactorCandidate",
    "FactorTemplateRegistry",
    "RegisteredFactorTemplate",
    "evaluate_existing_factor_candidates",
]
