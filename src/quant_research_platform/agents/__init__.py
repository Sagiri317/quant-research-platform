"""Agent proposal boundary for QRP 0.1.

This package does not call external LLM APIs and does not write private data.
"""

from quant_research_platform.agents.proposals import (
    FactorProposal,
    LabelProposal,
    ResearchDatasetProposal,
    WorkflowProposal,
)

__all__ = [
    "FactorProposal",
    "LabelProposal",
    "ResearchDatasetProposal",
    "WorkflowProposal",
]
