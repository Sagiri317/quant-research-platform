"""Package-wide exception types."""


class AlphaResearchError(Exception):
    """Base exception for quant-research-platform failures."""


class PanelValidationError(AlphaResearchError):
    """Raised when a panel violates the evaluation contract."""


class ArtifactValidationError(AlphaResearchError):
    """Raised when run artifacts violate the artifact contract."""


class UnsupportedEngineError(AlphaResearchError):
    """Raised when an evaluation engine is not supported."""


class UnsupportedBreakdownError(AlphaResearchError):
    """Raised when a requested breakdown is not supported."""


class RunExecutionError(AlphaResearchError):
    """Raised when a run lifecycle step fails."""

