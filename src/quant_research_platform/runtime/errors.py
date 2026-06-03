"""Runtime error classes for the QRP 0.1 surface."""


class QuantResearchPlatformError(Exception):
    """Base exception for quant-research-platform."""


class PanelValidationError(QuantResearchPlatformError):
    """Raised when a panel violates the research contract."""


class StorageBoundaryError(QuantResearchPlatformError):
    """Raised when a path violates the private storage boundary."""


class UnsupportedKernelError(QuantResearchPlatformError):
    """Raised when a compute kernel is not supported."""


class WorkflowExecutionError(QuantResearchPlatformError):
    """Raised when a workflow step fails."""


__all__ = [
    "PanelValidationError",
    "QuantResearchPlatformError",
    "StorageBoundaryError",
    "UnsupportedKernelError",
    "WorkflowExecutionError",
]
