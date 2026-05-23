"""定义包级异常类型。"""


class QuantResearchPlatformError(Exception):
    """表示 quant-research-platform 的基础异常。"""


class PanelValidationError(QuantResearchPlatformError):
    """在 panel 违反 evaluation contract 时抛出。"""


class ArtifactValidationError(QuantResearchPlatformError):
    """在 run artifacts 违反 artifact contract 时抛出。"""


class UnsupportedEngineError(QuantResearchPlatformError):
    """在 evaluation kernel 不受支持时抛出。"""


class UnsupportedBreakdownError(QuantResearchPlatformError):
    """请求的 breakdown 不受支持时抛出。"""


class RunExecutionError(QuantResearchPlatformError):
    """在 run lifecycle step 失败时抛出。"""
