"""定义 evaluation runs 的内存结果合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvaluationSetResult:
    """保存一个 evaluation set 的内存指标和 artifact paths。"""

    summary: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, pd.DataFrame]
    artifacts: Any
    combinations: pd.DataFrame
    combinations_path: Any
