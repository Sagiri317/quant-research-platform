"""内存中的 evaluation payload contract。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvaluationRunPayload:
    """记录 artifact 持久化前计算出的 evaluation-set 内容。"""

    summary: dict[str, Any]
    config: dict[str, Any]
    run_context: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    report: str
