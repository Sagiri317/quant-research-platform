"""In-memory result contracts for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EvaluationSetResult:
    """In-memory metrics and artifact paths for an evaluation set."""

    summary: dict[str, Any]
    metrics: dict[str, pd.DataFrame]
    artifacts: Any
    combinations: pd.DataFrame
    combinations_path: Any
