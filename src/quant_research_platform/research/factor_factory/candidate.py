"""定义 Factor Factory 候选对象和评估记录。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CandidateStatus = Literal["pending", "success", "failed"]
PRIVATE_PATH_PATTERN = re.compile(
    r"([A-Za-z]:)?/?(?:[^,\s:;]+/)*(?:raw_private|interim_private|processed_private|private_real)(?:/[^,\s:;]*)*",
    flags=re.IGNORECASE,
)
LOCAL_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:/|//)[^,\s;)\]]+"
)


@dataclass(frozen=True)
class FactorCandidate:
    """记录一个受控 factor candidate 的来源和依赖。"""

    candidate_id: str
    name: str
    version: str
    expression: str
    dependencies: tuple[str, ...]
    output_column: str | None
    sample_start: str
    sample_end: str
    label_columns: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    capabilities: tuple[str, ...]
    parameters: dict[str, object] = field(default_factory=dict)
    availability_rule: str = "trade_date_close"
    time_semantics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvaluationRecord:
    """记录一个 candidate 的评估状态或失败原因。"""

    candidate_id: str
    version: str
    expression: str
    dependencies: tuple[str, ...]
    output_column: str | None
    sample_start: str
    sample_end: str
    label_columns: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    capabilities: tuple[str, ...]
    parameters: dict[str, object]
    availability_rule: str
    time_semantics: dict[str, object]
    status: CandidateStatus
    run_dir: str | None = None
    failure_reason: str | None = None
    failure_code: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        """转换为 CSV/JSON 友好的扁平记录。"""

        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "expression": self.expression,
            "dependencies": ",".join(self.dependencies),
            "output_column": self.output_column or "",
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "label_columns": ",".join(self.label_columns),
            "inputs": ",".join(self.inputs),
            "outputs": ",".join(self.outputs),
            "capabilities": ",".join(self.capabilities),
            "parameters": _json_cell(self.parameters),
            "availability_rule": self.availability_rule,
            "time_semantics": _json_cell(self.time_semantics),
            "status": self.status,
            "run_ref": _run_ref(self.run_dir),
            "failure_code": self.failure_code or "",
            "failure_reason": _safe_failure_reason(self.failure_reason),
        }


def _json_cell(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _run_ref(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).name


def _safe_failure_reason(reason: str | None) -> str:
    if not reason:
        return ""
    sanitized = reason.replace("\\", "/")
    sanitized = PRIVATE_PATH_PATTERN.sub("[private-path]", sanitized)
    sanitized = LOCAL_ABSOLUTE_PATH_PATTERN.sub("[local-path]", sanitized)
    return sanitized


__all__ = ["CandidateEvaluationRecord", "CandidateStatus", "FactorCandidate"]
