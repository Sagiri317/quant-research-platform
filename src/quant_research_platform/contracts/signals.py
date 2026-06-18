"""Signal contracts shared by evaluation and portfolio research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_research_platform.contracts.filtering import reject_filter_input_columns

SignalSourceKind = Literal["factor", "feature", "prediction"]
SignalDirection = Literal["higher_is_better", "lower_is_better"]
SignalTransform = Literal["identity", "rank", "zscore"]


@dataclass(frozen=True, kw_only=True)
class SignalSpec:
    signal_id: str
    source_kind: SignalSourceKind
    source_column: str
    direction: SignalDirection = "higher_is_better"
    transform: SignalTransform = "identity"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_signal_contract_version")
        if not isinstance(self.signal_id, str) or not self.signal_id:
            raise ValueError("signal_id_required")
        if self.source_kind not in {"factor", "feature", "prediction"}:
            raise ValueError("unsupported_signal_source_kind")
        if not isinstance(self.source_column, str) or not self.source_column:
            raise ValueError("signal_source_column_required")
        if self.direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("unsupported_signal_direction")
        if self.transform not in {"identity", "rank", "zscore"}:
            raise ValueError("unsupported_signal_transform")
        reject_filter_input_columns((self.signal_id, self.source_column))


__all__ = [
    "SignalDirection",
    "SignalSourceKind",
    "SignalSpec",
    "SignalTransform",
]
