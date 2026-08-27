"""Output contracts for QRP 0.1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.run_id import validate_run_id

_OUTPUT_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def validate_output_identifier(value: str) -> str:
    if not isinstance(value, str) or _OUTPUT_IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("invalid_output_id")
    if value.casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("invalid_output_id")
    return value


class OutputScope(StrEnum):
    LOCAL_ONLY = "LOCAL_ONLY"
    REVIEW_SAFE = "REVIEW_SAFE"


@dataclass(frozen=True)
class OutputDefinition:
    output_definition_id: str
    display_name: str
    run_type: str
    scope: OutputScope

    def __post_init__(self) -> None:
        if type(self.display_name) is not str or not self.display_name:
            raise ValueError("output_display_name_required")
        if type(self.run_type) is not str or not self.run_type:
            raise ValueError("output_run_type_required")
        validate_output_identifier(self.output_definition_id)
        try:
            object.__setattr__(self, "scope", OutputScope(self.scope))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_output_scope") from exc


@dataclass(frozen=True)
class OutputSpec:
    output_id: str
    output_definition_id: str
    run_id: str
    market_id: MarketId
    run_type: str
    scope: OutputScope

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "scope", OutputScope(self.scope))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_output_scope") from exc
        validate_output_identifier(self.output_id)
        validate_output_identifier(self.output_definition_id)
        if self.output_id == self.output_definition_id:
            raise ValueError("output_id_must_identify_instance")
        validate_run_id(self.run_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        if type(self.run_type) is not str or not self.run_type:
            raise ValueError("output_run_type_required")
