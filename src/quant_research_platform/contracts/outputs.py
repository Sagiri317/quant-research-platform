"""Output contracts for QRP."""

from __future__ import annotations

import re
from dataclasses import dataclass

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


@dataclass(frozen=True)
class OutputSpec:
    output_id: str
    output_definition_id: str
    run_id: str
    market_id: MarketId

    def __post_init__(self) -> None:
        validate_output_identifier(self.output_id)
        if self.output_definition_id not in {
            "prediction_outputs",
            "evaluation_outputs",
        }:
            raise ValueError("invalid_output_definition_id")
        if self.output_id == self.output_definition_id:
            raise ValueError("output_id_must_identify_instance")
        validate_run_id(self.run_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
