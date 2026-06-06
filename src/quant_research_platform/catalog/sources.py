"""Internal source identity registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from quant_research_platform.contracts.ids import SourceId


class SourceVisibility(StrEnum):
    INTERNAL_PRIVATE = "internal_private"


@dataclass(frozen=True)
class SourceDefinition:
    source_id: SourceId
    provider_id: str
    display_name: str
    implementation_package: str
    visibility: SourceVisibility = SourceVisibility.INTERNAL_PRIVATE
    public_identifier: bool = False
    definition_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.visibility is not SourceVisibility.INTERNAL_PRIVATE:
            raise ValueError("source_definition_must_be_internal")
        if self.public_identifier:
            raise ValueError("source_definition_must_not_be_public")


@dataclass
class SourceRegistry:
    _items: dict[str, SourceDefinition] = field(default_factory=dict)

    def register(self, definition: SourceDefinition) -> None:
        key = str(definition.source_id)
        if key in self._items:
            raise ValueError(f"duplicate_source_definition: {key}")
        self._items[key] = definition

    def get(self, source_id: str) -> SourceDefinition:
        try:
            return self._items[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown_source_id: {source_id}") from exc

    def list(self) -> tuple[SourceDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["SourceDefinition", "SourceRegistry", "SourceVisibility"]
