"""Internal source identity registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from quant_research_platform.catalog._immutable import ImmutableCatalog
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


class SourceRegistry(ImmutableCatalog[SourceDefinition]):
    def __init__(self, items: Iterable[SourceDefinition] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for definition in values:
            key = str(definition.source_id)
            if key in seen:
                raise ValueError(f"duplicate_source_definition: {key}")
            seen.add(key)
        super().__init__(sorted(values, key=lambda item: str(item.source_id)))

    def get(self, source_id: str) -> SourceDefinition:
        try:
            return next(
                definition
                for definition in self._items
                if str(definition.source_id) == source_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown_source_id: {source_id}") from exc
