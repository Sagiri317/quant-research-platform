"""Source adapter registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.ids import SourceId


@dataclass(frozen=True)
class SourceDefinition:
    source_id: SourceId
    display_name: str
    interim_spec_ids: tuple[str, ...]
    public_identifier: bool = True
    definition_version: str = "0.1"


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

    @classmethod
    def bootstrap_default(cls) -> SourceRegistry:
        registry = cls()
        registry.register(
            SourceDefinition(
                source_id=SourceId("csmar"),
                display_name="CSMAR source adapter",
                interim_spec_ids=("csmar_daily_market_interim",),
            )
        )
        return registry


__all__ = ["SourceDefinition", "SourceRegistry"]
