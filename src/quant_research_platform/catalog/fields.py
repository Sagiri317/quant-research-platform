"""Platform canonical field catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.columns import ColumnRole, FieldSpec, LogicalType


@dataclass(frozen=True)
class CanonicalFieldNames:
    market_id: str
    asset_id: str
    timestamp: str
    session_date: str
    available_at: str
    source_id: str
    source_asset_id: str


@dataclass
class FieldCatalog:
    _items: dict[str, FieldSpec] = field(default_factory=dict)

    def register(self, spec: FieldSpec) -> None:
        if spec.name in self._items:
            raise ValueError(f"duplicate_field_spec: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> FieldSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown_field: {name}") from exc

    def list(self) -> tuple[FieldSpec, ...]:
        return tuple(self._items[name] for name in self.list_names())

    def list_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def require(self, names: tuple[str, ...]) -> tuple[FieldSpec, ...]:
        return tuple(self.get(name) for name in names)


_CANONICAL_FIELD_NAMES = CanonicalFieldNames(
    market_id="market_id",
    asset_id="asset_id",
    timestamp="timestamp",
    session_date="session_date",
    available_at="available_at",
    source_id="source_id",
    source_asset_id="source_asset_id",
)


def canonical_field_names() -> CanonicalFieldNames:
    return _CANONICAL_FIELD_NAMES


def platform_interim_column_names() -> tuple[str, ...]:
    names = canonical_field_names()
    return (
        names.market_id,
        names.asset_id,
        names.timestamp,
        names.session_date,
        names.available_at,
        names.source_id,
        names.source_asset_id,
    )


def platform_entity_time_key() -> tuple[str, ...]:
    names = canonical_field_names()
    return (names.market_id, names.asset_id, names.timestamp)


def platform_field_catalog() -> FieldCatalog:
    names = canonical_field_names()
    catalog = FieldCatalog()
    for spec in (
        FieldSpec(
            name=names.market_id,
            role=ColumnRole.KEY,
            dtype="string",
            logical_type=LogicalType.STRING_ID,
            description="Platform market identifier.",
        ),
        FieldSpec(
            name=names.asset_id,
            role=ColumnRole.KEY,
            dtype="string",
            logical_type=LogicalType.STRING_ID,
            description="Platform asset identifier.",
        ),
        FieldSpec(
            name=names.timestamp,
            role=ColumnRole.TIMESTAMP,
            dtype="timestamp[ns]",
            logical_type=LogicalType.TIMESTAMP,
            description="Logical observation timestamp.",
        ),
        FieldSpec(
            name=names.session_date,
            role=ColumnRole.TIMESTAMP,
            dtype="date",
            logical_type=LogicalType.DATE,
            description="Market-local trading session date.",
        ),
        FieldSpec(
            name=names.available_at,
            role=ColumnRole.TIMESTAMP,
            dtype="timestamp[ns]",
            logical_type=LogicalType.TIMESTAMP,
            description="Earliest research availability timestamp.",
        ),
        FieldSpec(
            name=names.source_id,
            role=ColumnRole.SOURCE,
            dtype="string",
            logical_type=LogicalType.STRING_ID,
            description="Source adapter identifier.",
        ),
        FieldSpec(
            name=names.source_asset_id,
            role=ColumnRole.SOURCE,
            dtype="string",
            logical_type=LogicalType.STRING_ID,
            description="Source-native asset identifier.",
        ),
    ):
        catalog.register(spec)
    return catalog


__all__ = [
    "CanonicalFieldNames",
    "FieldCatalog",
    "canonical_field_names",
    "platform_entity_time_key",
    "platform_field_catalog",
    "platform_interim_column_names",
]
