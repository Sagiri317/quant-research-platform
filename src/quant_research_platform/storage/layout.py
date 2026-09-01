"""Portable disk layout for QRP 0.1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.storage.identity import validate_instance_identity


@dataclass(frozen=True)
class StorageLayout:
    project_root: Path = Path(".")
    _target_roots: Mapping[str, Path] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self._target_roots, Mapping) or any(
            type(target_id) is not str or not target_id or not isinstance(root, Path)
            for target_id, root in self._target_roots.items()
        ):
            raise ValueError("target_roots_invalid")
        object.__setattr__(
            self,
            "_target_roots",
            MappingProxyType(dict(self._target_roots)),
        )

    def with_target_roots(self, roots: Mapping[str, Path]) -> StorageLayout:
        if not isinstance(roots, Mapping):
            raise ValueError("target_roots_invalid")
        merged = dict(self._target_roots)
        merged.update(roots)
        return replace(self, _target_roots=merged)

    def target_root(self, target_id: str, default: Path) -> Path:
        return self._target_roots.get(target_id, default)

    def raw_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/raw" / str(source_id)

    def interim_source_root(self, source_id: SourceId) -> Path:
        return self.project_root / "data/interim" / str(source_id)

    def interim_data_root(self, source_id: SourceId, data_id: str) -> Path:
        return self.interim_source_root(source_id) / data_id

    def processed_market_root(self, market_id: MarketId) -> Path:
        return self.project_root / "data/processed" / str(market_id)

    def processed_data_root(
        self,
        market_id: MarketId,
        kind: ProcessedDataKind | str,
        data_id: str,
    ) -> Path:
        return self.processed_market_root(market_id) / str(kind) / data_id

    def research_dataset_root(self, market_id: MarketId, data_id: str) -> Path:
        return self.project_root / "data/research" / str(market_id) / data_id

    def instance_root(self, logical_root: Path, instance_identity: str) -> Path:
        return (
            Path(logical_root)
            / "instances"
            / validate_instance_identity(instance_identity)
        )
