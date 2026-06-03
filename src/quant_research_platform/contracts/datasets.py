"""Research dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.data import DataSpec


@dataclass(frozen=True, kw_only=True)
class ResearchDatasetSpec(DataSpec):
    dataset_id: str
    processed_data_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    filter_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    source_columns: tuple[str, ...] = ()
    universe_id: str | None = None
    mask_ids: tuple[str, ...] = ()


__all__ = ["ResearchDatasetSpec"]
