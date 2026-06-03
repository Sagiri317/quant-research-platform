"""Model contracts for non-DL local research."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    feature_columns: tuple[str, ...]
    label_column: str
    contract_version: str = "0.1"


__all__ = ["ModelSpec"]
