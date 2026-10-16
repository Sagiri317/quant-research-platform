"""ResearchDataset validation helpers."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.datasets import ResearchDatasetSpec


def validate_research_dataset(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    require_all_columns: bool = True,
) -> None:
    _validate_asset_time_key(spec.primary_key, spec.column_roles)
    duplicate_columns = tuple(
        dict.fromkeys(
            str(column)
            for column in frame.columns[frame.columns.duplicated(keep=False)]
        )
    )
    if duplicate_columns:
        raise ValueError(
            "duplicate_research_dataset_columns: " + ", ".join(duplicate_columns)
        )
    required = set(spec.required_columns if require_all_columns else spec.primary_key)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_research_required_fields: {', '.join(missing)}")

    if (
        "market_id" in frame
        and not frame["market_id"].eq(str(spec.market_id)).fillna(False).all()
    ):
        raise ValueError("research_dataset_market_mismatch")

    null_key_columns = [
        column for column in spec.primary_key if bool(frame[column].isna().any())
    ]
    if null_key_columns:
        joined = ", ".join(null_key_columns)
        raise ValueError(f"null_research_dataset_key_columns: {joined}")

    duplicates = frame.duplicated(list(spec.primary_key))
    if bool(duplicates.any()):
        raise ValueError("duplicate_research_dataset_keys")
