"""Minimal Choice adapter; raw provider headers are intentionally not guessed."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.validation import (
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.sources.choice.fields import (
    FIRST_DISCLOSURE_COLUMNS,
    field_roles,
)
from quant_research_platform.sources.choice.ids import (
    FIRST_DISCLOSURE_DATA_ID,
    SOURCE_ID,
)


@dataclass(frozen=True)
class ChoiceReadiness:
    status: str
    blocker: str


def first_disclosure_spec() -> InterimDataSpec:
    return InterimDataSpec(
        data_id=FIRST_DISCLOSURE_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
        column_roles=field_roles(),
        primary_key=FIRST_DISCLOSURE_COLUMNS,
        partition_columns=("year",),
    )


def interim_catalog() -> InterimDataCatalog:
    return InterimDataCatalog((first_disclosure_spec(),))


def normalize_first_disclosure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept only the canonical adapter boundary once a real export is supplied."""

    if tuple(frame.columns) != FIRST_DISCLOSURE_COLUMNS:
        raise ValueError("choice_first_disclosure_header_not_canonical")
    result = frame.loc[:, FIRST_DISCLOSURE_COLUMNS].copy()
    result["asset_id"] = strict_string_series(
        result["asset_id"], error="invalid_choice_asset_id"
    )
    for column in ("report_period", "first_disclosure_date"):
        result[column] = strict_timestamp_series(
            result[column], error="invalid_choice_disclosure_date"
        ).dt.normalize()
    if result.duplicated(["asset_id", "report_period"], keep=False).any():
        raise ValueError("duplicate_choice_first_disclosure_key")
    return result.sort_values(
        list(FIRST_DISCLOSURE_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def readiness() -> ChoiceReadiness:
    return ChoiceReadiness(
        status="BLOCKED_EXTERNAL_DATA",
        blocker="Choice first-disclosure export/header not supplied",
    )
