"""Private artifact membership validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_membership_series,
    strict_string_series,
)


def validate_string_membership_column(
    frame: pd.DataFrame,
    column: str,
    allowed: Iterable[str],
    error_name: str,
) -> None:
    if column not in frame.columns:
        raise ValueError(error_name)
    values = strict_string_series(frame[column], error=error_name)
    strict_membership_series(values, allowed=set(allowed), error=error_name)


__all__ = ["validate_string_membership_column"]
