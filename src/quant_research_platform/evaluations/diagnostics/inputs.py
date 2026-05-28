"""Diagnostics input bundle boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class DiagnosticInputBundle:
    """In-memory inputs available to post-run diagnostics.

    The bundle carries already-computed evaluation frames and sanitized metadata.
    It does not authorize direct private-root reads or API/frontend state access.
    """

    base_tables: dict[str, pd.DataFrame]
    optional_columns: dict[str, pd.Series] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


__all__ = ["DiagnosticInputBundle"]
