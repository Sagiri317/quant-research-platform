"""Crypto materialization payload ownership."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.runtime.resources import pandas_frame_payload_bytes


def require_frame_within_budget(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    error: str,
) -> None:
    if pandas_frame_payload_bytes(frame) > target_bytes:
        raise ValueError(error)
