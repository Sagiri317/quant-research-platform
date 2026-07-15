"""Crypto materialization payload and disk guards."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)


def iter_disk_guarded_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    scratch_parent: Path,
) -> Iterator[pd.DataFrame]:
    for chunk in chunks:
        chunk_bytes = pandas_frame_payload_bytes(chunk)
        require_free_disk_headroom(
            scratch_parent,
            reserve_bytes=chunk_bytes,
        )
        yield chunk


def require_frame_within_budget(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    error: str,
) -> None:
    if pandas_frame_payload_bytes(frame) > target_bytes:
        raise ValueError(error)
