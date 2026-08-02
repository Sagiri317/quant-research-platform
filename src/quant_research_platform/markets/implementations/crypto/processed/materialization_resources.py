"""Crypto materialization payload and disk guards."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator


def iter_disk_guarded_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    scratch_parent: Path,
) -> Iterator[pd.DataFrame]:
    with closing_dataset_iterator(iter(chunks)) as source_chunks:
        for chunk in source_chunks:
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
