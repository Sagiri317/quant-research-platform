"""Crypto materialization payload ownership."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

from quant_research_platform.runtime.resources import pandas_frame_payload_bytes
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator


def iter_write_chunks(
    chunks: Iterable[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    with closing_dataset_iterator(iter(chunks)) as source_chunks:
        yield from source_chunks


def require_frame_within_budget(
    frame: pd.DataFrame,
    *,
    target_bytes: int,
    error: str,
) -> None:
    if pandas_frame_payload_bytes(frame) > target_bytes:
        raise ValueError(error)
