from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.panel_keys import DEFAULT_PANEL_KEY_COLUMNS
from tests.helpers.synthetic_a_share import (
    processed_a_share_root,
    write_synthetic_interim_inputs,
)


def tiny_processed_chunk_bytes(data_id: str) -> int:
    return {
        frequency_ids.MARKET_PANEL_DATA_ID: 512 * 1024,
        "execution": 512 * 1024,
        "price_volume": 512 * 1024,
        "forward_returns": 512 * 1024,
    }[data_id]


def assert_forced_tiny_processed_chunks_match_default(
    tmp_path,
    *,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    tiny_chunk_bytes = tiny_processed_chunk_bytes(data_id)
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    source_id = SourceId("csmar")
    write_synthetic_interim_inputs(
        default_layout,
        source_id,
        end_date=end_date,
        calendar_end_date=end_date,
    )
    write_synthetic_interim_inputs(
        tiny_layout,
        source_id,
        end_date=end_date,
        calendar_end_date=end_date,
    )

    materializer.materialize_processed_data(
        layout=default_layout,
        source_id=source_id,
        time_window=time_window,
        data_ids=(data_id,),
    )
    materializer.materialize_processed_data(
        layout=tiny_layout,
        source_id=source_id,
        time_window=time_window,
        data_ids=(data_id,),
        resource_policy=RuntimeResourcePolicy.from_environment(
            available_memory_bytes=tiny_chunk_bytes,
            environ={},
        ),
    )

    default_root = processed_a_share_root(tmp_path / "default", kind, data_id)
    tiny_root = processed_a_share_root(tmp_path / "tiny", kind, data_id)
    keys = list(DEFAULT_PANEL_KEY_COLUMNS)
    default = read_dataset(default_root).sort_values(keys)
    tiny = read_dataset(tiny_root).sort_values(keys)

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
