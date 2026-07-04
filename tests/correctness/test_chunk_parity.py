from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import (
    assemble_research_dataset,
    read_research_dataset,
)
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.processed_chunk_parity import (
    assert_forced_tiny_processed_chunks_match_default,
)
from tests.helpers.research_processed_inputs import (
    research_rows_for_dates,
    write_correctness_processed_inputs,
)
from tests.helpers.research_specs import (
    correctness_processed_catalog,
    correctness_research_dataset_catalog,
)


@pytest.mark.parametrize(
    ("data_id", "kind", "end_date", "time_window"),
    [
        (
            frequency_ids.MARKET_PANEL_DATA_ID,
            "panel",
            "2014-01-08",
            TimeWindow("2014-01-02", "2014-01-07"),
        ),
        (
            "execution",
            "annotation",
            "2014-01-08",
            TimeWindow("2014-01-02", "2014-01-07"),
        ),
        pytest.param(
            "price_volume",
            "factor",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "forward_returns",
            "label",
            "2014-08-29",
            TimeWindow("2014-01-02", "2014-02-28"),
            marks=pytest.mark.slow,
        ),
    ],
)
def test_processed_forced_tiny_session_chunks_equal_default(
    tmp_path,
    data_id: str,
    kind: str,
    end_date: str,
    time_window: TimeWindow,
) -> None:
    assert_forced_tiny_processed_chunks_match_default(
        tmp_path,
        data_id=data_id,
        kind=kind,
        end_date=end_date,
        time_window=time_window,
    )


def test_research_assembly_session_chunks_equal_default_and_keep_nan_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    frame.loc[frame["timestamp"].eq(pd.Timestamp("2014-01-06")), "mom_20d"] = pd.NA
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = correctness_processed_catalog()
    dataset_catalog = correctness_research_dataset_catalog()
    write_correctness_processed_inputs(
        default_layout,
        processed_catalog,
        frame=frame,
    )
    write_correctness_processed_inputs(
        tiny_layout,
        processed_catalog,
        frame=frame,
    )

    default_root = assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
    tiny_root = assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )

    default = read_research_dataset(
        default_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = read_research_dataset(
        tiny_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    assert default["mom_20d"].isna().tolist() == tiny["mom_20d"].isna().tolist()
    assert default.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    assert (
        inspect_dataset(tiny_root).file_count
        > inspect_dataset(default_root).file_count
    )


def test_research_feature_transform_session_chunks_equal_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = research_rows_for_dates(
        ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    )
    default_layout = StorageLayout(project_root=tmp_path / "default")
    tiny_layout = StorageLayout(project_root=tmp_path / "tiny")
    processed_catalog = correctness_processed_catalog()
    dataset_catalog = correctness_research_dataset_catalog(
        include_feature_transform=True
    )
    write_correctness_processed_inputs(
        default_layout,
        processed_catalog,
        frame=frame,
    )
    write_correctness_processed_inputs(
        tiny_layout,
        processed_catalog,
        frame=frame,
    )

    assemble_research_dataset(
        layout=default_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
    assemble_research_dataset(
        layout=tiny_layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    )

    default = read_research_dataset(
        default_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])
    tiny = read_research_dataset(
        tiny_layout,
        market_id=MarketId("a_share"),
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
    ).sort_values(["market_id", "asset_id", "timestamp"])

    pd.testing.assert_frame_equal(
        default.reset_index(drop=True),
        tiny.reset_index(drop=True),
    )
    timestamp_feature_mean = default.groupby("timestamp")["mom_20d_zscore"].mean()
    assert timestamp_feature_mean.abs().max() == pytest.approx(0.0)
    assert set(default["mom_20d_zscore"].dropna().abs()) == {1.0}
