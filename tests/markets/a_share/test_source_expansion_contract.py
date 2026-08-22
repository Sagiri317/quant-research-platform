from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as materializer,
)
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.synthetic_a_share import write_synthetic_interim_inputs


@pytest.mark.parametrize(
    "source_id",
    [SourceId("synthetic_vendor_a"), SourceId("synthetic_vendor_b")],
)
def test_synthetic_normalized_interim_can_generate_market_panel_daily(
    tmp_path: Path,
    source_id: SourceId,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        source_id,
        start_date="2024-01-02",
        end_date="2024-01-03",
        asset_id="asset_a",
        market_type=1,
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        data_ids=(frequency_ids.MARKET_PANEL_DATA_ID,),
    )

    panel = read_dataset(
        frequency_ids.market_panel_processed_root(tmp_path),
    )
    assert panel["asset_id"].tolist() == ["asset_a", "asset_a"]
    assert "source_asset_id" not in panel.columns


def test_two_synthetic_sources_generate_same_processed_schema(tmp_path: Path) -> None:
    schemas: list[tuple[str, ...]] = []
    for source_name in ("synthetic_vendor_a", "synthetic_vendor_b"):
        source_id = SourceId(source_name)
        layout = StorageLayout(project_root=tmp_path / source_name)
        write_synthetic_interim_inputs(
            layout,
            source_id,
            start_date="2024-01-02",
            end_date="2024-01-03",
            asset_id="asset_a",
            market_type=1,
        )
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            data_ids=(frequency_ids.MARKET_PANEL_DATA_ID,),
        )
        frame = read_dataset(
            frequency_ids.market_panel_processed_root(layout.project_root),
        )
        schemas.append(tuple(frame.columns))

    assert schemas[0] == schemas[1]


def test_extra_source_columns_do_not_leak_into_processed_schema(tmp_path: Path) -> None:
    source_id = SourceId("synthetic_vendor_a")
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        source_id,
        start_date="2024-01-02",
        end_date="2024-01-03",
        asset_id="asset_a",
        market_type=1,
        include_extra_daily_column=True,
    )

    materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        data_ids=(frequency_ids.MARKET_PANEL_DATA_ID,),
    )

    panel = read_dataset(frequency_ids.market_panel_processed_root(tmp_path))
    assert "vendor_extra_metric" not in panel.columns


def test_missing_required_source_column_fails_closed_with_data_id_and_column(
    tmp_path: Path,
) -> None:
    source_id = SourceId("synthetic_vendor_a")
    layout = StorageLayout(project_root=tmp_path)
    write_synthetic_interim_inputs(
        layout,
        source_id,
        start_date="2024-01-02",
        end_date="2024-01-03",
        asset_id="asset_a",
        market_type=1,
        drop_daily_column="open",
    )

    with pytest.raises(
        ValueError,
        match=rf"{frequency_ids.MARKET_BARS_DATA_ID}.*open",
    ):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            data_ids=(frequency_ids.MARKET_PANEL_DATA_ID,),
        )
