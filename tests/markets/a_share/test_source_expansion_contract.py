from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_materializer_uses_source_id_parameter_not_csmar_constant() -> None:
    source = inspect.getsource(materializer)

    assert "CSMAR_SOURCE_ID" not in source
    assert "SourceId(\"csmar\")" not in source


@pytest.mark.parametrize(
    "source_id",
    [SourceId("synthetic_vendor_a"), SourceId("synthetic_vendor_b")],
)
def test_synthetic_normalized_interim_can_generate_daily_panel(
    tmp_path: Path,
    source_id: SourceId,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout, source_id)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        data_ids=("daily_panel",),
    )

    panel = read_dataset(
        tmp_path / "data/processed/a_share/panel/daily_panel",
    )
    assert panel["asset_id"].tolist() == ["asset_a", "asset_a"]
    assert "source_asset_id" not in panel.columns


def test_two_synthetic_sources_generate_same_processed_schema(tmp_path: Path) -> None:
    schemas: list[tuple[str, ...]] = []
    for source_name in ("synthetic_vendor_a", "synthetic_vendor_b"):
        source_id = SourceId(source_name)
        layout = StorageLayout(project_root=tmp_path / source_name)
        _write_interim_inputs(layout, source_id)
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            data_ids=("daily_panel",),
        )
        frame = read_dataset(
            layout.project_root / "data/processed/a_share/panel/daily_panel",
        )
        schemas.append(tuple(frame.columns))

    assert schemas[0] == schemas[1]


def test_extra_source_columns_do_not_leak_into_processed_schema(tmp_path: Path) -> None:
    source_id = SourceId("synthetic_vendor_a")
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout, source_id, include_extra_daily_column=True)

    materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
        data_ids=("daily_panel",),
    )

    panel = read_dataset(tmp_path / "data/processed/a_share/panel/daily_panel")
    assert "vendor_extra_metric" not in panel.columns


def test_missing_required_source_column_fails_closed_with_data_id_and_column(
    tmp_path: Path,
) -> None:
    source_id = SourceId("synthetic_vendor_a")
    layout = StorageLayout(project_root=tmp_path)
    _write_interim_inputs(layout, source_id, drop_daily_column="open")

    with pytest.raises(ValueError, match=r"daily_market.*open"):
        materializer.materialize_processed_data(
            layout=layout,
            source_id=source_id,
            time_window=TimeWindow("2024-01-02", "2024-01-03"),
            data_ids=("daily_panel",),
        )


def _write_interim_inputs(
    layout: StorageLayout,
    source_id: SourceId,
    *,
    include_extra_daily_column: bool = False,
    drop_daily_column: str | None = None,
) -> None:
    catalog = _interim_catalog(
        source_id,
        include_extra_daily_column=include_extra_daily_column,
        drop_daily_column=drop_daily_column,
    )
    frames = {
        "daily_market": _daily_market_frame(),
        "stock_master": _stock_master_frame(),
        "market_cap": _market_cap_frame(),
        "trading_calendar": _trading_calendar_frame(),
    }
    if include_extra_daily_column:
        frames["daily_market"]["vendor_extra_metric"] = [1.0, 2.0]
    if drop_daily_column is not None:
        frames["daily_market"] = frames["daily_market"].drop(
            columns=[drop_daily_column]
        )
    for data_id, frame in frames.items():
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame,
            spec=catalog.get(data_id),
        )


def _interim_catalog(
    source_id: SourceId,
    *,
    include_extra_daily_column: bool,
    drop_daily_column: str | None,
) -> InterimDataCatalog:
    specs = {
        "daily_market": (
            "source_asset_id",
            "timestamp",
            "trade_date",
            "market_type",
            "trading_status_code",
            "filling_flag",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
            "limit_status",
            "return_1d",
            "adj_close",
            "adj_open",
            "float_market_cap",
            "total_market_cap",
            "turnover",
            *(("vendor_extra_metric",) if include_extra_daily_column else ()),
        ),
        "stock_master": (
            "source_asset_id",
            "timestamp",
            "stock_name",
            "listing_date",
            "delisting_date",
            "company_status",
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
            "status_change_date",
        ),
        "market_cap": (
            "source_asset_id",
            "timestamp",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        ),
        "trading_calendar": (
            "timestamp",
            "calendar_date",
            "source_market_type",
            "is_trading_day",
            "source_day_of_week",
            "source_state",
        ),
    }
    if drop_daily_column is not None:
        specs["daily_market"] = tuple(
            column for column in specs["daily_market"] if column != drop_daily_column
        )
    catalog = InterimDataCatalog()
    for data_id, columns in specs.items():
        primary_key = (
            ("timestamp",)
            if data_id == "trading_calendar"
            else ("source_asset_id", "timestamp")
        )
        roles = {
            column: (
                ColumnRole.TIMESTAMP
                if column in {"timestamp", "calendar_date"}
                else ColumnRole.KEY
                if column in primary_key
                else ColumnRole.CONTEXT
            )
            for column in columns
        }
        catalog.register(
            InterimDataSpec(
                data_id=data_id,
                source_id=source_id,
                frequency=DataFrequency.DAILY,
                grain=DataGrain.ASSET_TIMESTAMP,
                column_roles=roles,
                primary_key=primary_key,
                partition_columns=("year",),
            )
        )
    return catalog


def _daily_market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "market_type": [1, 1],
            "trading_status_code": [1, 1],
            "filling_flag": ["0", "0"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [100.0, 100.0],
            "amount": [1000.0, 1000.0],
            "prev_close": [9.9, 10.2],
            "limit_status": [0, 0],
            "return_1d": [0.01, 0.02],
            "adj_close": [10.2, 11.2],
            "adj_open": [10.0, 11.0],
            "float_market_cap": [100.0, 101.0],
            "total_market_cap": [200.0, 201.0],
            "turnover": [0.1, 0.1],
        }
    )


def _stock_master_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "stock_name": ["Asset A"],
            "listing_date": pd.to_datetime(["2023-01-01"]),
            "delisting_date": [pd.NaT],
            "company_status": ["normal"],
            "csrc_2012_industry_code": ["I1"],
            "csrc_2012_industry_name": ["Industry"],
            "status_change_date": pd.to_datetime(["2023-01-01"]),
        }
    )


def _market_cap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "share_change_date": pd.to_datetime(["2023-01-01"]),
            "total_shares": [1000.0],
            "a_float_shares": [500.0],
        }
    )


def _trading_calendar_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2024-01-03", freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "calendar_date": dates,
            "source_market_type": [1] * len(dates),
            "is_trading_day": [True] * len(dates),
            "source_day_of_week": [1] * len(dates),
            "source_state": ["O"] * len(dates),
        }
    )
