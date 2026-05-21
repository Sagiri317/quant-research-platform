from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.a_share.factors import build_factors as factors


def test_select_slim_factor_columns_drops_panel_only_fields() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "market_type": [4],
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "amount": [1000.0],
            "turnover": [0.01],
            "total_market_cap": [2000.0],
            "float_market_cap": [1500.0],
            "open": [10.0],
            "close": [10.2],
            "stock_name": ["平安银行"],
            "mom_20d": [0.05],
            "turnover_20d": [0.02],
            "forward_return_5d": [0.03],
        }
    )

    slim = factors.select_slim_factor_columns(frame)

    assert list(slim.columns) == [
        "stock_code",
        "trade_date",
        "market_type",
        "eligible",
        "can_enter",
        "can_exit",
        "amount",
        "turnover",
        "total_market_cap",
        "float_market_cap",
        "mom_20d",
        "turnover_20d",
        "forward_return_5d",
    ]
    assert "open" not in slim.columns
    assert "close" not in slim.columns
    assert "stock_name" not in slim.columns


def test_read_panel_prunes_parquet_columns_and_keeps_stock_code(tmp_path) -> None:
    path = tmp_path / "research_panel_2024.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "amount": [1000.0, 2000.0],
            "open": [10.0, 20.0],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    loaded = factors.read_panel(path, columns=["stock_code", "amount"])

    assert list(loaded.columns) == ["stock_code", "amount"]
    assert loaded["stock_code"].tolist() == ["000001", "000002"]


def _small_panel(start_date: str, rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_code": ["000001"] * rows,
            "trade_date": pd.date_range(start_date, periods=rows, freq="B"),
            "market_type": [4] * rows,
            "eligible": [True] * rows,
            "can_enter": [True] * rows,
            "can_exit": [True] * rows,
            "adj_open": [10.0 + idx for idx in range(rows)],
            "adj_close": [10.2 + idx for idx in range(rows)],
            "return_1d": [0.01] * rows,
            "amount": [10000.0 + idx for idx in range(rows)],
            "turnover": [0.01 + idx * 0.001 for idx in range(rows)],
            "high": [10.5 + idx for idx in range(rows)],
            "low": [9.8 + idx for idx in range(rows)],
            "prev_close": [9.9 + idx for idx in range(rows)],
            "total_market_cap": [2000.0 + idx for idx in range(rows)],
            "float_market_cap": [1500.0 + idx for idx in range(rows)],
        }
    )


def test_calculate_price_volume_factor_full_reads_full_panel(tmp_path) -> None:
    processed_root = tmp_path / "processed"
    panel_dir = processed_root / "panel_full"
    panel_dir.mkdir(parents=True)
    panel_2024 = _small_panel("2024-12-23", 7)
    panel_2025 = _small_panel("2025-01-01", 7)
    pd.concat([panel_2024, panel_2025], ignore_index=True).to_parquet(
        panel_dir / "research_panel_2024_2025.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    full_path, rows = factors.calculate_price_volume_factor_full(
        processed_root=processed_root,
        start_year=2024,
        end_year=2025,
        windows=(3,),
        horizons=(1, 3),
    )

    full = pd.read_parquet(full_path, engine="pyarrow")
    assert rows == len(full)
    assert full_path == (
        processed_root
        / "price_volume_factors_full"
        / "price_volume_factors_2024_2025.parquet"
    )
    assert full["trade_date"].dt.year.tolist() == [2024] * 7 + [2025] * 7
