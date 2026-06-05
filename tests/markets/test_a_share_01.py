from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.factors import (
    build_factors,
    price_volume,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    exact_literature_price_volume_factor_input_columns,
    expected_literature_price_volume_factor_columns,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    build_labels,
    forward_returns,
)
from quant_research_platform.markets.implementations.a_share.masks import (
    build as build_masks,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    contexts,
    panels,
)
from quant_research_platform.markets.implementations.a_share.universes import (
    build as build_universes,
)
from quant_research_platform.storage.layout import (
    processed_data_manifest_path,
    processed_data_parquet_root,
)
from quant_research_platform.storage.parquet import read_partitioned_parquet_dataset


def _panel() -> pd.DataFrame:
    rows = []
    for day in range(1, 6):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "stock_code": "000001",
                "timestamp": pd.Timestamp(f"2014-01-0{day}"),
                "session_date": pd.Timestamp(f"2014-01-0{day}"),
                "available_at": pd.Timestamp(f"2014-01-0{day}"),
                "trade_date": pd.Timestamp(f"2014-01-0{day}"),
                "adj_open": float(day + 9),
                "adj_close": float(day + 10),
                "return_1d": 0.01 * day,
                "amount": 1000.0 + day,
                "turnover": 0.1 + day / 100.0,
                "market_type": "A",
                "csrc_2012_industry_code": "I",
                "csrc_2012_industry_name": "industry",
                "total_market_cap": 10000.0 + day,
                "float_market_cap": 9000.0 + day,
                "eligible": True,
                "can_enter": True,
                "can_exit": True,
                "high": float(day + 11),
                "low": float(day + 9),
                "prev_close": float(day + 9),
            }
        )
    return pd.DataFrame(rows)


def test_a_share_price_volume_factor_contract_is_56_columns() -> None:
    columns = expected_literature_price_volume_factor_columns()

    assert len(columns) == 56
    assert "mom_20d" in columns
    assert "forward_return_20d" not in columns
    assert expected_literature_price_volume_factor_columns(factor_set="core")
    for factor_id in columns:
        assert exact_literature_price_volume_factor_input_columns(factor_id)
    with pytest.raises(ValueError, match="invalid_factor_set"):
        expected_literature_price_volume_factor_columns(
            factor_set="core" + "_long_period"
        )


def test_a_share_factor_and_label_helpers_use_past_and_forward_domains() -> None:
    panel = _panel()
    factored = price_volume.calculate_price_volume_factors(
        panel,
        windows=(2,),
        asset_column="asset_id",
        date_column="timestamp",
    )
    labeled = forward_returns.add_forward_return_labels(panel, horizons=(1,))

    assert "mom_2d" in factored.columns
    assert "forward_return_1d" not in factored.columns
    assert "forward_return_1d" in labeled.columns


def test_a_share_factor_and_label_builders_write_split_layers(tmp_path) -> None:
    panel = _panel()
    processed_root = tmp_path / "data" / "processed" / "a_share"
    panel_path = (
        processed_data_parquet_root(processed_root, "panels", "daily_panel")
        / "year=2014"
        / "part-000.parquet"
    )
    panel_path.parent.mkdir(parents=True)
    panel.to_parquet(panel_path, index=False)
    factored = build_factors.calculate_panel_factors(panel, windows=(2,))
    labels = build_labels.calculate_panel_labels(panel)
    factor_path, factor_rows = build_factors.calculate_price_volume_factor_full(
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    label_path, label_rows = build_labels.write_forward_labels(
        labels,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    factor_manifest = processed_data_manifest_path(
        processed_root, "factors", "price_volume_56"
    )
    label_manifest = processed_data_manifest_path(
        processed_root, "labels", "forward_returns"
    )

    assert "mom_2d" in factored.columns
    assert not any(column.startswith("forward_return_") for column in factored)
    assert "forward_return_1d" in labels.columns
    assert factor_rows == len(panel)
    assert label_rows == len(panel)
    assert factor_path.as_posix().endswith(
        "processed/a_share/factors/price_volume_56/data"
    )
    assert label_path.as_posix().endswith(
        "processed/a_share/labels/forward_returns/data"
    )
    factor_payload = json.loads(factor_manifest.read_text(encoding="utf-8"))
    label_payload = json.loads(label_manifest.read_text(encoding="utf-8"))
    assert factor_payload["schema_version"] == "0.1"
    assert label_payload["schema_version"] == "0.1"


def test_a_share_context_universe_and_mask_builders_write_split_layers(
    tmp_path,
) -> None:
    panel = _panel()
    processed_root = tmp_path / "data" / "processed" / "a_share"

    context_path, context_rows = contexts.write_contexts(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    universe_path, universe_rows = build_universes.write_universe(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )
    mask_outputs = build_masks.write_masks(
        panel,
        processed_root=processed_root,
        start_year=2014,
        end_year=2014,
    )

    assert context_rows == len(panel)
    assert universe_rows == len(panel)
    assert context_path.as_posix().endswith(
        "processed/a_share/contexts/daily_context/data"
    )
    assert universe_path.as_posix().endswith(
        "processed/a_share/universes/a_share_default/data"
    )
    assert sorted(mask_outputs) == ["as_of", "evaluation", "execution"]
    assert mask_outputs["execution"][0].as_posix().endswith(
        "processed/a_share/masks/execution/a_share_execution/data"
    )
    assert json.loads(
        processed_data_manifest_path(
            processed_root, "masks/execution", "a_share_execution"
        ).read_text(encoding="utf-8")
    )["schema_version"] == "0.1"


def test_a_share_panel_from_interim_applies_exact_end_date(tmp_path) -> None:
    interim_root = tmp_path / "data" / "interim" / "csmar"
    processed_root = tmp_path / "data" / "processed" / "a_share"
    _write_interim_fixture(
        interim_root,
        "daily_market",
        pd.DataFrame(
            [
                _daily_market_row("2024-06-28"),
                _daily_market_row("2024-07-01"),
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "stock_master",
        pd.DataFrame(
            [
                {
                    "stock_code": "000001",
                    "stock_name": "demo",
                    "listing_date": pd.Timestamp("2020-01-01"),
                    "delisting_date": pd.NaT,
                    "company_status": "active",
                    "status_change_date": pd.Timestamp("2020-01-01"),
                    "csrc_2012_industry_code": "I",
                    "csrc_2012_industry_name": "industry",
                }
            ]
        ),
    )
    _write_interim_fixture(
        interim_root,
        "market_cap",
        pd.DataFrame(
            [
                {
                    "stock_code": "000001",
                    "share_change_date": pd.Timestamp("2020-01-01"),
                    "total_shares": 1000.0,
                    "a_float_shares": 500.0,
                }
            ]
        ),
    )

    output_root, rows = panels.build_panel_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2024,
        end_year=2024,
        end_date="2024-06-30",
        min_listing_trading_days=1,
    )
    written = read_partitioned_parquet_dataset(output_root, years=(2024,))

    assert rows == 1
    assert written["trade_date"].tolist() == [pd.Timestamp("2024-06-28")]


def _daily_market_row(trade_date: str) -> dict[str, object]:
    timestamp = pd.Timestamp(trade_date)
    return {
        "market_id": "a_share",
        "asset_id": "000001",
        "timestamp": timestamp,
        "session_date": timestamp,
        "available_at": timestamp,
        "stock_code": "000001",
        "trade_date": timestamp,
        "market_type": "A",
        "trading_status_code": 1,
        "filling_flag": "0",
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "adj_open": 10.0,
        "adj_close": 10.2,
        "prev_close": 9.9,
        "volume": 1000.0,
        "amount": 10200.0,
        "return_1d": 0.03,
        "limit_status": 0,
        "is_limit_up": False,
        "is_limit_down": False,
        "is_trading": True,
        "is_st": False,
        "is_pt": False,
        "is_a_share_scope": True,
        "eligible_base": True,
        "total_market_cap": 200000.0,
        "float_market_cap": 100000.0,
    }


def _write_interim_fixture(
    interim_root,
    interim_data_id: str,
    frame: pd.DataFrame,
) -> None:
    data_root = interim_root / interim_data_id / "data" / "year=2024"
    data_root.mkdir(parents=True)
    frame.to_parquet(data_root / "part-000.parquet", index=False)
