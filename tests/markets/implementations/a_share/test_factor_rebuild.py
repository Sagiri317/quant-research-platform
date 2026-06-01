from __future__ import annotations

import json

import pandas as pd

from quant_research_platform.markets.implementations.a_share.factor_rebuild import (
    FactorRebuildConfig,
    run_factor_rebuild,
)


def test_factor_rebuild_overwrite_processed_with_optional_backup(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    target = (
        tmp_path
        / "data"
        / "processed_private"
        / "csmar"
        / "price_volume_factors_full"
        / "price_volume_factors_2024_2024.parquet"
    )
    target.parent.mkdir(parents=True)
    original = pd.DataFrame(
        {
            "stock_code": ["old"],
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "rev_1d": [999.0],
            "forward_return_1d": [0.0],
        }
    )
    original.to_parquet(target, engine="pyarrow", compression="zstd", index=False)
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    result = run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="overwrite_run",
            factor_parquet=target,
            overwrite_processed=True,
            backup_existing=True,
            memory_budget_gb=1,
        )
    )

    assert result.status == "ready"
    replacement = pd.read_parquet(target)
    assert "sentinel" not in replacement.columns
    assert set(replacement["stock_code"]) == {"000001", "000002"}
    backup_path = target.with_name(f"{target.stem}.overwrite_run.backup.parquet")
    assert backup_path.is_file()
    pd.testing.assert_frame_equal(pd.read_parquet(backup_path), original)
    manifest = json.loads(
        (output_root / "overwrite_run" / "overwrite_manifest.json").read_text("utf-8")
    )
    assert manifest["overwrite_processed"] is True
    assert manifest["backup_existing"] is True
    assert manifest["backup_file_name"] == backup_path.name
    assert str(tmp_path) not in json.dumps(manifest)


def _panel() -> pd.DataFrame:
    rows = []
    for stock_code, base in (("000001", 10.0), ("000002", 20.0)):
        for offset in range(5):
            close = base + offset
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": pd.Timestamp("2024-01-01")
                    + pd.offsets.BDay(offset),
                    "market_type": "A",
                    "is_a_share_scope": True,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "adj_open": close - 0.1,
                    "adj_close": close,
                    "return_1d": 0.01 * offset,
                    "amount": 100.0 + offset,
                    "turnover": 0.1 + offset / 100.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": close - 0.5,
                    "float_market_cap": 1000.0 + offset,
                    "total_market_cap": 2000.0 + offset,
                }
            )
    return pd.DataFrame(rows)
