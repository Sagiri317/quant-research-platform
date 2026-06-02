from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.factor_rebuild import (
    FactorRebuildConfig,
    refresh_factor_rebuild_eda,
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
    feature_manifest = json.loads(
        (output_root / "overwrite_run" / "feature_set_manifest.json").read_text(
            "utf-8"
        )
    )
    assert manifest["overwrite_processed"] is True
    assert manifest["backup_existing"] is True
    assert manifest["backup_file_name"] == backup_path.name
    assert feature_manifest["feature_set_id"] == (
        "a_share_literature_price_volume_preset_v1"
    )
    assert feature_manifest["unsupported_factor_count"] == 0
    assert (output_root / "overwrite_run" / "eda" / "eda_manifest.json").is_file()
    assert (output_root / "overwrite_run" / "eda" / "eda_report.md").is_file()
    assert str(tmp_path) not in json.dumps(manifest)


def test_factor_rebuild_date_filter_degraded_smoke(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel_with_dates().to_parquet(
        panel_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    result = run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="filtered_run",
            memory_budget_gb=1,
            start_date="2024-02-01",
            end_date="2024-02-29",
        )
    )

    output = pd.read_parquet(output_root / "filtered_run" / "factor_rebuild.parquet")
    assert result.status == "ready"
    assert output["trade_date"].min() >= pd.Timestamp("2024-02-01")
    assert output["trade_date"].max() <= pd.Timestamp("2024-02-29")
    preflight = json.loads(
        (output_root / "filtered_run" / "memory_preflight.json").read_text("utf-8")
    )
    assert preflight["date_filter"] == {
        "start_date": "2024-02-01",
        "end_date": "2024-02-29",
    }


def test_factor_rebuild_uses_warmup_then_crops_output_window(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel_with_long_warmup().to_parquet(
        panel_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    result = run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="warmup_core_run",
            memory_budget_gb=1,
            calculation_start_date="2024-01-01",
            evaluation_start_date="2024-08-01",
            end_date="2024-10-31",
            factor_set_tier="core_long_period",
        )
    )

    run_dir = output_root / "warmup_core_run"
    output = pd.read_parquet(run_dir / "factor_rebuild.parquet")
    manifest = json.loads((run_dir / "feature_set_manifest.json").read_text("utf-8"))
    summary = json.loads((run_dir / "runtime_summary.json").read_text("utf-8"))
    assert result.status == "ready"
    assert output["trade_date"].min() >= pd.Timestamp("2024-08-01")
    assert output["trade_date"].max() <= pd.Timestamp("2024-10-31")
    assert output["mom_120d"].notna().any()
    assert "mom_252d" not in output.columns
    assert "drawdown_20d" not in output.columns
    assert "forward_return_20d" in output.columns
    assert manifest["factor_set_tier"] == "core_long_period"
    assert "warmup_sufficient" in manifest["warnings"]
    assert "date_filtered_without_warmup" not in manifest["warnings"]
    assert summary["warmup_status"] == "warmup_sufficient"
    assert summary["warmup"] == {
        "warmup_status": "warmup_sufficient",
        "largest_lookback_days": 120,
        "warmup_observation_count": 153,
        "calculation_start_date": "2024-01-01",
        "evaluation_start_date": "2024-08-01",
        "end_date": "2024-10-31",
        "warmup_sufficient": True,
    }
    eda_manifest = json.loads(
        (run_dir / "eda" / "eda_manifest.json").read_text("utf-8")
    )
    assert eda_manifest["chart_dependency_status"] == "ready"
    assert eda_manifest["chart_count"] == 2
    assert eda_manifest["chart_files"] == [
        "factor_coverage_by_year.png",
        "factor_missingness_top20.png",
    ]
    assert (run_dir / "eda" / "factor_coverage_by_year.png").is_file()
    assert (run_dir / "eda" / "factor_missingness_top20.png").is_file()


def test_refresh_factor_rebuild_eda_regenerates_static_charts(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="refresh_factor_eda",
            memory_budget_gb=1,
            factor_set_tier="smoke_minimal",
        )
    )

    run_dir = output_root / "refresh_factor_eda"
    (run_dir / "eda" / "factor_coverage_by_year.png").unlink()
    (run_dir / "eda" / "factor_missingness_top20.png").unlink()
    result = refresh_factor_rebuild_eda(run_dir)
    manifest = json.loads((run_dir / "eda" / "eda_manifest.json").read_text("utf-8"))

    assert result["chart_dependency_status"] == "ready"
    assert result["chart_count"] == 2
    assert manifest["chart_files"] == [
        "factor_coverage_by_year.png",
        "factor_missingness_top20.png",
    ]
    assert (run_dir / "eda" / "factor_coverage_by_year.png").is_file()
    assert (run_dir / "eda" / "factor_missingness_top20.png").is_file()


def test_refresh_factor_rebuild_eda_rejects_public_run_root(tmp_path) -> None:
    run_dir = tmp_path / "reports" / "public" / "factor_run"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_factor_rebuild_eda(run_dir)


def test_refresh_factor_rebuild_eda_rejects_unapproved_private_marker_root(
    tmp_path,
) -> None:
    run_dir = tmp_path / "reports" / "private_tmp" / "factor_run"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_factor_rebuild_eda(run_dir)


def test_refresh_factor_rebuild_eda_rejects_traversal_out_of_private_root(
    tmp_path,
) -> None:
    run_dir = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share_factor_rebuilds"
        / ".."
        / ".."
        / "public"
        / "factor_run"
    )
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_factor_rebuild_eda(run_dir)


def test_factor_rebuild_full_range_reports_full_range_warmup(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    result = run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="full_range_dry",
            memory_budget_gb=1,
            dry_run=True,
        )
    )

    summary = json.loads(
        (output_root / "full_range_dry" / "runtime_summary.json").read_text("utf-8")
    )
    warmup = json.loads(
        (output_root / "full_range_dry" / "warmup_sufficiency.json").read_text(
            "utf-8"
        )
    )
    assert result.status == "dry_run"
    assert summary["warmup_status"] == "full_range"
    assert warmup["warmup_status"] == "full_range"
    assert warmup["warmup_sufficient"] is True


def test_factor_rebuild_warmup_sufficiency_statuses_and_lookbacks(
    tmp_path,
) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel_with_long_warmup().to_parquet(
        panel_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="sufficient_core",
            memory_budget_gb=1,
            dry_run=True,
            calculation_start_date="2024-01-01",
            evaluation_start_date="2024-08-01",
            end_date="2024-10-31",
            factor_set_tier="core_long_period",
            require_sufficient_warmup=True,
        )
    )
    run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="date_filter_no_warmup",
            memory_budget_gb=1,
            dry_run=True,
            start_date="2024-08-01",
            end_date="2024-10-31",
            factor_set_tier="core_long_period",
        )
    )
    run_factor_rebuild(
        FactorRebuildConfig(
            panel_parquet=panel_path,
            output_root=output_root,
            run_id="full_tier_lookback",
            memory_budget_gb=1,
            dry_run=True,
            calculation_start_date="2024-01-01",
            evaluation_start_date="2024-08-01",
            end_date="2024-10-31",
            factor_set_tier="full",
        )
    )

    sufficient = json.loads(
        (output_root / "sufficient_core" / "warmup_sufficiency.json").read_text(
            "utf-8"
        )
    )
    no_warmup = json.loads(
        (output_root / "date_filter_no_warmup" / "warmup_sufficiency.json").read_text(
            "utf-8"
        )
    )
    full_tier = json.loads(
        (output_root / "full_tier_lookback" / "warmup_sufficiency.json").read_text(
            "utf-8"
        )
    )
    assert sufficient["warmup_status"] == "warmup_sufficient"
    assert sufficient["largest_lookback_days"] == 120
    assert sufficient["warmup_observation_count"] == 153
    assert no_warmup["warmup_status"] == "date_filtered_without_warmup"
    assert no_warmup["warmup_sufficient"] is False
    assert full_tier["warmup_status"] == "warmup_insufficient"
    assert full_tier["largest_lookback_days"] == 252


def test_factor_rebuild_strict_mode_rejects_insufficient_warmup(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel_with_long_warmup().to_parquet(
        panel_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"

    with pytest.raises(ValueError, match="warmup_insufficient"):
        run_factor_rebuild(
            FactorRebuildConfig(
                panel_parquet=panel_path,
                output_root=output_root,
                run_id="strict_insufficient_full",
                memory_budget_gb=1,
                dry_run=True,
                calculation_start_date="2024-01-01",
                evaluation_start_date="2024-08-01",
                end_date="2024-10-31",
                factor_set_tier="full",
                require_sufficient_warmup=True,
            )
        )

    summary = json.loads(
        (
            output_root
            / "strict_insufficient_full"
            / "runtime_summary.json"
        ).read_text("utf-8")
    )
    assert summary["failure_code"] == "warmup_insufficient"


def test_factor_rebuild_rejects_private_input_to_public_output_before_run_dir(
    tmp_path,
) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    public_output = tmp_path / "reports" / "public_runs"

    with pytest.raises(ValueError, match="private_output_policy_violation"):
        run_factor_rebuild(
            FactorRebuildConfig(
                panel_parquet=panel_path,
                output_root=public_output,
                run_id="public_output_blocked",
                dry_run=True,
            )
        )

    assert not public_output.exists()


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


def _panel_with_dates() -> pd.DataFrame:
    rows = []
    for stock_code, base in (("000001", 10.0), ("000002", 20.0)):
        for offset, date in enumerate(pd.bdate_range("2024-01-01", periods=60)):
            close = base + offset / 10.0
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": date,
                    "market_type": "A",
                    "is_a_share_scope": True,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "adj_open": close - 0.1,
                    "adj_close": close,
                    "return_1d": 0.001 * offset,
                    "amount": 100.0 + offset,
                    "turnover": 0.1 + offset / 1000.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": close - 0.5,
                    "float_market_cap": 1000.0 + offset,
                    "total_market_cap": 2000.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def _panel_with_long_warmup() -> pd.DataFrame:
    rows = []
    for stock_code, base in (("000001", 10.0), ("000002", 20.0)):
        for offset, date in enumerate(pd.bdate_range("2024-01-01", periods=230)):
            close = base + offset / 100.0
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": date,
                    "market_type": "A",
                    "is_a_share_scope": True,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "adj_open": close - 0.1,
                    "adj_close": close,
                    "return_1d": 0.0005 * offset,
                    "amount": 100.0 + offset,
                    "turnover": 0.1 + offset / 1000.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": close - 0.5,
                    "float_market_cap": 1000.0 + offset,
                    "total_market_cap": 2000.0 + offset,
                }
            )
    return pd.DataFrame(rows)
