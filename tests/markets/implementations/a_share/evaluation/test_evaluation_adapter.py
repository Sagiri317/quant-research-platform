from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.evaluations.diagnostics import DiagnosticsSpec
from quant_research_platform.markets.implementations.a_share.evaluation import (
    evaluation_adapter as evaluation,
)
from quant_research_platform.markets.implementations.a_share.evaluation import (
    run_factor_evaluation,
)


def test_discover_factor_and_label_columns_from_slim_parquet(tmp_path) -> None:
    path = tmp_path / "price_volume_factors_2025.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "mom_20d": [0.1],
            "amihud_20d": [0.2],
            "forward_return_5d": [0.03],
            "forward_return_20d": [0.05],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    assert evaluation.discover_factor_columns(path) == ("amihud_20d", "mom_20d")
    assert evaluation.discover_label_columns(path) == (
        "forward_return_20d",
        "forward_return_5d",
    )


def test_choose_default_factor_columns_prefers_representative_short_list() -> None:
    factors = (
        "mom_20d",
        "rev_20d",
        "vol_20d",
        "amihud_20d",
        "mom_60d",
    )

    assert evaluation.choose_default_factor_columns(factors) == (
        "mom_20d",
        "vol_20d",
        "amihud_20d",
    )


def test_label_availability_summary_counts_filter_columns() -> None:
    frame = pd.DataFrame(
        {
            "eligible": [True, True, False],
            "can_enter": [True, False, True],
            "can_exit": [True, True, False],
            "forward_return_5d": [0.1, None, 0.2],
        }
    )

    summary = evaluation.summarize_label_availability(
        frame,
        label_columns=("forward_return_5d",),
        year=2025,
    )

    row = summary.iloc[0]
    assert row["year"] == 2025
    assert row["label"] == "forward_return_5d"
    assert row["rows"] == 3
    assert row["non_null_rows"] == 2
    assert row["eligible_rows"] == 2
    assert row["can_enter_rows"] == 2
    assert row["can_exit_rows"] == 2


def test_date_filtered_evaluation_frame_keeps_only_requested_trade_dates(
    tmp_path,
) -> None:
    path = tmp_path / "price_volume_factors_2024.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001"],
            "trade_date": pd.to_datetime(["2024-09-23", "2024-09-24", "2024-09-25"]),
            "eligible": [True, True, True],
            "can_enter": [True, True, True],
            "can_exit": [True, True, True],
            "rev_20d": [0.1, 0.2, 0.3],
            "forward_return_1d": [0.001, 0.002, 0.003],
            "forward_return_20d": [0.01, 0.02, 0.03],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    filtered = evaluation.read_evaluation_frame(
        path,
        feature_columns=("rev_20d",),
        label_columns=("forward_return_20d",),
        start_date="2024-09-24",
        end_date="2024-09-25",
    )

    assert filtered["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-09-24",
        "2024-09-25",
    ]
    assert "rev_20d" in filtered.columns
    assert "forward_return_20d" in filtered.columns
    assert "forward_return_1d" in filtered.columns


def test_diagnostics_enabled_evaluation_frame_reads_only_allowed_optional_context(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_read_factor_frame(path, *, columns=None, start_date=None, end_date=None):
        captured["path"] = path
        captured["columns"] = columns
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return pd.DataFrame(
            {
                "stock_code": ["000001"],
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "eligible": [True],
                "can_enter": [True],
                "can_exit": [True],
                "rev_20d": [0.1],
                "forward_return_1d": [0.001],
                "forward_return_20d": [0.01],
            }
        )

    monkeypatch.setattr(evaluation, "schema_columns", lambda path: (
        "stock_code",
        "trade_date",
        "eligible",
        "can_enter",
        "can_exit",
        "rev_20d",
        "forward_return_1d",
        "forward_return_20d",
        "amount",
        "turnover",
        "avg_amount_20d",
        "amihud_20d",
        "amplitude_20d",
        "total_market_cap",
        "float_market_cap",
        "market_type",
        "industry",
        "sector",
        "stock_name",
        "listing_date",
        "open",
        "close",
    ))
    monkeypatch.setattr(evaluation, "read_factor_frame", fake_read_factor_frame)

    evaluation.read_evaluation_frame(
        Path("factors.parquet"),
        feature_columns=("rev_20d",),
        label_columns=("forward_return_20d",),
        diagnostics_spec=DiagnosticsSpec.from_request({"enabled": True}),
    )

    columns = set(captured["columns"])
    assert {
        "amount",
        "turnover",
        "avg_amount_20d",
        "amihud_20d",
        "amplitude_20d",
        "total_market_cap",
        "float_market_cap",
        "market_type",
    }.issubset(columns)
    assert {
        "industry",
        "sector",
        "stock_name",
        "listing_date",
        "open",
        "close",
    }.isdisjoint(columns)


def test_read_factor_frame_pushes_trade_date_filters_to_parquet(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_read_parquet(path, *, columns=None, engine=None, filters=None):
        captured["path"] = path
        captured["columns"] = columns
        captured["engine"] = engine
        captured["filters"] = filters
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "rev_20d": [0.1],
            }
        )

    monkeypatch.setattr(evaluation.pd, "read_parquet", fake_read_parquet)

    frame = evaluation.read_factor_frame(
        Path("factors.parquet"),
        columns=("rev_20d",),
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert len(frame) == 1
    assert captured["columns"] == ["rev_20d", "trade_date"]
    assert captured["engine"] == "pyarrow"
    assert captured["filters"] == [
        ("trade_date", ">=", pd.Timestamp("2024-01-01").to_pydatetime()),
        ("trade_date", "<=", pd.Timestamp("2024-12-31").to_pydatetime()),
    ]


def test_evaluation_spec_uses_filtered_panel_without_input_parquet(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-09-24"]),
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "rev_20d": [0.2],
            "forward_return_1d": [0.002],
            "forward_return_20d": [0.02],
        }
    )

    spec = evaluation.make_a_share_evaluation_spec(
        panel=panel,
        start_year=2024,
        end_year=2024,
        label_columns=("forward_return_20d",),
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "run",
        run_id="run",
        n_groups=5,
        source_file="data/processed_private/csmar/example.parquet",
        start_date="2024-09-24",
        end_date="2024-09-24",
    )

    assert spec.panel is panel
    assert spec.metadata["start_date"] == "2024-09-24"
    assert spec.metadata["source_file"] == (
        "data/processed_private/csmar/example.parquet"
    )
    assert spec.feature_columns == ("rev_20d",)
    assert spec.label_columns == ("forward_return_20d",)
    assert spec.return_source_column == "forward_return_1d"
    assert spec.time_breakdowns == ("year",)
    assert spec.time_semantics_config.decision_time_column == "trade_date"
    assert spec.metadata["time_semantics"]["decision_time_column"] == "trade_date"
    assert spec.metadata["time_semantics"]["timezone"] == "Asia/Shanghai"


def test_a_share_diagnostics_enabled_adds_month_time_breakdown(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-09-24"]),
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "rev_20d": [0.2],
            "forward_return_1d": [0.002],
            "forward_return_20d": [0.02],
        }
    )

    disabled = evaluation.make_a_share_evaluation_spec(
        panel=panel,
        start_year=2024,
        end_year=2024,
        label_columns=("forward_return_20d",),
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "disabled",
        run_id="disabled",
        n_groups=5,
        source_file="data/processed_private/csmar/example.parquet",
    )
    enabled = evaluation.make_a_share_evaluation_spec(
        panel=panel,
        start_year=2024,
        end_year=2024,
        label_columns=("forward_return_20d",),
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "enabled",
        run_id="enabled",
        n_groups=5,
        source_file="data/processed_private/csmar/example.parquet",
        diagnostics_spec=DiagnosticsSpec.from_request({"enabled": True}),
    )

    assert disabled.time_breakdowns == ("year",)
    assert enabled.time_breakdowns == ("year", "month")


def test_evaluation_spec_rejects_non_forward_return_label_name(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-09-24"]),
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "rev_20d": [0.2],
            "future_return_20d": [0.02],
            "forward_return_1d": [0.002],
        }
    )

    with pytest.raises(ValueError, match="forward_return"):
        evaluation.make_a_share_evaluation_spec(
            panel=panel,
            start_year=2024,
            end_year=2024,
            label_columns=("future_return_20d",),
            feature_columns=("rev_20d",),
            output_dir=tmp_path / "reports" / "run",
            run_id="run",
            n_groups=5,
            source_file="data/processed_private/csmar/example.parquet",
        )


def test_evaluation_spec_exposes_evaluation_runtime_options(tmp_path) -> None:
    panel = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-09-24"]),
            "industry": ["bank"],
            "size_exposure": [1.0],
            "risk_20d": [0.2],
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "rev_20d": [0.2],
            "forward_return_1d": [0.002],
            "forward_return_5d": [0.01],
            "forward_return_20d": [0.02],
        }
    )

    spec = evaluation.make_a_share_evaluation_spec(
        panel=panel,
        start_year=2024,
        end_year=2024,
        label_columns=("forward_return_20d",),
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "run",
        run_id="run",
        n_groups=5,
        source_file="data/processed_private/csmar/example.parquet",
        return_source_column="forward_return_5d",
        cost_bps=3.5,
        portfolio_mode="long_only",
        kernel="tensor",
        segment_columns=("industry",),
        risk_column="risk_20d",
        neutralization_method="cross_sectional_ols",
        neutralization_group_columns=(),
        neutralization_exposure_columns=("size_exposure",),
        diagnostics_spec=DiagnosticsSpec.from_request(
            {
                "enabled": True,
                "materialized_views": ["scenario_metrics"],
            }
        ),
    )

    assert spec.return_source_column == "forward_return_5d"
    assert spec.weight_configs[0].cost_bps == 3.5
    assert spec.weight_configs[0].portfolio_mode == "long_only"
    assert spec.weight_configs[0].risk_column == "risk_20d"
    assert spec.metadata["portfolio_mode"] == "long_only"
    assert spec.kernel == "tensor"
    assert spec.segment_columns == ("industry",)
    assert spec.neutralization_config.method == "cross_sectional_ols"
    assert spec.neutralization_config.exposure_columns == ("size_exposure",)
    assert spec.diagnostics_spec.enabled is True
    assert spec.diagnostics_spec.materialized_views == ("scenario_metrics",)


def test_label_availability_frame_applies_date_filter(tmp_path) -> None:
    path = tmp_path / "price_volume_factors_2024.parquet"
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-09-24"]),
            "eligible": [True, False],
            "can_enter": [True, False],
            "can_exit": [True, False],
            "forward_return_20d": [0.01, None],
        }
    ).to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    frame = evaluation.read_label_availability_frame(
        path,
        label_columns=("forward_return_20d",),
        start_date="2024-09-24",
        end_date="2024-09-24",
    )

    assert len(frame) == 1
    assert bool(frame.iloc[0]["eligible"]) is False


def test_processed_factor_evaluation_reads_full_factor_file_once(
    tmp_path,
) -> None:
    processed_root = tmp_path / "processed"
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    factor_path = factor_dir / "price_volume_factors_2014_2025.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002", "000001", "000002"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2025-01-02", "2025-01-02"]
            ),
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "rev_20d": [0.1, 0.2, 0.3, 0.4],
            "forward_return_1d": [0.001, 0.002, 0.003, 0.004],
            "forward_return_20d": [0.01, 0.02, 0.03, 0.04],
        }
    ).to_parquet(factor_path, engine="pyarrow", compression="zstd", index=False)
    report_root = tmp_path / "factor_evaluation" / "full_2024_2025"

    result = evaluation.run_processed_factor_evaluation(
        processed_root=processed_root,
        report_root=report_root,
        start_year=2024,
        end_year=2025,
        requested_labels=("forward_return_20d",),
        requested_factors=("rev_20d",),
        all_factors=False,
        n_groups=2,
    )
    label_summary, manifest_path, run_records = result

    assert sorted(label_summary["year"].unique().tolist()) == [2024, 2025]
    assert run_records[0]["run_id"] == "full_2024_2025"
    assert run_records[0]["labels"] == ["forward_return_20d"]
    assert run_records[0]["features"] == ["rev_20d"]
    assert run_records[0]["portfolio_modes"] == ["long_short"]
    assert run_records[0]["kernels"] == ["dataframe"]
    assert run_records[0]["output_dir"] == "."
    assert "input_file" not in run_records[0]
    assert run_records[0]["input_source"] == "processed_factor_table"
    assert run_records[0]["input_year_range"] == "2014_2025"
    assert manifest_path.name == "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combinations = pd.read_csv(report_root / "dimensions" / "combinations.csv")
    assert combinations.columns.tolist() == [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    ]
    manifest_text = json.dumps(manifest)
    assert not (report_root / "config.json").exists()
    assert not (report_root / "lineage.json").exists()
    run_context = json.loads(
        (report_root / "run_context.json").read_text(encoding="utf-8")
    )
    run_context_text = json.dumps(run_context)
    assert "data/processed_private" not in manifest_text
    assert factor_path.as_posix() not in manifest_text
    assert factor_path.as_posix() not in run_context_text
    assert "source_alias" not in run_context_text
    assert "evaluation_id" not in run_context_text
    assert run_context["provenance"]["source_file"] == factor_path.name
    assert not (report_root / "label_availability.csv").exists()
    assert (report_root / "input_quality" / "coverage_summary.csv").is_file()


def test_processed_factor_evaluation_combined_summary_counts_all_payload_tables(
    tmp_path,
) -> None:
    processed_root = tmp_path / "processed"
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "rev_20d": [0.1, 0.2],
            "forward_return_1d": [0.001, 0.002],
        }
    ).to_parquet(
        factor_dir / "price_volume_factors_2024_2024.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    report_root = tmp_path / "factor_evaluation" / "multi_mode_2024"

    _, manifest_path, _ = evaluation.run_processed_factor_evaluation(
        processed_root=processed_root,
        report_root=report_root,
        start_year=2024,
        end_year=2024,
        requested_labels=("forward_return_1d",),
        requested_factors=("rev_20d",),
        all_factors=False,
        n_groups=2,
        portfolio_modes=("long_short", "long_only"),
    )

    summary = json.loads((manifest_path.parent / "summary.json").read_text())
    run_context = json.loads((manifest_path.parent / "run_context.json").read_text())
    config = run_context["config"]["evaluation_spec"]
    assert summary["n_combinations"] == 2
    assert summary["tables"]["combinations"] == 2
    assert summary["tables"]["returns"] == len(
        pd.read_csv(report_root / "portfolio" / "returns.csv")
    )
    assert summary["tables"]["evaluation_metrics_summary"] == len(
        pd.read_csv(report_root / "metrics" / "evaluation_metrics_summary.csv")
    )
    assert config["portfolio_modes"] == ["long_short", "long_only"]
    assert "portfolio_mode" not in config["metadata"]
    assert "portfolio_mode" not in config["weight_configs"][0]


def test_processed_factor_evaluation_artifact_uses_full_report_identity(
    tmp_path,
) -> None:
    processed_root = tmp_path / "processed"
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "rev_20d": [0.1, 0.2],
            "forward_return_1d": [0.001, 0.002],
        }
    ).to_parquet(
        factor_dir / "price_volume_factors_2024_2024.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    _, manifest_path, _ = evaluation.run_processed_factor_evaluation(
        processed_root=processed_root,
        report_root=tmp_path / "factor_evaluation" / "full_2024_2024",
        start_year=2024,
        end_year=2024,
        requested_labels=("forward_return_1d",),
        requested_factors=("rev_20d",),
        all_factors=False,
        n_groups=2,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "full_2024_2024"
    assert "evaluation_id" not in manifest
    run_context = json.loads((manifest_path.parent / "run_context.json").read_text())
    run_context_text = json.dumps(run_context)
    assert run_context["provenance"]["run_id"] == "full_2024_2024"
    assert "source_alias" not in run_context_text
    assert "evaluation_id" not in run_context_text


def test_processed_factor_evaluation_artifact_preserves_pipeline_identity(
    tmp_path,
) -> None:
    processed_root = tmp_path / "processed"
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "rev_20d": [0.1, 0.2],
            "forward_return_1d": [0.001, 0.002],
        }
    ).to_parquet(
        factor_dir / "price_volume_factors_2024_2024.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    _, manifest_path, _ = evaluation.run_processed_factor_evaluation(
        processed_root=processed_root,
        report_root=tmp_path / "pipeline_run" / "evaluation",
        start_year=2024,
        end_year=2024,
        requested_labels=("forward_return_1d",),
        requested_factors=("rev_20d",),
        all_factors=False,
        n_groups=2,
        source_alias="pipeline",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "pipeline_run"
    assert "evaluation_id" not in manifest
    run_context = json.loads((manifest_path.parent / "run_context.json").read_text())
    run_context_text = json.dumps(run_context)
    assert run_context["provenance"]["run_id"] == "pipeline_run"
    assert "source_alias" not in run_context_text
    assert "evaluation_id" not in run_context_text


def test_run_factor_evaluation_cli_passes_evaluation_runtime_options(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_a_share_evaluation(request):
        captured.update(request.__dict__)
        return SimpleNamespace(manifest_path=tmp_path / "manifest.json", run_records=[])

    monkeypatch.setattr(
        run_factor_evaluation,
        "run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )

    exit_code = run_factor_evaluation.main(
        [
            "--processed-root",
            str(tmp_path / "processed"),
            "--report-root",
            str(tmp_path / "reports"),
            "--start-year",
            "2024",
            "--end-year",
            "2025",
            "--labels",
            "forward_return_20d",
            "--features",
            "rev_20d",
            "--return-source-column",
            "forward_return_5d",
            "--cost-bps",
            "2.5",
            "--portfolio-modes",
            "long_only",
            "--kernels",
            "tensor",
            "--segment-columns",
            "industry,market",
            "--risk-column",
            "risk_20d",
            "--neutralization-method",
            "cross_sectional_ols",
            "--neutralization-group-columns",
            "industry",
            "--neutralization-exposure-columns",
            "ln_mktcap,beta",
        ]
    )

    assert exit_code == 0
    assert captured["return_source_column"] == "forward_return_5d"
    assert captured["cost_bps"] == 2.5
    assert captured["portfolio_modes"] == ("long_only",)
    assert captured["kernels"] == ("tensor",)
    assert captured["segment_columns"] == ("industry", "market")
    assert captured["risk_column"] == "risk_20d"
    assert captured["neutralization_method"] == "cross_sectional_ols"
    assert captured["neutralization_group_columns"] == ("industry",)
    assert captured["neutralization_exposure_columns"] == ("ln_mktcap", "beta")


def test_run_factor_evaluation_cli_rejects_invalid_kernel_before_io() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_factor_evaluation.main(["--features", "rev_20d", "--kernels", "bad"])

    assert exc_info.value.code == 2


def test_run_factor_evaluation_dry_run_checks_date_coverage_without_artifact(
    tmp_path,
    capsys,
) -> None:
    input_file = tmp_path / "factors.parquet"
    report_root = tmp_path / "reports"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "rev_20d": [0.1, 0.2],
            "forward_return_1d": [0.001, 0.002],
            "forward_return_20d": [0.01, 0.02],
        }
    ).to_parquet(input_file, engine="pyarrow", compression="zstd", index=False)

    exit_code = run_factor_evaluation.main(
        [
            "--input-file",
            str(input_file),
            "--report-root",
            str(report_root),
            "--start-year",
            "2024",
            "--end-year",
            "2024",
            "--start-date",
            "2024-01-01",
            "--end-date",
            "2024-01-31",
            "--features",
            "rev_20d",
            "--labels",
            "forward_return_20d",
            "--kernels",
            "dataframe",
            "tensor",
            "--portfolio-modes",
            "long_short",
            "long_only",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "date_coverage=2024-01-01..2024-01-31" in output
    assert "rows_in_range=2" in output
    assert "combinations=4" in output
    assert not (report_root / "manifest.json").exists()


def test_run_factor_evaluation_dry_run_rejects_missing_date_coverage(
    tmp_path,
    capsys,
) -> None:
    input_file = tmp_path / "factors.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "eligible": [True],
            "can_enter": [True],
            "can_exit": [True],
            "rev_20d": [0.1],
            "forward_return_1d": [0.001],
            "forward_return_20d": [0.01],
        }
    ).to_parquet(input_file, engine="pyarrow", compression="zstd", index=False)

    exit_code = run_factor_evaluation.main(
        [
            "--input-file",
            str(input_file),
            "--start-year",
            "2024",
            "--end-year",
            "2024",
            "--start-date",
            "2024-02-01",
            "--end-date",
            "2024-02-29",
            "--features",
            "rev_20d",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    assert "no rows in requested date coverage" in capsys.readouterr().out


def test_processed_factor_evaluation_rejects_invalid_options_before_parquet_io(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_if_parquet_path_is_resolved(*args, **kwargs):
        raise AssertionError("invalid options must fail before private parquet IO")

    monkeypatch.setattr(
        evaluation,
        "factor_full_path_covering_range",
        fail_if_parquet_path_is_resolved,
    )

    with pytest.raises(ValueError, match="kernel"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            kernels=("bad",),
        )

    with pytest.raises(ValueError, match="portfolio_mode"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            portfolio_modes=("bad",),
        )

    with pytest.raises(ValueError, match="unsafe label"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d/evil",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
        )

    with pytest.raises(ValueError, match="n_groups"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=1,
        )

    with pytest.raises(ValueError, match="cost_bps"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            cost_bps=-0.1,
        )

    with pytest.raises(ValueError, match="start_year"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2025,
            end_year=2024,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
        )

    with pytest.raises(ValueError, match="start_date"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            start_date="2025-01-01",
            end_date="2024-12-31",
        )

    with pytest.raises(ValueError, match="within start_year/end_year"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2024,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            start_date="2023-12-29",
            end_date="2024-12-31",
        )


def test_run_factor_evaluation_cli_defaults_to_full_report_root(
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_a_share_evaluation(request):
        captured.update(request.__dict__)
        return SimpleNamespace(manifest_path="manifest.json", run_records=[])

    monkeypatch.setattr(
        run_factor_evaluation,
        "run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )

    exit_code = run_factor_evaluation.main(["--features", "rev_20d"])

    assert exit_code == 0
    assert captured["report_root"].as_posix() == (
        "reports/private_real/factor_evaluation/full_2024_2025"
    )
    assert captured["portfolio_modes"] == ("long_short",)
    assert captured["kernels"] == ("dataframe",)


def test_run_factor_evaluation_cli_derives_default_report_root_from_years(
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_a_share_evaluation(request):
        captured.update(request.__dict__)
        return SimpleNamespace(manifest_path="manifest.json", run_records=[])

    monkeypatch.setattr(
        run_factor_evaluation,
        "run_a_share_evaluation",
        fake_run_a_share_evaluation,
    )

    exit_code = run_factor_evaluation.main(
        ["--start-year", "2023", "--end-year", "2025", "--features", "rev_20d"]
    )

    assert exit_code == 0
    assert captured["report_root"].as_posix() == (
        "reports/private_real/factor_evaluation/full_2023_2025"
    )
