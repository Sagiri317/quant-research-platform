from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.a_share.evaluation import core_bridge as evaluation
from quant_research_platform.markets.a_share.evaluation import run_factor_evaluation


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
            "trade_date": pd.to_datetime(
                ["2024-09-23", "2024-09-24", "2024-09-25"]
            ),
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
        label_column="forward_return_20d",
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
        label_column="forward_return_20d",
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "core_runs" / "run",
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
    assert spec.pnl_return_column == "forward_return_1d"
    assert spec.time_breakdowns == ("year",)


def test_evaluation_spec_exposes_core_runtime_options(tmp_path) -> None:
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
        label_column="forward_return_20d",
        feature_columns=("rev_20d",),
        output_dir=tmp_path / "reports" / "core_runs" / "run",
        run_id="run",
        n_groups=5,
        source_file="data/processed_private/csmar/example.parquet",
        pnl_return_column="forward_return_5d",
        cost_bps=3.5,
        portfolio_mode="long_only",
        engine="tensor",
        segment_columns=("industry",),
        risk_column="risk_20d",
        neutralization_method="cross_sectional_ols",
        neutralization_group_columns=(),
        neutralization_exposure_columns=("size_exposure",),
    )

    assert spec.pnl_return_column == "forward_return_5d"
    assert spec.weight_configs[0].cost_bps == 3.5
    assert spec.weight_configs[0].portfolio_mode == "long_only"
    assert spec.weight_configs[0].risk_column == "risk_20d"
    assert spec.metadata["portfolio_mode"] == "long_only"
    assert spec.engine == "tensor"
    assert spec.segment_columns == ("industry",)
    assert spec.neutralization_config.method == "cross_sectional_ols"
    assert spec.neutralization_config.exposure_columns == ("size_exposure",)


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
    monkeypatch,
) -> None:
    processed_root = tmp_path / "processed"
    factor_dir = processed_root / "price_volume_factors_full"
    factor_dir.mkdir(parents=True)
    factor_path = factor_dir / "price_volume_factors_2014_2025.parquet"
    pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001"],
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2025-01-02", "2025-01-03"]
            ),
            "eligible": [True, True, False],
            "can_enter": [True, True, False],
            "can_exit": [True, True, False],
            "rev_20d": [0.1, 0.2, 0.3],
            "forward_return_1d": [0.001, 0.002, 0.003],
            "forward_return_20d": [0.01, 0.02, 0.03],
        }
    ).to_parquet(factor_path, engine="pyarrow", compression="zstd", index=False)
    captured_specs = []

    def fake_run_evaluation(spec):
        captured_specs.append(spec)

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run_evaluation)

    result = evaluation.run_processed_factor_evaluation(
        processed_root=processed_root,
        report_root=tmp_path / "reports",
        start_year=2024,
        end_year=2025,
        requested_labels=("forward_return_20d",),
        requested_factors=("rev_20d",),
        all_factors=False,
        n_groups=5,
    )
    label_summary, manifest_path, core_runs = result

    assert len(captured_specs) == 1
    assert captured_specs[0].metadata["start_year"] == 2024
    assert captured_specs[0].metadata["end_year"] == 2025
    assert captured_specs[0].metadata["source_file"] == factor_path.as_posix()
    assert captured_specs[0].metadata["portfolio_mode"] == "long_short"
    assert captured_specs[0].weight_configs[0].portfolio_mode == "long_short"
    assert captured_specs[0].panel["trade_date"].dt.year.tolist() == [
        2024,
        2025,
        2025,
    ]
    assert sorted(label_summary["year"].unique().tolist()) == [2024, 2025]
    assert core_runs[0]["run_id"] == "csmar_2024_2025_forward_return_20d"
    assert core_runs[0]["portfolio_mode"] == "long_short"
    assert core_runs[0]["output_dir"] == (
        "core_runs/csmar_2024_2025_forward_return_20d"
    )
    assert manifest_path.name == "evaluation_manifest.json"


def test_run_label_evaluation_rejects_unsafe_label_in_run_id(tmp_path):
    panel = pd.DataFrame(
        {
            "stock_code": ["000001", "000002"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
            "rev_20d": [0.1, 0.2],
            "forward_return_1d": [0.001, 0.002],
            "forward_return_20d/evil": [0.01, 0.02],
        }
    )

    with pytest.raises(ValueError, match="unsafe core run_id"):
        evaluation.run_label_evaluation(
            panel=panel,
            source_file="factor.parquet",
            start_year=2024,
            end_year=2024,
            label_column="forward_return_20d/evil",
            feature_columns=("rev_20d",),
            output_root=tmp_path / "reports",
            n_groups=2,
        )


def test_run_factor_evaluation_cli_passes_core_runtime_options(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_processed_factor_evaluation(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(), tmp_path / "evaluation_manifest.json", []

    monkeypatch.setattr(
        run_factor_evaluation,
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
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
            "--factors",
            "rev_20d",
            "--pnl-return-column",
            "forward_return_5d",
            "--cost-bps",
            "2.5",
            "--portfolio-mode",
            "long_only",
            "--engine",
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
    assert captured["pnl_return_column"] == "forward_return_5d"
    assert captured["cost_bps"] == 2.5
    assert captured["portfolio_mode"] == "long_only"
    assert captured["engine"] == "tensor"
    assert captured["segment_columns"] == ("industry", "market")
    assert captured["risk_column"] == "risk_20d"
    assert captured["neutralization_method"] == "cross_sectional_ols"
    assert captured["neutralization_group_columns"] == ("industry",)
    assert captured["neutralization_exposure_columns"] == ("ln_mktcap", "beta")


def test_run_factor_evaluation_cli_rejects_invalid_engine_before_io() -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_factor_evaluation.main(["--engine", "bad"])

    assert exc_info.value.code == 2


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

    with pytest.raises(ValueError, match="engine"):
        evaluation.run_processed_factor_evaluation(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "reports",
            start_year=2024,
            end_year=2025,
            requested_labels=("forward_return_20d",),
            requested_factors=("rev_20d",),
            all_factors=False,
            n_groups=5,
            engine="bad",
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
            portfolio_mode="bad",
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


def test_run_factor_evaluation_cli_defaults_to_full_report_root(
    monkeypatch,
) -> None:
    captured = {}

    def fake_run_processed_factor_evaluation(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(), "manifest.json", []

    monkeypatch.setattr(
        run_factor_evaluation,
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )

    exit_code = run_factor_evaluation.main([])

    assert exit_code == 0
    assert captured["report_root"].as_posix() == (
        "reports/private_real/factor_evaluation/full_2024_2025"
    )
    assert captured["portfolio_mode"] == "long_short"
