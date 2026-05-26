from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.artifact_schema import (
    EVALUATION_SET_REQUIRED_TABLES,
)
from quant_research_platform.artifacts.artifact_writer import (
    write_evaluation_set_artifacts,
)


def test_schema_06_contract_names_required_first_stage_tables() -> None:
    assert ARTIFACT_SCHEMA_VERSION == "0.6"
    assert set(EVALUATION_SET_REQUIRED_TABLES) == {
        "combinations",
        "ic",
        "ic_summary",
        "grouped_returns",
        "group_spread_summary",
        "performance_summary",
        "returns",
        "cumulative_returns",
        "weights",
        "tradability_diagnostics",
        "label_availability",
        "factor_coverage",
        "tradability_coverage",
        "evaluation_summary",
    }
    assert "pnl" not in EVALUATION_SET_REQUIRED_TABLES
    assert "positions" not in EVALUATION_SET_REQUIRED_TABLES
    assert "turnover" not in EVALUATION_SET_REQUIRED_TABLES


def test_writer_persists_schema_06_star_artifact_with_parquet_weights(tmp_path) -> None:
    output_dir = tmp_path / "n7_eval"

    write_evaluation_set_artifacts(
        output_dir,
        config={
            "metadata": _metadata(),
            "source_file": (
                "E:/quant-workspace/projects/quant-research-platform/"
                "data/processed_private/csmar/factors.parquet"
            ),
            "nested": {
                "path": "C:/Users/DELL/private/factors.parquet",
                "input_path": Path("data/processed_private/csmar/factors.parquet"),
                "public_label": "safe",
            },
            "cost_config": {"cost_bps": 0.0},
            "weight_config": {"method": "equal_weight_quantile"},
            "benchmark_profiles": [],
            "universe_policy": "fixture",
            "eligibility_policy": "fixture",
            "tradability_policy": "fixture",
        },
        lineage={
            "input_file": "fixture.parquet",
            "input_reference": "fixture",
            "input_fingerprint": "sha256:fixture",
            "source_file": "data/processed_private/csmar/factors.parquet",
            "nested": {
                "path": "C:/Users/DELL/private/factors.parquet",
                "input_path": Path("data/processed_private/csmar/factors.parquet"),
                "public_label": "safe",
            },
            "actual_date_coverage": {
                "start_date": "2026-01-01",
                "end_date": "2026-01-02",
            },
            "artifact_schema_version": "0.6",
        },
        summary={
            **_metadata(),
            "evaluation_id": "n7_eval",
            "n_rows_evaluated": 8,
            "n_combinations": 1,
            "schema": {
                "timestamp_column": "timestamp",
                "asset_column": "asset",
                "feature_columns": ["factor"],
                "label_columns": ["forward_return"],
                "return_source_column": "forward_return",
            },
            "kernel": "dataframe",
            "segment_columns": [],
            "breakdown_enabled": False,
            "breakdown_columns": [],
            "period_breakdown_policy": "disabled",
        },
        tables=_schema_06_tables(),
        report="# N7 fixture\n",
    )

    assert (output_dir / "lineage.json").is_file()
    assert (output_dir / "dimensions" / "combinations.csv").is_file()
    assert (output_dir / "portfolio" / "weights.parquet").is_file()
    assert (output_dir / "portfolio" / "cumulative_returns.csv").is_file()
    assert not (output_dir / "portfolio" / "pnl.csv").exists()
    assert not (output_dir / "portfolio" / "positions.csv").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    lineage = json.loads((output_dir / "lineage.json").read_text(encoding="utf-8"))
    combinations = pd.read_csv(output_dir / "dimensions" / "combinations.csv")
    weights = pd.read_parquet(output_dir / "portfolio" / "weights.parquet")

    assert manifest["artifact_schema_version"] == "0.6"
    assert summary["artifact_schema_version"] == "0.6"
    assert manifest["kernel"] == "dataframe"
    assert "engine" not in manifest
    assert "engine" not in summary
    assert config["cost_config"] == {"cost_bps": 0.0}
    assert config["benchmark_profiles"] == []
    assert config["source_file"] == "[redacted-path]"
    assert config["nested"]["path"] == "[redacted-path]"
    assert config["nested"]["input_path"] == "[redacted-path]"
    assert config["nested"]["public_label"] == "safe"
    assert "input_file" not in lineage
    assert lineage["input_reference"] == "fixture"
    assert lineage["source_file"] == "[redacted-path]"
    assert lineage["nested"]["path"] == "[redacted-path]"
    assert lineage["nested"]["input_path"] == "[redacted-path]"
    assert lineage["nested"]["public_label"] == "safe"
    assert list(combinations.columns) == [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    ]
    assert list(weights.columns) == [
        "combination_id",
        "timestamp",
        "asset",
        "target_weight",
    ]

    display = manifest["display"]
    assert display["schema_version"] == "0.6"
    assert display["dimension_source"] == "dimensions/combinations.csv"
    assert display["default_selection"] == {
        "feature": "factor",
        "label": "forward_return",
        "kernel": "dataframe",
        "portfolio_mode": "long_only",
    }
    display_tables = {table["key"]: table for table in display["tables"]}
    assert display_tables["weights"]["format"] == "parquet"
    assert display_tables["weights"]["path"] == "portfolio/weights.parquet"
    assert display_tables["weights"]["join_key"] == "combination_id"
    assert display_tables["weights"]["preview_rows"] == 20
    assert display_tables["weights"]["supports_combination_filter"] is True
    assert display_tables["weights"]["preview"] == {
        "kind": "table",
        "table_key": "weights",
        "default_limit": 20,
        "max_limit": 1000,
        "supports_offset": True,
        "supports_combination_filter": True,
        "filter_parameter": "combination_id",
    }
    assert display_tables["label_availability"]["join_key"] is None
    assert display_tables["label_availability"]["supports_combination_filter"] is False
    assert display_tables["label_availability"]["preview"] == {
        "kind": "table",
        "table_key": "label_availability",
        "default_limit": 20,
        "max_limit": 1000,
        "supports_offset": True,
        "supports_combination_filter": False,
        "filter_parameter": None,
    }


def _metadata() -> dict[str, object]:
    return {
        "run_id": "n7_eval",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "evaluation_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _schema_06_tables() -> dict[str, pd.DataFrame]:
    dimensions = {
        "combination_id": ["factor__forward_return__dataframe__long_only"],
        "feature": ["factor"],
        "label": ["forward_return"],
        "kernel": ["dataframe"],
        "portfolio_mode": ["long_only"],
    }
    combination_id = dimensions["combination_id"][0]
    timestamp = "2026-01-01"
    return {
        "combinations": pd.DataFrame(dimensions),
        "ic": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "ic": [0.1],
                "rankic": [0.2],
                "n_obs": [4],
            }
        ),
        "ic_summary": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "mean_ic": [0.1],
                "std_ic": [0.0],
                "icir": [0.0],
                "positive_ic_rate": [1.0],
                "mean_rankic": [0.2],
                "std_rankic": [0.0],
                "rankicir": [0.0],
                "positive_rankic_rate": [1.0],
            }
        ),
        "grouped_returns": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "group": [1],
                "mean_forward_return": [0.01],
                "n_obs": [2],
            }
        ),
        "group_spread_summary": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "top_group_mean_return": [0.02],
                "bottom_group_mean_return": [0.01],
                "top_bottom_spread_mean": [0.01],
                "top_bottom_spread_hit_rate": [1.0],
                "group_monotonicity_score": [1.0],
            }
        ),
        "performance_summary": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "periods": [1],
                "final_cumulative_gross_return": [0.01],
                "final_cumulative_net_return": [0.01],
                "mean_net_portfolio_return": [0.01],
                "mean_turnover": [0.0],
                "max_drawdown": [0.0],
                "sharpe_like": [0.0],
                "sortino_like": [0.0],
                "total_cost": [0.0],
                "win_rate": [1.0],
            }
        ),
        "returns": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "portfolio_mode": ["long_only"],
                "long_return": [0.01],
                "short_return": [0.0],
                "gross_portfolio_return": [0.01],
                "turnover": [0.0],
                "cost": [0.0],
                "net_portfolio_return": [0.01],
                "long_count": [2],
                "short_count": [0],
                "return_source_column": ["forward_return"],
            }
        ),
        "cumulative_returns": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "gross_return": [0.01],
                "net_return": [0.01],
                "cost": [0.0],
                "cumulative_gross_return": [0.01],
                "cumulative_net_return": [0.01],
                "cumulative_cost": [0.0],
                "high_water_mark": [0.01],
                "drawdown": [0.0],
            }
        ),
        "weights": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "asset": ["asset_001"],
                "target_weight": [1.0],
            }
        ),
        "tradability_diagnostics": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "timestamp": [timestamp],
                "method": ["equal_weight_quantile"],
                "direction": ["long_high"],
                "universe_count": [4],
                "eligible_count": [4],
                "can_enter_count": [4],
                "can_exit_count": [4],
                "entry_blocked_count": [0],
                "exit_blocked_count": [0],
                "eligible_rate": [1.0],
                "can_enter_rate": [1.0],
                "can_exit_rate": [1.0],
                "entry_blocked_rate": [0.0],
                "exit_blocked_rate": [0.0],
                "fallback_reason": [""],
            }
        ),
        "label_availability": pd.DataFrame(
            {
                "label": ["forward_return"],
                "available_rows": [8],
                "missing_rows": [0],
                "coverage_rate": [1.0],
            }
        ),
        "factor_coverage": pd.DataFrame(
            {
                "feature": ["factor"],
                "available_rows": [8],
                "missing_rows": [0],
                "coverage_rate": [1.0],
            }
        ),
        "tradability_coverage": pd.DataFrame(
            {
                "field": ["can_enter"],
                "available_rows": [8],
                "missing_rows": [0],
                "coverage_rate": [1.0],
            }
        ),
        "evaluation_summary": pd.DataFrame(
            {
                "combination_id": [combination_id],
                "feature": ["factor"],
                "label": ["forward_return"],
                "kernel": ["dataframe"],
                "portfolio_mode": ["long_only"],
                "mean_ic": [0.1],
                "final_cumulative_net_return": [0.01],
                "mean_turnover": [0.0],
                "total_cost": [0.0],
            }
        ),
    }
