from __future__ import annotations

import json

import pandas as pd

from quant_research_platform.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    load_evaluation_set,
)
from quant_research_platform.evaluations import (
    EvaluationSpec,
    WeightConfig,
    run_evaluation,
)


def test_run_evaluation_writes_batch_native_evaluation_set(tmp_path):
    spec = EvaluationSpec(
        panel=_panel(),
        timestamp_column="timestamp",
        asset_column="asset",
        feature_columns=("factor_a", "factor_b"),
        label_columns=("ret_5d", "ret_20d"),
        return_source_column="forward_return_1d",
        metadata={
            "evaluation_id": "evaluation_demo",
            "project": "quant-research-platform",
            "kind": "factor",
            "domain": "evaluation_synthetic",
            "data_stage": "synthetic_only",
            "uses_external_data": False,
        },
        output_dir=tmp_path / "evaluation_demo",
        weight_configs=(
            WeightConfig(
                portfolio_mode="long_only",
                top_quantile=0.25,
                bottom_quantile=0.25,
            ),
        ),
        n_groups=2,
    )

    result = run_evaluation(spec)

    output_dir = tmp_path / "evaluation_demo"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    combinations = pd.read_csv(output_dir / "dimensions" / "combinations.csv")
    returns = pd.read_csv(output_dir / "portfolio" / "returns.csv")
    cumulative_returns = pd.read_csv(
        output_dir / "portfolio" / "cumulative_returns.csv"
    )
    weights = pd.read_parquet(output_dir / "portfolio" / "weights.parquet")
    ic = pd.read_csv(output_dir / "metrics" / "ic.csv")

    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["artifact_type"] == "evaluation_set"
    assert summary["artifact_type"] == "evaluation_set"
    assert summary["n_combinations"] == 4
    assert manifest["kernel"] == "dataframe"
    assert "engine" not in manifest
    assert result.combinations_path == output_dir / "dimensions" / "combinations.csv"
    assert not (output_dir / "portfolio" / "long_short.csv").exists()
    assert not (output_dir / "portfolio" / "pnl.csv").exists()
    assert not (output_dir / "portfolio" / "positions.csv").exists()
    assert len(combinations) == 4
    assert list(combinations.columns) == [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    ]
    assert set(combinations["portfolio_mode"]) == {"long_only"}
    assert set(combinations["kernel"]) == {"dataframe"}
    assert set(combinations["feature"]) == {"factor_a", "factor_b"}
    assert set(combinations["label"]) == {"ret_5d", "ret_20d"}
    assert {
        "gross_portfolio_return",
        "net_portfolio_return",
        "short_count",
    }.issubset(returns.columns)
    assert returns["short_count"].eq(0).all()
    assert "return_source_column" in returns.columns
    assert "target_weight" in weights.columns
    for frame in (returns, cumulative_returns, weights, ic):
        assert "combination_id" in frame.columns
        assert "weight_id" not in frame.columns
        assert "cost_id" not in frame.columns
    loaded = load_evaluation_set(output_dir)
    assert loaded.summary["evaluation_id"] == "evaluation_demo"
    assert set(loaded.tables) >= {
        "combinations",
        "returns",
        "cumulative_returns",
        "weights",
        "ic",
    }
    assert "long_short" not in loaded.tables


def test_public_api_does_not_export_single_evaluation_contracts():
    import quant_research_platform.evaluations as quant_research_platform

    assert not hasattr(quant_research_platform, "EvaluationRequest")
    assert not hasattr(quant_research_platform, "run_factor_evaluation")
    assert not hasattr(quant_research_platform, "BatchEvaluationSpec")
    assert not hasattr(quant_research_platform, "run_batch_factor_evaluations")


def _panel() -> pd.DataFrame:
    rows = []
    for timestamp in pd.date_range("2026-01-01", periods=4):
        for offset, asset in enumerate(("A", "B", "C", "D")):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "factor_a": offset + 0.1,
                    "factor_b": 4 - offset,
                    "forward_return_1d": (offset - 1.5) / 100,
                    "ret_5d": (1.5 - offset) / 50,
                    "ret_20d": (offset - 1.5) / 25,
                }
            )
    return pd.DataFrame(rows)
