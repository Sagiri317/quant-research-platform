from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.evaluations import EvaluationSpec, WeightConfig
from quant_research_platform.kernels.kernel_protocol import CostConfig


def test_evaluation_spec_rejects_multiple_weight_configs_with_schema_07_wording(
    tmp_path,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        EvaluationSpec(
            panel=_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("multi_weight"),
            weight_configs=(WeightConfig(), WeightConfig(portfolio_mode="long_only")),
            output_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "schema 0.7" in message
    assert "schema 0.6" not in message


def test_evaluation_spec_rejects_multiple_cost_configs_with_run_context_wording(
    tmp_path,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        EvaluationSpec(
            panel=_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("multi_cost"),
            cost_configs=(CostConfig(slippage_bps=1), CostConfig(slippage_bps=2)),
            output_dir=tmp_path,
        )

    message = str(exc_info.value)
    assert "schema 0.7" in message
    assert "run_context.json" in message
    assert "schema 0.6" not in message
    assert "config.json/lineage.json" not in message


def _metadata(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "evaluation_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "asset": ["A", "B"],
            "factor": [0.1, 0.2],
            "forward_return": [0.01, -0.02],
            "forward_return_1d": [0.01, -0.02],
        }
    )
