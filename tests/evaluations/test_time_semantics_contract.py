from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.evaluations import EvaluationSpec
from quant_research_platform.evaluations.evaluation_spec import TimeSemanticsConfig


def test_time_semantics_requires_declared_columns(tmp_path):
    with pytest.raises(ValueError, match="decision_time_column"):
        EvaluationSpec(
            panel=_panel().drop(columns=["decision_time"]),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("missing_decision_time"),
            output_dir=tmp_path,
            time_semantics_config=TimeSemanticsConfig(
                decision_time_column="decision_time",
                label_time_columns={"forward_return": "label_time"},
            ),
        )


def test_time_semantics_rejects_label_time_before_decision_time(tmp_path):
    panel = _panel()
    panel.loc[0, "label_time"] = pd.Timestamp("2025-12-31")

    with pytest.raises(ValueError, match="label_time"):
        EvaluationSpec(
            panel=panel,
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("bad_label_time"),
            output_dir=tmp_path,
            time_semantics_config=TimeSemanticsConfig(
                decision_time_column="decision_time",
                label_time_columns={"forward_return": "label_time"},
            ),
        )


def test_time_semantics_rejects_as_of_time_after_decision_time(tmp_path):
    panel = _panel()
    panel.loc[0, "factor_as_of_time"] = pd.Timestamp("2026-01-03")

    with pytest.raises(ValueError, match="as_of_time"):
        EvaluationSpec(
            panel=panel,
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata=_metadata("bad_as_of_time"),
            output_dir=tmp_path,
            time_semantics_config=TimeSemanticsConfig(
                decision_time_column="decision_time",
                label_time_columns={"forward_return": "label_time"},
                as_of_time_columns={"factor": "factor_as_of_time"},
            ),
        )


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
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "asset": ["A", "A"],
            "decision_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "label_time": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "factor_as_of_time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "factor": [0.1, 0.2],
            "forward_return": [0.01, 0.02],
        }
    )
