"""Training feedback uses native causal formulas, maturity and independent ranks."""

import pandas as pd
import pytest

from quant_research_platform.agents.mining import training_feedback
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.learning import learning_panel


def test_training_feedback_keeps_failures_undefined_and_coverage(tmp_path):
    frame, spec = learning_panel(tmp_path / "input")
    # The last training day's labels are unavailable; later rows reverse the signal.
    frame.loc[frame.timestamp.eq("2020-01-03"), "label_available_at"] += pd.Timedelta(
        days=30
    )
    frame.loc[frame.timestamp.ge("2020-01-05"), "label"] *= -1
    write_dataset(tmp_path / "panel", frame, spec=spec)
    config = {
        "dataset": str(tmp_path / "panel"),
        "columns": ["x"],
        "baseline_columns": [],
        "label": "label",
        "train_window": {"start": "2020-01-01", "end": "2020-01-03"},
        "train_cutoff": "2020-01-04",
        "expressions": ["x", "-x", "delay(x,1)", "0", "delay(x,-1)"],
    }
    result = training_feedback(config)
    assert result["sample_role"] == "training_only"
    assert result["eligible_rows"] == 6
    rows = result["candidates"]
    assert rows[0]["train_rank_ic"] == pytest.approx(1)
    assert rows[1]["train_rank_ic"] == pytest.approx(-1)
    assert rows[2]["covered_rows"] == 3 and rows[2]["coverage"] == 0.5
    assert rows[3]["status"] == "undefined" and rows[3]["train_rank_ic"] is None
    assert rows[4]["status"] == "failed"
    assert "validation" not in str(result)
    config["baseline_columns"] = ["x"]
    config["expressions"] = ["x"]
    redundant = training_feedback(config)["candidates"][0]
    assert redundant["train_rank_ic"] == pytest.approx(1)
    assert redundant["train_incremental_rank_ic"] is None
    config["validation_window"] = {"start": "2020-01-05", "end": "2020-01-07"}
    with pytest.raises(ValueError, match="operation_config_fields_invalid"):
        training_feedback(config)
