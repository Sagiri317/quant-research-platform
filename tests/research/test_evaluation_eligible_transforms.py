"""Proposed native regression for eligible-domain EVAL transformations."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from tests.research.crypto_evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


@pytest.mark.parametrize("transform", ("identity", "rank", "zscore"))
def test_native_evaluation_transforms_only_selected_cross_section(tmp_path, transform):
    frame = pd.DataFrame(
        {
            "asset_id": list("abcdef"),
            "timestamp": pd.Timestamp("2023-01-02"),
            "x": [1.0, 2.0, 3.0, 1.5, 1.6, 1.7],
            "label": [1.0, 2.0, 3.0, 10.0, 10.0, 10.0],
            "available_at": pd.Timestamp("2023-01-02"),
            "eligible": [True] * 3 + [False] * 3,
        }
    )
    if transform == "zscore":
        frame.loc[3, "x"] = 1e308
    for name, selected in (
        ("eligible_only", frame.iloc[:3]),
        ("with_ineligible", frame),
    ):
        inputs = write_evaluation_input(
            tmp_path / name / "inputs", selected, factors=("x",), labels=("label",)
        )
        spec = EvaluationSpec(
            evaluation_id=name,
            market_id="a_share",
            research_dataset_id="fixture",
            signals=(
                SignalSpec(
                    signal_id="score",
                    source_kind="factor",
                    source_column="x",
                    transform=transform,
                ),
            ),
            label_columns=("label",),
            label_availability_columns={"label": "available_at"},
            filter_mode="eligible",
            n_groups=3,
        )
        output = evaluate(
            evaluation_config(
                inputs,
                tmp_path / name / "evaluation",
                spec,
                start="2023-01-02",
                end="2023-01-03",
            )
        )
        manifest, tables = read_evaluation_outputs(output)
        assert tables["metrics"].ic.tolist() == pytest.approx([1.0])
        assert tables["metrics"].rankic.tolist() == pytest.approx([1.0])
        assert tables["metrics"].n_obs.tolist() == [3]
        assert tables["metrics"].ic_reason.isna().all()
        diagnostics = manifest["metadata"]["evaluation_diagnostics"]
        assert diagnostics["rows_input"] == len(selected)
        assert diagnostics["rows_after_filter_mode"] == 3
        assert diagnostics["rows_after_signal_label_dropna"] == 3
        assert diagnostics["filter_columns_used"] == ["eligible"]
        assert diagnostics["filter_definition"] == "eligible == True"
