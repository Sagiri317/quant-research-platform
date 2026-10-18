"""Native regressions for eligible-domain EVAL transformations."""

import numpy as np
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
            "x": [1.0, 2.0, 8.0, 1.5, 1.6, 1.7],
            "label": [1.0, 4.0, 2.0, 10.0, 10.0, 10.0],
            "available_at": pd.Timestamp("2023-01-02"),
            "eligible": [True] * 3 + [False] * 3,
        }
    )
    if transform == "zscore":
        frame.loc[3, "x"] = 1000.0
    raw_ic = np.corrcoef([1.0, 2.0, 8.0], [1.0, 4.0, 2.0])[0, 1]
    rank_ic = np.corrcoef([1.0, 2.0, 3.0], [1.0, 4.0, 2.0])[0, 1]
    assert abs(rank_ic - raw_ic) > 0.1
    expected_ic = rank_ic if transform == "rank" else raw_ic
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
        assert tables["metrics"].ic.tolist() == pytest.approx(
            [expected_ic], rel=1e-12, abs=1e-15
        )
        assert tables["metrics"].rankic.tolist() == pytest.approx([0.5])
        assert tables["metrics"].n_obs.tolist() == [3]
        assert tables["metrics"].ic_reason.isna().all()
        diagnostics = manifest["metadata"]["evaluation_diagnostics"]
        assert diagnostics["rows_input"] == len(selected)
        assert diagnostics["rows_after_filter_mode"] == 3
        assert diagnostics["rows_after_signal_label_dropna"] == 3
        assert diagnostics["filter_columns_used"] == ["eligible"]
        assert diagnostics["filter_definition"] == "eligible == True"


def test_native_zscore_is_undefined_on_singleton_eligible_cross_section(tmp_path):
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2023-01-02"),
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
            "available_at": pd.Timestamp("2023-01-02"),
            "eligible": [True, False],
        }
    )
    inputs = write_evaluation_input(
        tmp_path / "inputs", frame, factors=("x",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="singleton",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="score",
                source_kind="factor",
                source_column="x",
                transform="zscore",
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
            tmp_path / "evaluation",
            spec,
            start="2023-01-02",
            end="2023-01-03",
        )
    )
    _, tables = read_evaluation_outputs(output)
    # A singleton has zero variance; an ineligible peer must not define its score.
    assert tables["metrics"].n_obs.tolist() == [0]
    assert tables["grouped_label_values"].empty
