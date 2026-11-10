"""Evaluation readers validate persisted members without reopening unrelated data."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import (
    read_evaluation_outputs,
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)


def _spec():
    return EvaluationSpec(
        evaluation_id="native_eval",
        market_id="a_share",
        research_dataset_id="explicit_input",
        signals=(
            SignalSpec(
                signal_id="feature_signal",
                source_kind="feature",
                source_column="feature",
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )


def _frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-02"] * 3),
            "asset_id": ["a", "b", "c"],
            "feature": [1.0, 2.0, 3.0],
            "label": [2.0, 3.0, 5.0],
            "eligible": True,
        }
    )


def test_evaluation_publishes_numeric_members_and_reads_only_requested_summaries(
    tmp_path,
    monkeypatch,
):
    spec = _spec()
    output = tmp_path / "evaluation"
    with build_evaluation_result_from_frames(
        (_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=output,
        )
    assert not (output / "summary.md").exists()
    manifest, tables = read_evaluation_outputs(output)
    assert len(tables["metrics"]) == 1
    assert len(tables["ic_summary"]) == 1
    assert manifest["metadata"]["evaluation_diagnostics"]["metric_rows"] == 1
    # Selecting summaries must not open the metrics member.
    from quant_research_platform.artifacts.outputs import evaluation

    opened = []
    original = evaluation._iter_evaluation_member_batches

    def observe(*args, **kwargs):
        opened.append(kwargs["name"])
        yield from original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_iter_evaluation_member_batches", observe)
    _, selected = read_evaluation_outputs(output, tables=("ic_summary",))
    assert opened == ["ic_summary"]
    assert selected["ic_summary"].mean_ic.iloc[0] == pytest.approx(
        (27 / 28) ** 0.5, rel=1e-13
    )
