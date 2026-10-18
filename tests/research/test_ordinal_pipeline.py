"""Order-based consumers preserve native observations through real artifact chains."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import _evaluation_strata
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.signals.normalization import (
    normalize_signal_frame,
)
from tests.research.crypto_evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)
from tests.research.direct_model_test_support import fit_frame, predict_frame


@pytest.mark.parametrize(
    "offset,dtype,direction",
    [(0, "Int64", "higher_is_better"), (2**63, "UInt64", "lower_is_better")],
)
def test_native_evaluation_keeps_order_ties_and_pair_domain(
    tmp_path, offset, dtype, direction
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": list("abcdefg"),
            "x": pd.array(
                [
                    offset + v if v is not None else None
                    for v in [0, 1, 1, 2, 3, 4, None]
                ],
                dtype=dtype,
            ),
            "label": pd.array(
                [2**53 + v for v in [4, 0, 2, 1, 3, 5, 6]], dtype="int64"
            ),
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-01"),
        }
    )
    frame.loc[3, "label_available_at"] = pd.Timestamp("2030-01-01")
    frame = frame.iloc[[6, 2, 5, 0, 3, 1, 4]].reset_index(drop=True)
    inputs = write_evaluation_input(
        tmp_path / "input", frame, factors=("x",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="ordinal",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="score",
                source_kind="factor",
                source_column="x",
                direction=direction,
            ),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        n_groups=3,
    )
    config = evaluation_config(
        inputs,
        tmp_path / "eval",
        spec,
        start="2020-01-01",
        end="2020-01-01",
        cutoff="2020-01-02",
    )
    _, tables = read_evaluation_outputs(
        evaluate(config), tables=("metrics", "grouped_label_values")
    )
    sign = -1 if direction == "lower_is_better" else 1
    # Only a,b,c,e,f pair. Their independent ranks exclude d, not merely its rank value.
    expected = sign * np.corrcoef([1, 2.5, 2.5, 4, 5], [4, 1, 2, 3, 5])[0, 1]
    assert tables["metrics"].rankic.tolist() == pytest.approx([expected])
    assert tables["metrics"].n_obs.tolist() == [5]
    groups = tables["grouped_label_values"].set_index("group")
    assert groups.n_obs.to_dict() == (
        {1: 3, 2: 0, 3: 2} if sign == 1 else {1: 2, 2: 2, 3: 1}
    )
    if direction == "lower_is_better":
        # The unscoped runner has its own maturity mask before the shared kernel.
        config["evaluation"]["filter_mode"] = "unfiltered"
        config["output"] = str(tmp_path / "unfiltered")
        _, unscoped = read_evaluation_outputs(
            evaluate(config), tables=("metrics", "grouped_label_values")
        )
        for name in tables:
            pd.testing.assert_frame_equal(
                tables[name], unscoped[name], check_exact=True
            )
    ranked = normalize_signal_frame(
        frame,
        (
            SignalSpec(
                signal_id="score",
                source_kind="factor",
                source_column="x",
                transform="rank",
                direction=direction,
            ),
        ),
    )
    expected_ranks = (
        pd.Series([1, 2.5, 2.5, 4, 5, 6, np.nan], index=list("abcdefg")) * sign
    )
    np.testing.assert_allclose(
        ranked.score, expected_ranks.reindex(frame.asset_id), equal_nan=True
    )


def test_native_rank_model_state_prediction_and_trial_labels(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": list("abcde"),
            "x": pd.array([2**53 + v for v in [0, 1, 1, 2, 3]], dtype="Int64"),
            "label": [0, 1, 1, 2, 3],
            "label_available_at": pd.Timestamp("2020-01-01"),
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="ordinal",
        model_algorithm="ic_weighted_rank_ensemble",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=19,
    )
    case = fit_frame(tmp_path / "fit", frame, spec)
    future = (
        frame.assign(timestamp=pd.Timestamp("2020-01-02"))
        .iloc[[4, 1, 3, 0, 2]]
        .reset_index(drop=True)
    )
    predictions, prediction_path, _ = predict_frame(
        tmp_path / "pred",
        future,
        case,
        window={"start": "2020-01-02", "end": "2020-01-02"},
    )
    expected = pd.Series([-0.4, -0.1, -0.1, 0.2, 0.4], index=list("abcde"))
    np.testing.assert_allclose(
        predictions.prediction_score, expected.reindex(predictions.asset_id), atol=1e-15
    )
    assert case.result.metadata["rank_parameters"]["weights"] == [1.0]
    trial = (
        predictions.rename(columns={"prediction_score": "prediction"})
        .merge(frame[["asset_id", "x"]], on="asset_id")
        .rename(columns={"x": "target"})
    )
    metrics = evaluate_model_trial_metrics_from_chunks(
        [trial.iloc[:2], trial.iloc[2:]],
        model_id="ordinal",
        label_column="target",
        prediction_column="prediction",
    )
    assert metrics["validation_rankic"] == pytest.approx(1.0)
    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain

    # Rank diagnostics are translation-invariant even when labels are native integers.
    future["label"] = future["label"] + 2**53
    path = tmp_path / "explain-input.parquet"
    future.to_parquet(path, index=False)
    output = explain(
        {
            "model": str(case.root),
            "inputs": str(path),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "explained"),
            "window": {"start": "2020-01-02", "end": "2020-01-02"},
            "sample_role": "test",
            "methods": ["permutation"],
            "output_space": "model_output",
            "selection": {"max_rows": 5, "max_dates": 1, "root_seed": 17},
            "permutation": {
                "metrics": ["rankic"],
                "cutoff": "2020-01-03",
                "axis": "within_timestamp",
                "groups": {"x": ["x"]},
                "repeats": 1,
                "root_seed": 12,
            },
        }
    )
    _, tables = read_explanation(output)
    assert tables["permutation_repeats"].baseline.tolist() == pytest.approx([1.0])
    import json

    from quant_research_platform.research.models.trial_selection import (
        read_selection,
        score,
        select,
    )

    window = {"start": "2020-01-02", "end": "2020-01-02"}
    scores = score(
        {
            "predictions": [str(prediction_path)],
            "candidate_ids": ["ordinal"],
            "labels": str(path),
            "label_spec": case.config["input_spec"],
            "validation_windows": [window],
            "scoring_window": window,
            "cutoff": "2020-01-03",
            "selection_group": "quarterly",
            "validation_axis": "cross_sectional",
            "label_unit": "decimal_return",
            "output": str(tmp_path / "scores.json"),
        }
    )
    assert json.loads(scores.read_text())["metrics"][0][
        "validation_rankic"
    ] == pytest.approx(1.0)
    selected = select(
        {
            "scores": str(scores),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["ordinal"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
            "refit_cutoff": "2020-01-03",
        }
    )
    assert read_selection(selected)["selected_model_id"] == "ordinal"


def test_quantile_boundary_and_local_invalid_strata_preserve_order():
    signal = pd.Series([2**63, 2**63 + 1, 2**63 + 2], index=list("abc"), dtype="uint64")
    weights = build_simple_target_weights(
        signal, eligible=pd.Series(True, index=signal.index), top_quantile=1 / 3
    )
    assert weights.to_dict() == {"a": 0.0, "b": 0.0, "c": 1.0}
    frame = pd.DataFrame(
        {
            "label": 1.0,
            "amount": pd.Series([2**53, 2**53 + 1, 2**53 + 2, -1], dtype="int64"),
        }
    )
    groups, quality = _evaluation_strata(
        frame, {"primary_label": "label", "kinds": ["liquidity"]}
    )
    assert groups["liquidity"].iloc[:3].tolist() == ["1", "2", "3"]
    assert pd.isna(groups["liquidity"].iloc[3])
    assert quality["liquidity_invalid"] == 1


def test_native_response_uses_exact_frozen_integer_boundaries(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.evaluations.engine import (
        training_quantile_cutpoints,
    )
    from quant_research_platform.storage.datasets import write_dataset

    values = pd.Series([2**63 + i for i in range(4)], dtype="uint64")
    assert training_quantile_cutpoints(values, n_groups=2) == [2**63 + 1]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="s"),
            "asset_id": "btc",
            "x": values,
            "label": [0.0001, 0.0002, 0.0003, 0.0004],
            "label_available_at": pd.Timestamp("2020-01-01 01:00:00"),
            "eligible": True,
        }
    )
    source = tmp_path / "input"
    write_dataset(
        source,
        frame,
        spec=DataSpec(
            data_id="ordinal",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
            column_roles={
                "timestamp": "timestamp",
                "asset_id": "key",
                "x": "factor",
                "label": "label",
                "label_available_at": "context",
                "eligible": "mask",
            },
            column_units={"label": "decimal_return"},
            label_availability_columns={"label": "label_available_at"},
        ),
    )
    for direction, edge, counts, means in [
        ("higher_is_better", 2**63 + 1, [2, 2], [1.5, 3.5]),
        ("lower_is_better", -(2**63 + 1), [3, 1], [3.0, 1.0]),
    ]:
        spec = EvaluationSpec(
            evaluation_id="ordinal",
            market_id="crypto",
            research_dataset_id="ordinal",
            signals=(
                SignalSpec(
                    signal_id="score",
                    source_kind="factor",
                    source_column="x",
                    direction=direction,
                ),
            ),
            label_columns=("label",),
            label_availability_columns={"label": "label_available_at"},
            evaluation_axis="single_asset_time_series",
        )
        config = evaluation_config(
            source,
            tmp_path / direction,
            spec,
            start="2020-01-01",
            end="2020-01-01",
            cutoff="2020-01-02",
        )
        config["response_diagnostics"] = {"cutpoints": {"score": [edge]}}
        _, tables = read_evaluation_outputs(
            evaluate(config), tables=("response_groups",)
        )
        groups = (
            tables["response_groups"]
            .query("period == 'pooled' and sample_scope == 'own'")
            .sort_values("group")
        )
        assert groups.n_obs.tolist() == counts
        assert groups.mean_return_bp.tolist() == pytest.approx(means)
