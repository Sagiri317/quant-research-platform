"""Order-based consumers preserve native observations through real artifact chains."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import _evaluation_strata
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.signals.normalization import (
    normalize_signal_frame,
)
from tests.research.evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
def test_native_evaluation_keeps_order_ties_and_pair_domain(tmp_path, direction):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": list("abcdefg"),
            "x": pd.array([0, 0.1, 0.1, 0.2, 0.3, 0.4, None], dtype="Float64"),
            "label": [0.04, 0.0, 0.02, 0.01, 0.03, 0.05, 0.06],
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


def test_quantile_boundary_and_local_invalid_strata_preserve_order():
    signal = pd.Series([0.1, 0.2, 0.3], index=list("abc"))
    weights = build_simple_target_weights(
        signal, eligible=pd.Series(True, index=signal.index), top_quantile=1 / 3
    )
    assert weights.to_dict() == {"a": 0.0, "b": 0.0, "c": 1.0}
    frame = pd.DataFrame(
        {
            "label": 1.0,
            "amount": [1e6, 2e6, 3e6, -1],
        }
    )
    groups, quality = _evaluation_strata(
        frame, {"primary_label": "label", "kinds": ["liquidity"]}
    )
    assert groups["liquidity"].iloc[:3].tolist() == ["1", "2", "3"]
    assert pd.isna(groups["liquidity"].iloc[3])
    assert quality["liquidity_invalid"] == 1


def test_native_response_uses_frozen_boundaries_after_direction(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.evaluations.engine import (
        training_quantile_cutpoints,
    )
    from quant_research_platform.storage.datasets import write_dataset

    values = pd.Series([0.1, 0.2, 0.3, 0.4])
    assert training_quantile_cutpoints(values, n_groups=2) == [0.2]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="s"),
            "asset_id": "asset_a",
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
        ("higher_is_better", 0.2, [2, 2], [1.5, 3.5]),
        ("lower_is_better", -0.2, [3, 1], [3.0, 1.0]),
    ]:
        spec = EvaluationSpec(
            evaluation_id="ordinal",
            market_id="other_market",
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
