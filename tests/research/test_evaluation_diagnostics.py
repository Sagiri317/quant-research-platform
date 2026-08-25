from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.evaluations import engine as evaluation_engine
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.output_frames import make_artifact_signal_spec
from tests.helpers.runtime_resources import capture_test_policy


def test_evaluation_chunk_producer_does_not_terminal_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_stream",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    monkeypatch.setattr(
        pd,
        "concat",
        lambda *_args, **_kwargs: pytest.fail(
            "single_label_managed_evaluation_must_not_concat"
        ),
    )
    policy = capture_test_policy(available_memory_bytes=256 * 1024)
    with runtime_resource_policy_scope(policy):
        chunks = evaluation_engine._iter_evaluation_output_chunks(
            (_evaluation_frame(),),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            resource_policy=policy,
        )
        metrics, grouped, _ = next(chunks)
        chunks.close()

    assert list(metrics.columns) == [
        "timestamp",
        "signal",
        "label",
        "ic",
        "rankic",
        "n_obs",
        "ic_reason",
        "rankic_reason",
    ]
    assert list(grouped.columns) == [
        "timestamp",
        "signal",
        "label",
        "group",
        "mean_label_value",
        "n_obs",
    ]


@pytest.mark.parametrize("epsilon", (1e-7, 1e-8))
def test_streaming_ic_summary_matches_stable_canonical_values(
    epsilon: float,
) -> None:
    values = (1.0 - epsilon, 1.0, 1.0 - 2 * epsilon)
    metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=3),
            "signal": ["signal"] * 3,
            "label": ["label"] * 3,
            "ic": values,
            "rankic": values,
            "n_obs": [2, 2, 2],
            "ic_reason": [None, None, None],
            "rankic_reason": [None, None, None],
        }
    )
    spec = EvaluationSpec(
        run_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
        n_groups=2,
    )
    try:
        state.consume(
            metrics,
            pd.DataFrame(),
            {
                "rows_input": 6,
                "rows_after_filter_mode": 6,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible == True",
                "timestamp_count": 3,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-04",
                "skipped_timestamp_count": 0,
            },
        )
        streaming = state.ic_summary().iloc[0]
    finally:
        state.close()

    expected_mean = sum(values) / len(values)
    expected_std = (
        sum((value - expected_mean) ** 2 for value in values) / len(values)
    ) ** 0.5
    expected_ratio = expected_mean / expected_std
    for column in ("mean_ic", "std_ic", "mean_rankic", "std_rankic"):
        expected = {
            "mean_ic": expected_mean,
            "std_ic": expected_std,
            "mean_rankic": expected_mean,
            "std_rankic": expected_std,
        }[column]
        assert streaming[column] == pytest.approx(
            expected,
            rel=0.0,
            abs=1e-12,
        )
    for column in ("icir", "rankicir"):
        assert streaming[column] == pytest.approx(
            expected_ratio,
            rel=1e-8,
            abs=1e-12,
        )


def test_streaming_group_spread_reducer_avoids_chunk_materialization() -> None:
    grouped = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "signal": ["signal"] * 4,
            "label": ["label"] * 4,
            "group": [1, 2, 1, 2],
            "mean_label_value": [0.1, 0.3, 0.2, 0.1],
            "n_obs": [2, 2, 2, 2],
        }
    )
    spec = EvaluationSpec(
        run_id="streaming_group_spread",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
        n_groups=2,
    )
    try:
        state.consume(
            pd.DataFrame(),
            grouped,
            {
                "rows_input": 4,
                "rows_after_filter_mode": 4,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible == True",
                "timestamp_count": 2,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-03",
                "skipped_timestamp_count": 0,
            },
        )
        actual = state.group_spread_summary()
    finally:
        state.close()

    assert len(actual) == 1
    row = actual.iloc[0]
    assert row["signal"] == "signal"
    assert row["label"] == "label"
    assert row["top_group_mean_label_value"] == pytest.approx(0.2)
    assert row["bottom_group_mean_label_value"] == pytest.approx(0.15)
    assert row["top_bottom_label_value_spread_mean"] == pytest.approx(0.05)
    assert row["top_bottom_label_value_spread_hit_rate"] == pytest.approx(0.5)
    assert row["group_monotonicity_score"] == pytest.approx(0.0)
    assert pd.isna(row["group_monotonicity_score_reason"])


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset_id": ["asset_a", "asset_b"],
            "feature": [1.0, 2.0],
            "label": [0.1, 0.2],
            "eligible": [True, True],
        }
    )
