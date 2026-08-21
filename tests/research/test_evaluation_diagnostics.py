from __future__ import annotations

import warnings
from contextlib import contextmanager
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.evaluations import engine as evaluation_engine
from quant_research_platform.research.evaluations.diagnostics import (
    build_evaluation_diagnostics,
    summarize_group_spread,
    summarize_period_ic,
    summarize_signal_label_coverage,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.report_output_frames import make_artifact_signal_spec
from tests.helpers.runtime_resources import capture_test_policy


def test_summarize_signal_label_coverage_handles_empty_metrics() -> None:
    summary = summarize_signal_label_coverage(pd.DataFrame())

    assert summary == {
        "metric_rows": 0,
        "rows_after_signal_label_dropna": 0,
        "timestamp_count": 0,
        "min_timestamp": None,
        "max_timestamp": None,
        "skipped_timestamp_count": 0,
    }


def test_summarize_signal_label_coverage_counts_minimum_observation_skips() -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
            "signal": ["s1", "s1", "s2"],
            "label": ["l1", "l1", "l1"],
            "ic": [0.1, float("nan"), 0.2],
            "rankic": [0.2, float("nan"), 0.3],
            "n_obs": [3, 1, 2],
        }
    )

    summary = summarize_signal_label_coverage(metrics)

    assert summary["metric_rows"] == 3
    assert summary["rows_after_signal_label_dropna"] == 6
    assert summary["timestamp_count"] == 2
    assert summary["min_timestamp"] == "2024-01-02"
    assert summary["max_timestamp"] == "2024-01-03"
    assert summary["skipped_timestamp_count"] == 1


def test_summarize_signal_label_coverage_accepts_equivalent_timezones() -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=ZoneInfo("Europe/Paris"),
                ),
            ],
            "n_obs": [2, 2],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = summarize_signal_label_coverage(metrics)

    assert summary["timestamp_count"] == 1
    assert summary["min_timestamp"] == "2024-01-15"
    assert summary["max_timestamp"] == "2024-01-15"


def test_summarize_period_ic_makes_all_nan_behavior_explicit() -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "signal": ["s1", "s1"],
            "label": ["l1", "l1"],
            "ic": [float("nan"), float("nan")],
            "rankic": [float("nan"), float("nan")],
            "n_obs": [1, 1],
        }
    )

    summary = summarize_period_ic(metrics)

    assert summary.loc[0, "valid_ic_count"] == 0
    assert pd.isna(summary.loc[0, "mean_ic"])


def test_summarize_group_spread_is_deterministically_sorted() -> None:
    grouped = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "signal": ["s1", "s1", "s1", "s1"],
            "label": ["l1", "l1", "l1", "l1"],
            "group": [2, 2, 1, 1],
            "mean_label_value": [0.4, 0.3, 0.1, 0.2],
            "n_obs": [2, 2, 2, 2],
        }
    )

    summary = summarize_group_spread(grouped)

    assert summary.loc[0, "signal"] == "s1"
    assert summary.loc[0, "label"] == "l1"
    assert summary.loc[0, "top_bottom_label_value_spread_mean"] == pytest.approx(0.2)


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
        metrics, grouped, metadata = next(chunks)
        chunks.close()

    assert list(metrics.columns) == [
        "timestamp",
        "signal",
        "label",
        "ic",
        "rankic",
        "n_obs",
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
    values = (1.0 - epsilon, 1.0, 1.0 + epsilon)
    metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=3),
            "signal": ["signal"] * 3,
            "label": ["label"] * 3,
            "ic": values,
            "rankic": values,
            "n_obs": [2, 2, 2],
        }
    )
    spec = EvaluationSpec(
        run_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    canonical = summarize_period_ic(metrics).iloc[0]
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

    for column in ("mean_ic", "std_ic", "mean_rankic", "std_rankic"):
        assert streaming[column] == pytest.approx(
            canonical[column],
            rel=0.0,
            abs=1e-12,
        )
    for column in ("icir", "rankicir"):
        if pd.isna(canonical[column]):
            assert pd.isna(streaming[column])
        else:
            assert streaming[column] == pytest.approx(
                canonical[column],
                rel=1e-9,
                abs=1e-12,
            )


def test_streaming_group_spread_reducer_avoids_chunk_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    expected = summarize_group_spread(grouped)
    monkeypatch.setattr(
        evaluation_engine,
        "summarize_group_spread",
        lambda *_args, **_kwargs: pytest.fail(
            "streaming_group_spread_must_not_materialize_summary_frame"
        ),
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
    pd.testing.assert_frame_equal(actual, expected, check_exact=False)


def test_evaluation_diagnostics_reports_fail_closed_on_bad_inputs() -> None:
    with pytest.raises(ValueError, match="missing_evaluation_metric_columns"):
        summarize_signal_label_coverage(pd.DataFrame({"timestamp": ["2024-01-02"]}))

    with pytest.raises(ValueError, match="invalid_evaluation_timestamp_values"):
        summarize_signal_label_coverage(
            pd.DataFrame({"timestamp": ["bad"], "n_obs": [1]})
        )

    with pytest.raises(ValueError, match="invalid_evaluation_observation_counts"):
        summarize_signal_label_coverage(
            pd.DataFrame({"timestamp": ["2024-01-02"], "n_obs": [1.5]})
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_build_evaluation_diagnostics_rejects_invalid_filter_metadata_counts(
    field: str,
    value: object,
) -> None:
    metadata = _filter_metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match="invalid_evaluation_diagnostics_count"):
        _build_diagnostics(filter_metadata=metadata)


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_build_evaluation_diagnostics_rejects_invalid_dropna_count(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_evaluation_diagnostics_count"):
        _build_diagnostics(rows_after_signal_label_dropna=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_build_evaluation_diagnostics_rejects_invalid_group_count(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_evaluation_diagnostics_count"):
        _build_diagnostics(n_groups=value)  # type: ignore[arg-type]


def test_build_evaluation_diagnostics_accepts_valid_counts() -> None:
    diagnostics = _build_diagnostics()

    assert diagnostics.rows_input == 2
    assert diagnostics.rows_after_filter_mode == 2
    assert diagnostics.rows_after_signal_label_dropna == 2
    assert diagnostics.n_groups == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_evaluation_engine_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    metadata = {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }
    metadata[field] = value

    @contextmanager
    def fake_filter_mode_scope(frame: pd.DataFrame, **_: object):
        yield frame, metadata

    monkeypatch.setattr(
        "quant_research_platform.research.filtering._filter_mode_scope",
        fake_filter_mode_scope,
    )

    with pytest.raises(ValueError, match="invalid_evaluation_diagnostics_count"):
        evaluation_engine.evaluate_signal_frame_chunks(
            (_evaluation_frame(),),
            EvaluationSpec(
                run_id="eval_bad_metadata_count",
                market_id=MarketId("a_share"),
                research_dataset_id="research_demo",
                signals=(make_artifact_signal_spec(),),
                label_columns=("label",),
                n_groups=2,
            ),
            n_groups=2,
        )


def test_evaluation_engine_accepts_valid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def fake_filter_mode_scope(frame: pd.DataFrame, **_: object):
        yield frame, {
            "filter_columns_used": ["eligible"],
            "filter_definition": "eligible == True",
            "rows_input": 2,
            "rows_after_filter_mode": 2,
        }

    monkeypatch.setattr(
        "quant_research_platform.research.filtering._filter_mode_scope",
        fake_filter_mode_scope,
    )

    result = evaluation_engine.evaluate_signal_frame_chunks(
        (_evaluation_frame(),),
        EvaluationSpec(
            run_id="eval_valid_metadata_count",
            market_id=MarketId("a_share"),
            research_dataset_id="research_demo",
            signals=(make_artifact_signal_spec(),),
            label_columns=("label",),
            n_groups=2,
        ),
        n_groups=2,
    )

    assert result.diagnostics.rows_input == 2
    assert result.diagnostics.rows_after_filter_mode == 2


def _build_diagnostics(
    *,
    filter_metadata: dict[str, object] | None = None,
    rows_after_signal_label_dropna: int = 2,
    n_groups: int = 2,
):
    return build_evaluation_diagnostics(
        spec=EvaluationSpec(
            run_id="eval_demo",
            market_id=MarketId("a_share"),
            research_dataset_id="research_demo",
            signals=(make_artifact_signal_spec(),),
            label_columns=("label",),
            n_groups=2,
        ),
        filter_metadata=filter_metadata or _filter_metadata(),
        rows_after_signal_label_dropna=rows_after_signal_label_dropna,
        metrics=pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "signal": ["signal"],
                "label": ["label"],
                "ic": [0.1],
                "rankic": [0.2],
                "n_obs": [2],
            }
        ),
        grouped_label_values=pd.DataFrame(),
        ic_summary=pd.DataFrame(),
        group_spread_summary=pd.DataFrame(),
        n_groups=n_groups,
        timestamp_column="timestamp",
    )


def _filter_metadata() -> dict[str, object]:
    return {
        "filter_columns_used": ("eligible",),
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }


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
