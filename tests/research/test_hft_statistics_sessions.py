"""Intraday IC inference uses explicit sessions and equal-weight daily estimates."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.factors.statistics import factor_statistics
from quant_research_platform.research.models.statistics import (
    build_a_share_statistics_tables,
)
from quant_research_platform.research.statistics import (
    common_correlation_mask,
    daily_mean_frame,
)
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.research.evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _case(tmp_path, *, interval=3, missing_session=False):
    first, last = pd.to_datetime(["2020-01-02 09:30", "2020-01-06 09:30"])
    timestamps = [first + pd.Timedelta(seconds=interval * i) for i in range(3)]
    timestamps.append(last)
    frame = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "asset_id": asset,
                "eligible": True,
                "x": float(i),
                "constant": 1.0,
                "label": float(i if block < 2 else 3 - i),
                "label_available_at": timestamp + pd.Timedelta(seconds=30),
            }
            for block, timestamp in enumerate(timestamps)
            for i, asset in enumerate("abcd")
        ]
    )
    source = write_evaluation_input(
        tmp_path / "input", frame, factors=("x", "constant"), labels=("label",)
    )
    sessions = [
        "2020-01-02",
        *(["2020-01-03"] if missing_session else []),
        "2020-01-06",
    ]
    config = {
        "dataset": str(source),
        "output": str(tmp_path / "statistics"),
        "window": {"start": sessions[0], "end": sessions[-1]},
        "sessions": sessions,
        "evaluation": {
            "factors": ["x", "constant"],
            "primary_label": "label",
            "label_availability_columns": {"label": "label_available_at"},
        },
        "analyses": ["rank_ic"],
        "statistics": {
            "hac_lag": 1,
            "factor_fdr_families": {"declared": ["x", "constant"]},
        },
    }
    return frame, source, config


@pytest.mark.parametrize("interval", [3, 30])
@pytest.mark.parametrize("missing_session", [False, True])
def test_factor_rankic_averages_sessions_before_hac_and_keeps_fdr_family(
    tmp_path, interval, missing_session
):
    _, _, config = _case(tmp_path, interval=interval, missing_session=missing_session)
    artifact = read_statistics_artifact(factor_statistics(config))
    with artifact.table("factor_diagnostics.parquet") as table:
        actual = table.loc[table.factor_id.eq("x")]
        # Timestamp RankIC=(1,1,-1,-1), daily=(1/3,-1). Pooled mean is zero.
        assert actual["mean"].to_numpy() == pytest.approx([-1 / 3] * 2)
        assert actual.observation_count.eq(2).all()
        assert actual.input_date_count.eq(3 if missing_session else 2).all()
        assert table.planned_hypothesis_count.eq(2).all()
        if missing_session:
            assert actual.reason.eq("incomplete_session_clock").all()
            assert table.p_value.isna().all()
            assert table.fdr_adjusted_p_value.isna().all()
        else:
            assert actual.status.eq("DEFINED").all()
            # Bartlett HAC lag one on residuals (+2/3,-2/3) has SE=1/3.
            assert actual.standard_error.to_numpy() == pytest.approx([1 / 3] * 2)
            assert actual.defined_hypothesis_count.eq(1).all()
            assert actual.missing_hypothesis_count.eq(1).all()
    assert artifact.metadata["hac_lag_unit"] == "trading_session"


def test_intraday_statistics_requires_a_declared_session_axis(tmp_path):
    _, _, config = _case(tmp_path)
    del config["sessions"]
    with pytest.raises(ValueError, match="statistics_sessions_required"):
        factor_statistics(config)
    assert not (tmp_path / "statistics").exists()
    legacy = pd.DataFrame(
        {"x": [1.0, -1.0]}, index=pd.to_datetime(["2020-01-02", "2020-01-06"])
    )
    pd.testing.assert_frame_equal(
        daily_mean_frame(legacy), legacy.rename_axis("timestamp")
    )
    with pytest.raises(ValueError, match="statistics_input_session_not_declared"):
        daily_mean_frame(legacy, sessions=["2020-01-02"])
    empty = daily_mean_frame(legacy.iloc[:0], sessions=[])
    assert empty.empty
    assert empty.columns.tolist() == ["x"]
    assert isinstance(empty.index, pd.DatetimeIndex) and empty.index.name == "timestamp"
    with pytest.raises(ValueError, match="statistics_input_session_not_declared"):
        daily_mean_frame(legacy, sessions=[])


def test_common_correlation_mask_keeps_metric_and_stratum_domains_separate():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2020-01-02 09:30")] * 4,
            "label": ["label"] * 4,
            "stratum": [None, None, "B", "B"],
            "ic": [np.nan, 1.0, 1.0, 1.0],
            "rankic": [1.0, 1.0, np.nan, 1.0],
        }
    )
    mask = common_correlation_mask(frame, groups=("timestamp", "label", "stratum"))
    np.testing.assert_array_equal(
        mask.to_numpy(), [[False, True], [False, True], [True, False], [True, False]]
    )


@pytest.mark.parametrize("scope", ["own", "common"])
def test_saved_common_correlation_uses_identical_timestamp_blocks(tmp_path, scope):
    frame, _, config = _case(tmp_path)
    times = frame.timestamp.drop_duplicates().tolist()
    frame["label"] = np.tile(np.arange(4, dtype=float), 4)
    frame["z"] = np.tile(np.arange(4, dtype=float), 4)
    frame.loc[frame.timestamp.isin(times[1:3]), "z"] = np.tile([3, 2, 1, 0], 2)
    frame.loc[frame.timestamp.eq(times[0]), "x"] = 1.0
    frame.loc[frame.timestamp.eq(times[-1]), "x"] = [3, 2, 1, 0]
    frame["industry_level1_code"] = ["A", "A", "B", "B"] * 4
    source = write_evaluation_input(
        tmp_path / "paired-input", frame, factors=("x", "z"), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="paired-blocks",
        market_id="a_share",
        research_dataset_id="evaluation_fixture",
        signals=tuple(
            SignalSpec(signal_id=name, source_kind="factor", source_column=name)
            for name in ("x", "z")
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        correlation_clock="trading_day",
        n_groups=2,
    )
    saved = evaluate(
        {
            **evaluation_config(
                source,
                tmp_path / "paired-evaluation",
                spec,
                cutoff="2020-01-06T16:00:00",
                **config["window"],
            ),
            "sessions": config["sessions"],
            "sample_scope": scope,
            "strata": {"primary_label": "label", "kinds": ["industry"]},
        }
    )
    _, raw = read_evaluation_outputs(saved, tables=("metrics", "ic_summary"))
    # The constant-x first block remains visible as a defined z diagnostic.
    z_blocks = raw["metrics"].loc[raw["metrics"].signal.eq("z")]
    assert z_blocks.rankic.tolist() == pytest.approx([1, -1, -1, 1])
    # Two groups select half of four assets per side. Each two-point side has
    # the same slope sign: (+1, -1, -1, +1). Own daily means are (-1/3, 1);
    # common excludes the constant-x first block, leaving daily means (-1, 1).
    for side in ("long", "short"):
        assert z_blocks[f"{side}_n_obs"].tolist() == [2] * 4
        for metric in ("ic", "rankic"):
            assert z_blocks[f"{side}_{metric}"].tolist() == pytest.approx(
                [1, -1, -1, 1]
            )
    expected = 0.0 if scope == "common" else 1 / 3
    summary = raw["ic_summary"].set_index("signal")
    assert summary.loc["z", "mean_rankic"] == pytest.approx(expected)
    artifact = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(saved)],
                "output": str(tmp_path / "paired-statistics"),
                "analyses": ["rank_ic", "stratified_ic"],
                "evaluation": {"factors": ["x", "z"], "primary_label": "label"},
                "statistics": {
                    "hac_lag": 0,
                    "factor_fdr_families": {"declared": ["x", "z"]},
                },
            }
        )
    )
    with artifact.table("factor_stratified_diagnostics.parquet") as table:
        selected = table.loc[table.signal.eq("z")]
        assert set(selected.stratification) == {"all", "industry", "side"}
        sides = selected.loc[selected.stratification.eq("side")]
        assert len(sides) == 4
        assert sides["mean"].tolist() == pytest.approx([expected] * 4)
        assert sides.observation_count.eq(2).all()
        assert sides.observation_pairs.eq(8).all()
        selected = selected.loc[~selected.stratification.eq("side")]
        assert set(selected.stratification) == {"all", "industry"}
        assert selected["mean"].to_numpy() == pytest.approx([expected] * len(selected))
    with artifact.table("factor_diagnostics.parquet") as table:
        assert table.loc[table.factor_id.eq("z"), "mean"].tolist() == pytest.approx(
            [expected] * 2
        )


@pytest.mark.parametrize("missing_session", [False, True])
def test_saved_intraday_evaluation_uses_the_same_daily_ic_rankic_axis(
    tmp_path, missing_session
):
    _, source, config = _case(tmp_path, missing_session=missing_session)
    spec = EvaluationSpec(
        evaluation_id="intraday",
        market_id="a_share",
        research_dataset_id="evaluation_fixture",
        signals=(SignalSpec(signal_id="x", source_kind="factor", source_column="x"),),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        n_groups=2,
    )
    saved = evaluate(
        {
            **evaluation_config(
                source,
                tmp_path / "evaluation",
                spec,
                cutoff="2020-01-06T16:00:00",
                **config["window"],
            ),
            "sessions": config["sessions"],
        }
    )
    published = factor_statistics(
        {
            "evaluation_inputs": [str(saved)],
            "output": str(tmp_path / "saved-statistics"),
            "analyses": ["rank_ic", "stratified_ic"],
            "evaluation": {"factors": ["x"], "primary_label": "label"},
            "statistics": {"hac_lag": 1, "factor_fdr_families": {"declared": ["x"]}},
        }
    )
    artifact = read_statistics_artifact(published)
    with artifact.table("factor_stratified_diagnostics.parquet") as table:
        assert set(table.statistic) == {"ic", "rankic"}
        sides = table.loc[table.stratification.eq("side")]
        # With n_groups=2, each side pairs two assets per timestamp. The side
        # correlations are (+1, +1, -1, -1), just like the full cross sections:
        # daily means (1/3, -1) give mean -1/3 and lag-one HAC SE 1/3.
        assert len(sides) == 4 and len(table) == 6
        assert sides.observation_pairs.eq(8).all()
        assert table["mean"].to_numpy() == pytest.approx([-1 / 3] * 6)
        assert table.observation_count.eq(2).all()
        assert table.observed_date_count.eq(2).all()
        assert (
            table.loc[table.stratification.eq("all"), "observation_pairs"].eq(16).all()
        )
        assert table.input_date_count.eq(len(config["sessions"])).all()
        if missing_session:
            assert table.reason.eq("incomplete_session_clock").all()
            assert table.fdr_adjusted_p_value.isna().all()
        else:
            assert table.standard_error.to_numpy() == pytest.approx([1 / 3] * 6)
    with artifact.table("factor_diagnostics.parquet") as table:
        assert table["mean"].to_numpy() == pytest.approx([-1 / 3] * 2)
        assert table.input_date_count.eq(len(config["sessions"])).all()
    empty = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(saved)],
                "output": str(tmp_path / "empty-window-statistics"),
                "window": {"start": "2020-01-04", "end": "2020-01-05"},
                "analyses": ["rank_ic", "stratified_ic"],
                "evaluation": {"factors": ["x"], "primary_label": "label"},
                "statistics": {
                    "hac_lag": 1,
                    "factor_fdr_families": {"declared": ["x"]},
                },
            }
        )
    )
    for name in ("factor_diagnostics", "factor_stratified_diagnostics"):
        with empty.table(name + ".parquet") as table:
            assert table.input_date_count.eq(0).all()
            assert table.observation_count.eq(0).all()
            assert table.status.eq("UNDEFINED").all()


@pytest.mark.parametrize("missing_session", [False, True])
def test_model_common_reranks_assets_and_pairs_timestamp_blocks_before_daily_mean(
    tmp_path, missing_session
):
    frame, source, config = _case(tmp_path, missing_session=missing_session)
    timestamps = frame.timestamp.drop_duplicates().tolist()
    scores = {
        (timestamp, asset, family): values[i]
        for block, timestamp in enumerate(timestamps)
        for family, values in (
            ("timestamp", [1, 1, 1, 1] if block == 1 else [3, 1, 2, 100]),
            ("peer", [1, 2, 3, np.nan] if block in (1, 3) else [3, 1, 2, np.nan]),
        )
        for i, asset in enumerate("abcd")
    }
    # Keep label ranks ascending on every timestamp for this comparison oracle.
    frame["label"] = np.tile(np.arange(4, dtype=float), 4)
    source = write_evaluation_input(
        tmp_path / "model-input", frame, factors=("x",), labels=("label",)
    )

    class Lookup:
        def values_for_frame(self, keys, *, mark_matched):
            assert mark_matched
            values = np.array(
                [scores[key] for key in keys.itertuples(index=False, name=None)]
            )
            return values, np.isfinite(values)

    model_config = {
        **config,
        "sample_role": "exploratory",
        "family_ids": ["timestamp", "peer"],
        "comparison_model_id": "timestamp",
        "evaluation": {
            "primary_label": "label",
            "label_availability_columns": {"label": "label_available_at"},
        },
        "analyses": ["prediction_hac", "model_comparison"],
        "statistics": {"hac_lag": 0, "bootstrap_block": 1, "bootstrap_repetitions": 31},
    }
    window = TimeWindow(**config["window"])
    tables = build_a_share_statistics_tables(
        dataset_session=open_dataset_session(source, time_window=window),
        prediction_id="unused",
        model_entries={
            ("period", family): {
                "model_metadata": {},
                "model_facts": {"model": {"target_column": "label"}},
            }
            for family in model_config["family_ids"]
        },
        selections={},
        period_windows={"period": window},
        trial_rows=pd.DataFrame(),
        config=model_config,
        time_window=window,
        prediction_lookup=Lookup(),
    )
    own = tables["prediction_diagnostics"].set_index("prediction_id")
    assert own.loc["timestamp", "mean"] == pytest.approx(0.4)
    assert own.loc["peer", "mean"] == pytest.approx(0.5)
    comparison = tables["model_comparison"].iloc[0]
    # On the 3-asset common sample, day-one A/B both have RankIC=-.5;
    # the constant-A timestamp is excluded from BOTH daily means. Day-two
    # A=-.5 and B=1, so the equal-day difference is (0 + 1.5)/2.
    assert comparison["mean"] == pytest.approx(0.75)
    assert comparison.observation_count == 2
    assert comparison.input_date_count == len(config["sessions"])
    if missing_session:
        assert comparison.reason == "incomplete_session_clock"
        assert np.isnan(comparison.p_value)
    else:
        assert comparison.status == "DEFINED"


def test_intraday_loss_details_do_not_require_an_inference_session_axis(tmp_path):
    frame, source, config = _case(tmp_path)
    config.pop("sessions")
    config.update(
        analyses=["model_losses"],
        family_ids=["model"],
        sample_role="exploratory",
        statistics={},
    )

    class Lookup:
        def values_for_frame(self, keys, *, mark_matched):
            assert mark_matched
            return np.arange(len(keys), dtype=float), np.ones(len(keys), dtype=bool)

    window = TimeWindow(**config["window"])
    tables = build_a_share_statistics_tables(
        dataset_session=open_dataset_session(source, time_window=window),
        prediction_id="unused",
        model_entries={
            ("period", "model"): {
                "model_id": "model",
                "model_metadata": {},
                "model_facts": {
                    "model": {
                        "model_algorithm": "linear_regression",
                        "target_column": "label",
                    }
                },
            }
        },
        selections={},
        period_windows={"period": window},
        trial_rows=pd.DataFrame(),
        config=config,
        time_window=window,
        prediction_lookup=Lookup(),
    )
    assert set(tables) == {"model_losses", "model_prediction_metrics"}
    assert (
        tables["model_losses"].timestamp.tolist() == frame.timestamp.unique().tolist()
    )
    assert tables["model_losses"].loss.tolist() == pytest.approx([-1, -1, 1, 1])
