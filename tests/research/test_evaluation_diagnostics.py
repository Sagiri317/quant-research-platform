from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.evaluations import engine as evaluation_engine
from tests.helpers.output_frames import make_artifact_signal_spec


def _assert_recomputed_evaluation_equal(left, right):
    """Compare rerun evaluation values with exact domains and missing masks."""
    continuous = left.select_dtypes(include="floating").columns
    pd.testing.assert_frame_equal(
        left.drop(columns=continuous), right.drop(columns=continuous), check_exact=True
    )
    pd.testing.assert_frame_equal(left.isna(), right.isna(), check_exact=True)
    pd.testing.assert_frame_equal(
        left[continuous], right[continuous], check_exact=False, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("valid_pairs", (None, 0, 1))
def test_single_asset_insufficient_pairs_publish_with_rankic_not_applicable(
    tmp_path: Path, valid_pairs: int | None
) -> None:
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )

    spec = EvaluationSpec(
        evaluation_id="single_undefined",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-02")],
            "asset_id": "asset",
            "feature": 1.0,
            "label": 2.0 if valid_pairs else np.nan,
            "eligible": True,
        }
    )
    if valid_pairs is None:
        frame = frame.iloc[:0]
    with evaluation_engine.build_evaluation_result_from_frames(
        (frame,),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        root = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / spec.evaluation_id,
        )
    metrics = pd.read_parquet(root / "metrics.parquet")
    assert metrics.n_obs.tolist() == ([] if valid_pairs is None else [valid_pairs])
    assert (
        metrics.ic.isna().all()
        and metrics.ic_reason.eq("insufficient_observations").all()
    )
    assert (
        metrics.rankic.isna().all() and metrics.rankic_reason.eq("not_applicable").all()
    )


def test_nanosecond_evaluation_diagnostics_survive_actual_writer_roundtrip(
    tmp_path: Path,
) -> None:
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )

    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "asset",
            "feature": [1.0, 2.0],
            "label": [2.0, 4.0],
            "eligible": True,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="nanosecond_eval",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    with evaluation_engine.build_evaluation_result_from_frames(
        (frame,),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        root = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / spec.evaluation_id,
        )
    validate_output_bundle(root)
    diagnostics = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
        "metadata"
    ]["evaluation_diagnostics"]
    assert diagnostics["min_timestamp"] == times[0].isoformat()
    assert diagnostics["max_timestamp"] == times[-1].isoformat()


def test_evaluation_chunk_producer_releases_consumed_frames(monkeypatch) -> None:
    import weakref

    spec = EvaluationSpec(
        evaluation_id="eval_stream",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    references = []
    original = evaluation_engine._iter_evaluation_work_frames

    def observed(*args, **kwargs):
        with evaluation_engine.closing_iterator(original(*args, **kwargs)) as source:
            for frame in source:
                assert sum(ref() is not None for ref in references) <= 2
                references.append(weakref.ref(frame[0]))
                yield frame
                del frame

    monkeypatch.setattr(evaluation_engine, "_iter_evaluation_work_frames", observed)

    def frames():
        for day in range(16):
            frame = _evaluation_frame()
            frame["timestamp"] += pd.Timedelta(days=day)
            yield frame

    count = 0
    for metrics, grouped, _ in evaluation_engine._iter_evaluation_output_chunks(
        frames(), spec=spec, asset_column="asset_id"
    ):
        count += 1
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
        assert metrics["n_obs"].tolist() == [2]
        assert metrics["ic"].tolist() == pytest.approx([1.0])
    assert count == 16
    assert all(ref() is None for ref in references)


def test_single_asset_time_series_evaluation_is_stable_across_chunks() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="100ms"),
            "asset_id": ["btc_usdt"] * 4,
            "feature": [1.0, 2.0, 3.0, 4.0],
            "label": [2.0, 4.0, 6.0, 8.0],
            "eligible": [True] * 4,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="single_asset_time_series",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    outputs: list[pd.DataFrame] = []
    for frames in ((frame,), (frame.iloc[:2], frame.iloc[2:])):
        with evaluation_engine.build_evaluation_result_from_frames(
            frames,
            spec=spec,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
        ) as result:
            chunks = iter(result.output_chunks)
            owned = next(chunks)
            metrics, grouped, metadata = owned
            with pytest.raises(StopIteration):
                next(chunks)
            summary = result.summaries()[1]
        assert grouped.empty
        assert metadata["timestamp_count"] == 4
        assert metrics.loc[0, "n_obs"] == 4
        assert metrics.loc[0, "ic"] == pytest.approx(1.0)
        assert pd.isna(metrics.loc[0, "rankic"])
        assert metrics.loc[0, "rankic_reason"] == "not_applicable"
        assert summary.loc[0, "valid_ic_count"] == 1
        assert summary.loc[0, "valid_rankic_count"] == 0
        outputs.append(metrics)
    _assert_recomputed_evaluation_equal(outputs[0], outputs[1])


@pytest.mark.parametrize("case", ("ordinary", "missing", "offset"))
def test_single_asset_time_series_values_are_chunk_stable(case: str) -> None:
    row_count = 10007
    random = np.random.default_rng(0)
    if case == "offset":
        feature = random.normal(10000.0, 1.0, size=row_count)
        label = feature * 0.17 + random.normal(-1000.0, 2.0, size=row_count)
    else:
        feature = random.normal(size=row_count)
        label = random.normal(size=row_count)
        if case == "missing":
            feature[::997] = np.nan
            label[::613] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="100ms"),
            "asset_id": ["btc_usdt"] * row_count,
            "feature": feature,
            "label": label,
            "eligible": [True] * row_count,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="single_asset_chunk_stable",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )

    def evaluate(frames) -> pd.DataFrame:
        with evaluation_engine.build_evaluation_result_from_frames(
            frames,
            spec=spec,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
        ) as result:
            chunks = iter(result.output_chunks)
            metrics = next(chunks)[0].copy()
            with pytest.raises(StopIteration):
                next(chunks)
        return metrics

    valid = np.isfinite(feature) & np.isfinite(label)
    x = feature[valid] - np.mean(feature[valid])
    y = label[valid] - np.mean(label[valid])
    expected_ic = np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    chunks_127 = evaluate(
        tuple(frame.iloc[start : start + 127] for start in range(0, row_count, 127))
    )
    chunks_251 = evaluate(
        tuple(frame.iloc[start : start + 251] for start in range(0, row_count, 251))
    )
    for metrics in (chunks_127, chunks_251):
        assert metrics.n_obs.tolist() == [int(valid.sum())]
        assert metrics.ic.tolist() == pytest.approx([expected_ic], rel=1e-11, abs=1e-14)
        assert metrics.ic_reason.isna().all()
        assert metrics.rankic.isna().all()
        assert metrics.rankic_reason.tolist() == ["not_applicable"]
    _assert_recomputed_evaluation_equal(chunks_127, chunks_251)


@pytest.mark.parametrize("epsilon", (1e-07, 1e-08))
def test_streaming_ic_summary_matches_stable_canonical_values(epsilon: float) -> None:
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
        evaluation_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
    )
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
        assert streaming[column] == pytest.approx(expected, rel=0.0, abs=1e-12)
    for column in ("icir", "rankicir"):
        assert streaming[column] == pytest.approx(expected_ratio, rel=1e-08, abs=1e-12)


def test_streaming_group_spread_reducer_avoids_chunk_materialization() -> None:
    import weakref

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
        evaluation_id="streaming_group_spread",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
    )
    references = []

    def chunks():
        for offset in (0, 2):
            assert all(reference() is None for reference in references)
            chunk = grouped.iloc[offset : offset + 2].copy()
            references.append(weakref.ref(chunk))
            yield chunk
            del chunk

    metadata = {
        "rows_input": 4,
        "rows_after_filter_mode": 4,
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "timestamp_count": 2,
        "min_timestamp": "2024-01-02",
        "max_timestamp": "2024-01-03",
        "skipped_timestamp_count": 0,
    }
    for chunk in chunks():
        state.consume(pd.DataFrame(), chunk, metadata)
        del chunk
    assert all(reference() is None for reference in references)
    actual = state.group_spread_summary()
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


def test_evaluation_writer_roundtrips_long_windows_path(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.storage.identity import native_filesystem_path

    destination = tmp_path / ("long" * 40) / ("path" * 40) / "eval"
    spec = EvaluationSpec(
        evaluation_id="eval_long",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    with evaluation_engine.build_evaluation_result_from_frames(
        (_evaluation_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        root = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=destination,
        )
    assert len(str(root / "grouped_label_values.parquet")) > 260
    with pytest.raises(ValueError, match="output_already_exists"):
        write_evaluation_outputs_from_chunks(
            (),
            summary_factory=lambda: pytest.fail("existing_output_was_rewritten"),
            spec=spec,
            output_path=destination,
        )
    metrics = pd.read_parquet(native_filesystem_path(root / "metrics.parquet"))
    assert metrics.n_obs.tolist() == [2]
    assert metrics.ic.iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_evaluation_failure_closes_source_with_traceback_retained(
    tmp_path, monkeypatch, cleanup_fails
):
    from quant_research_platform.research.evaluations import runner
    from tests.research.crypto_evaluation_test_support import (
        evaluation_config,
        write_evaluation_input,
    )

    frame = _evaluation_frame().assign(label_available_at=pd.Timestamp("2024-01-02"))
    root = write_evaluation_input(
        tmp_path / "inputs", frame, features=("feature",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="error",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    primary = PermissionError("input_read_failure")
    closed = []

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary

        def close(self):
            closed.append(True)
            if cleanup_fails:
                raise RuntimeError("secondary_close")

    monkeypatch.setattr(runner, "iter_dataset_chunks", lambda *a, **k: Source())
    with pytest.raises(PermissionError) as caught:
        runner.evaluate(
            evaluation_config(
                root, tmp_path / "out", spec, start="2024-01-02", end="2024-01-03"
            )
        )
    assert caught.value is primary and primary.__traceback__ is not None
    assert closed == [True]
    assert not (tmp_path / "out").exists()
    if cleanup_fails:
        assert any(
            "iterator close also failed: RuntimeError" in note
            for note in primary.__notes__
        )


@pytest.mark.parametrize(
    "values",
    [
        (0.1, 0.2, 0.3),
        (0.0, 0.0001),
        (-0.2, 0.0, 0.2),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
def test_ic_summary_nonzero_variance_survives_chunking_and_persistence(
    values, reverse, chunk_size, tmp_path
):
    from statistics import mean, pstdev

    values = values[::-1] if reverse else values
    metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=len(values)),
            "signal": "signal",
            "label": "label",
            "ic": values,
            "rankic": values,
            "n_obs": 2,
            "ic_reason": None,
            "rankic_reason": None,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
    )
    for start in range(0, len(metrics), chunk_size):
        state.consume(
            metrics.iloc[start : start + chunk_size],
            pd.DataFrame(),
            {
                "rows_input": 2,
                "rows_after_filter_mode": 2,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible",
                "timestamp_count": 1,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-04",
                "skipped_timestamp_count": 0,
            },
        )
    summary = state.ic_summary()
    path = tmp_path / "summary.parquet"
    summary.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(restored, summary, check_exact=True)
    row = restored.iloc[0]
    expected_std = pstdev(values)
    for prefix in ("ic", "rankic"):
        assert row[f"std_{prefix}"] == pytest.approx(expected_std, rel=1e-12, abs=1e-15)
        assert row[f"{prefix}ir"] == pytest.approx(
            mean(values) / expected_std, rel=1e-12, abs=1e-14
        )
        assert row[f"{prefix}ir_reason"] is None


def test_persisted_unfiltered_evaluation_keeps_ineligible_mature_labels(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import (
        evaluation_config,
        write_evaluation_input,
    )

    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": pd.Timestamp("2023-01-02"),
            "signal": [1.0, 2.0, 3.0, 100.0],
            "label": [2.0, 4.0, 6.0, -100.0],
            "available_at": pd.to_datetime(["2023-01-02"] * 3 + ["2023-01-04"]),
            "eligible": [True, False, True, True],
        }
    )
    root = write_evaluation_input(
        tmp_path / "inputs", frame, factors=("signal",), labels=("label",)
    )
    for mode, count in (("unfiltered", 3), ("eligible", 2)):
        spec = EvaluationSpec(
            evaluation_id=mode,
            market_id="a_share",
            research_dataset_id="fixture",
            signals=(
                SignalSpec(
                    signal_id="signal", source_kind="factor", source_column="signal"
                ),
            ),
            label_columns=("label",),
            label_availability_columns={"label": "available_at"},
            filter_mode=mode,
            n_groups=2,
        )
        config = evaluation_config(
            root, tmp_path / mode, spec, start="2023-01-02", end="2023-01-03"
        )
        output = evaluate(config)
        manifest, tables = read_evaluation_outputs(output)
        assert tables["metrics"].n_obs.tolist() == [count]
        assert tables["metrics"].ic.tolist() == pytest.approx([1.0])
        for field in ("window", "cutoff", "sample_role"):
            assert manifest["metadata"][field] == config[field]


def test_explicit_crypto_direction_reaches_native_metadata_metrics_and_groups(tmp_path):
    from dataclasses import replace

    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.signals import SignalSpec

    factors = ["spread", "relative_spread", "ofi_500ms"]
    label = "future_mid_return_500ms"
    spec = EvaluationSpec(
        evaluation_id="direction",
        market_id="crypto",
        research_dataset_id="fixture",
        signals=tuple(
            SignalSpec(
                signal_id=name,
                source_column=name,
                source_kind="factor",
                direction="lower_is_better"
                if name in factors[:2]
                else "higher_is_better",
            )
            for name in factors
        ),
        label_columns=(label,),
        n_groups=3,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-08-01T13:00:00"),
            "asset_id": ["a", "b", "c"],
            "spread": [1.0, 2.0, 3.0],
            "relative_spread": [0.01, 0.02, 0.03],
            "ofi_500ms": [1.0, 2.0, 3.0],
            label: [0.1, 0.2, 0.3],
            "eligible": True,
        }
    )
    old = replace(
        spec,
        signals=tuple(
            replace(signal, direction="higher_is_better") for signal in spec.signals
        ),
    )
    roots = []
    for index, active in enumerate((old, spec)):
        with evaluation_engine.build_evaluation_result_from_frames(
            (frame,),
            spec=active,
            asset_column="asset_id",
            source_columns_by_kind={"factor": tuple(factors)},
        ) as result:
            root = write_evaluation_outputs_from_chunks(
                result.output_chunks,
                summary_factory=result.summaries,
                spec=active,
                output_path=tmp_path / str(index),
            )
        roots.append(root)
        manifest, tables = read_evaluation_outputs(root)
        assert [item["direction"] for item in manifest["metadata"]["signals"]] == [
            item.direction for item in active.signals
        ]
        expected = [1.0, 1.0, 1.0] if index == 0 else [-1.0, -1.0, 1.0]
        metrics = tables["metrics"].set_index("signal")
        for column in ("ic", "rankic"):
            assert metrics.loc[factors, column].tolist() == pytest.approx(expected)
        if index == 0:
            old_bytes = {p.name: p.read_bytes() for p in root.iterdir()}
        else:
            groups = tables["grouped_label_values"]
            top = groups.loc[groups.group.eq(3)].set_index("signal")
            assert top.loc[factors, "mean_label_value"].tolist() == pytest.approx(
                [0.1, 0.1, 0.3]
            )
    assert {p.name: p.read_bytes() for p in roots[0].iterdir()} == old_bytes


def _write_count_evaluation(tmp_path, count):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.evaluations import (
        EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        EVALUATION_METRICS_COLUMNS,
    )

    spec = EvaluationSpec(
        evaluation_id="count_summary",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    metrics = pd.DataFrame(
        [
            [
                pd.Timestamp("2023-01-02"),
                "signal",
                "label",
                1.0,
                None,
                count,
                None,
                "not_applicable",
            ]
        ],
        columns=EVALUATION_METRICS_COLUMNS,
    )
    grouped = pd.DataFrame(columns=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS)
    metadata = {
        "rows_input": count,
        "rows_after_filter_mode": count,
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "timestamp_count": count,
        "min_timestamp": "2023-01-02T00:00:00",
        "max_timestamp": "2023-01-03T00:00:00",
        "skipped_timestamp_count": 0,
    }
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
    )
    state.consume(metrics, grouped, metadata)

    def summaries():
        return (
            state.diagnostics(),
            state.ic_summary(),
            state.group_spread_summary(),
        )

    root = write_evaluation_outputs_from_chunks(
        [(metrics, grouped, metadata)],
        summary_factory=summaries,
        spec=spec,
        output_path=tmp_path / "counts",
    )
    return root, spec, (metrics, grouped, metadata)


@pytest.mark.parametrize("count", [99999, 100000, 303767, 400000, 500000, 600000])
def test_evaluation_writer_preserves_counts_without_text_workarounds(tmp_path, count):
    from quant_research_platform.artifacts.outputs.manifest import (
        validate_output_bundle,
    )

    root, _, _ = _write_count_evaluation(tmp_path, count)
    manifest, _ = validate_output_bundle(root)
    diagnostics = manifest["metadata"]["evaluation_diagnostics"]
    for field in ("rows_input", "rows_after_filter_mode", "timestamp_count"):
        assert type(diagnostics[field]) is int and diagnostics[field] == count
    assert pd.read_parquet(root / "metrics.parquet").n_obs.tolist() == [count]


def test_evaluation_summary_failure_preserves_existing_output(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )

    root, spec, chunk = _write_count_evaluation(tmp_path, 3)
    old_bytes = {p.name: p.read_bytes() for p in root.iterdir()}
    primary = RuntimeError("injected_report_failure")

    def fail_summary():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        write_evaluation_outputs_from_chunks(
            [chunk],
            summary_factory=fail_summary,
            spec=spec,
            output_path=tmp_path / "failed_summary",
        )
    assert caught.value is primary
    assert {p.name: p.read_bytes() for p in root.iterdir()} == old_bytes
    assert not (root.parent / "failed_summary").exists()


def test_evaluation_early_failure_closes_native_source_without_gc(
    tmp_path, monkeypatch
):
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.evaluations import runner
    from tests.research.crypto_evaluation_test_support import (
        evaluation_config,
        write_evaluation_input,
    )

    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"] * 3,
            "timestamp": pd.date_range("2023-01-02", periods=3).repeat(3),
            "signal": [1.0, 2.0, 3.0] * 3,
            "label": [2.0, 4.0, 6.0] * 3,
            "available_at": pd.date_range("2023-01-02", periods=3).repeat(3),
            "eligible": True,
        }
    )
    root = write_evaluation_input(
        tmp_path / "input", frame, factors=("signal",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="close",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="signal", source_column="signal", source_kind="factor"
            ),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "available_at"},
        n_groups=2,
    )
    read = runner.iter_dataset_chunks
    closed = []

    def chunks(*args, **kwargs):
        with closing_iterator(read(*args, **kwargs)) as source:
            try:
                for chunk in source:
                    for _, rows in chunk.groupby("timestamp", sort=False):
                        yield rows
            finally:
                closed.append(True)

    primary = KeyboardInterrupt("stop_before_consumption_completed")

    def interrupt(chunks, **kwargs):
        next(iter(chunks))
        raise primary

    monkeypatch.setattr(runner, "iter_dataset_chunks", chunks)
    monkeypatch.setattr(runner, "write_evaluation_outputs_from_chunks", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.evaluate(
            evaluation_config(
                root, tmp_path / "out", spec, start="2023-01-02", end="2023-01-04"
            )
        )
    assert caught.value is primary
    assert closed == [True]
    assert not (tmp_path / "out").exists()


def test_single_asset_cancel_preserves_close_failure(monkeypatch):
    import sys
    import traceback

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2),
            "asset_id": ["btc_usdt"] * 2,
            "feature": [1.0, 2.0],
            "label": [2.0, 4.0],
            "eligible": [True, True],
        }
    )
    spec = EvaluationSpec(
        evaluation_id="cancel",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    failure = KeyboardInterrupt("cancel")
    closed = []
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            return frame

        def close(self):
            closed.append(True)
            raise RuntimeError("source_close_failed")

    def interrupt(*args, **kwargs):
        raise failure

    monkeypatch.setattr(evaluation_engine, "normalize_signal_frame", interrupt)
    with evaluation_engine.build_evaluation_result_from_frames(
        Source(),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        with pytest.raises(KeyboardInterrupt) as caught:
            next(iter(result.output_chunks))
        assert caught.value is failure
        assert "interrupt" in [
            item.name for item in traceback.extract_tb(failure.__traceback__)
        ]
        assert any(
            "iterator close also failed: RuntimeError" in note
            for note in failure.__notes__
        )
        assert closed == [True]
        assert not unraisable
    assert closed == [True]


@pytest.fixture
def explicit_evaluation(tmp_path):
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
    from tests.research.crypto_evaluation_test_support import (
        evaluation_config,
        write_evaluation_input,
    )

    times = pd.date_range("2026-01-01", periods=8, freq="1s")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "btc_usdt",
            "x": np.arange(1.0, 9.0),
            "spread": np.arange(1.0, 9.0) / 100,
            "short": np.arange(1.0, 9.0) * 2,
            "long": np.arange(1.0, 9.0) ** 2,
            "short_available_at": times,
            "long_available_at": times,
            "eligible": [True] * 7 + [False],
        }
    )
    frame.loc[4, "x"] = np.nan
    frame.loc[5, "short_available_at"] = times[-1] + pd.Timedelta(seconds=1)
    frame["label_available_at"] = frame.short_available_at
    inputs = write_evaluation_input(
        tmp_path / "inputs",
        frame,
        features=("x",),
        factors=("spread",),
        labels=("short", "long"),
    )
    model = fit(
        {
            "inputs": str(inputs),
            "output": str(tmp_path / "model"),
            "market_id": "crypto",
            "model": {
                "model_id": "linear",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["x"],
                "target_column": "short",
                "random_seed": 17,
            },
            "fit_windows": [
                {"start": times[0].isoformat(), "end": times[2].isoformat()}
            ],
            "fit_cutoff": times[2].isoformat(),
            "label_unit": "decimal_return",
        }
    )
    window = {"start": times[3].isoformat(), "end": times[-1].isoformat()}
    predictions = predict(
        {
            "inputs": str(inputs),
            "model": str(model),
            "output": str(tmp_path / "predictions"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    spec = EvaluationSpec(
        evaluation_id="pairwise",
        market_id="crypto",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="pred", source_kind="prediction", source_column="pred"
            ),
            SignalSpec(
                signal_id="spread",
                source_kind="factor",
                source_column="spread",
                direction="lower_is_better",
            ),
        ),
        label_columns=("short", "long"),
        label_availability_columns={
            "short": "short_available_at",
            "long": "long_available_at",
        },
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    config = evaluation_config(inputs, tmp_path / "evaluation", spec, **window)
    config["predictions"] = {"pred": str(predictions)}
    return frame, spec, config


def test_direct_evaluation_preserves_each_mature_pair_and_late_label_invariance(
    tmp_path,
    explicit_evaluation,
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import (
        assert_pairwise_maturity_and_ic,
        write_evaluation_input,
    )

    frame, spec, config = explicit_evaluation
    predictions, _ = read_predictions(Path(config["predictions"]["pred"]))
    work = frame.loc[frame.timestamp.ge(pd.Timestamp(config["window"]["start"]))].merge(
        predictions[["timestamp", "asset_id", "prediction_score"]].rename(
            columns={"prediction_score": "pred"}
        ),
        on=["timestamp", "asset_id"],
        how="left",
        validate="one_to_one",
    )
    output = evaluate(config)
    manifest, tables = read_evaluation_outputs(output)
    assert_pairwise_maturity_and_ic(tables["metrics"], work, spec, config["cutoff"])
    assert tables["metrics"].set_index(["signal", "label"]).n_obs.to_dict() == {
        ("pred", "short"): 2,
        ("pred", "long"): 3,
        ("spread", "short"): 3,
        ("spread", "long"): 4,
    }
    diagnostics = manifest["metadata"]["evaluation_diagnostics"]
    assert diagnostics["min_timestamp"] == config["window"]["start"]
    # Last row is ineligible; the metadata's observed input extent may include it,
    # but no sample past the requested evaluation interval is consumed.
    assert pd.Timestamp(diagnostics["max_timestamp"]) <= pd.Timestamp(
        config["window"]["end"]
    )
    changed = frame.copy()
    changed.loc[
        changed.short_available_at.gt(pd.Timestamp(config["cutoff"])), "short"
    ] = 999.0
    changed_input = write_evaluation_input(
        tmp_path / "late-labels",
        changed,
        features=("x",),
        factors=("spread",),
        labels=("short", "long"),
    )
    other = evaluate(
        {
            **config,
            "inputs": str(changed_input),
            "output": str(tmp_path / "late-output"),
        }
    )
    _, actual = read_evaluation_outputs(other)
    for name in tables:
        _assert_recomputed_evaluation_equal(tables[name], actual[name])


@pytest.mark.parametrize("change", ("drop", "null"))
def test_direct_evaluation_does_not_requalify_saved_predictions(
    tmp_path, explicit_evaluation, change
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame, _spec, config = explicit_evaluation
    _, expected = read_evaluation_outputs(evaluate(config))
    changed = frame.drop(columns="x") if change == "drop" else frame.assign(x=np.nan)
    inputs = write_evaluation_input(
        tmp_path / "current-labels",
        changed,
        features=() if change == "drop" else ("x",),
        factors=("spread",),
        labels=("short", "long"),
    )
    _, actual = read_evaluation_outputs(
        evaluate(
            {
                **config,
                "inputs": str(inputs),
                "output": str(tmp_path / "current-evaluation"),
            }
        )
    )
    for name in expected:
        _assert_recomputed_evaluation_equal(expected[name], actual[name])


@pytest.mark.parametrize("damage", ("missing", "extra", "unusable_extra"))
def test_direct_evaluation_pairs_actual_prediction_keys_with_current_mature_labels(
    tmp_path,
    explicit_evaluation,
    damage,
    monkeypatch,
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )
    from quant_research_platform.research.evaluations import runner
    from tests.research.crypto_evaluation_test_support import (
        assert_pairwise_maturity_and_ic,
    )

    frame, spec, config = explicit_evaluation
    predictions, metadata = read_predictions(Path(config["predictions"]["pred"]))
    if damage == "missing":
        # A missing saved prediction affects only pairs that actually use its key.
        predictions = predictions.loc[predictions.timestamp.ne(frame.timestamp.iloc[5])]
    else:
        extra = predictions.iloc[[0]].copy()
        if damage == "extra":
            extra.asset_id = "eth_usdt"
        else:
            extra.timestamp = frame.timestamp.iloc[4]
        predictions = pd.concat((predictions, extra)).sort_values(
            ["timestamp", "asset_id"]
        )
    mutated = write_prediction_outputs(
        output_path=tmp_path / "mutated",
        output_chunks=(predictions,),
        metadata={
            **metadata,
            "prediction_rows": None,
            "effective_prediction_end": None,
        },
    )
    source_chunks = runner.iter_dataset_chunks
    merge = pd.DataFrame.merge
    chunk_sizes = []

    def small_chunks(*args, **kwargs):
        for chunk in source_chunks(*args, **kwargs):
            for offset in range(0, len(chunk), 2):
                piece = chunk.iloc[offset : offset + 2].copy()
                piece.index = np.arange(100 + offset, 100 + offset + len(piece))
                chunk_sizes.append(len(piece))
                yield piece

    def no_prediction_merge(left, right, *args, **kwargs):
        assert "pred" not in right, "repeated whole-prediction merge"
        return merge(left, right, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(runner, "iter_dataset_chunks", small_chunks)
        patch.setattr(pd.DataFrame, "merge", no_prediction_merge)
        _, tables = read_evaluation_outputs(
            runner.evaluate({**config, "predictions": {"pred": str(mutated)}})
        )
    assert len(chunk_sizes) > 1
    work = frame.loc[frame.timestamp.ge(pd.Timestamp(config["window"]["start"]))].merge(
        predictions[["timestamp", "asset_id", "prediction_score"]].rename(
            columns={"prediction_score": "pred"}
        ),
        on=["timestamp", "asset_id"],
        how="left",
        validate="one_to_one",
    )
    assert_pairwise_maturity_and_ic(tables["metrics"], work, spec, config["cutoff"])


@pytest.mark.parametrize("overlap", ("head", "tail", "none", "different_windows"))
def test_prediction_windows_only_constrain_their_actual_pairs(
    tmp_path, explicit_evaluation, overlap
):
    from dataclasses import replace

    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import (
        assert_pairwise_maturity_and_ic,
    )

    frame, spec, config = explicit_evaluation
    _, baseline = read_evaluation_outputs(evaluate(config))
    config["output"] = str(tmp_path / "partial-windows")
    predictions, metadata = read_predictions(Path(config["predictions"]["pred"]))
    bounds = (5, 7) if overlap in {"head", "different_windows"} else (3, 5)
    selections = {"pred": bounds}
    if overlap == "different_windows":
        selections["other"] = (3, 5)
        other = SignalSpec(
            signal_id="other", source_kind="prediction", source_column="other"
        )
        spec = replace(spec, signals=(*spec.signals, other))
        config["evaluation"]["signals"].append(
            {
                "signal_id": "other",
                "source_kind": "prediction",
                "source_column": "other",
            }
        )
    if overlap == "none":
        config["window"] = {
            "start": frame.timestamp.iloc[0].isoformat(),
            "end": frame.timestamp.iloc[2].isoformat(),
        }
    work = frame.loc[
        frame.timestamp.between(
            pd.Timestamp(config["window"]["start"]),
            pd.Timestamp(config["window"]["end"]),
        )
    ]
    for name, (first, last) in selections.items():
        start, end = frame.timestamp.iloc[[first, last]]
        selected = predictions.loc[predictions.timestamp.between(start, end)]
        root = write_prediction_outputs(
            output_path=tmp_path / f"prediction-{name}",
            output_chunks=(selected,),
            metadata={
                **metadata,
                "prediction_window": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                "prediction_rows": None,
                "effective_prediction_end": None,
            },
        )
        config["predictions"][name] = str(root)
        work = work.merge(
            selected[["timestamp", "asset_id", "prediction_score"]].rename(
                columns={"prediction_score": name}
            ),
            on=["timestamp", "asset_id"],
            how="left",
            validate="one_to_one",
        )
    _, tables = read_evaluation_outputs(evaluate(config))
    assert_pairwise_maturity_and_ic(tables["metrics"], work, spec, config["cutoff"])
    if overlap == "none":
        empty = tables["metrics"].loc[tables["metrics"].signal.eq("pred")]
        assert empty.n_obs.eq(0).all() and empty.ic.isna().all()
    else:
        for name in tables:
            # Other signals' new reasons can change pandas' nullable-string
            # inference; the native reader already enforces the persisted schema.
            left, right = (
                tables[name]
                .loc[tables[name].signal.eq("spread")]
                .reset_index(drop=True),
                baseline[name]
                .loc[baseline[name].signal.eq("spread")]
                .reset_index(drop=True),
            )
            continuous = left.select_dtypes(include="floating").columns
            pd.testing.assert_frame_equal(
                left.drop(columns=continuous),
                right.drop(columns=continuous),
                check_exact=True,
                check_dtype=False,
            )
            pd.testing.assert_frame_equal(
                left.isna(), right.isna(), check_exact=True, check_dtype=False
            )
            pd.testing.assert_frame_equal(
                left[continuous],
                right[continuous],
                check_exact=False,
                rtol=1e-12,
                atol=1e-15,
            )


def test_evaluation_collision_never_opens_input_or_overwrites(tmp_path, monkeypatch):
    from quant_research_platform.research.evaluations import runner

    output = tmp_path / "old-good"
    output.mkdir()
    marker = output / "retained"
    marker.write_bytes(b"old-good")
    monkeypatch.setattr(
        runner,
        "iter_dataset_chunks",
        lambda *a, **k: pytest.fail("collision consumed input"),
    )
    with pytest.raises(ValueError, match="already_exists"):
        runner.evaluate(
            {
                "inputs": "missing",
                "output": str(output),
                "evaluation": {},
                "window": {},
                "cutoff": "NaT",
                "sample_role": "test",
            }
        )
    assert marker.read_bytes() == b"old-good"
    assert tuple(output.iterdir()) == (marker,)


@pytest.mark.parametrize(
    "damage", ("role", "empty_role", "cutoff", "window", "missing_field")
)
def test_evaluation_native_science_metadata_is_validated_on_read(
    tmp_path,
    explicit_evaluation,
    damage,
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate

    _frame, _spec, config = explicit_evaluation
    root = evaluate(config)
    path = root / "manifest.json"
    value = json.loads(path.read_text())
    if damage == "role":
        value["metadata"]["sample_role"] = "holdout"
    elif damage == "empty_role":
        value["metadata"]["sample_role"] = ""
    elif damage == "cutoff":
        value["metadata"]["cutoff"] = "NaT"
    elif damage == "window":
        value["metadata"]["window"]["start"] = "2027-01-01"
    else:
        value["metadata"].pop("cutoff")
    path.write_text(json.dumps(value))
    if damage == "role":
        manifest, _tables = read_evaluation_outputs(root)
        assert manifest["metadata"]["sample_role"] == "holdout"
    else:
        with pytest.raises(ValueError):
            read_evaluation_outputs(root)


def test_direct_evaluation_partial_publication_keeps_primary_close_note_and_old_good(
    tmp_path,
    explicit_evaluation,
    monkeypatch,
):
    from quant_research_platform.research.evaluations import runner
    from quant_research_platform.research.evaluations.engine import EvaluationResult

    _frame, _spec, config = explicit_evaluation
    root = runner.evaluate(config)
    old = {p.name: p.read_bytes() for p in root.iterdir()}
    build = runner.build_evaluation_result_from_frames
    primary = KeyboardInterrupt("after_native_output_chunk")
    closed = []

    def interrupted(*args, **kwargs):
        result = build(*args, **kwargs)

        def chunks():
            for chunk in result.output_chunks:
                yield chunk
                raise primary

        source = chunks()

        def close():
            source.close()
            result.close()
            closed.append(True)
            raise RuntimeError("evaluation_cleanup_failed")

        return EvaluationResult(source, result.summaries, close)

    monkeypatch.setattr(runner, "build_evaluation_result_from_frames", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.evaluate({**config, "output": str(tmp_path / "interrupted")})
    assert caught.value is primary and closed == [True]
    assert any("RuntimeError" in note for note in primary.__notes__)
    assert not (tmp_path / "interrupted").exists()
    assert old == {p.name: p.read_bytes() for p in root.iterdir()}


def test_evaluation_preserves_each_explicit_prediction_asset_scope(
    tmp_path,
    explicit_evaluation,
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.predictions.runner import predict
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame, _spec, config = explicit_evaluation
    eth = frame.copy()
    eth.asset_id = "eth_usdt"
    both = pd.concat((frame, eth)).sort_values(["timestamp", "asset_id"])
    inputs = write_evaluation_input(
        tmp_path / "two-assets",
        both,
        features=("x",),
        factors=("spread",),
        labels=("short", "long"),
    )
    _, metadata = read_predictions(Path(config["predictions"]["pred"]))
    prediction = predict(
        {
            "inputs": str(inputs),
            "output": str(tmp_path / "scoped"),
            "model": metadata["model"],
            "prediction_window": config["window"],
            "sample_role": "validation",
            "asset_ids": ["btc_usdt"],
        }
    )
    # The factor retains both assets, so a single-asset IC would be ill-defined;
    # here evaluate only the scoped prediction to keep its actual scientific axis.
    evaluation = {
        **config["evaluation"],
        "signals": [config["evaluation"]["signals"][0]],
    }
    root = evaluate(
        {
            **config,
            "inputs": str(inputs),
            "evaluation": evaluation,
            "predictions": {"pred": str(prediction)},
        }
    )
    _, tables = read_evaluation_outputs(root)
    assert tables["metrics"].set_index("label").n_obs.to_dict() == {
        "short": 2,
        "long": 3,
    }


@pytest.mark.parametrize(
    "window,stamp,accepted",
    (
        ({"start": "2020-01-02", "end": "2020-01-02"}, "2020-01-02T12:00:00", True),
        (
            {"start": "2020-01-02", "end": "2020-01-02"},
            "2020-01-02T23:59:59.999999999",
            True,
        ),
        ({"start": "2020-01-02", "end": "2020-01-02"}, "2020-01-03T00:00:00", False),
        (
            {"start": "2020-01-02T00:00:00", "end": "2020-01-02T00:00:00.000000001"},
            "2020-01-02T00:00:00.000000001",
            True,
        ),
        (
            {"start": "2020-01-02T00:00:00", "end": "2020-01-02T00:00:00.000000001"},
            "2020-01-02T00:00:00.000000002",
            False,
        ),
    ),
)
def test_evaluation_science_window_distinguishes_calendar_day_from_exact_ns(
    tmp_path,
    window,
    stamp,
    accepted,
):
    from contextlib import nullcontext

    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
        write_evaluation_outputs_from_chunks,
    )

    frame = _evaluation_frame().assign(timestamp=pd.Timestamp(stamp))
    spec = EvaluationSpec(
        evaluation_id="day_bounds",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    with (
        nullcontext() if accepted else pytest.raises(ValueError, match="outside_window")
    ):
        with evaluation_engine.build_evaluation_result_from_frames(
            (frame,),
            spec=spec,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
        ) as result:
            root = write_evaluation_outputs_from_chunks(
                result.output_chunks,
                summary_factory=result.summaries,
                spec=spec,
                output_path=tmp_path / "out",
                metadata={
                    "window": window,
                    "cutoff": "2020-01-03",
                    "sample_role": "validation",
                },
            )
        metadata, restored = read_evaluation_outputs(root)
        assert metadata["metadata"]["window"] == window
        assert restored["metrics"].timestamp.iloc[0] == pd.Timestamp(stamp)
    assert (tmp_path / "out").exists() is accepted


@pytest.mark.parametrize("member", ("metrics", "grouped_label_values"))
@pytest.mark.parametrize("damage", ("string", "nat", "outside"))
def test_native_evaluation_consumed_timestamp_must_be_typed_and_in_window(
    tmp_path,
    member,
    damage,
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
        write_evaluation_outputs_from_chunks,
    )

    spec = EvaluationSpec(
        evaluation_id="timestamp",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    science = {
        "window": {"start": "2024-01-02", "end": "2024-01-02"},
        "cutoff": "2024-01-03",
        "sample_role": "validation",
    }
    with evaluation_engine.build_evaluation_result_from_frames(
        (_evaluation_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        metrics, groups, diagnostics = next(result.output_chunks)
        root = write_evaluation_outputs_from_chunks(
            [(metrics, groups, diagnostics)],
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / "good",
            metadata=science,
        )
        broken = (metrics if member == "metrics" else groups).copy()
        broken["timestamp"] = {
            "string": "not-a-time",
            "nat": pd.NaT,
            "outside": pd.Timestamp("2024-01-03"),
        }[damage]
        changed = (
            (broken, groups, diagnostics)
            if member == "metrics"
            else (metrics, broken, diagnostics)
        )
        with pytest.raises(
            ValueError, match="timestamp|outside_window|primary_key_null"
        ):
            write_evaluation_outputs_from_chunks(
                [changed],
                summary_factory=result.summaries,
                spec=spec,
                output_path=tmp_path / "bad",
                metadata=science,
            )
    assert not (tmp_path / "bad").exists()
    broken.to_parquet(root / f"{member}.parquet", index=False)
    with pytest.raises(ValueError, match="timestamp|outside_window|primary_key_null"):
        read_evaluation_outputs(root, tables=(member,))
    # Summary consumption never requests the now damaged unselected raw member.
    _, summaries = read_evaluation_outputs(root, tables=("ic_summary",))
    assert summaries["ic_summary"].mean_ic.iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize("damage", ("key", "count", "missing_group"))
def test_native_evaluation_groups_bind_to_their_actual_metric_pair(tmp_path, damage):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
        write_evaluation_outputs_from_chunks,
    )

    spec = EvaluationSpec(
        evaluation_id="binding",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    science = {
        "window": {"start": "2024-01-02", "end": "2024-01-02"},
        "cutoff": "2024-01-03",
        "sample_role": "validation",
    }
    with evaluation_engine.build_evaluation_result_from_frames(
        (_evaluation_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        metrics, groups, diagnostics = next(result.output_chunks)
        root = write_evaluation_outputs_from_chunks(
            [(metrics, groups, diagnostics)],
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / "good",
            metadata=science,
        )
        broken = groups.copy()
        if damage == "key":
            broken["timestamp"] += pd.Timedelta(nanoseconds=1)
        elif damage == "count":
            broken.loc[broken.index[0], "n_obs"] += 1
        else:
            broken = broken.iloc[1:]
        with pytest.raises(ValueError, match="metric_group_observations_mismatch"):
            write_evaluation_outputs_from_chunks(
                [(metrics, broken, diagnostics)],
                summary_factory=result.summaries,
                spec=spec,
                output_path=tmp_path / "bad",
                metadata=science,
            )
    assert not (tmp_path / "bad").exists()
    broken.to_parquet(root / "grouped_label_values.parquet", index=False)
    with pytest.raises(
        ValueError, match="metric_group_observations_mismatch|row_count_mismatch"
    ):
        read_evaluation_outputs(root, tables=("metrics", "grouped_label_values"))
    # No preflight or hidden read of either unrequested observation table.
    _, summaries = read_evaluation_outputs(
        root, tables=("ic_summary", "group_spread_summary")
    )
    assert summaries["ic_summary"].mean_ic.iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "change", ("missing_labels", "immature", "ineligible", "different_keys")
)
def test_direct_evaluation_uses_current_label_domain(
    tmp_path, explicit_evaluation, change
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import (
        assert_pairwise_maturity_and_ic,
        write_evaluation_input,
    )

    frame, spec, config = explicit_evaluation
    frame = frame.drop(columns="x")
    if change == "missing_labels":
        frame[["short", "long"]] = np.nan
    elif change == "immature":
        frame[["short_available_at", "long_available_at"]] = pd.Timestamp(
            config["cutoff"]
        ) + pd.Timedelta(days=1)
    elif change == "ineligible":
        frame["eligible"] = False
    else:
        frame["asset_id"] = "eth_usdt"
    inputs = write_evaluation_input(
        tmp_path / "current-domain",
        frame,
        factors=("spread",),
        labels=("short", "long"),
    )
    _, tables = read_evaluation_outputs(
        evaluate(
            {
                **config,
                "inputs": str(inputs),
                "output": str(tmp_path / "current-domain-eval"),
            }
        )
    )
    predictions, _ = read_predictions(Path(config["predictions"]["pred"]))
    work = frame.loc[frame.timestamp.ge(pd.Timestamp(config["window"]["start"]))].merge(
        predictions[["timestamp", "asset_id", "prediction_score"]].rename(
            columns={"prediction_score": "pred"}
        ),
        on=["timestamp", "asset_id"],
        how="left",
        validate="one_to_one",
    )
    assert_pairwise_maturity_and_ic(tables["metrics"], work, spec, config["cutoff"])
    prediction_metrics = tables["metrics"].loc[tables["metrics"].signal.eq("pred")]
    assert prediction_metrics.n_obs.eq(0).all()
    assert prediction_metrics.ic.isna().all()
    assert prediction_metrics.ic_reason.eq("insufficient_observations").all()


def test_prediction_only_evaluation_needs_no_factor_or_model_input(
    tmp_path, explicit_evaluation
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame, _spec, config = explicit_evaluation
    inputs = write_evaluation_input(
        tmp_path / "only-labels",
        frame.drop(columns=["x", "spread"]),
        labels=("short", "long"),
    )
    config = {**config, "inputs": str(inputs)}
    # A requested factor is still required even alongside persisted predictions.
    with pytest.raises(ValueError, match="evaluation_signal_role_mismatch"):
        evaluate(config)
    config["evaluation"] = {
        **config["evaluation"],
        "signals": config["evaluation"]["signals"][:1],
    }
    _, tables = read_evaluation_outputs(evaluate(config))
    assert tables["metrics"].set_index("label").n_obs.to_dict() == {
        "short": 2,
        "long": 3,
    }
