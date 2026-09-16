"""Regression oracles for real-number, identity and live-payload boundaries."""

from __future__ import annotations

import gc
import json
import tracemalloc
import weakref
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
    _stable_zscore,
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
    strict_non_negative_count_series,
    strict_non_negative_count_value,
    strict_positive_group_series,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.validation import (
    strict_finite_float,
    strict_finite_numeric_series,
    strict_real_numeric_series,
)
from quant_research_platform.research.evaluations.engine import (
    _add_group_spread_observation,
    _EvaluationStreamingState,
    _merge_time_series_moments,
    _time_series_metrics_frame,
)
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.risk import MarketModelRiskState
from quant_research_platform.research.statistics import cross_sectional_rank_ic
from quant_research_platform.runtime.dataframes import iter_complete_time_sized_frames
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage import semantic_artifacts
from quant_research_platform.storage.datasets.schema import (
    _typed_category_value_payload,
    parse_logical_dtype_metadata,
)
from tests.storage.test_semantic_artifacts import _envelope


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.0, 1e308, 1e308], [-(2**0.5), 2**-0.5, 2**-0.5]),
        ([-1e308, 0.0, 1e308], [-(1.5**0.5), 0.0, 1.5**0.5]),
        ([0.0, 1e-200, 1e-200], [-(2**0.5), 2**-0.5, 2**-0.5]),
        ([1e-320, 2e-320, 3e-320], [-(1.5**0.5), 0.0, 1.5**0.5]),
        (
            [
                1e308,
                np.nextafter(1e308, np.inf),
                np.nextafter(np.nextafter(1e308, np.inf), np.inf),
            ],
            [-(1.5**0.5), 0.0, 1.5**0.5],
        ),
    ],
)
def test_centered_kernels_preserve_finite_variation(values, expected):
    values = np.asarray(values)
    correlation, reason = _pearson_correlation_result(values, values)
    assert reason is None
    assert correlation == pytest.approx(1.0)
    np.testing.assert_allclose(_stable_zscore(pd.Series(values)), expected, atol=1e-14)


@pytest.mark.parametrize("scale", [1e308, 1e-200, 1e-320])
@pytest.mark.parametrize("block_size", [1, 3, 11])
def test_online_pearson_rescales_previous_moments(scale, block_size):
    left = np.array([0, 0, 0.1, 0.2, 0.2, -0.2, -1, 1, 1, 0, -1]) * scale
    right = left / scale
    stats = dict.fromkeys(
        (
            "count",
            "left_mean",
            "right_mean",
            "left_origin",
            "right_origin",
            "left_scale",
            "right_scale",
            "left_m2",
            "right_m2",
            "covariance",
        ),
        0.0,
    )
    for start in range(0, len(left), block_size):
        _merge_time_series_moments(
            stats, left[start : start + block_size], right[start : start + block_size]
        )
    result = _time_series_metrics_frame(
        {("signal", "label"): stats}, timestamp=pd.Timestamp("2023-01-01")
    )
    assert result.n_obs.iloc[0] == len(left)
    assert pd.isna(result.ic_reason.iloc[0])
    assert result.ic.iloc[0] == pytest.approx(1.0)


def test_rank_ic_is_invariant_to_tiny_factor_units():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-01"),
            "asset_id": list("abcdef"),
            "factor": np.arange(6) * 1e-200,
            "label": [0, 1, 1, 3, 4, 5],
        }
    )
    expected = pd.Series(range(6)).corr(pd.Series(frame.label), method="spearman")
    result = cross_sectional_rank_ic(
        frame, value_columns=("factor",), label_column="label"
    )
    assert result.factor.iloc[0] == pytest.approx(expected)


def test_market_beta_matches_centered_ols_for_nearly_constant_returns():
    state = MarketModelRiskState()
    markets, returns = [], []
    for index in range(320):
        market = 0.01 + 1e-10 * ((index % 11) - 5)
        values = [1.2 * market, 0.8 * market]
        if index:
            markets.append((100.0 * values[0] + 100.0 * values[1]) / 200.0)
            returns.append(values[0])
        result = state.observe(
            pd.DataFrame(
                {
                    "asset_id": ["a", "b"],
                    "period_return": values,
                    "float_market_cap": [100.0, 100.0],
                    "eligible": True,
                }
            )
        )
    with localcontext() as context:
        context.prec = 80
        x = [Decimal.from_float(value) for value in markets[-252:]]
        y = [Decimal.from_float(value) for value in returns[-252:]]
        mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
        covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
        variance = sum((a - mean_x) ** 2 for a in x)
        expected = float(covariance / variance)
    assert result.beta.loc["a"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1 + 0j, np.complex128(1 + 3j)])
def test_real_number_boundaries_reject_wrong_numeric_kinds(value):
    with pytest.raises(ValueError):
        strict_finite_float(value, error="numeric_invalid")
    for dtype in (None, object):
        for nullable in (False, True):
            with pytest.raises(ValueError):
                strict_finite_numeric_series(
                    pd.Series([value], dtype=dtype),
                    error="numeric_invalid",
                    allow_null=nullable,
                )
    with pytest.raises(ValueError):
        PortfolioCapital("synthetic", value, "USDT", "linear_perpetual")
    with pytest.raises(ValueError):
        build_simple_target_weights(
            pd.Series([value, 2], index=["a", "b"], dtype=object),
            eligible=pd.Series(True, index=["a", "b"]),
        )


def test_real_number_nullable_values_are_preserved():
    result = strict_finite_numeric_series(
        pd.Series(["2.5", None, pd.NA]), error="numeric_invalid", allow_null=True
    )
    pd.testing.assert_series_equal(
        result, pd.Series([2.5, pd.NA, pd.NA], dtype="Float64")
    )


@pytest.mark.parametrize("dtype", ["int64", "uint64", "object"])
def test_evaluation_counts_keep_exact_int64_identity(dtype):
    values = pd.Series([2**53 + 1, 2**63 - 1], dtype=dtype)
    for validate in (strict_non_negative_count_series, strict_positive_group_series):
        assert validate(values).tolist() == values.tolist()
    assert strict_non_negative_count_value(2**53 + 1) == 2**53 + 1
    with pytest.raises(ValueError):
        strict_non_negative_count_series(pd.Series([2**63], dtype="uint64"))


@pytest.mark.parametrize("count", [1000, 10000])
def test_complete_frame_releases_consumed_concat_before_next_output(count):
    def inputs():
        for dates in ([1] * count, [1] * count + [2], [3, 4]):
            yield pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(dates, unit="D"),
                    "value": range(len(dates)),
                }
            )

    with runtime_resource_policy_scope() as policy:
        stream = iter_complete_time_sized_frames(
            inputs(), timestamp_column="timestamp", resource_policy=policy
        )
        try:
            first = next(stream)
            assert len(first) == 2 * count
            reference = weakref.ref(first)
            del first
            second = next(stream)
            assert len(second) == 1
            gc.collect()
            assert reference() is None
            del second
        finally:
            stream.close()
        assert policy.used_memory_bytes == 0


def test_decimal_metadata_rejects_exponents_without_expansion():
    payload = {
        "schema_id": "qrp-dataset-part-logical-dtypes",
        "columns": [
            {
                "name": "category",
                "kind": "categorical",
                "ordered": False,
                "categories": [{"type": "decimal", "value": "1e1000000"}],
            }
        ],
    }
    raw = json.dumps(payload).encode()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            parse_logical_dtype_metadata(raw)
        assert tracemalloc.get_traced_memory()[1] < 100000
    finally:
        tracemalloc.stop()
    with pytest.raises(ValueError):
        _typed_category_value_payload(Decimal("1e2000000"))
    assert _typed_category_value_payload(Decimal("1.2300")) == {
        "type": "decimal",
        "value": "1.2300",
    }


@pytest.mark.parametrize("target", ["member", "manifest"])
def test_semantic_read_rechecks_earlier_members_and_manifest(
    tmp_path, monkeypatch, target
):
    def writer(path):
        (path / "a.bin").write_bytes(b"first")
        (path / "b.bin").write_bytes(b"last")

    root = semantic_artifacts.write_semantic_artifact(tmp_path, _envelope(), writer)
    original = semantic_artifacts._member_record
    visited = []

    def record(relative, path):
        result = original(relative, path)
        visited.append(path)
        if len(visited) == 2:
            (visited[0] if target == "member" else root / "manifest.json").write_bytes(
                b"changed"
            )
        return result

    monkeypatch.setattr(semantic_artifacts, "_member_record", record)
    with pytest.raises(ValueError, match="semantic_artifact_member_identity_changed"):
        semantic_artifacts.read_semantic_artifact_record(root)


def test_semantic_hash_workspace_uses_active_budget(tmp_path, monkeypatch):
    def writer(path):
        with (path / "payload.bin").open("wb") as handle:
            for _ in range(64):
                handle.write(b"x" * 65536)

    root = semantic_artifacts.write_semantic_artifact(tmp_path, _envelope(), writer)
    original = semantic_artifacts._hash_file_content
    observed = []
    with runtime_resource_policy_scope(explicit_cap_bytes=65536) as policy:

        def hashing(path, identity, error, *, chunk_size):
            observed.append(chunk_size)
            assert 0 < chunk_size <= policy.used_memory_bytes <= 65536
            return original(path, identity, error, chunk_size=chunk_size)

        monkeypatch.setattr(semantic_artifacts, "_hash_file_content", hashing)
        semantic_artifacts.read_semantic_artifact_record(root)
        assert observed
        assert policy.used_memory_bytes == 0


def test_real_observations_preserve_nonfinite_domain_handling():
    values = pd.Series([1.0, np.inf, -np.inf, np.nan])
    pd.testing.assert_series_equal(
        strict_real_numeric_series(values, error="real_invalid"), values
    )
    target = build_simple_target_weights(
        pd.Series({"a": np.inf, "b": 1.0}), eligible=pd.Series({"a": False, "b": True})
    )
    assert target.tolist() == [0.0, 1.0]


@pytest.mark.parametrize(
    "axis,market",
    [("cross_sectional", "a_share"), ("single_asset_time_series", "crypto")],
)
def test_evaluation_releases_previous_filtered_input(monkeypatch, axis, market):
    from contextlib import contextmanager

    import quant_research_platform.research.filtering as filtering
    from quant_research_platform.contracts.evaluations import EvaluationSpec
    from quant_research_platform.contracts.ids import MarketId
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )

    original = filtering._filter_mode_scope
    references = []

    @contextmanager
    def inspect(*args, **kwargs):
        if references:
            assert references[-1]() is None
        with original(*args, **kwargs) as pair:
            references.append(weakref.ref(pair[0]))
            yield pair

    monkeypatch.setattr(filtering, "_filter_mode_scope", inspect)

    def frames():
        for number in range(4):
            dates = pd.date_range("2023-01-01", periods=5) + pd.Timedelta(
                days=5 * number
            )
            yield pd.DataFrame(
                {
                    "timestamp": dates if market == "crypto" else dates.repeat(3),
                    "asset_id": ["a"] * 5
                    if market == "crypto"
                    else ["a", "b", "c"] * 5,
                    "feature": list(range(5))
                    if market == "crypto"
                    else [0.0, 1.0, 2.0] * 5,
                    "label": list(range(5))
                    if market == "crypto"
                    else [0.0, 1.0, 2.0] * 5,
                    "eligible": True,
                }
            )

    spec = EvaluationSpec(
        evaluation_id="lifetime",
        market_id=MarketId(market),
        research_dataset_id="dataset",
        signals=(
            SignalSpec(
                signal_id="signal", source_column="feature", source_kind="feature"
            ),
        ),
        label_columns=("label",),
        evaluation_axis=axis,
        n_groups=2,
    )
    with runtime_resource_policy_scope() as policy:
        with build_evaluation_result_from_frames(
            frames(),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
            resource_policy=policy,
        ) as result:
            for item in result.output_chunks:
                item.release(owner="test.evaluation")
        assert len(references) > 1
        assert policy.used_memory_bytes == 0


def test_prediction_releases_previous_model_array():
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.research.predictions.outputs import (
        _iter_prediction_output_frames,
        _PredictionBuildState,
    )

    references = []

    def predict(frame):
        if references:
            assert references[-1]() is None
        value = np.arange(len(frame), dtype=float)
        references.append(weakref.ref(value))
        return value

    def frames():
        for number in range(3):
            yield pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=5)
                    + pd.Timedelta(days=5 * number),
                    "asset_id": "a",
                    "feature": np.arange(5, dtype=float),
                    "eligible": True,
                }
            )

    spec = PredictionSpec("prediction", "model", "dataset", "2022-12-31", "2023-01-01")
    with runtime_resource_policy_scope() as policy:
        stream = _iter_prediction_output_frames(
            frames(),
            spec=spec,
            input_columns=("feature",),
            predictor=predict,
            build_state=_PredictionBuildState(),
        )
        try:
            for item in stream:
                item.release(owner="test.prediction")
        finally:
            stream.close()
        assert len(references) == 3
        assert policy.used_memory_bytes == 0


def test_semantic_hash_io_failure_preserves_error_and_releases_budget(
    tmp_path, monkeypatch
):
    root = semantic_artifacts.write_semantic_artifact(
        tmp_path,
        _envelope(),
        lambda path: (path / "payload.bin").write_bytes(b"payload"),
    )
    primary = OSError("synthetic_io_failure")

    def failed_hash(*args, **kwargs):
        raise primary

    monkeypatch.setattr(semantic_artifacts, "_hash_file_content", failed_hash)
    with runtime_resource_policy_scope(explicit_cap_bytes=65536) as policy:
        with pytest.raises(
            ValueError, match="semantic_artifact_member_invalid"
        ) as caught:
            semantic_artifacts.read_semantic_artifact_record(root)
        assert caught.value.__cause__ is primary
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([1e308, 1e308], 1e308),
        ([1e308, 1.0, -1e308], 1 / 3),
        ([1e-320, 1e-320], 1e-320),
        ([1e308, np.nextafter(1e308, np.inf)], 1e308),
    ],
)
def test_grouped_label_mean_preserves_finite_results(labels, expected):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-01"),
            "asset_id": [str(i) for i in range(len(labels))],
            "signal": 1.0,
            "label": labels,
        }
    )
    _, grouped = compute_ic_and_grouped_label_values(
        frame,
        ComputePanelSpec(
            signal_columns=("signal",), label_column="label", asset_column="asset_id"
        ),
        n_groups=2,
    )
    assert len(grouped) == 1
    assert grouped.mean_label_value.iloc[0] == expected


@pytest.mark.parametrize(
    "tops, bottoms, expected_top, expected_bottom, expected_spread",
    [
        ([1e308] * 3, [0.0] * 3, 1e308, 0.0, 1e308),
        ([1e308, 1.0, -1e308], [0.0] * 3, 1 / 3, 0.0, 1 / 3),
        ([1e308, -1e308], [-1e308, 1e308], 0.0, 0.0, 0.0),
        ([1e-320] * 3, [0.0] * 3, 1e-320, 0.0, 1e-320),
        ([1e308], [-1e308], 1e308, -1e308, None),
    ],
)
def test_group_spread_aggregates_before_float_conversion(
    tops, bottoms, expected_top, expected_bottom, expected_spread
):
    spec = EvaluationSpec(
        evaluation_id="finite_mean",
        market_id=MarketId("a_share"),
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(
                signal_id="signal", source_column="feature", source_kind="feature"
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    state = _EvaluationStreamingState(spec=spec, n_groups=2)
    try:
        for top, bottom in zip(tops, bottoms, strict=True):
            _add_group_spread_observation(
                state._spread, "signal", "label", top, bottom, float(top > bottom), None
            )
        if expected_spread is None:
            with pytest.raises(ValueError, match="^invalid_evaluation_numeric_values$"):
                state.group_spread_summary()
        else:
            summary = state.group_spread_summary().iloc[0]
            assert summary.top_group_mean_label_value == expected_top
            assert summary.bottom_group_mean_label_value == expected_bottom
            assert summary.top_bottom_label_value_spread_mean == expected_spread
    finally:
        state.close()


@pytest.mark.parametrize("timezone", [None, "UTC"])
@pytest.mark.parametrize("count", [100_000, 500_000])
def test_timestamp_validation_fits_time_operation_reservation(timezone, count):
    from quant_research_platform.contracts.validation import strict_timestamp_series
    from quant_research_platform.runtime.dataframes import _time_chunk_operation_bytes

    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2020", periods=count, freq="s", tz=timezone)}
    )
    bound = _time_chunk_operation_bytes(frame, timestamp_column="timestamp")
    gc.collect()
    tracemalloc.start()
    try:
        result = strict_timestamp_series(frame.timestamp, error="timestamp_invalid")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(result) == len(frame)
    assert result.dtype == np.dtype("datetime64[ns]")
    assert peak <= bound + 65536


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "datetime64[ns, UTC]", "timedelta64[ns]"]
)
def test_persisted_temporal_values_cannot_become_portfolio_signals(tmp_path, dtype):
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    frame = make_default_dataset_frame(
        asset_ids=("a", "b"),
        dates=("2024-01-02", "2024-01-02"),
        values=(1, 2),
    )
    frame["value"] = pd.Series([1, 2], dtype=dtype)
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(root, frame, spec=make_processed_dataset_spec())
    restored = read_dataset(root).set_index("asset_id")
    assert restored.value.dtype == frame.value.dtype
    with pytest.raises(ValueError, match="target_signal_invalid"):
        build_simple_target_weights(
            restored.value, eligible=pd.Series(True, index=restored.index)
        )


def test_evaluation_mixed_timestamp_strings_follow_chronological_order():
    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )

    frame = pd.DataFrame(
        {
            "timestamp": ["2024-2-01"] * 3 + ["2024-10-01"] * 3 + ["2024-1-01"] * 3,
            "asset_id": ["a", "b", "c"] * 3,
            "feature": [1.0, 2.0, 3.0] * 3,
            "label": [1.0, 2.0, 3.0] * 3,
            "eligible": True,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="time_order",
        market_id=MarketId("a_share"),
        research_dataset_id="dataset",
        signals=(
            SignalSpec(
                signal_id="signal", source_column="feature", source_kind="feature"
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    observed = []
    with runtime_resource_policy_scope() as policy:
        with build_evaluation_result_from_frames(
            [frame],
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            resource_policy=policy,
            source_columns_by_kind={"feature": ("feature",)},
        ) as result:
            for output in result.output_chunks:
                metrics, _groups, _metadata = output.value
                observed.extend(metrics.timestamp.drop_duplicates().tolist())
                output.release(owner="test.evaluation")
        assert policy.used_memory_bytes == 0
    assert list(pd.to_datetime(observed, format="mixed")) == list(
        pd.to_datetime(["2024-01-01", "2024-02-01", "2024-10-01"])
    )
