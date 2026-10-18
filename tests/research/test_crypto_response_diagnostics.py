"""Independent arithmetic, frozen bins and native response-table publication."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    _response_tables,
    training_quantile_cutpoints,
)
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.storage.datasets.writing import write_dataset


def _spec():
    return EvaluationSpec(
        evaluation_id="response",
        market_id="crypto",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(signal_id="score", source_kind="feature", source_column="score"),
        ),
        label_columns=("r100", "r500"),
        label_availability_columns={"r100": "a100", "r500": "a500"},
        evaluation_axis="single_asset_time_series",
    )


def _response(cuts=None):
    return {
        "cutpoints": {"score": [2.0] if cuts is None else cuts},
        "reference_window": {
            "start": "2026-01-01T00:00:00",
            "end": "2026-01-01T12:00:00",
        },
    }


def _frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02T00:00:00",
                    "2026-01-02T00:00:01",
                    "2026-01-02T00:00:02",
                    "2026-01-02T00:00:03",
                    "2026-01-03T00:00:00",
                    "2026-01-03T00:00:01",
                ]
            ),
            "asset_id": "btc",
            "score": [1.0, 2.0, 3.0, 4.0, 1.0, 4.0],
            "r100": [0.0001, 0.0002, 0.0003, 0.0004, 0.0004, 0.0001],
            "r500": [0.0002, 0.0004, 0.0006, np.nan, 0.0008, 0.0002],
            "a100": pd.Timestamp("2026-01-03T01:00:00"),
            "a500": pd.Timestamp("2026-01-03T01:00:00"),
            "eligible": True,
        }
    )


def _rows(table, **criteria):
    for name, value in criteria.items():
        table = table.loc[table[name].eq(value)]
    return table


def test_training_quantiles_do_not_split_ties_or_make_constant_spread():
    assert training_quantile_cutpoints([3.0] * 20) == []
    assert training_quantile_cutpoints([0.0, 0.0, 0.0, 0.0, 1.0]) == [0.0]
    assert training_quantile_cutpoints(np.arange(1.0, 11.0)) == [2.0, 4.0, 6.0, 8.0]
    tables = _response_tables(_frame(), spec=_spec(), config=_response([]))
    assert tables["response_groups"].group.eq(1).all()
    assert tables["response_metrics"].top_bottom_mean_bp.isna().all()
    assert tables["response_metrics"].spread_reason.eq("one_signal_group").all()


@pytest.mark.parametrize("values", [[], [1.0, np.nan], [1.0, np.inf], [[1.0, 2.0]]])
def test_bad_reference_is_not_silently_redefined(values):
    with pytest.raises(ValueError, match="reference_scores_invalid"):
        training_quantile_cutpoints(values)


def test_hand_calculation_common_own_daily_and_pooled_are_distinct():
    tables = _response_tables(_frame(), spec=_spec(), config=_response())
    metrics, groups = tables["response_metrics"], tables["response_groups"]
    first = _rows(metrics, period="daily", sample_scope="common", label="r100").iloc[0]
    assert first.n_obs == 3 and first.signal_count == 4
    assert first.coverage == 0.75 and first.ic == pytest.approx(1.0)
    assert first.top_bottom_mean_bp == pytest.approx(1.5)
    own = _rows(metrics, period="daily", sample_scope="own", label="r100").iloc[0]
    assert own.n_obs == 4 and own.top_bottom_mean_bp == pytest.approx(2.0)
    equal = _rows(
        metrics, period="daily_equal_weight", sample_scope="own", label="r100"
    ).iloc[0]
    assert equal.ic == pytest.approx(0.0, abs=1e-15)
    assert equal.ic_std == pytest.approx(2**0.5)
    assert equal.valid_day_count == 2 and equal.n_obs == 6
    pooled = _rows(metrics, period="pooled", sample_scope="own", label="r100").iloc[0]
    # Centered cross-product is 0.5; both squared-deviation sums are 9.5.
    assert pooled.ic == pytest.approx(0.5 / 9.5)
    low = _rows(
        groups, period="daily", sample_scope="common", label="r100", group=1
    ).iloc[0]
    assert low.n_obs == 2 and low.mean_return_bp == pytest.approx(1.5)
    assert low.median_return_bp == pytest.approx(1.5)
    assert _rows(groups, period="daily_equal_weight").median_return_bp.isna().all()


def test_empty_fixed_groups_are_retained_and_extremes_not_reassigned():
    tables = _response_tables(
        _frame(), spec=_spec(), config=_response([-100.0, 2.0, 100.0])
    )
    pooled = _rows(
        tables["response_groups"], period="pooled", sample_scope="common", label="r100"
    )
    assert pooled.group.tolist() == [1, 2, 3, 4]
    assert pooled.n_obs.tolist() == [0, 3, 2, 0]
    assert pooled.iloc[[0, 3]].mean_return_bp.isna().all()
    metrics = _rows(tables["response_metrics"], period="pooled")
    assert metrics.spread_reason.eq("empty_extreme_group").all()


def _independent_exact_pearson(left, right):
    left, right = (
        [Fraction(float(x)) for x in left],
        [Fraction(float(y)) for y in right],
    )
    xmean, ymean = sum(left) / len(left), sum(right) / len(right)
    cross = sum((x - xmean) * (y - ymean) for x, y in zip(left, right, strict=True))
    xx = sum((x - xmean) ** 2 for x in left)
    yy = sum((y - ymean) ** 2 for y in right)
    return np.copysign(
        float(cross**2 / xx / yy) ** 0.5, float(cross > 0) - float(cross < 0)
    )


@pytest.mark.parametrize("case", ["ordinary", "large_offset"])
def test_batch_pearson_against_independent_exact_hand_moments_and_permutation(case):
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = np.array([3.0, -2.0, 7.0, 5.0, 13.0])
    if case == "large_offset":
        x, y = 10000 + x, -10000 + y
    expected = _independent_exact_pearson(x, y)
    for order in [np.arange(5), np.arange(4, -1, -1), [3, 1, 0, 4, 2]]:
        value, reason = _pearson_correlation_result(x[order], y[order])
        assert reason is None and value == pytest.approx(expected, abs=2e-14)


def test_batch_pearson_undefined_reasons_and_local_pair_masks():
    assert _pearson_correlation_result([1.0], [2.0]) == (
        None,
        "insufficient_observations",
    )
    assert _pearson_correlation_result([1.0, 1.0], [2.0, 3.0]) == (
        None,
        "zero_variance",
    )
    assert _pearson_correlation_result([1.0, np.inf], [2.0, 3.0]) == (
        None,
        "nonfinite_result",
    )
    frame = _frame()
    frame.loc[0, "r500"] = np.inf
    table = _response_tables(frame, spec=_spec(), config=_response())[
        "response_metrics"
    ]
    assert (
        _rows(table, period="pooled", sample_scope="own", label="r100").n_obs.item()
        == 6
    )
    assert (
        _rows(table, period="pooled", sample_scope="common", label="r100").n_obs.item()
        == 4
    )


def _native_config(tmp_path, frame=None):
    frame = _frame() if frame is None else frame
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(
        timestamp="timestamp",
        asset_id="key",
        score="feature",
        r100="label",
        r500="label",
        eligible="mask",
    )
    root = tmp_path / "input"
    write_dataset(
        root,
        frame,
        spec=DataSpec(
            data_id="response_fixture",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
            column_units={"r100": "decimal_return", "r500": "decimal_return"},
            label_availability_columns={"r100": "a100", "r500": "a500"},
        ),
    )
    spec = _spec()
    from dataclasses import asdict

    return {
        "inputs": str(root),
        "output": str(tmp_path / "evaluation"),
        "evaluation": dict(
            evaluation_id=spec.evaluation_id,
            market_id="crypto",
            research_dataset_id="fixture",
            signals=[asdict(spec.signals[0])],
            label_columns=list(spec.label_columns),
            evaluation_axis=spec.evaluation_axis,
        ),
        "window": {"start": "2026-01-02", "end": "2026-01-03"},
        "cutoff": "2026-01-04",
        "response_diagnostics": _response(),
    }


def test_native_roundtrip_and_shared_scan(tmp_path, monkeypatch):
    from quant_research_platform.research.evaluations import runner

    original = runner.iter_dataset_chunks
    scans = []

    def scan(*args, **kwargs):
        scans.append(1)
        yield from original(*args, **kwargs)

    monkeypatch.setattr(runner, "iter_dataset_chunks", scan)
    config = _native_config(tmp_path)
    root = evaluate(config)
    assert len(scans) == 1
    manifest, tables = read_evaluation_outputs(
        root, tables=("response_metrics", "response_groups", "metrics")
    )
    assert (
        manifest["metadata"]["response_diagnostics"] == config["response_diagnostics"]
    )
    assert len(tables["response_metrics"]) == 16
    assert len(tables["response_groups"]) == 32
    assert tables["metrics"].n_obs.tolist() == [6, 5]
    assert tables["response_groups"].median_return_bp.notna().sum() == 24


@pytest.mark.parametrize("change", ["duplicate_edges", "bool_edge", "unit"])
def test_bad_response_request_is_rejected_before_publication(tmp_path, change):
    config = _native_config(tmp_path)
    if change == "duplicate_edges":
        config["response_diagnostics"]["cutpoints"]["score"] = [2.0, 2.0]
    elif change == "bool_edge":
        config["response_diagnostics"]["cutpoints"]["score"] = [True]
    else:
        import json
        from pathlib import Path

        spec_path = Path(config["inputs"]) / "spec.json"
        data = json.loads(spec_path.read_text())
        data["column_units"]["r100"] = "percent"
        spec_path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evaluate(config)


def test_label_maturity_remains_horizon_local_and_no_anchors_are_created(tmp_path):
    config = _native_config(tmp_path)
    # An earlier cutoff invalidates both labels without inventing replacements.
    config["cutoff"] = "2026-01-02"
    root = evaluate(config)
    _, tables = read_evaluation_outputs(
        root, tables=("response_metrics", "response_groups")
    )
    metrics = tables["response_metrics"]
    assert metrics.n_obs.eq(0).all()
    assert metrics.signal_count.max() == 6
    assert metrics.ic.isna().all()
    assert tables["response_groups"].n_obs.eq(0).all()


def test_chunk_boundaries_and_empty_dates_do_not_change_response(tmp_path, monkeypatch):
    from quant_research_platform.research.evaluations import runner

    original = runner.iter_dataset_chunks
    config = _native_config(tmp_path)
    output = evaluate(config)
    _, reference = read_evaluation_outputs(
        output, tables=("response_metrics", "response_groups")
    )

    def single_rows(*args, **kwargs):
        for frame in original(*args, **kwargs):
            for index in range(len(frame)):
                yield frame.iloc[index : index + 1]

    monkeypatch.setattr(runner, "iter_dataset_chunks", single_rows)
    config["output"] += "_chunks"
    output = evaluate(config)
    _, split = read_evaluation_outputs(output, tables=tuple(reference))
    for name in reference:
        pd.testing.assert_frame_equal(
            reference[name], split[name], check_exact=False, rtol=1e-12, atol=1e-12
        )
        assert reference[name].n_obs.tolist() == split[name].n_obs.tolist()


def test_native_prediction_is_consumed_without_model_or_trial(tmp_path):
    import shutil

    from quant_research_platform.contracts.models import ModelSpec
    from tests.research.direct_model_test_support import fit_frame, predict_frame

    frame = _frame()
    train = frame.iloc[:4].copy()
    train.timestamp -= pd.Timedelta(days=1)
    train["label_available_at"] = train.timestamp
    case = fit_frame(
        tmp_path / "fit",
        train,
        ModelSpec(
            model_id="signal",
            model_algorithm="linear_regression",
            input_feature_columns=("score",),
            target_column="r100",
        ),
    )
    window = {"start": "2026-01-02", "end": "2026-01-03"}
    _, prediction, _ = predict_frame(
        tmp_path / "prediction", frame, case, window=window
    )
    shutil.rmtree(case.root)
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    config = _native_config(tmp_path)
    config["evaluation"]["signals"] = [
        {"signal_id": "score", "source_kind": "prediction", "source_column": "p"}
    ]
    config["predictions"] = {"p": str(prediction)}
    config["response_diagnostics"]["cutpoints"]["score"] = [0.0002]
    _, tables = read_evaluation_outputs(evaluate(config), tables=("response_metrics",))
    pooled = _rows(tables["response_metrics"], period="pooled", sample_scope="common")
    assert pooled.n_obs.tolist() == [5, 5]


def test_both_chinese_response_charts_reuse_existing_series_renderer(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from quant_research_platform.reports.plots import plot_series

    config = _native_config(tmp_path)
    _, tables = read_evaluation_outputs(
        evaluate(config), tables=("response_metrics", "response_groups")
    )
    metrics = _rows(
        tables["response_metrics"], period="daily_equal_weight", sample_scope="common"
    )
    groups = _rows(
        tables["response_groups"], period="daily_equal_weight", sample_scope="common"
    )
    horizon = {"r100": "100毫秒", "r500": "500毫秒"}
    rows = [
        {"预测期限": horizon[row.label], "日等权IC": row.ic}
        for row in metrics.itertuples()
    ]
    figure, axes = plot_series(
        rows,
        x_column="预测期限",
        value_columns=["日等权IC"],
        title="同一冻结信号的IC响应（合成验收）",
        units="相关系数",
        source_reference="合成数值验收",
        sample_role="test",
        timezone="UTC",
        categorical=True,
    )
    figure.savefig(tmp_path / "ic.png")
    assert axes[0].get_xlabel() == "预测期限"
    assert len(axes[0].lines[0].get_xdata()) == 2
    plt.close(figure)
    rows = []
    for label, block in groups.groupby("label", sort=False):
        rows.append(
            {
                "预测期限": horizon[label],
                **{
                    f"第{row.group}组均值": row.mean_return_bp
                    for row in block.itertuples()
                },
            }
        )
    figure, axes = plot_series(
        rows,
        x_column="预测期限",
        value_columns=["第1组均值", "第2组均值"],
        title="训练期固定分组的未来收益响应（合成验收）",
        units="基点（bp，未扣交易成本）",
        source_reference="同一合成EVAL数值",
        sample_role="test",
        timezone="UTC",
        categorical=True,
    )
    figure.savefig(tmp_path / "groups.png")
    assert len(axes[0].lines) == 2
    figure.canvas.draw()
    legend_box = axes[0].get_legend().get_window_extent()
    assert legend_box.x0 >= axes[0].get_window_extent().x1
    assert legend_box.x1 <= figure.bbox.x1
    assert all(
        (tmp_path / name).stat().st_size > 1000 for name in ["ic.png", "groups.png"]
    )
    plt.close(figure)


def test_bp_scaling_preserves_small_return_mean_and_median():
    frame = _frame().iloc[:2].copy()
    tiny = 0.000001
    frame["r100"] = [0.0, tiny]
    frame["r500"] = [0.0, tiny]
    groups = _response_tables(frame, spec=_spec(), config=_response([]))[
        "response_groups"
    ]
    expected = float(Fraction(float(tiny)) * 5000)
    actual = _rows(groups, period="pooled", label="r100", sample_scope="common").iloc[0]
    assert actual.mean_return_bp == pytest.approx(expected, rel=1e-12, abs=0)
    assert actual.median_return_bp == pytest.approx(expected, rel=1e-12, abs=0)


def test_explicit_cutpoints_do_not_require_reference_admission(tmp_path):
    config = _native_config(tmp_path)
    config["response_diagnostics"] = {"cutpoints": {"score": [2.0]}}
    evaluate(config)


def test_empty_group_cannot_be_persisted_as_a_fake_zero(tmp_path):
    from pathlib import Path

    config = _native_config(tmp_path)
    config["response_diagnostics"]["cutpoints"]["score"] = [-100.0, 2.0, 100.0]
    root = evaluate(config)
    path = Path(root) / "response_groups.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, ["mean_return_bp", "mean_reason"]] = [0.0, None]
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS

    pq.write_table(
        pa.Table.from_pandas(
            frame, schema=DAILY_SCHEMAS["response_groups"], preserve_index=False
        ),
        path,
    )
    with pytest.raises(ValueError, match="response_groups_invalid"):
        read_evaluation_outputs(root, tables=("response_groups",))


def test_cancellation_and_near_zero_spread_use_financial_tolerance():
    frame = _frame().iloc[:3].copy()
    frame["r100"] = [-0.01, 1e-6, 0.01]
    frame["r500"] = frame.r100
    groups = _response_tables(frame, spec=_spec(), config=_response([]))[
        "response_groups"
    ]
    actual = _rows(
        groups, period="pooled", label="r100", sample_scope="common"
    ).mean_return_bp.item()
    assert actual == pytest.approx(1e-6 * 10000 / 3, rel=1e-12, abs=1e-12)
    frame = _frame().copy()
    frame["score"] = [1.0, 1.0, 1.0, 4.0, 4.0, 4.0]
    frame["r100"] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.010001]
    frame["r500"] = frame.r100
    table = _response_tables(frame, spec=_spec(), config=_response())[
        "response_metrics"
    ]
    actual = _rows(
        table, period="pooled", label="r100", sample_scope="common"
    ).top_bottom_mean_bp.item()
    assert actual == pytest.approx(1e-6 * 10000 / 3, abs=1e-12)


def test_native_partial_maturity_and_empty_date_preserve_domains(tmp_path):
    frame = _frame()
    frame.loc[0, "a100"] = pd.Timestamp("2026-02-01")
    frame.loc[1, "a500"] = pd.NaT
    frame.loc[4:, "eligible"] = False
    root = evaluate(_native_config(tmp_path, frame))
    _, tables = read_evaluation_outputs(
        root, tables=("response_metrics", "response_groups")
    )
    metrics = tables["response_metrics"]
    assert _rows(metrics, period="pooled", sample_scope="common").n_obs.tolist() == [
        1,
        1,
    ]
    assert _rows(metrics, period="pooled", sample_scope="own").n_obs.tolist() == [3, 2]
    assert _rows(metrics, period="pooled").signal_count.eq(4).all()
    empty = _rows(
        metrics, period="daily", timestamp=pd.Timestamp("2026-01-03T00:00:01")
    )
    assert len(empty) == 4 and empty.n_obs.eq(0).all()
    assert empty.signal_count.eq(0).all() and empty.coverage.isna().all()


@pytest.mark.parametrize(
    "case",
    ["ordinary", "large_offset", "zero_correlation", "constant", "empty"],
)
def test_time_series_batch_pairs_match_independent_oracle_across_chunks(case):
    from dataclasses import replace

    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )

    frame = _frame()
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    y = np.array([3.0, -2.0, 7.0, 5.0, 13.0, -1.0])
    if case == "large_offset":
        x, y = 10000 + x, -10000 + y
    elif case == "zero_correlation":
        x = np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0])
        y = np.array([-1.0, 1.0, -1.0, 1.0, 0.0, 0.0])
    elif case == "constant":
        x[:] = 1.0
    elif case == "empty":
        x[:] = np.nan
    frame["score"], frame["r100"] = x, y
    frame["other"] = x
    frame.loc[1, "other"] = np.nan
    frame["r500"] = y
    frame.loc[2, "r500"] = np.nan
    spec = replace(
        _spec(),
        signals=(
            *_spec().signals,
            SignalSpec(signal_id="other", source_kind="feature", source_column="other"),
        ),
    )
    outputs = []
    for size in (1, 2, 6):
        with build_evaluation_result_from_frames(
            (frame.iloc[start : start + size] for start in range(0, len(frame), size)),
            spec=spec,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("score", "other")},
        ) as result:
            metrics, grouped, facts = list(result.output_chunks)[0]
            assert grouped.empty and facts["timestamp_count"] == len(frame)
            assert "response_metrics" not in facts
            outputs.append(metrics)
    for metrics in outputs:
        pd.testing.assert_frame_equal(
            metrics, outputs[0], check_exact=False, rtol=1e-12, atol=1e-14
        )
        for row in metrics.itertuples():
            left, right = frame[row.signal].to_numpy(), frame[row.label].to_numpy()
            valid = np.isfinite(left) & np.isfinite(right)
            assert row.n_obs == int(valid.sum())
            assert row.rankic_reason == "not_applicable"
            if case in {"constant", "empty"}:
                assert pd.isna(row.ic)
                assert row.ic_reason == (
                    "zero_variance"
                    if case == "constant"
                    else "insufficient_observations"
                )
            else:
                expected = _independent_exact_pearson(left[valid], right[valid])
                assert pd.isna(row.ic_reason)
                assert row.ic == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_native_time_series_optional_response_preserves_mature_pair_domain(tmp_path):
    frame = _frame()
    frame.loc[0, "a100"] = pd.Timestamp("2026-01-05")
    frame.loc[1, "score"] = np.nan
    frame.loc[4, "eligible"] = False
    config = _native_config(tmp_path, frame)
    with_response = evaluate(config)
    _, expected = read_evaluation_outputs(
        with_response, tables=("metrics", "ic_summary")
    )
    config.pop("response_diagnostics")
    config["output"] = str(tmp_path / "plain")
    _, actual = read_evaluation_outputs(
        evaluate(config), tables=("metrics", "ic_summary")
    )
    for name in actual:
        numeric = list(actual[name].select_dtypes(include="floating").columns)
        pd.testing.assert_frame_equal(
            actual[name].drop(columns=numeric),
            expected[name].drop(columns=numeric),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            actual[name][numeric].isna(),
            expected[name][numeric].isna(),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            actual[name][numeric],
            expected[name][numeric],
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
        )
    assert actual["metrics"].n_obs.tolist() == [3, 3]
    valid = frame.loc[[2, 3, 5]]
    assert actual["metrics"].iloc[0].ic == pytest.approx(
        _independent_exact_pearson(valid.score, valid.r100), abs=2e-14
    )


@pytest.mark.parametrize("empty", [False, True])
def test_native_response_pair_domain_and_independent_members(tmp_path, empty):
    frame = _frame().iloc[:0] if empty else _frame()
    root = evaluate(_native_config(tmp_path, frame))
    members = ("response_metrics", "response_groups")
    _, tables = read_evaluation_outputs(root, tables=members)
    for name in members:
        _, selected = read_evaluation_outputs(root, tables=(name,))
        pd.testing.assert_frame_equal(selected[name], tables[name], check_exact=True)
    if empty:
        assert all(table.empty for table in tables.values())
    else:
        metrics = tables["response_metrics"]
        assert _rows(metrics, period="pooled", sample_scope="own").n_obs.tolist() == [
            6,
            5,
        ]
        assert _rows(
            metrics, period="pooled", sample_scope="common"
        ).n_obs.tolist() == [5, 5]
        assert _rows(
            tables["response_groups"],
            period="pooled",
            sample_scope="common",
            label="r100",
        ).n_obs.tolist() == [3, 2]


@pytest.mark.parametrize("change", ["empty_metrics", "missing_common"])
def test_response_pair_contract_at_publication_and_selected_read(tmp_path, change):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS
    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )

    root = evaluate(_native_config(tmp_path))
    members = ("response_metrics", "response_groups")
    manifest, tables = read_evaluation_outputs(root, tables=members)
    science = {
        key: manifest["metadata"][key]
        for key in (
            "window",
            "cutoff",
            "sample_role",
            "sample_scope",
            "input_facts",
            "comparison_inputs",
            "strata",
            "stratum_domains",
            "optional_member_rows",
            "response_diagnostics",
            "common_rule",
        )
    }
    science["optional_member_rows"] = dict.fromkeys(members, 0)

    def altered(name, frame):
        if change == "empty_metrics":
            return frame.iloc[:0] if name == "response_metrics" else frame
        return frame.loc[~(frame.label.eq("r500") & frame.sample_scope.eq("common"))]

    error = (
        "evaluation_response_group_count_mismatch"
        if change == "empty_metrics"
        else "evaluation_response_common_count_mismatch"
    )
    unpublished = tmp_path / "invalid"
    with build_evaluation_result_from_frames(
        [_frame()],
        spec=_spec(),
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("score",)},
        response_diagnostics=_response(),
    ) as result:
        chunks = [
            (
                metrics,
                grouped,
                {**facts, **{name: altered(name, facts[name]) for name in members}},
            )
            for metrics, grouped, facts in result.output_chunks
        ]
        with pytest.raises(ValueError, match=f"^{error}$"):
            write_evaluation_outputs_from_chunks(
                chunks,
                summary_factory=result.summaries,
                spec=_spec(),
                output_path=unpublished,
                metadata=science,
            )
    assert not unpublished.exists()
    # Each member remains locally valid; only their consumed relationship is broken.
    for name in members:
        frame = altered(name, tables[name])
        pq.write_table(
            pa.Table.from_pandas(
                frame, schema=DAILY_SCHEMAS[name], preserve_index=False
            ),
            root / f"{name}.parquet",
        )
        manifest["metadata"]["optional_member_rows"][name] = len(frame)
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name in members:
        read_evaluation_outputs(root, tables=(name,))
    with pytest.raises(ValueError, match=f"^{error}$"):
        read_evaluation_outputs(root, tables=members)


def test_response_count_owner_distinguishes_unconsumed_empty_and_incomplete_pairs():
    from quant_research_platform.artifacts.outputs.evaluation_daily import (
        validate_response_counts,
    )

    tables = _response_tables(_frame(), spec=_spec(), config=_response())
    metrics, groups = tables["response_metrics"], tables["response_groups"]
    science = {"response_diagnostics": _response()}
    validate_response_counts(None, groups, science=science)
    validate_response_counts(metrics, None, science=science)
    with pytest.raises(ValueError, match="^evaluation_response_group_count_mismatch$"):
        validate_response_counts(metrics, groups.iloc[:0], science=science)
    for name in tables:
        tables[name] = tables[name].loc[
            ~(tables[name].label.eq("r500") & tables[name].sample_scope.eq("own"))
        ]
    with pytest.raises(ValueError, match="^evaluation_response_common_count_mismatch$"):
        validate_response_counts(
            tables["response_metrics"], tables["response_groups"], science=science
        )
    wrong_counts = groups.copy()
    wrong_counts.loc[0, "n_obs"] += 1
    with pytest.raises(ValueError, match="^evaluation_response_group_count_mismatch$"):
        validate_response_counts(metrics, wrong_counts, science=science)
