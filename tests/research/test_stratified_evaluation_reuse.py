"""Persisted factor scope and stratification acceptance through public operations."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.statistics_production import factor_statistics
from tests.research.crypto_evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _fixture(tmp_path, *, days=1):
    dates = pd.date_range("2023-01-02", periods=days)
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(4),
            "asset_id": list("abcd") * days,
            "x": [1.0, 2.0, 4.0, np.nan] * days,
            "z": [3.0, 2.0, np.nan, 1.0] * days,
            "label": [1.0, 3.0, 2.0, 4.0] * days,
            "label_available_at": pd.Timestamp("2023-01-02"),
            "eligible": True,
            "industry_level1_code": ["A", "A", None, "<NA>"] * days,
            "total_market_cap": [0.0, 1.0, 2.0, 3.0] * days,
            "amount": [1.0, 1.0, 2.0, 3.0] * days,
        }
    )
    inputs = write_evaluation_input(
        tmp_path / "input", frame, factors=("x", "z"), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="scope",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(signal_id="score", source_kind="factor", source_column="x"),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        filter_mode="eligible",
        n_groups=2,
    )
    config = evaluation_config(
        inputs,
        tmp_path / "own",
        spec,
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        cutoff="2024-01-01",
    )
    config["strata"] = {
        "primary_label": "label",
        "kinds": ["industry", "size", "liquidity"],
    }
    config["sample_scope"] = "own"
    return frame, config


def test_size_strata_use_total_cap_with_opposite_float_cap_order(tmp_path):
    frame, config = _fixture(tmp_path)
    frame["total_market_cap"] = [4.0, 3.0, 2.0, 1.0]
    frame["float_market_cap"] = [0.1, 0.2, 0.3, 0.4]
    config["inputs"] = str(
        write_evaluation_input(
            tmp_path / "opposite", frame, factors=("x", "z"), labels=("label",)
        )
    )
    _, tables = read_evaluation_outputs(
        evaluate(config), tables=("metrics", "daily_strata")
    )
    size = tables["daily_strata"].query("stratification == 'size'")
    # The smallest total-cap company has a missing signal. Float-cap ordering
    # would instead assign the first, observed company to the smallest bin.
    assert size.set_index("stratum").n_obs.to_dict() == {"1": 0, "2": 1, "3": 2}
    liquidity = tables["daily_strata"].query("stratification == 'liquidity'")
    assert liquidity.set_index("stratum").n_obs.to_dict() == {"2": 2, "3": 1}
    assert tables["metrics"].n_obs.tolist() == [3]


@pytest.mark.parametrize("missing", ([0, 1], [0, 1, 2, 3]))
def test_missing_total_cap_is_local_to_size_strata(tmp_path, missing):
    frame, config = _fixture(tmp_path)
    _, baseline = read_evaluation_outputs(
        evaluate(config),
        tables=(
            "metrics",
            "grouped_label_values",
            "ic_summary",
            "group_spread_summary",
            "daily_strata",
        ),
    )
    frame.loc[missing, "total_market_cap"] = np.nan
    frame["float_market_cap"] = [1.0, 2.0, 3.0, 4.0]
    config["inputs"] = str(
        write_evaluation_input(
            tmp_path / "missing", frame, factors=("x", "z"), labels=("label",)
        )
    )
    config["output"] = str(tmp_path / "missing-evaluation")
    _, actual = read_evaluation_outputs(
        evaluate(config),
        tables=(
            "metrics",
            "grouped_label_values",
            "ic_summary",
            "group_spread_summary",
            "daily_strata",
        ),
    )
    for name in (
        "metrics",
        "grouped_label_values",
        "ic_summary",
        "group_spread_summary",
    ):
        pd.testing.assert_frame_equal(actual[name], baseline[name], check_exact=True)
    for kind in ("industry", "liquidity"):
        pd.testing.assert_frame_equal(
            actual["daily_strata"]
            .loc[actual["daily_strata"].stratification.eq(kind)]
            .reset_index(drop=True),
            baseline["daily_strata"]
            .loc[baseline["daily_strata"].stratification.eq(kind)]
            .reset_index(drop=True),
            check_exact=True,
        )
    size = actual["daily_strata"].query("stratification == 'size'")
    undefined = size.loc[size.stratum.isna()]
    assert undefined.stratum_rows.tolist() == [len(missing)]
    assert undefined.n_obs.tolist() == [sum(frame.loc[missing, "x"].notna())]
    assert size.stratum_rows.sum() == len(frame)


def test_native_own_common_strata_and_persisted_reduction(tmp_path, monkeypatch):
    _, config = _fixture(tmp_path)
    own = evaluate(config)
    common = evaluate(
        {
            **config,
            "output": str(tmp_path / "common"),
            "sample_scope": "common",
            "evaluation": {
                **config["evaluation"],
                "evaluation_id": "different-nickname",
                "research_dataset_id": "another-description",
            },
            "comparison_inputs": [
                {"inputs": config["inputs"], "signal_columns": ["z"]}
            ],
        }
    )
    own_manifest, own_tables = read_evaluation_outputs(
        own, tables=("metrics", "daily_coverage", "daily_strata")
    )
    common_manifest, common_tables = read_evaluation_outputs(
        common, tables=("metrics", "daily_coverage", "daily_strata")
    )
    assert own_tables["metrics"].n_obs.tolist() == [3]
    assert own_tables["metrics"].rankic.tolist() == pytest.approx([0.5])
    assert common_tables["metrics"].n_obs.tolist() == [2]
    assert common_tables["metrics"].rankic.tolist() == pytest.approx([1.0])
    assert own_tables["daily_coverage"].domain_rows.tolist() == [4]
    assert common_tables["daily_coverage"].domain_rows.tolist() == [2]
    assert (
        own_tables["daily_strata"].stratum_rows.tolist()
        == common_tables["daily_strata"].stratum_rows.tolist()
    )
    assert own_tables["daily_strata"].stratum.isna().any()
    assert "<NA>" in own_tables["daily_strata"].stratum.dropna().tolist()
    assert (
        own_manifest["metadata"]["input_facts"]
        == common_manifest["metadata"]["input_facts"]
    )
    from quant_research_platform.research import statistics_production as production

    def forbidden(*args, **kwargs):
        raise AssertionError("persisted reduction must not rescan or recompute")

    monkeypatch.setattr(production, "read_dataset", forbidden)
    monkeypatch.setattr(production, "open_dataset_session", forbidden)
    monkeypatch.setattr(production, "cross_sectional_rank_ic", forbidden)
    monkeypatch.setattr(production, "fama_macbeth_period", forbidden)
    output = factor_statistics(
        {
            "evaluation_inputs": [str(own), str(common)],
            "output": str(tmp_path / "statistics"),
            "window": config["window"],
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    artifact = read_statistics_artifact(output)
    with artifact.table("factor_stratified_diagnostics.parquet") as result:
        all_rows = result.loc[result.stratification.eq("all")]
        assert set(all_rows.sample_scope) == {"own", "common"}
        assert all_rows.status.eq("UNDEFINED").all()
        assert all_rows["mean"].notna().all()
        assert all_rows.observation_count.eq(1).all()
        assert set(all_rows.observation_pairs) == {2, 3}


def test_common_final_domain_labels_only_and_missing_dates(tmp_path):
    frame, config = _fixture(tmp_path, days=2)
    peer = frame.loc[frame.timestamp.eq(frame.timestamp.min())].copy()
    peer.loc[peer.asset_id.eq("c"), "label"] = 99.0
    peer_path = write_evaluation_input(
        tmp_path / "peer", peer, factors=("z",), labels=("label",)
    )
    result = evaluate(
        {
            **config,
            "sample_scope": "common",
            "comparison_inputs": [{"inputs": str(peer_path), "signal_columns": ["z"]}],
        }
    )
    _, tables = read_evaluation_outputs(result, tables=("metrics", "daily_coverage"))
    assert tables["metrics"].n_obs.tolist() == [2, 0]
    assert tables["daily_coverage"].domain_rows.tolist() == [2, 0]


def test_factor_evaluation_pair_rejects_real_input_change(tmp_path):
    _, config = _fixture(tmp_path)
    own = evaluate(config)
    common = evaluate(
        {**config, "sample_scope": "common", "output": str(tmp_path / "common")}
    )
    manifest_path = common / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["input_facts"]["label_semantics"] = {"label": {"horizon": 99}}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation.*incompatible"):
        factor_statistics(
            {
                "evaluation_inputs": [str(own), str(common)],
                "output": str(tmp_path / "stats"),
                "window": config["window"],
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )


def _write_native(root, frame, *, signals, labels, availability, units=None, raw=False):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.storage.datasets import write_dataset

    roles = dict.fromkeys(frame.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    roles.update(
        {
            name: "feature" if name.startswith("feature_") else "factor"
            for name in signals
        }
    )
    roles.update(dict.fromkeys(labels, "label"))
    values = dict(
        data_id="fixture",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles=roles,
        label_availability_columns=availability,
        column_units=units or {},
    )
    if raw:
        path = root.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
        return {"path": str(path), "spec": values}
    write_dataset(root, frame, spec=DataSpec(**values))
    return str(root)


def test_real_observed_values_and_per_label_maturity(tmp_path):
    frame, config = _fixture(tmp_path)
    frame = frame.rename(columns={"x": "feature_x"})
    frame["feature_x"] = [0.0, 0.0, 2.0, 3.0]
    frame["feature_observed_x"] = [1, 0, 1, 1]
    frame["later"] = [3.0, 2.0, 1.0, 4.0]
    frame["later_clock"] = [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-02"),
    ]
    config["inputs"] = _write_native(
        tmp_path / "focal",
        frame,
        signals=("feature_x",),
        labels=("label", "later"),
        availability={"label": "label_available_at", "later": "later_clock"},
    )
    config["evaluation"]["signals"][0].update(
        source_column="feature_x", source_kind="feature"
    )
    config["evaluation"]["label_columns"] = ["label", "later"]
    config["evaluation"]["label_availability_columns"] = {
        "label": "label_available_at",
        "later": "later_clock",
    }
    peer = frame.rename(
        columns={
            "feature_x": "feature_z",
            "feature_observed_x": "feature_observed_z",
            "later_clock": "peer_later_clock",
        }
    ).copy()
    peer["feature_z"] = [9.0, 8.0, 7.0, 6.0]
    peer["feature_observed_z"] = [1, 1, 1, 1]
    peer.loc[peer.asset_id.eq("c"), "peer_later_clock"] = pd.Timestamp("2025-01-01")
    peer_path = _write_native(
        tmp_path / "peer-observed",
        peer,
        signals=("feature_z",),
        labels=("label", "later"),
        availability={"label": "label_available_at", "later": "peer_later_clock"},
    )
    out = evaluate(
        {
            **config,
            "sample_scope": "common",
            "comparison_inputs": [
                {"inputs": peer_path, "signal_columns": ["feature_z"]}
            ],
        }
    )
    _, tables = read_evaluation_outputs(out, tables=("metrics", "daily_coverage"))
    counts = tables["metrics"].set_index("label").n_obs.to_dict()
    assert counts == {"label": 3, "later": 2}
    assert tables["metrics"].set_index("label").loc["later", "rankic"] == pytest.approx(
        1.0
    )


def test_local_bad_classification_and_literal_fractional_age(tmp_path):
    _, config = _fixture(tmp_path)
    age = "median_forecast_age_days_fy1"
    ages = [
        0.0,
        7.0,
        7.5,
        8.0,
        30.0,
        30.5,
        31.0,
        60.0,
        60.5,
        61.0,
        90.0,
        90.5,
        91.0,
        180.0,
        np.nan,
        -1.0,
        np.inf,
        181.0,
    ]
    n = len(ages)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [f"a{i}" for i in range(n)],
            age: ages,
            "x": np.arange(n, dtype=float),
            "label": np.arange(n, dtype=float),
            "label_available_at": pd.Timestamp("2023-01-02"),
            "eligible": True,
            "total_market_cap": [-1.0, np.inf, np.nan] + [0.0] * (n - 3),
            "amount": [-2.0, np.inf, np.nan] + [1.0] * (n - 3),
        }
    )
    config["inputs"] = _write_native(
        tmp_path / "ages",
        frame,
        signals=(age, "x"),
        labels=("label",),
        availability={"label": "label_available_at"},
        units={age: "days"},
        raw=True,
    )
    config["evaluation"]["signals"].append(
        {
            "signal_id": "age",
            "source_kind": "factor",
            "source_column": age,
            "transform": "identity",
            "direction": "higher_is_better",
        }
    )
    config["strata"] = {
        "primary_label": "label",
        "kinds": ["size", "liquidity", "forecast_age"],
        "forecast_freshness_days": 180,
    }
    out = evaluate(config)
    _, tables = read_evaluation_outputs(
        out, tables=("daily_coverage", "daily_strata", "metrics")
    )
    coverage = tables["daily_coverage"].iloc[0]
    assert coverage.age_missing == 1
    assert coverage.age_invalid == 3
    assert coverage.fractional_boundary_unassigned == 4
    assert coverage.size_invalid == 2 and coverage.liquidity_invalid == 2
    bins = tables["daily_strata"].query(
        "signal == 'score' and stratification == 'forecast_age'"
    )
    assert bins.set_index("stratum").loc[
        ["0-7", "8-30", "31-60", "61-90", "91-180"], "stratum_rows"
    ].tolist() == [2, 2, 2, 2, 2]
    assert bins.stratum_rows.sum() == n - 4
    assert tables["metrics"].query("signal == 'score'").n_obs.iloc[0] == n


def test_scans_each_actual_input_selection_once(tmp_path, monkeypatch):
    _, config = _fixture(tmp_path, days=3)
    from quant_research_platform.research.evaluations import runner

    original = runner.iter_dataset_chunks
    scans = []

    def counted(session):
        scans.append(tuple(session.logical_columns))
        yield from original(session)

    monkeypatch.setattr(runner, "iter_dataset_chunks", counted)
    evaluate(
        {
            **config,
            "sample_scope": "common",
            "comparison_inputs": [
                {"inputs": config["inputs"], "signal_columns": ["z"]},
                {"inputs": config["inputs"], "signal_columns": ["x"]},
            ],
        }
    )
    assert len(scans) == 1
    assert {"x", "z", "total_market_cap", "amount"} <= set(scans[0])


def test_hac20_by_is_one_full_family_and_alt_policy_applies(tmp_path):
    from quant_research_platform.research.statistics import fdr_adjust, hac_mean

    _, config = _fixture(tmp_path, days=28)
    rng = np.random.default_rng(102)
    dates = pd.date_range("2023-01-02", periods=28)
    n = 12
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(n),
            "asset_id": [f"a{i}" for i in range(n)] * len(dates),
            "x": rng.normal(size=n * len(dates)),
            "z": rng.normal(size=n * len(dates)),
            "label": rng.normal(size=n * len(dates)),
            "label_available_at": pd.Timestamp("2023-01-02"),
            "eligible": True,
            "industry_level1_code": ["A"] * 6 * len(dates) + ["B"] * 6 * len(dates),
            "total_market_cap": np.tile(np.arange(n, dtype=float), len(dates)),
            "amount": rng.uniform(size=n * len(dates)),
        }
    )
    frame["x"] += 0.3 * frame["label"]
    config["inputs"] = _write_native(
        tmp_path / "random",
        frame,
        signals=("x", "z"),
        labels=("label",),
        availability={"label": "label_available_at"},
        units={"label": "decimal_return"},
    )
    config["strata"]["kinds"] = ["size"]
    own = evaluate(config)
    common = evaluate(
        {
            **config,
            "output": str(tmp_path / "common"),
            "sample_scope": "common",
            "comparison_inputs": [
                {"inputs": config["inputs"], "signal_columns": ["z"]}
            ],
        }
    )
    expected = []
    numeric_sequences = {}
    for scope, path in (("own", own), ("common", common)):
        _, tables = read_evaluation_outputs(path, tables=("metrics", "daily_strata"))
        for kind, group in [
            ("all", tables["metrics"]),
            *[
                (stratum, block)
                for stratum, block in tables["daily_strata"].groupby("stratum")
            ],
        ]:
            for statistic in ("ic", "rankic"):
                values = group[statistic].dropna().to_numpy()
                numeric_sequences[(scope, kind, statistic)] = values
                inference = hac_mean(values, max_lags=20)
                expected.append((scope, kind, statistic, inference))
    adjusted = fdr_adjust(
        [row[3].p_value for row in expected], method="BY"
    ).adjusted_p_values
    per_layer = np.concatenate(
        [
            fdr_adjust(
                [row[3].p_value for row in expected[index : index + 2]], method="BY"
            ).adjusted_p_values
            for index in range(0, len(expected), 2)
        ]
    )
    assert np.max(np.abs(adjusted - per_layer)) > 0.1
    for lag, method in ((20, "BY"), (1, "BH")):
        expected = [
            (scope, kind, statistic, hac_mean(values, max_lags=lag))
            for (scope, kind, statistic), values in numeric_sequences.items()
        ]
        adjusted = fdr_adjust(
            [row[3].p_value for row in expected], method=method
        ).adjusted_p_values
        output = factor_statistics(
            {
                "evaluation_inputs": [str(own), str(common)],
                "output": str(tmp_path / f"stats-{lag}"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": lag, "fdr_method": method, "fdr_alpha": 0.05},
            }
        )
        artifact = read_statistics_artifact(output)
        with artifact.table("factor_stratified_diagnostics.parquet") as result:
            assert result.hac_lag.eq(lag).all() and result.fdr_method.eq(method).all()
            assert result.defined_hypothesis_count.eq(len(expected)).all()
            for (scope, kind, statistic, inference), p in zip(
                expected, adjusted, strict=True
            ):
                selected = result.loc[
                    result.sample_scope.eq(scope)
                    & result.statistic.eq(statistic)
                    & (
                        result.stratification.eq("all")
                        if kind == "all"
                        else result.stratum.eq(kind)
                    )
                ].iloc[0]
                assert selected.standard_error == pytest.approx(
                    inference.standard_error
                )
                assert selected.fdr_adjusted_p_value == pytest.approx(p)


@pytest.mark.parametrize(
    "mutation", ("duplicate", "stratum", "range", "count", "reason")
)
def test_selected_daily_reader_and_reducer_reject_same_corruption(
    tmp_path, monkeypatch, mutation
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS
    from quant_research_platform.storage.datasets import reading

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "daily_strata.parquet"
    frame = pq.read_table(path).to_pandas()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "stratum":
        frame.loc[frame.stratification.eq("size"), "stratum"] = "4"
    elif mutation == "range":
        frame.loc[0, "ic"] = 1.1
        frame.loc[0, "ic_reason"] = None
    elif mutation == "count":
        frame.loc[0, "n_obs"] = int(frame.loc[0, "stratum_rows"]) + 1
    else:
        frame.loc[0, "ic_reason"] = "insufficient_observations"
    pq.write_table(
        pa.Table.from_pandas(
            frame, schema=DAILY_SCHEMAS["daily_strata"], preserve_index=False
        ),
        path,
    )
    original = reading.iter_declared_parquet_member_batches

    def batches(*args, **kwargs):
        kwargs["batch_rows"] = 1
        yield from original(*args, **kwargs)

    monkeypatch.setattr(reading, "iter_declared_parquet_member_batches", batches)
    with pytest.raises(ValueError):
        read_evaluation_outputs(output, tables=("daily_strata",))
    with pytest.raises(ValueError):
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "stats"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )


def test_cut_window_uses_only_selected_daily_members(tmp_path, monkeypatch):
    from quant_research_platform.storage.datasets import reading

    _, config = _fixture(tmp_path, days=4)
    output = evaluate(config)
    (output / "grouped_label_values.parquet").write_bytes(b"unselected corrupt member")
    (output / "ic_summary.parquet").write_bytes(b"unselected corrupt member")
    original = reading.iter_declared_parquet_member_batches
    scans = []

    def observed(path, **kwargs):
        scans.append((path.name, kwargs.get("parquet_filters")))
        yield from original(path, **kwargs)

    monkeypatch.setattr(reading, "iter_declared_parquet_member_batches", observed)
    reduced = factor_statistics(
        {
            "evaluation_inputs": [str(output)],
            "output": str(tmp_path / "slice"),
            "window": {"start": "2023-01-03", "end": "2023-01-04"},
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 1, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    assert {name for name, _ in scans} == {
        "metrics.parquet",
        "daily_coverage.parquet",
        "daily_strata.parquet",
    }
    assert len(scans) == 3 and all(filters is not None for _, filters in scans)
    with read_statistics_artifact(reduced).table(
        "factor_stratified_diagnostics.parquet"
    ) as frame:
        assert frame.input_date_count.eq(2).all()
        assert frame.loc[frame.stratification.eq("all"), "mean"].notna().all()


@pytest.mark.parametrize("scenario", ("ineligible", "immature", "empty_common"))
def test_no_legal_samples_still_preserve_every_focal_date(tmp_path, scenario):
    frame, config = _fixture(tmp_path, days=3)
    if scenario == "ineligible":
        frame["eligible"] = False
    elif scenario == "immature":
        frame["label_available_at"] = pd.Timestamp("2025-01-01")
    else:
        frame["z"] = np.nan
    config["inputs"] = _write_native(
        tmp_path / "empty-domain",
        frame,
        signals=("x", "z"),
        labels=("label",),
        availability={"label": "label_available_at"},
    )
    if scenario == "empty_common":
        config.update(
            sample_scope="common",
            comparison_inputs=[{"inputs": config["inputs"], "signal_columns": ["z"]}],
        )
    output = evaluate(config)
    _, tables = read_evaluation_outputs(
        output, tables=("metrics", "daily_coverage", "daily_strata")
    )
    assert len(tables["metrics"]) == len(tables["daily_coverage"]) == 3
    assert tables["metrics"].n_obs.eq(0).all()
    assert tables["metrics"].ic_reason.eq("insufficient_observations").all()
    assert tables["metrics"].ic.isna().all()


def test_common_does_not_renormalize_selected_ranks(tmp_path):
    _, config = _fixture(tmp_path)
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": list("abcde"),
            "x": [0.0, 1.0, 4.0, 5.0, 9.0],
            "z": [1.0, 1.0, np.nan, np.nan, 1.0],
            "label": [1.0, 3.0, 2.0, 9.0, 4.0],
            "label_available_at": pd.Timestamp("2023-01-02"),
            "eligible": True,
        }
    )
    config["inputs"] = _write_native(
        tmp_path / "ranks",
        frame,
        signals=("x", "z"),
        labels=("label",),
        availability={"label": "label_available_at"},
    )
    del config["strata"]
    config["evaluation"]["signals"][0]["transform"] = "rank"
    output = evaluate(
        {
            **config,
            "sample_scope": "common",
            "comparison_inputs": [
                {"inputs": config["inputs"], "signal_columns": ["z"]}
            ],
        }
    )
    _, tables = read_evaluation_outputs(output, tables=("metrics",))
    actual = tables["metrics"].ic.iloc[0]
    assert actual == pytest.approx(np.corrcoef([1.0, 2.0, 5.0], [1.0, 3.0, 4.0])[0, 1])
    assert abs(actual - np.corrcoef([1.0, 2.0, 3.0], [1.0, 3.0, 4.0])[0, 1]) > 0.01


@pytest.mark.parametrize(
    "change",
    (
        {"hac_lag": -1},
        {"hac_lag": True},
        {"fdr_method": "holm"},
        {"fdr_alpha": 0.0},
        {"fdr_alpha": 1.0},
    ),
)
def test_invalid_persisted_policy_fails_before_member_scans(
    tmp_path, monkeypatch, change
):
    from quant_research_platform.artifacts.outputs import evaluation as outputs

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid policy must not read")

    monkeypatch.setattr(outputs, "_iter_evaluation_member_batches", forbidden)
    with pytest.raises(ValueError):
        factor_statistics(
            {
                "evaluation_inputs": ["not-opened"],
                "output": str(tmp_path / "invalid"),
                "analyses": ["stratified_ic"],
                "statistics": {
                    "hac_lag": 20,
                    "fdr_method": "BY",
                    "fdr_alpha": 0.05,
                    **change,
                },
            }
        )


def test_legacy_rank_ic_reduces_one_persisted_evaluation(tmp_path):
    _, config = _fixture(tmp_path, days=3)
    output = evaluate(config)
    result = factor_statistics(
        {
            "evaluation_inputs": [str(output)],
            "output": str(tmp_path / "legacy"),
            "analyses": ["rank_ic"],
            "evaluation": {"primary_label": "label", "factors": ["score"]},
            "statistics": {
                "hac_lag": 1,
                "factor_fdr_families": {"selected": ["score"]},
                "fdr_primary": "BH_5pct",
                "fdr_robustness": "BY_5pct",
            },
        }
    )
    with read_statistics_artifact(result).table("factor_diagnostics.parquet") as frame:
        assert set(frame.fdr_method) == {"BH", "BY"}
        assert frame.input_date_count.eq(3).all()
        assert frame.observation_count.eq(3).all()


def test_selected_metrics_reader_rejects_non_native_physical_type(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "metrics.parquet"
    table = pq.read_table(path)
    schema = pa.schema(
        [
            pa.field(field.name, pa.float32() if field.name == "ic" else field.type)
            for field in table.schema
        ]
    )
    pq.write_table(table.cast(schema), path)
    with pytest.raises(ValueError, match="metrics_schema_invalid"):
        read_evaluation_outputs(output, tables=("metrics",))
    with pytest.raises(ValueError, match="metrics_schema_invalid"):
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "stats"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )


@pytest.mark.parametrize(
    "mutation",
    ("duplicate", "scope", "stratum", "pairs", "reason", "family_count", "missing_all"),
)
def test_stratified_statistics_semantics_shared_by_writer_and_reader(
    tmp_path, mutation
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.statistics import write_statistics_artifact

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    root = factor_statistics(
        {
            "evaluation_inputs": [str(output)],
            "output": str(tmp_path / "stats"),
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    artifact = read_statistics_artifact(root)
    path = root / "factor_stratified_diagnostics.parquet"
    table = pq.read_table(path)
    frame = table.to_pandas()
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing_all":
        frame = frame.loc[
            ~(frame.stratification.eq("all") & frame.statistic.eq("ic"))
        ].copy()
    elif mutation == "scope":
        frame.loc[0, "sample_scope"] = "common"
    elif mutation == "stratum":
        frame.loc[0, "stratum"] = "literal-null"
    elif mutation == "pairs":
        frame.loc[0, "observation_pairs"] = 0
    elif mutation == "reason":
        frame.loc[0, "reason"] = "unexplained"
    else:
        frame["defined_hypothesis_count"] = 1
    with pytest.raises(ValueError):
        write_statistics_artifact(
            tmp_path / "rejected",
            metadata=artifact.metadata,
            tables={"factor_stratified_diagnostics": frame},
            table_group="factor",
        )
    pq.write_table(
        pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False), path
    )
    with pytest.raises(ValueError):
        with artifact.table(path.name):
            pass


def test_exhaustive_strata_observation_totals_match_all(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "daily_strata.parquet"
    frame = pq.read_table(path).to_pandas()
    mask = frame.stratification.eq("industry") & frame.stratum.eq("<NA>")
    assert frame.loc[mask, "n_obs"].tolist() == [0]
    frame.loc[mask, "n_obs"] = 1
    pq.write_table(
        pa.Table.from_pandas(
            frame, schema=DAILY_SCHEMAS["daily_strata"], preserve_index=False
        ),
        path,
    )
    with pytest.raises(ValueError, match="stratum_observations_mismatch"):
        read_evaluation_outputs(
            output, tables=("metrics", "daily_coverage", "daily_strata")
        )
    with pytest.raises(ValueError, match="stratum_observations_mismatch"):
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "stats"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )


def test_summary_nonempty_values_require_valid_observations(tmp_path):
    import pyarrow.parquet as pq

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "ic_summary.parquet"
    frame = pq.read_table(path).to_pandas()
    frame["valid_ic_count"] = 0
    frame["std_ic"] = np.nan
    frame["std_ic_reason"] = "no_valid_observations"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="summary_count_values"):
        read_evaluation_outputs(output, tables=("ic_summary",))


def test_empty_native_window_publishes_and_reduces_undefined(tmp_path):
    _, config = _fixture(tmp_path)
    config["window"] = {"start": "2023-02-01", "end": "2023-02-02"}
    output = evaluate(config)
    _, tables = read_evaluation_outputs(
        output, tables=("metrics", "daily_coverage", "daily_strata")
    )
    assert all(frame.empty for frame in tables.values())
    reduced = factor_statistics(
        {
            "evaluation_inputs": [str(output)],
            "output": str(tmp_path / "empty-statistics"),
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    with read_statistics_artifact(reduced).table(
        "factor_stratified_diagnostics.parquet"
    ) as frame:
        assert len(frame) == 2
        assert frame.input_date_count.eq(0).all()
        assert frame.observation_count.eq(0).all()
        assert frame.status.eq("UNDEFINED").all()
        assert frame["mean"].isna().all()


def test_declared_strata_requires_daily_members_without_scanning_them(tmp_path):
    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["optional_member_rows"] = {}
    manifest["files"] = [
        name for name in manifest["files"] if not name.startswith("daily_")
    ]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evaluation_optional_members_invalid"):
        read_evaluation_outputs(output, tables=("metrics",))
    with pytest.raises(ValueError, match="evaluation_optional_members_invalid"):
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "missing-daily-statistics"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )


def test_summary_valid_count_cannot_exceed_declared_dates(tmp_path):
    import pyarrow.parquet as pq

    _, config = _fixture(tmp_path)
    output = evaluate(config)
    path = output / "ic_summary.parquet"
    frame = pq.read_table(path).to_pandas()
    frame["valid_ic_count"] = 1000000
    frame.to_parquet(path, index=False)
    (output / "metrics.parquet").write_bytes(b"unselected")
    with pytest.raises(ValueError, match="summary_count_values"):
        read_evaluation_outputs(output, tables=("ic_summary",))


def test_daily_key_memory_is_one_date_for_producer_and_consumer(tmp_path, monkeypatch):
    from quant_research_platform.artifacts.outputs import evaluation as artifacts
    from quant_research_platform.research.evaluations import runner

    _, config = _fixture(tmp_path, days=8)
    sizes = []
    original = artifacts._EvaluationDateKeys.add_table

    def observe(self, table):
        result = original(self, table)
        if self.index is not None:
            sizes.append(sum(len(values) for values in self.index._namespaces.values()))
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("scoped evaluation must not duplicate dataset PK history")

    monkeypatch.setattr(artifacts._EvaluationDateKeys, "add_table", observe)
    monkeypatch.setattr(runner, "ExactPrimaryKeyIndex", forbidden)
    output = evaluate(config)
    producer_max = max(sizes)
    sizes.clear()
    read_evaluation_outputs(
        output, tables=("metrics", "daily_coverage", "daily_strata")
    )
    assert max(sizes) == producer_max == 8


@pytest.mark.parametrize("member", ["metrics", "daily_coverage", "daily_strata"])
def test_daily_reader_rejects_backward_date_even_without_duplicate(tmp_path, member):
    import pyarrow.parquet as pq

    _, config = _fixture(tmp_path, days=2)
    output = evaluate(config)
    path = output / f"{member}.parquet"
    table = pq.read_table(path)
    order = table.to_pandas().timestamp.argsort()[::-1]
    pq.write_table(table.take(order.to_numpy()), path, row_group_size=1)
    with pytest.raises(ValueError, match="timestamp_order_invalid"):
        read_evaluation_outputs(output, tables=(member,))


def test_age_unassigned_observations_bounded_by_fractional_population(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS

    frame, config = _fixture(tmp_path)
    age = "median_forecast_age_days_fy1"
    frame[age] = [0.0, 0.0, 7.5, 0.0]
    config["inputs"] = _write_native(
        tmp_path / "age-bounds",
        frame,
        signals=("x", age),
        labels=("label",),
        availability={"label": "label_available_at"},
        units={age: "days"},
    )
    config["evaluation"]["signals"].append(
        {"signal_id": "age", "source_kind": "factor", "source_column": age}
    )
    config["strata"] = {
        "primary_label": "label",
        "kinds": ["forecast_age"],
        "forecast_freshness_days": 90,
    }
    output = evaluate(config)
    path = output / "daily_strata.parquet"
    frame = pq.read_table(path).to_pandas()
    row = frame.signal.eq("score") & frame.stratum.eq("0-7")
    assert frame.loc[row, "n_obs"].tolist() == [2]
    frame.loc[row, "n_obs"] = 0
    frame.loc[row, ["ic", "rankic"]] = np.nan
    frame.loc[row, ["ic_reason", "rankic_reason"]] = "insufficient_observations"
    pq.write_table(
        pa.Table.from_pandas(
            frame, schema=DAILY_SCHEMAS["daily_strata"], preserve_index=False
        ),
        path,
    )
    with pytest.raises(ValueError, match="stratum_observations_mismatch"):
        read_evaluation_outputs(
            output, tables=("metrics", "daily_coverage", "daily_strata")
        )


def test_null_stratum_must_belong_to_actual_published_domain(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.evaluation_daily import DAILY_SCHEMAS

    frame, config = _fixture(tmp_path)
    frame["industry_level1_code"] = "A"
    config["inputs"] = _write_native(
        tmp_path / "known-industry",
        frame,
        signals=("x",),
        labels=("label",),
        availability={"label": "label_available_at"},
    )
    config["strata"]["kinds"] = ["industry"]
    output = evaluate(config)
    path = output / "daily_strata.parquet"
    frame = pq.read_table(path).to_pandas()
    frame["stratum"] = None
    pq.write_table(
        pa.Table.from_pandas(
            frame, schema=DAILY_SCHEMAS["daily_strata"], preserve_index=False
        ),
        path,
    )
    with pytest.raises(ValueError, match="stratum_domain_invalid"):
        read_evaluation_outputs(output, tables=("daily_strata",))
    with pytest.raises(ValueError, match="stratum_domain_invalid"):
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "unknown-domain-statistics"),
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )
