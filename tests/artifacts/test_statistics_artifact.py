from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import statistics


@pytest.fixture
def tables(request):
    group = getattr(request, "param", "research")
    row = dict(
        signal="factor",
        label="label",
        sample_scope="own",
        stratification="all",
        stratum=None,
        statistic="rankic",
        observed_date_count=32,
        observation_pairs=320,
        hac_lag=20,
        family_id="family",
        factor_id="factor",
        comparison_factor_id="peer",
        year=2023,
        mean_absolute=0.2,
        prediction_id="prediction",
        period_id="2023Q1",
        selection_group="global",
        decision_id="final",
        config_id="config",
        metric_id="validation_rankic",
        test_id="spa_reality_check",
        regression_id="single:factor",
        coefficient_id="factor",
        sample_role="validation",
        benchmark_id="benchmark",
        statistic_id="p_value",
        selection_status="CHAMPION",
        status="DEFINED",
        reason="",
        white_reason="",
        timestamp=pd.Timestamp("2023-01-02"),
        observation_count=32,
        prediction_count=32,
        target_count=32,
        mse=0.2,
        mse_reason="",
        baseline_mse=0.4,
        oos_r2=0.5,
        oos_r2_reason="",
        head_selected_count=7,
        head_evaluable_count=7,
        head_coverage=1.0,
        pool_evaluable_count=32,
        head_mean_raw_return=0.3,
        pool_mean_raw_return=0.1,
        head_excess_raw_return=0.2,
        input_date_count=32,
        period_count=32,
        mean=0.2,
        standard_error=0.1,
        t_statistic=2.0,
        p_value=0.05,
        ci_lower=0.0,
        ci_upper=0.4,
        bootstrap_status="DEFINED",
        bootstrap_reason="",
        bootstrap_lower=0.0,
        bootstrap_upper=0.4,
        fdr_method="BH",
        fdr_alpha=0.05,
        fdr_adjusted_p_value=0.05,
        planned_hypothesis_count=1,
        defined_hypothesis_count=1,
        missing_hypothesis_count=0,
        bootstrap_block_size=20,
        bootstrap_repetitions=10000,
        bootstrap_seed=42,
        value=0.2,
        loss=0.2,
        white_p_value=0.05,
        white_defined_period_count=32,
        white_undefined_period_count=0,
        common_date=1,
        common_observation_count=32,
        block_size=20,
        repetitions=10000,
        alpha=0.05,
        seed=42,
    )
    result = {
        member.removesuffix(".parquet"): pd.DataFrame(
            [
                {
                    column: row[column]
                    for column in statistics.STATISTICS_TABLE_COLUMNS[member]
                }
            ]
        )
        for member in statistics.STATISTICS_TABLE_GROUPS[group]
        if member != "model_group_comparison.parquet"
    }
    if group == "research":
        result["model_group_comparison"] = pa.Table.from_pylist(
            [], schema=statistics._ARROW_SCHEMAS["model_group_comparison.parquet"]
        ).to_pandas()
    if "factor_correlation" in result:
        result["factor_correlation"]["statistic"] = "pearson"
    if "factor_stratified_diagnostics" in result:
        new = result["factor_stratified_diagnostics"]
        result["factor_stratified_diagnostics"] = pd.concat(
            [new.assign(statistic="ic"), new], ignore_index=True
        )
        result["factor_stratified_diagnostics"]["defined_hypothesis_count"] = 2
        result["factor_stratified_diagnostics"]["stratum"] = result[
            "factor_stratified_diagnostics"
        ]["stratum"].astype("string")
    for frame in result.values():
        if "timestamp" in frame:
            frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    return result


@pytest.fixture
def metadata(tables):
    return {
        "evaluation_binding": {
            "signals": [{"signal_id": "factor"}],
            "label_columns": ["label"],
            "strata": {},
        },
        "family_scopes": ["own"],
        "input_date_counts": {"own": 32},
        "statistics": {"hac_lag": 20, "fdr_method": "BH", "fdr_alpha": 0.05},
        "sample_role": "validation",
        "window": {"start": "2023-01-02", "end": "2023-03-31"},
    }


def test_table_consumer_primary_exception_is_preserved(tmp_path, tables, metadata):
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics", metadata=metadata, tables=tables
    )
    artifact = statistics.read_statistics_artifact(root)
    primary = KeyboardInterrupt("consumer_primary")
    with pytest.raises(KeyboardInterrupt) as caught:
        with artifact.table("model_trials.parquet") as frame:
            assert not frame.empty
            raise primary
    assert caught.value is primary


@pytest.mark.parametrize(
    "tables,member",
    [
        (
            "factor"
            if member.removesuffix(".parquet") + ".parquet"
            in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research",
            member,
        )
        for member in (
            *(
                member
                for member in statistics.STATISTICS_TABLE_GROUPS["research"]
                if member != "model_prediction_metrics.parquet"
            ),
            *statistics.STATISTICS_TABLE_GROUPS["factor"],
        )
    ],
    indirect=["tables"],
)
@pytest.mark.parametrize("mutation", ["status", "reason", "count", "undefined"])
def test_research_table_rejects_invalid_values_at_write(
    tmp_path, tables, metadata, member, mutation
):
    name = member.removesuffix(".parquet")
    if name == "model_group_comparison":
        tables[name] = pd.DataFrame([_defined_group_comparison_row()])
    if mutation == "status":
        tables[name].loc[0, "status"] = "UNKNOWN"
    elif mutation == "reason":
        tables[name].loc[0, "reason"] = "contradicts_defined_status"
    elif mutation == "undefined":
        tables[name].loc[0, "status"] = "UNDEFINED"
    else:
        count = (
            "period_count"
            if name == "fama_macbeth"
            else "prediction_count"
            if name == "model_prediction_metrics"
            else "observation_count"
        )
        tables[name].loc[0, count] = -1
    error = (
        "statistics_table_count_invalid"
        if mutation == "count"
        else "statistics_group_comparison_invalid"
        if name == "model_group_comparison"
        else "statistics_table_status_invalid"
        if mutation == "status"
        else "statistics_table_status_value_invalid"
    )
    if name == "prediction_diagnostics" or mutation == "undefined":
        _assert_rejected_at_write(
            tmp_path, tables, metadata, member, error=f"^{error}$"
        )
    else:
        with pytest.raises(ValueError, match=f"^{error}$"):
            statistics._validated_frame(tables[name], member=member, metadata=metadata)


def _assert_rejected_at_write(
    tmp_path, tables, metadata, member, *, table_group=None, error="statistics_table_"
):
    if table_group is None:
        table_group = (
            "factor"
            if member in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research"
        )
    with pytest.raises(ValueError, match=error):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            metadata=metadata,
            tables=tables,
            table_group=table_group,
        )
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize(
    "tables,name,column",
    [
        (
            "factor"
            if name + ".parquet" in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research",
            name,
            column,
        )
        for name, column in [
            ("factor_diagnostics", "mean"),
            ("prediction_diagnostics", "mean"),
            ("model_trials", "value"),
            ("fama_macbeth", "mean"),
            ("fama_macbeth_periods", "value"),
            ("model_losses", "loss"),
            ("model_comparison", "mean"),
            ("model_reality_checks", "value"),
        ]
    ],
    indirect=["tables"],
)
def test_defined_research_measurements_cannot_be_missing(
    tmp_path, tables, metadata, name, column
):
    tables[name].loc[0, column] = float("nan")
    _assert_rejected_at_write(tmp_path, tables, metadata, f"{name}.parquet")


@pytest.mark.parametrize(
    "tables,member",
    [
        (
            "factor"
            if member.removesuffix(".parquet") + ".parquet"
            in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research",
            member,
        )
        for member in (
            *statistics.STATISTICS_TABLE_GROUPS["research"],
            *statistics.STATISTICS_TABLE_GROUPS["factor"],
        )
    ],
    indirect=["tables"],
)
def test_research_table_counts_reject_lossy_coercion(tables, metadata, member):
    name = member.removesuffix(".parquet")
    if name == "model_group_comparison":
        tables[name] = pd.DataFrame([_defined_group_comparison_row()])
    count = (
        "period_count"
        if name == "fama_macbeth"
        else "prediction_count"
        if name == "model_prediction_metrics"
        else "observation_count"
    )
    tables[name][count] = pd.Series([1.5] * len(tables[name]), dtype=object)
    with pytest.raises(ValueError, match="statistics_table_count_invalid"):
        statistics._validated_frame(tables[name], member=member, metadata=metadata)


def test_group_comparison_integral_counts_round_trip(tmp_path, tables, metadata):
    count = 32
    metadata = _test_statistics_metadata(metadata, tables)
    tables["model_group_comparison"] = pd.DataFrame([_defined_group_comparison_row()])
    tables["model_group_comparison"]["observation_count"] = pd.Series(
        [count], dtype=object
    )
    tables["model_group_comparison"]["calendar_count"] = pd.Series(
        [count], dtype=object
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "valid",
        metadata=metadata,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("model_group_comparison.parquet") as frame:
        assert frame.observation_count.tolist() == [count]
        assert frame.calendar_count.tolist() == [count]


@pytest.mark.parametrize("tables", ["factor"], indirect=True)
@pytest.mark.parametrize(
    "factor_case", ["formal", "diagnostic_only", "no_factor_columns"]
)
def test_research_statistics_tables_round_trip_by_factor_case(
    tmp_path, tables, metadata, factor_case
):
    factor = tables["factor_diagnostics"]
    if factor_case != "formal":
        factor["fdr_method"] = "NOT_APPLICABLE"
        factor["fdr_alpha"] = factor["fdr_adjusted_p_value"] = float("nan")
        factor["family_id"] = factor_case
    if factor_case == "no_factor_columns":
        factor["family_id"] = factor["factor_id"] = "none"
        factor["status"] = "UNDEFINED"
        factor["reason"] = factor_case
        factor.loc[
            :,
            [
                "mean",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
        ] = float("nan")
        factor.loc[
            :,
            [
                "observation_count",
                "planned_hypothesis_count",
                "defined_hypothesis_count",
                "missing_hypothesis_count",
            ],
        ] = 0
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics",
        metadata=metadata,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(root)
    for member in statistics.STATISTICS_TABLE_GROUPS["factor"]:
        with artifact.table(member) as actual:
            pd.testing.assert_frame_equal(
                actual, tables[member.removesuffix(".parquet")], check_exact=True
            )


def test_writer_cleanup_preserves_primary(tmp_path, tables, metadata, monkeypatch):
    primary = KeyboardInterrupt("write_interrupted")
    original = pq.ParquetWriter.close

    def cleanup_failure(self):
        original(self)
        raise RuntimeError("writer_close_failed")

    def interrupt(*args, **kwargs):
        raise primary

    monkeypatch.setattr(pq.ParquetWriter, "close", cleanup_failure)
    monkeypatch.setattr(pq.ParquetWriter, "write_table", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        statistics.write_statistics_artifact(
            tmp_path / "statistics",
            metadata=metadata,
            tables=tables,
            table_group="factor" if "factor_diagnostics" in tables else "research",
        )
    assert caught.value is primary
    assert any("RuntimeError" in note for note in primary.__notes__)
    assert not (tmp_path / "statistics").exists()


@pytest.mark.parametrize("name", ("prediction_diagnostics", "model_comparison"))
def test_undefined_model_inference_preserves_descriptive_mean(
    tmp_path, tables, metadata, name
):
    frame = tables[name]
    frame.loc[0, "status"] = "UNDEFINED"
    frame.loc[0, "reason"] = "global_validation_selection_reuses_validation"
    frame.loc[0, "bootstrap_status"] = "UNDEFINED"
    frame.loc[0, "bootstrap_reason"] = "global_validation_selection_reuses_validation"
    inference_columns = [
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "bootstrap_lower",
        "bootstrap_upper",
    ]
    frame.loc[0, inference_columns] = float("nan")
    root = statistics.write_statistics_artifact(
        tmp_path / "descriptive",
        metadata=metadata,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table(f"{name}.parquet") as result:
        assert result["mean"].iloc[0] == 0.2
        assert result.observation_count.iloc[0] == 32
        assert result.status.iloc[0] == "UNDEFINED"
        assert result.loc[:, inference_columns].isna().all().all()


@pytest.mark.parametrize("name", ("prediction_diagnostics", "model_comparison"))
def test_undefined_model_inference_rejects_inferential_values(
    tmp_path, tables, metadata, name
):
    frame = tables[name]
    frame.loc[0, "status"] = "UNDEFINED"
    frame.loc[0, "reason"] = "global_validation_selection_reuses_validation"
    frame.loc[0, "bootstrap_status"] = "UNDEFINED"
    frame.loc[0, "bootstrap_reason"] = "global_validation_selection_reuses_validation"
    frame.loc[
        0,
        [
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
            "bootstrap_lower",
            "bootstrap_upper",
        ],
    ] = float("nan")
    root = statistics.write_statistics_artifact(
        tmp_path / "valid_baseline", metadata=metadata, tables=tables
    )
    with statistics.read_statistics_artifact(root).table(f"{name}.parquet") as restored:
        assert restored.loc[0, "mean"] == frame.loc[0, "mean"]
        assert restored.loc[0, "bootstrap_status"] == "UNDEFINED"
    frame.loc[0, "p_value"] = 0.01
    _assert_rejected_at_write(
        tmp_path,
        tables,
        metadata,
        f"{name}.parquet",
        error="^statistics_table_status_value_invalid$",
    )


@pytest.mark.parametrize(
    "corruption,fdr_method",
    [
        ("none", "BH"),
        ("none", "BY"),
        ("none", "NOT_APPLICABLE"),
        ("global_mean", "BY"),
        ("quarterly_mean", "BY"),
        ("mean_difference", "BY"),
        ("sample_role", "BY"),
    ],
)
def test_defined_group_comparison_requires_descriptive_facts(
    tmp_path, tables, metadata, corruption, fdr_method
):
    metadata = _test_statistics_metadata(metadata, tables)
    row = _defined_group_comparison_row()
    row["fdr_method"] = fdr_method
    if fdr_method == "NOT_APPLICABLE":
        row["fdr_adjusted_p_value"] = float("nan")
    if corruption == "sample_role":
        row[corruption] = "validation"
    elif corruption != "none":
        row[corruption] = (
            0 if corruption in {"observation_count", "standard_error"} else float("nan")
        )
    tables["model_group_comparison"] = pd.DataFrame([row])
    if corruption != "none":
        _assert_rejected_at_write(
            tmp_path,
            tables,
            metadata,
            "model_group_comparison.parquet",
            error="^statistics_sample_role_mismatch$"
            if corruption == "sample_role"
            else "^statistics_group_comparison_invalid$",
        )
    else:
        root = statistics.write_statistics_artifact(
            tmp_path / "group",
            metadata=metadata,
            tables=tables,
            table_group="factor" if "factor_diagnostics" in tables else "research",
        )
        artifact = statistics.read_statistics_artifact(root)
        with artifact.table("model_group_comparison.parquet") as frame:
            assert frame.global_mean.iloc[0] == 0.2
            assert frame.observation_count.iloc[0] == 64
            assert frame.fdr_method.iloc[0] == fdr_method
            assert frame.fdr_adjusted_p_value.isna().all() == (
                fdr_method == "NOT_APPLICABLE"
            )


def _defined_group_comparison_row():
    return dict(
        family_id="family",
        sample_role="test",
        status="DEFINED",
        reason="",
        fdr_method="BY",
        bootstrap_status="UNDEFINED",
        bootstrap_reason="insufficient_blocks",
        observation_count=64,
        calendar_count=64,
        fdr_family_size=1,
        hac_lag=19,
        bootstrap_block_size=20,
        bootstrap_repetitions=10000,
        bootstrap_seed=42,
        global_mean=0.2,
        quarterly_mean=0.3,
        mean_difference=0.1,
        standard_error=0.1,
        t_statistic=1.0,
        p_value=0.3,
        ci_lower=-0.1,
        ci_upper=0.3,
        fdr_adjusted_p_value=0.3,
        fdr_alpha=0.05,
        confidence=0.95,
        bootstrap_lower=float("nan"),
        bootstrap_upper=float("nan"),
    )


def _test_statistics_metadata(metadata, tables):
    for frame in tables.values():
        if "sample_role" in frame:
            frame["sample_role"] = "test"
    return {**metadata, "sample_role": "test"}


@pytest.mark.parametrize("boundary", ("access", "outside", "factor_history"))
def test_research_statistics_bind_sample_role_and_explicit_window(
    tmp_path, tables, metadata, boundary
):
    if boundary == "access":
        tables["model_losses"]["sample_role"] = "test"
        error = "statistics_sample_role_mismatch"
    else:
        tables["model_losses"]["timestamp"] = pd.Timestamp(
            "2030-01-02" if boundary == "outside" else "2014-01-02"
        )
        error = "statistics_table_window_mismatch"
    _assert_rejected_at_write(
        tmp_path, tables, metadata, "model_losses.parquet", error=error
    )


def test_statistics_table_group_matches_declared_members(tmp_path, tables, metadata):
    with pytest.raises(ValueError, match="statistics_tables_invalid"):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            metadata=metadata,
            tables=tables,
            table_group="factor",
        )
    root = statistics.write_statistics_artifact(
        tmp_path / "valid", metadata=metadata, tables=tables
    )
    artifact = statistics.read_statistics_artifact(root)
    with pytest.raises(ValueError, match="statistics_table_member_invalid"):
        with artifact.table("fama_macbeth.parquet"):
            pass


def test_legal_internal_model_name_survives_joint_test_publication(
    tmp_path, tables, metadata
):
    import numpy as np

    from quant_research_platform.research.models.statistics import _model_comparison

    family = "__benchmark_loss__"
    dates = pd.bdate_range("2023-01-02", periods=40)
    rank = pd.DataFrame(
        {
            "baseline": 0.2 + 0.1 * np.sin(np.arange(40)),
            family: 0.3 + 0.1 * np.cos(np.arange(40)),
        },
        index=dates,
    )
    comparison, checks, robustness = _model_comparison(
        rank,
        family_ids=("baseline", family),
        benchmark_id="baseline",
        sample_role="validation",
        inference_blocker=None,
        max_lags=2,
        block_size=3,
        repetitions=99,
        root_seed=42,
        dataset_id="probe",
    )
    assert robustness.status.eq("DEFINED").all()
    assert family in set(checks.family_id)
    tables.update(
        model_comparison=comparison, model_reality_checks=checks, robustness=robustness
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "joint", metadata=metadata, tables=tables
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("model_reality_checks.parquet") as actual:
        pd.testing.assert_frame_equal(
            actual,
            checks.assign(reason=checks.reason.fillna("")),
            check_dtype=False,
            check_exact=True,
        )
        assert family in set(actual.family_id)


@pytest.mark.parametrize("access", ("validation", "test"))
def test_paired_rankic_and_champion_diagnostics_reach_native_artifacts(
    tmp_path, tables, metadata, access
):
    import numpy as np

    import quant_research_platform.research.models.statistics as model_stats

    dates = pd.bdate_range("2023-01-02", periods=5)
    permutations = {
        "global": [(1, 2, 3), (1, 3, 2), (1, 1, 1), (1, 3, 2), (3, 1, 2)],
        "quarterly": [(1, 3, 2), (1, 2, 3), (1, 2, 3), (3, 1, 2), (1, 2, 3)],
    }
    ranks = {}
    for group, values in permutations.items():
        frame = pd.DataFrame(
            {
                "timestamp": dates.repeat(3),
                "asset_id": ["a", "b", "c"] * 5,
                "prediction": np.asarray(values).ravel(),
                "label": [1.0, 2.0, 3.0] * 5,
            }
        )
        ranks[group] = (
            model_stats.cross_sectional_rank_ic(
                frame, value_columns=("prediction",), label_column="label"
            )
            .set_index("timestamp")
            .prediction
        )
    losses = {
        group: pd.DataFrame(
            dict(
                timestamp=dates,
                sample_role=access,
                period_id="2023Q1",
                family_id="family",
                config_id="config",
                loss=-values.to_numpy(),
                observation_count=3,
                status=np.where(values.notna(), "DEFINED", "UNDEFINED"),
                reason=np.where(values.notna(), "", "cross_sectional_rankic_undefined"),
                common_date=values.notna().astype(int).to_numpy(),
                common_observation_count=3,
            )
        )
        for group, values in ranks.items()
    }
    config = {
        "sample_role": access,
        "statistics": {
            "hac_lag": 1,
            "bootstrap_block": 2,
            "bootstrap_repetitions": 32,
            "group_comparison": {
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    expected = {}
    for group in ("global", "quarterly"):
        bound = {
            **metadata,
            "sample_role": access,
            "selection_group": group,
            "coverage_basis": "model_prediction_domain",
            "evaluation": {"primary_label": "label"},
            "evaluation_axis": "cross_sectional",
            "column_roles": {"label": "label"},
            "column_units": {},
        }
        if group == "global" and access == "validation":
            bound["selection_cutoffs"] = [metadata["window"]["end"]]
        selected = {name: table.copy(deep=True) for name, table in tables.items()}
        for table in selected.values():
            if "sample_role" in table:
                table["sample_role"] = access
        selected["model_losses"] = losses[group]
        selected["prediction_diagnostics"] = model_stats._prediction_summary(
            ranks[group].to_frame(model_stats._PREDICTION_COLUMN),
            prediction_id=f"factors_only_{group}_persisted_predictions",
            inference_blocker="selection_reuses_evaluation_window"
            if group == "global" and access == "validation"
            else None,
            max_lags=1,
            block_size=2,
            repetitions=32,
            seed=42,
        )
        if group == "quarterly":
            selected["model_group_comparison"] = model_stats._paired_group_rankic(
                losses[group].assign(
                    **{
                        "global": -losses["global"].loss,
                        "quarterly": -losses[group].loss,
                    }
                ),
                inference_blocker="selection_reuses_evaluation_window"
                if access == "validation"
                else None,
                selection_group=group,
                family_ids=("family",),
                config=config,
            )
        logical = tmp_path / group
        root = statistics.write_statistics_artifact(
            logical, metadata=bound, tables=selected
        )
        artifact = statistics.read_statistics_artifact(root)
        if group == "quarterly":
            with artifact.table("model_group_comparison.parquet") as paired:
                row = paired.iloc[0]
                assert row.observation_count == 4 and row.calendar_count == 5
                assert row.global_mean == 0.375
                assert row.quarterly_mean == 0.5
                assert (
                    row.mean_difference == row.quarterly_mean - row.global_mean == 0.125
                )
                assert row.fdr_family_size == 1
                assert row.status == "UNDEFINED"
                assert row.reason == (
                    "selection_reuses_evaluation_window"
                    if access == "validation"
                    else "incomplete_session_clock"
                )
        with artifact.table("prediction_diagnostics.parquet") as diagnostic:
            expected[group] = (
                diagnostic.astype(object)
                .where(diagnostic.notna(), None)
                .to_dict("records")
            )
    for group in ("global", "quarterly"):
        rows = expected[group]
        assert rows[0]["input_date_count"] == 5
        assert rows[0]["input_date_count"] - rows[0]["observation_count"] == (
            1 if group == "global" else 0
        )
        assert rows[0]["status"] == ("UNDEFINED" if group == "global" else "DEFINED")


@pytest.mark.parametrize("varying_ic", (False, True))
def test_native_model_persistence_preserves_std_reason_in_statistics(
    tmp_path, tables, metadata, varying_ic
):
    import numpy as np

    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.models.statistics import _selection_trial_rows
    from quant_research_platform.research.models.trial_selection import (
        read_selection,
        score,
        select,
    )
    from quant_research_platform.research.predictions.runner import predict
    from tests.research.direct_model_test_support import fit_frame

    dates = pd.date_range("2020-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(3),
            "asset_id": ["asset_0", "asset_1", "asset_2"] * 5,
            "eligible": True,
            "x": np.tile([-1.0, 0.0, 1.0], 5),
            "label": np.tile([0.0, 1.0, 0.0], 5),
        }
    )
    frame["label_available_at"] = frame.timestamp
    frame.loc[
        frame.groupby("timestamp", sort=False).ngroup().mod(2).eq(1)
        & frame.asset_id.eq("asset_2"),
        "label",
    ] = 0.1 if varying_ic else 0.0
    frame.loc[frame.timestamp.eq(dates[0]), "label"] = [-1.0, 0.0, 1.0]
    spec = ModelSpec(
        model_id="rank",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    trained = fit_frame(
        tmp_path / "train",
        frame.loc[frame.timestamp.eq(dates[0])].copy(),
        spec,
        market_id="a_share",
    )
    source = tmp_path / "validation.parquet"
    frame.to_parquet(source, index=False)
    window = {"start": "2020-01-02", "end": "2020-01-05"}
    prediction = predict(
        {
            "model": str(trained.root),
            "inputs": str(source),
            "input_spec": trained.config["input_spec"],
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    scored = score(
        {
            "predictions": [str(prediction)],
            "labels": str(source),
            "label_spec": trained.config["input_spec"],
            "candidate_ids": ["rank"],
            "scoring_window": window,
            "validation_windows": [window],
            "cutoff": "2020-01-05",
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "output": str(tmp_path / "scores.json"),
            "label_unit": "decimal_return",
        }
    )
    selected = select(
        {
            "scores": str(scored),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["rank"],
            "refit_windows": [*trained.config["fit_windows"], window],
            "refit_cutoff": "2020-01-05",
        }
    )
    tables["model_trials"] = _selection_trial_rows(
        {"final": read_selection(selected)}, "global"
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics", metadata=metadata, tables=tables
    )
    with statistics.read_statistics_artifact(root).table(
        "model_trials.parquet"
    ) as persisted:
        rows = persisted.set_index("metric_id")
    daily_ic = np.array(
        [
            np.corrcoef(group.x, group.label)[0, 1]
            for _, group in frame.loc[frame.timestamp.isin(dates[1:])].groupby(
                "timestamp"
            )
        ]
    )
    expected_std = float(np.std(daily_ic))
    assert rows.loc["validation_std_ic", "status"] == "DEFINED"
    assert rows.loc["validation_std_ic", "value"] == pytest.approx(expected_std)
    assert rows.loc["validation_std_ic", "reason"] == ""
    if varying_ic:
        assert rows.loc["validation_icir", "status"] == "DEFINED"
        assert rows.loc["validation_icir", "value"] == pytest.approx(
            daily_ic.mean() / expected_std
        )
        assert rows.loc["validation_icir", "reason"] == ""
    else:
        assert rows.loc["validation_icir", "status"] == "UNDEFINED"
        assert pd.isna(rows.loc["validation_icir", "value"])
        assert rows.loc["validation_icir", "reason"] == "zero_variance"


@pytest.mark.parametrize(
    "tables,member",
    [
        (
            "factor"
            if member.removesuffix(".parquet") + ".parquet"
            in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research",
            member,
        )
        for member in [
            "model_trials",
            "prediction_diagnostics",
            "robustness",
            "factor_diagnostics",
            "fama_macbeth",
            "fama_macbeth_periods",
        ]
    ],
    indirect=["tables"],
)
def test_statistics_all_numeric_tables_reject_duplicate_primary_keys(
    tmp_path, tables, metadata, member
):
    tables[member] = pd.concat([tables[member], tables[member]], ignore_index=True)
    group = (
        "factor"
        if member in {"factor_diagnostics", "fama_macbeth", "fama_macbeth_periods"}
        else "research"
    )
    with pytest.raises(ValueError, match="statistics_table_duplicate_keys"):
        statistics.write_statistics_artifact(
            tmp_path / "unpublished",
            metadata=metadata,
            tables=tables,
            table_group=group,
        )
    assert not (tmp_path / "unpublished").exists()


@pytest.mark.parametrize("start,pnl", ((1000.0, 0.01), (1000.0, 1e-10)))
def test_equity_return_uses_money_rounding_bound_without_erasing_small_pnl(start, pnl):
    from fractions import Fraction

    from quant_research_platform.contracts.time import TimeWindow
    from tests.helpers.statistics_artifacts import statistics_metadata

    member = "portfolio_daily.parquet"
    net_return = float(Fraction(pnl) / Fraction(start))
    end = float(Fraction(start) + Fraction(pnl))
    row = dict(
        candidate_id="raw",
        sample_role="validation",
        entry_at=pd.Timestamp("2024-01-02"),
        end_at=pd.Timestamp("2024-01-03"),
        nav_before=start,
        nav_post=start,
        nav_end=end,
        cash_post=0.0,
        cash_weight_post=0.0,
        gross_return=net_return,
        net_return=net_return,
        cost_drag=0.0,
        target_turnover=0.0,
        requested_turnover=0.0,
        realized_turnover=0.0,
        return_status="DEFINED",
        return_reason="",
    )
    frame = pd.DataFrame([row], columns=statistics.STATISTICS_TABLE_COLUMNS[member])
    metadata = statistics_metadata(
        TimeWindow("2024-01-01", "2024-01-04"), candidate_ids=["raw"]
    )
    result = statistics._validated_frame(frame, member=member, metadata=metadata)
    assert result.net_return.iloc[0] == net_return
    assert result.net_return.iloc[0] > 0


@pytest.mark.parametrize("method", ("BH", "BY"))
def test_paired_group_fdr_methods_reach_native_artifact(
    tmp_path, tables, metadata, method
):
    from quant_research_platform.research.models.statistics import _paired_group_rankic

    metadata = _test_statistics_metadata(metadata, tables)
    left = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=8),
            "family_id": "family",
            "loss": [0.1, 0.2, -0.1, 0.3, 0.1, -0.2, 0.05, 0.1],
            "status": "DEFINED",
            "observation_count": 3,
            "sample_role": "test",
        }
    )
    right = left.copy()
    right["loss"] += [0.1, 0.05, 0.07, -0.03, 0.1, 0.02, -0.01, 0.09]

    policy = {"fdr_method": method, "alpha": 0.05, "seed": 42, "confidence": 0.95}
    config = {
        "sample_role": "test",
        "statistics": {
            "hac_lag": 0,
            "bootstrap_block": 2,
            "bootstrap_repetitions": 32,
            "group_comparison": policy,
        },
    }
    produced = _paired_group_rankic(
        right.assign(**{"global": -left.loss, "quarterly": -right.loss}),
        selection_group="quarterly",
        family_ids=("family",),
        config=config,
    )
    assert produced.status.eq("DEFINED").all()
    assert produced.fdr_adjusted_p_value.notna().all()
    tables["model_group_comparison"] = produced
    metadata = {
        **metadata,
        "statistics": {**metadata["statistics"], "group_comparison": policy},
    }
    root = statistics.write_statistics_artifact(
        tmp_path / "paired", metadata=metadata, tables=tables
    )
    artifact = statistics.read_statistics_artifact(root)
    assert artifact.metadata["statistics"]["group_comparison"] == policy
    with artifact.table("model_group_comparison.parquet") as restored:
        assert restored.fdr_method.tolist() == [method]
        pd.testing.assert_frame_equal(
            restored,
            produced.assign(
                reason=produced.reason.fillna(""),
                bootstrap_reason=produced.bootstrap_reason.fillna(""),
            ),
            check_dtype=False,
            check_exact=True,
        )


@pytest.mark.parametrize(
    "tables,member,column,value,error",
    [
        (
            "factor",
            "factor_annual_diagnostics",
            "year",
            2024,
            "statistics_table_window_mismatch",
        ),
    ],
    indirect=["tables"],
)
def test_factor_comparison_artifacts_reject_inconsistent_summary_facts(
    tmp_path, tables, metadata, member, column, value, error
):
    tables[member].loc[0, column] = value
    _assert_rejected_at_write(
        tmp_path, tables, metadata, member + ".parquet", error=error
    )


def test_inventory_attribution_clock_keeps_actual_interval_endpoints(tmp_path):
    entry = pd.Timestamp("2024-01-02T01:30:00")
    frame = pd.DataFrame(
        [
            dict(
                attribution_group="a" * 64,
                decision_at=entry,
                entry_at=entry,
                end_at=entry + pd.Timedelta(seconds=seconds),
                factor_id="industry_one",
                value=0.01,
                observation_count=128,
            )
            for seconds in (3, 6)
        ],
        columns=statistics.STATISTICS_TABLE_COLUMNS["holding_factor_returns.parquet"],
    )
    metadata = {
        "window": {"start": "2024-01-02", "end": "2024-01-02"},
        "sample_role": "exploratory",
    }
    with pytest.raises(ValueError, match="statistics_holding_interval_invalid"):
        statistics.write_statistics_artifact(
            tmp_path / "daily-clock",
            metadata=metadata,
            tables={"holding_factor_returns": frame},
            table_group="portfolio",
        )
    metadata["attribution_clock"] = "inventory_intervals"
    root = statistics.write_statistics_artifact(
        tmp_path / "inventory-clock",
        metadata=metadata,
        tables={"holding_factor_returns": frame},
        table_group="portfolio",
    )
    with statistics.read_statistics_artifact(root).table(
        "holding_factor_returns.parquet"
    ) as actual:
        assert len(actual) == 2
        assert actual.end_at.tolist() == frame.end_at.tolist()
    with pytest.raises(ValueError, match="statistics_holding_interval_invalid"):
        statistics.write_statistics_artifact(
            tmp_path / "future-design",
            metadata=metadata,
            tables={
                "holding_factor_returns": frame.assign(
                    decision_at=entry + pd.Timedelta(nanoseconds=1)
                )
            },
            table_group="portfolio",
        )
