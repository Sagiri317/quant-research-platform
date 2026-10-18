import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.statistics_production import (
    build_a_share_factor_statistics_tables,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from tests.helpers.research_specs import make_demo_research_dataset_spec


def _factor_input(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=12)
    frame = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "timestamp": date,
                "asset_id": str(asset),
                "eligible": True,
                "factor": float(asset),
                "total_market_cap": float(asset + 1),
                "forward_return_20d": asset * 0.2 + np.sin(day + asset) * 0.01,
                "forward_return_20d_available_at": date + pd.Timedelta(days=1),
            }
            for day, date in enumerate(dates)
            for asset in range(5)
        ]
    )
    frame.loc[frame.timestamp.eq(dates[-1]), "forward_return_20d_available_at"] = (
        pd.Timestamp("2021-01-01")
    )
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    source = tmp_path / "input"
    write_dataset(source, frame.loc[:, list(spec.column_roles)], spec=spec)
    config = {
        "sample_role": "validation",
        "evaluation": {"primary_label": "forward_return_20d", "factors": ["factor"]},
        "statistics": {
            "factor_fdr_families": {"family": ["factor"]},
            "fama_macbeth_core_factors": [],
            "hac_lag": 1,
            "fdr_primary": "BY_5pct",
            "fdr_robustness": "BH_5pct",
        },
    }
    window = TimeWindow(dates[0].date(), dates[-1].date())
    metadata = {
        "sample_role": "validation",
        "window": {"start": str(window.start), "end": str(window.end)},
    }
    return spec, config, window, metadata, dates


def _factor_tables(tmp_path):
    spec, config, window, metadata, dates = _factor_input(tmp_path)
    tables = build_a_share_factor_statistics_tables(
        dataset_path=tmp_path / "input",
        dataset_spec=spec,
        config=config,
        time_window=window,
    )
    return tables, metadata, dates


def test_explicit_factor_statistics_native_roundtrip_and_maturity(tmp_path):
    tables, metadata, dates = _factor_tables(tmp_path)
    output = tmp_path / "arbitrary-results"
    write_statistics_artifact(
        output, metadata=metadata, tables=tables, table_group="factor"
    )
    artifact = read_statistics_artifact(output)
    with artifact.table("fama_macbeth_periods.parquet") as frame:
        assert frame.timestamp.max() < dates[-1]
        assert frame.observation_count.eq(5).all()
    with artifact.table("factor_diagnostics.parquet") as frame:
        assert frame.input_date_count.eq(12).all()
        assert frame.planned_hypothesis_count.eq(1).all()
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(OutputAlreadyExistsError):
        write_statistics_artifact(
            output, metadata=metadata, tables=tables, table_group="factor"
        )
    assert (output / "manifest.json").read_bytes() == before


def test_statistics_unconsumed_member_not_scanned(tmp_path):
    tables, metadata, _ = _factor_tables(tmp_path)
    root = tmp_path / "statistics"
    write_statistics_artifact(
        root, metadata=metadata, tables=tables, table_group="factor"
    )
    (root / "fama_macbeth.parquet").write_bytes(b"not parquet")
    artifact = read_statistics_artifact(root)
    with artifact.table("factor_diagnostics.parquet") as frame:
        assert not frame.empty
    with pytest.raises((pa.ArrowInvalid, ValueError)):
        with artifact.table("fama_macbeth.parquet"):
            pass


@pytest.mark.parametrize("role", ["holdout", "development", "", None])
def test_statistics_preserves_descriptive_roles_and_rejects_empty_roles(tmp_path, role):
    if role:
        tables, metadata, _ = _factor_tables(tmp_path)
        metadata["sample_role"] = role
        for name, frame in tables.items():
            if "sample_role" in frame:
                tables[name] = frame.assign(sample_role=role)
        root = write_statistics_artifact(
            tmp_path / "descriptive",
            metadata=metadata,
            tables=tables,
            table_group="factor",
        )
        artifact = read_statistics_artifact(root)
        assert artifact.metadata["sample_role"] == role
        with artifact.table("factor_diagnostics.parquet") as frame:
            assert not frame.empty
        return
    with pytest.raises(ValueError, match="sample_role"):
        write_statistics_artifact(
            tmp_path / "bad",
            metadata={"sample_role": role},
            tables={},
            table_group="factor",
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "bare,intraday,omit_fdr_defaults",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_public_factor_statistics_projects_custom_native_or_bare_input(
    tmp_path, bare, intraday, omit_fdr_defaults
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.statistics_production import factor_statistics
    from quant_research_platform.storage.datasets import read_dataset

    _, _, _, metadata, _ = _factor_input(tmp_path)
    frame = read_dataset(tmp_path / "input").rename(
        columns={
            "forward_return_20d": "custom_return",
            "forward_return_20d_available_at": "observable_at",
        }
    )
    if intraday:
        frame["timestamp"] += pd.Timedelta(hours=12)
        frame["observable_at"] = frame.timestamp + pd.Timedelta(hours=1)
    frame["unused_payload"] = "not a numeric factor"
    declared = {
        "data_id": "not_in_any_catalog",
        "frequency": "1d",
        "grain": "asset_timestamp",
        "primary_key": ["market_id", "asset_id", "timestamp"],
        "partition_columns": ["year"],
        "column_units": {"custom_return": "decimal_return"},
        "column_roles": {
            "market_id": "key",
            "asset_id": "key",
            "timestamp": "timestamp",
            "factor": "factor",
            "custom_return": "label",
            "observable_at": "context",
            "eligible": "universe",
            "total_market_cap": "context",
        },
    }
    source = tmp_path / ("custom.parquet" if bare else "custom-native")
    if bare:
        frame.to_parquet(source, index=False)
    else:
        values = {
            **declared,
            "primary_key": tuple(declared["primary_key"]),
            "partition_columns": ("year",),
        }
        write_dataset(
            source, frame.drop(columns="unused_payload"), spec=DataSpec(**values)
        )
    config = {
        "dataset": str(source),
        "output": str(tmp_path / "published"),
        **metadata,
        "evaluation": {
            "factors": ["factor"],
            "primary_label": "custom_return",
            "label_availability_columns": {"custom_return": "observable_at"},
        },
        "statistics": {
            "factor_fdr_families": {"family": ["factor"]},
            "fama_macbeth_core_factors": [],
            "hac_lag": 1,
            "fdr_primary": "BY_5pct",
            "fdr_robustness": "BH_5pct",
        },
        **({"input_spec": declared} if bare else {}),
    }
    if omit_fdr_defaults:
        config["statistics"].pop("fdr_primary")
        config["statistics"].pop("fdr_robustness")
    result = read_statistics_artifact(factor_statistics(config))
    assert result.metadata["statistics"]["fdr_primary"] == "BY_5pct"
    assert result.metadata["statistics"]["fdr_robustness"] == "BH_5pct"
    with result.table("factor_diagnostics.parquet") as actual:
        assert actual.factor_id.eq("factor").all()
        assert actual.input_date_count.eq(12).all()
        assert actual.observation_count.eq(12 if intraday else 11).all()
    if bare:
        broken = {**config, "output": str(tmp_path / "unpublished")}
        broken.pop("input_spec")
        with pytest.raises(ValueError, match="parquet_semantic_spec_required"):
            factor_statistics(broken)
        assert not (tmp_path / "unpublished").exists()


def test_nonstreaming_statistics_reject_callback_before_publishing(tmp_path):
    tables, metadata, _ = _factor_tables(tmp_path)

    def repeat(on_table):
        for name, frame in tables.items():
            on_table(name, frame)
            on_table(name, frame)

    with pytest.raises(ValueError, match="statistics_tables_invalid"):
        write_statistics_artifact(
            tmp_path / "unpublished",
            metadata=metadata,
            tables=repeat,
            table_group="factor",
        )
    assert not (tmp_path / "unpublished").exists()


@pytest.mark.parametrize("operation", ["factor", "model", "portfolio"])
def test_public_statistics_rejects_unused_nested_options_before_reading(
    tmp_path, operation
):
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from quant_research_platform.research.statistics_production import (
        factor_statistics,
        model_statistics,
    )

    common = {
        "output": str(tmp_path / "unpublished"),
        "window": {"start": "2023-01-01", "end": "2023-01-02"},
        "sample_role": "validation",
    }
    if operation == "factor":
        config = {
            **common,
            "dataset": "missing",
            "evaluation": {"factors": ["x"], "primary_label": "label"},
            "statistics": {
                "factor_fdr_families": {"f": ["x"]},
                "fama_macbeth_core_factors": [],
                "hac_lag": 1,
                "fdr_primary": "BY_5pct",
                "fdr_robustness": "BH_5pct",
                "hac_lga": 2,
            },
        }
        function = factor_statistics
    elif operation == "model":
        config = {
            **common,
            "dataset": "missing",
            "evaluation": {"primary_label": "label"},
            "statistics": {
                "hac_lag": 1,
                "bootstrap_block": 2,
                "bootstrap_repetitions": 20,
                "unused": True,
            },
            "root_seed": 1,
            "selection_group": "global",
            "comparison_model_id": "model",
            "periods": [],
            "prediction_id": "prediction",
        }
        function = model_statistics
    else:
        config = {
            **common,
            "ledgers": ["missing"],
            "inputs": {},
            "benchmark_total_return_indices": {},
            "risk": {},
            "adv_sensitivities": [0.1],
            "analyses": ["cscv"],
            "statistics": {"cscv_blocks": 10, "unused": True},
        }
        function = portfolio_statistics
    with pytest.raises(ValueError, match="statistics_policy_invalid"):
        function(config)
    assert not (tmp_path / "unpublished").exists()


def test_failed_family_keeps_planned_inference_members_without_fake_prediction(
    tmp_path,
):
    from quant_research_platform.research.statistics_production import (
        _selection_trial_rows,
        build_a_share_statistics_tables,
    )
    from tests.helpers.statistics_artifacts import (
        model_statistics_inputs,
        statistics_config,
    )

    _, _, _, metadata, dates = _factor_input(tmp_path)
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    config = statistics_config(factors=["factor"])
    config["statistics"]["bootstrap_repetitions"] = 20
    window = TimeWindow(dates[0].date(), dates[-1].date())
    inputs = model_statistics_inputs(
        spec, config, window, families=("working", "failed")
    )
    selection = inputs["selections"]["explicit_period"]
    failed = selection["families"][1]
    failed["selected_config_id"] = None
    metrics = failed["trial_metrics"][0]
    for key in tuple(metrics):
        if key.startswith("validation_"):
            metrics[key] = (
                "fit_failed"
                if key.endswith("_reason")
                else 0
                if key.endswith("_count") or key == "validation_rows"
                else None
            )
    inputs["model_entries"]["explicit_period", "failed"] = {
        "model_metadata": None,
        "model_id": "",
    }
    inputs["trial_rows"] = _selection_trial_rows({"final": selection}, "global")

    class Lookup:
        def values_for_frame(self, keys, *, mark_matched):
            assert keys.family_id.eq("working").all()
            return keys.asset_id.astype(float).to_numpy(), np.ones(
                len(keys), dtype=bool
            )

        def coverage_counts(self):
            return (0, 0)

    tables = build_a_share_statistics_tables(
        dataset_path=tmp_path / "input",
        dataset_spec=spec,
        config=config,
        time_window=window,
        prediction_id="predictions",
        prediction_lookup=Lookup(),
        global_prediction_lookup=None,
        **inputs,
    )
    lost = tables["model_losses"].loc[lambda frame: frame.family_id.eq("failed")]
    assert len(lost) == len(dates)
    assert lost.status.eq("UNDEFINED").all()
    assert lost.reason.eq("family_training_failed").all()
    assert lost.asset_coverage_id.eq(canonical_json_hash([])).all()
    assert lost.config_id.eq("").all()
    assert set(tables["model_trials"].family_id) == {"working", "failed"}
    assert tables["model_comparison"].prediction_id.tolist() == ["failed"]
    assert tables["model_comparison"].status.eq("UNDEFINED").all()
    assert tables["model_reality_checks"].status.eq("UNDEFINED").all()
    root = write_statistics_artifact(
        tmp_path / "failed-family-statistics", metadata=metadata, tables=tables
    )
    with read_statistics_artifact(root).table("model_losses.parquet") as actual:
        assert set(actual.family_id) == {"working", "failed"}


@pytest.mark.parametrize(
    "analyses", [["rank_ic"], ["fama_macbeth"], ["rank_ic", "fama_macbeth"]]
)
@pytest.mark.parametrize("empty_labels", [False, True])
def test_factor_statistics_complete_batches_preserve_dates_and_masks(
    tmp_path, monkeypatch, analyses, empty_labels
):
    from quant_research_platform.research import statistics_production as production
    from quant_research_platform.storage.datasets import read_dataset

    _, _, _, _, dates = _factor_input(tmp_path)
    frame = read_dataset(tmp_path / "input").rename(columns={"factor": "feature_x"})
    times = pd.date_range("2020-01-02", periods=len(dates), freq="100ms")
    frame["timestamp"] = frame.timestamp.map(dict(zip(dates, times, strict=True)))
    frame["forward_return_20d_available_at"] = frame.timestamp
    frame["feature_observed_x"] = 1.0
    frame.loc[frame.timestamp.eq(times[2]), "feature_observed_x"] = 0.0
    frame.loc[frame.timestamp.eq(times[4]), "forward_return_20d"] = np.nan
    frame.loc[frame.timestamp.eq(times[6]), "eligible"] = False
    frame.loc[frame.timestamp.eq(times[8]), "forward_return_20d_available_at"] = (
        pd.Timestamp("2021-01-01")
    )
    frame.loc[
        frame.timestamp.eq(times[1]) & frame.asset_id.eq("0"), "feature_observed_x"
    ] = 0.0
    if empty_labels:
        frame["forward_return_20d"] = np.nan
    spec = make_demo_research_dataset_spec(
        data_id="batched_statistics",
        factor_columns=(),
        feature_columns=("feature_x", "feature_observed_x"),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    source = tmp_path / "batched-input"
    write_dataset(source, frame.loc[:, list(spec.column_roles)], spec=spec)
    config = {
        "dataset": {
            "path": str(source),
            "row_filters": [["asset_id", "in", ["0", "1", "2", "3"]]],
        },
        "window": {"start": "2020-01-02", "end": "2020-01-02"},
        "analyses": analyses,
        "evaluation": {"primary_label": "forward_return_20d", "factors": ["feature_x"]},
        "statistics": {
            "hac_lag": 1,
            "factor_fdr_families": {"family": ["feature_x"]},
            "fama_macbeth_core_factors": [],
        },
    }
    original = production.iter_dataset_chunks
    artifacts = []
    closed = []
    for batch_rows in (1000, 7):

        def chunks(session, batch_rows=batch_rows):
            try:
                yield from original(session, batch_rows=batch_rows)
            finally:
                closed.append(batch_rows)

        monkeypatch.setattr(production, "iter_dataset_chunks", chunks)
        artifacts.append(
            read_statistics_artifact(
                production.factor_statistics(
                    {**config, "output": str(tmp_path / f"result-{batch_rows}")}
                )
            )
        )
    assert closed == [1000, 7]
    assert artifacts[0].metadata == artifacts[1].metadata
    names = (["factor_diagnostics.parquet"] if "rank_ic" in analyses else []) + (
        ["fama_macbeth.parquet", "fama_macbeth_periods.parquet"]
        if "fama_macbeth" in analyses
        else []
    )
    for name in names:
        with artifacts[0].table(name) as left, artifacts[1].table(name) as right:
            continuous = left.select_dtypes(include="floating").columns.difference(
                ["fdr_alpha"]
            )
            pd.testing.assert_frame_equal(
                left.drop(columns=continuous),
                right.drop(columns=continuous),
                check_exact=True,
            )
            pd.testing.assert_frame_equal(left.isna(), right.isna(), check_exact=True)
            pd.testing.assert_frame_equal(
                left[continuous],
                right[continuous],
                check_exact=False,
                rtol=1e-10,
                atol=0,
            )

            if name == "factor_diagnostics.parquet":
                assert left.input_date_count.eq(12).all()
                assert left.observation_count.eq(0 if empty_labels else 8).all()
            if name == "fama_macbeth_periods.parquet":
                assert left.timestamp.nunique() == (0 if empty_labels else 9)
                assert left.observation_count.le(4).all()
                if not empty_labels:
                    assert (
                        left.loc[left.timestamp.eq(times[2]), "status"]
                        .eq("UNDEFINED")
                        .all()
                    )


def test_factor_statistics_empty_selected_window_is_undefined(tmp_path):
    from quant_research_platform.research.statistics_production import factor_statistics

    _factor_input(tmp_path)
    config = {
        "dataset": str(tmp_path / "input"),
        "output": str(tmp_path / "empty-window"),
        "window": {"start": "2020-02-01", "end": "2020-02-02"},
        "analyses": ["rank_ic", "fama_macbeth"],
        "evaluation": {"primary_label": "forward_return_20d", "factors": ["factor"]},
        "statistics": {
            "hac_lag": 1,
            "factor_fdr_families": {"family": ["factor"]},
            "fama_macbeth_core_factors": [],
        },
    }
    artifact = read_statistics_artifact(factor_statistics(config))
    metrics = [
        "mean",
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    for name, count in (
        ("factor_diagnostics", "observation_count"),
        ("fama_macbeth", "period_count"),
    ):
        with artifact.table(name + ".parquet") as frame:
            assert not frame.empty
            assert frame.input_date_count.eq(0).all()
            assert frame[count].eq(0).all()
            assert frame.status.eq("UNDEFINED").all()
            assert frame.reason.eq("hac_lag_exceeds_observations").all()
            assert frame[metrics].isna().all().all()
            if name == "factor_diagnostics":
                assert frame.defined_hypothesis_count.eq(0).all()
                assert frame.fdr_adjusted_p_value.isna().all()
    with artifact.table("fama_macbeth_periods.parquet") as frame:
        assert frame.empty
    missing_output = tmp_path / "missing-source-output"
    with pytest.raises(FileNotFoundError):
        factor_statistics(
            {
                **config,
                "dataset": str(tmp_path / "absent"),
                "output": str(missing_output),
            }
        )
    assert not missing_output.exists()


@pytest.mark.parametrize("historical_x", ["excluded", "missing", "changed"])
def test_selected_statistics_consumes_saved_predictions_without_historical_x(
    tmp_path, historical_x
):
    from quant_research_platform.research.statistics_production import (
        build_a_share_statistics_tables,
    )
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from tests.helpers.statistics_artifacts import (
        model_statistics_inputs,
        statistics_config,
    )

    _, _, _, _, dates = _factor_input(tmp_path)
    session = open_dataset_session(tmp_path / "input")
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    # The persisted lookup is unchanged across historical feature mutations.
    if historical_x != "excluded":
        from quant_research_platform.storage.datasets import read_dataset

        frame = read_dataset(tmp_path / "input")
        frame["factor"] = np.nan if historical_x == "missing" else -1e9
        write_dataset(tmp_path / "changed", frame, spec=spec)
        session = open_dataset_session(tmp_path / "changed")
    session = session.projected_view(
        tuple(c for c in spec.column_roles if c != "factor")
    )
    config = statistics_config(factors=["factor"])
    config["analyses"] = ["model_losses"]
    config["statistics"].pop("group_comparison")
    window = TimeWindow(dates[0].date(), dates[-1].date())
    inputs = model_statistics_inputs(spec, config, window, families=("working",))

    class Lookup:
        def values_for_frame(self, keys, *, mark_matched):
            return keys.asset_id.astype(float).to_numpy(), np.ones(
                len(keys), dtype=bool
            )

        def coverage_counts(self):
            return (0, 0)

    tables = build_a_share_statistics_tables(
        dataset_path=tmp_path / "input",
        dataset_session=session,
        dataset_spec=spec,
        config=config,
        time_window=window,
        prediction_id="saved",
        prediction_lookup=Lookup(),
        global_prediction_lookup=None,
        **inputs,
    )
    assert len(tables["model_losses"]) == len(dates)
    assert tables["model_losses"].status.iloc[:-1].eq("DEFINED").all()
    assert tables["model_losses"].status.iloc[-1] == "UNDEFINED"


@pytest.mark.parametrize("invalid_cap", [0.0, -1.0])
def test_public_factor_statistics_preserves_equal_weight_with_bad_cap(
    tmp_path, invalid_cap
):
    from quant_research_platform.research.statistics_production import factor_statistics
    from quant_research_platform.storage.datasets import read_dataset

    tables, metadata, _ = _factor_tables(tmp_path)
    frame = read_dataset(tmp_path / "input")
    frame.loc[frame.asset_id.eq("0"), "total_market_cap"] = invalid_cap
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    write_dataset(tmp_path / "bad-cap", frame, spec=spec)
    path = factor_statistics(
        {
            "dataset": str(tmp_path / "bad-cap"),
            "output": str(tmp_path / "statistics"),
            **metadata,
            "analyses": ["fama_macbeth"],
            "evaluation": {
                "primary_label": "forward_return_20d",
                "factors": ["factor"],
            },
            "statistics": {"fama_macbeth_core_factors": [], "hac_lag": 1},
        }
    )
    with read_statistics_artifact(path).table("fama_macbeth_periods.parquet") as actual:
        equal = actual.regression_id.str.startswith("equal:")
        expected = tables["fama_macbeth_periods"].loc[equal].copy()
        expected["reason"] = expected.reason.fillna("").astype(str)
        expected["white_reason"] = expected.white_reason.fillna("").astype(str)
        pd.testing.assert_frame_equal(actual.loc[equal], expected)
        assert actual.loc[~equal, "status"].eq("DEFINED").all()
        assert actual.loc[~equal, "observation_count"].eq(4).all()
