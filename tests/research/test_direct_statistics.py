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


def _factor_tables(tmp_path):
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
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
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
    tables = build_a_share_factor_statistics_tables(
        dataset_path=source, dataset_spec=spec, config=config, time_window=window
    )
    metadata = {
        "sample_role": "validation",
        "window": {"start": str(window.start), "end": str(window.end)},
    }
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


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("intraday", [False, True])
@pytest.mark.parametrize("omit_fdr_defaults", [False, True])
def test_public_factor_statistics_projects_custom_native_or_bare_input(
    tmp_path, bare, intraday, omit_fdr_defaults
):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.statistics_production import factor_statistics
    from quant_research_platform.storage.datasets import read_dataset

    _, metadata, _ = _factor_tables(tmp_path)
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

    _, metadata, dates = _factor_tables(tmp_path)
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
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
        global_statistics=None,
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
