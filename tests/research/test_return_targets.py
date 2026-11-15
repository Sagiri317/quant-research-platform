"""One risk target owner binds every horizon to its actual source and clock."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.models import (
    _target_definition,
)
from quant_research_platform.contracts.risk import RISK_STYLE_COLUMNS
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.statistics import model_statistics
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.inspection import inspect_dataset


@pytest.mark.parametrize("explicit_label", [None, "forward_return_20d"])
def test_default_five_day_target_and_explicit_twenty_day_native_chain(
    tmp_path, explicit_label
):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-03-01"])
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": list("abcd") * 3,
            "timestamp": dates.repeat(4),
            "eligible": [True, True, True, False] * 3,
            "x": [-1.0, 0.0, 1.0, 10.0] * 3,
            "forward_return_5d": [-0.1, 0.0, 0.1, 10.0] * 3,
            "forward_return_20d": [0.2, 0.0, -0.2, -10.0] * 3,
            "forward_return_5d_available_at": pd.to_datetime(
                ["2024-01-10T07:00", "2024-01-11T07:00", "2024-03-11T07:00"]
            ).repeat(4),
            "forward_return_20d_available_at": pd.to_datetime(
                ["2024-02-01T07:00", "2024-02-02T07:00", "2024-04-01T07:00"]
            ).repeat(4),
        }
    )
    for horizon in (5, 20):
        frame.loc[4, f"forward_return_{horizon}d_available_at"] = pd.NaT
    roles = dict.fromkeys(frame, "context")
    roles.update(
        market_id="key",
        asset_id="key",
        timestamp="timestamp",
        x="factor",
        eligible="universe",
    )
    for horizon in (5, 20):
        roles[f"forward_return_{horizon}d"] = "label"
    source = tmp_path / "returns"
    write_dataset(
        source,
        frame,
        spec=ProcessedDataSpec(
            data_id="returns",
            market_id="a_share",
            kind="panel",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles=roles,
            column_units={f"forward_return_{h}d": "decimal_return" for h in (5, 20)},
            label_availability_columns={
                f"forward_return_{h}d": f"forward_return_{h}d_available_at"
                for h in (5, 20)
            },
        ),
    )
    request = dict(
        market_id="a_share",
        dataset_id="targets",
        inputs={"returns": str(source)},
        factor_columns=["x"],
        feature_columns=[],
        feature_recipe=None,
        label_columns=["model_target", "forward_return_5d", "forward_return_20d"],
        filter_columns=["eligible"],
        time_window={"start": "2024-01-02", "end": "2024-03-01"},
        output_root=str(tmp_path / "targets"),
    )
    if explicit_label is not None:
        request["target_label_column"] = explicit_label
    dataset = assemble(request)
    saved = read_dataset(dataset)
    label = explicit_label or "forward_return_5d"
    sign = -1 if explicit_label else 1
    expected = sign * np.array([-np.sqrt(1.5), 0.0, np.sqrt(1.5)])
    for _, rows in saved.groupby("timestamp", sort=False):
        np.testing.assert_allclose(rows.model_target.iloc[:3], expected, atol=1e-15)
        assert pd.isna(rows.model_target.iloc[3])
    assert (
        saved.loc[saved.timestamp.eq(dates[0]), "model_target_available_at"]
        .iloc[:3]
        .eq(pd.Timestamp("2024-02-01T07:00" if explicit_label else "2024-01-10T07:00"))
        .all()
    )
    assert (
        saved.loc[saved.timestamp.eq(dates[1]), "model_target_available_at"]
        .isna()
        .all()
    )
    for horizon in (5, 20):
        pd.testing.assert_series_equal(
            saved[f"forward_return_{horizon}d"], frame[f"forward_return_{horizon}d"]
        )
    spec = inspect_dataset(dataset).spec
    assert (
        spec["target_label_column"],
        spec["target_transform"],
        spec["target_clip"],
    ) == (label, "date_zscore", "none")
    assert spec["column_units"]["model_target"] == "dimensionless"
    model = fit(
        dict(
            market_id="a_share",
            inputs=str(dataset),
            output=str(tmp_path / "model"),
            model=dict(
                model_id="linear",
                model_algorithm="linear_regression",
                input_factor_columns=["x"],
                target_column="model_target",
            ),
            fit_windows=[dict(start="2024-01-02", end="2024-01-03")],
            fit_cutoff="2024-02-02T08:00",
        )
    )
    loaded, facts = load_model(model)
    assert loaded.metadata["rows_after_input_label_dropna"] == 3
    assert facts["target_definition"]["source_label"] == label
    assert facts["label_unit"] == "dimensionless"
    window = dict(start="2024-03-01", end="2024-03-01")
    prediction = predict(
        dict(
            model=str(model),
            inputs=str(dataset),
            output=str(tmp_path / "predictions"),
            prediction_window=window,
        )
    )
    output, metadata = read_predictions(prediction)
    assert output.asset_id.tolist() == list("abc")
    np.testing.assert_allclose(output.prediction_score, expected, atol=1e-14)
    assert metadata["model_facts"]["target_definition"] == facts["target_definition"]
    statistics = read_statistics_artifact(
        model_statistics(
            dict(
                dataset=str(dataset),
                output=str(tmp_path / "statistics"),
                window=window,
                label_cutoff="2024-04-02",
                family_ids=["linear"],
                evaluation={"primary_label": "model_target"},
                analyses=["prediction_hac", "model_losses"],
                statistics={"hac_lag": 0},
                periods=[
                    dict(
                        period_id="march", window=window, predictions=[str(prediction)]
                    )
                ],
            )
        )
    )
    assert statistics.metadata["prediction_metrics"]["raw_return_label"] == label
    with statistics.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.target_count.tolist() == [3]
        assert metrics.mse.iloc[0] == pytest.approx(0.0, abs=1e-28)
        assert metrics.head_mean_raw_return.iloc[0] == pytest.approx(
            0.2 if explicit_label else 0.1
        )


@pytest.mark.parametrize(
    "label",
    [
        "forward_return_1d",
        "forward_return_5d",
        "next_session_intraday_return",
        "future_mid_return_30s",
    ],
)
def test_explicit_risk_target_survives_native_assembly_and_keeps_label_semantics(
    tmp_path, label
):
    signs = 2.0 * ((np.arange(128)[:, None] >> np.arange(6)) & 1) - 1
    residual = 0.003 * signs[:, 5]
    clock = label + "_available_at"
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"a{i:03d}" for i in range(128)],
            "timestamp": pd.Timestamp("2024-01-02T02:00:00"),
            "eligible": True,
            "industry_standard_id": "test",
            "industry_level1_code": "A",
            "total_market_cap": 1e8,
            **dict(zip(RISK_STYLE_COLUMNS, signs[:, :5].T, strict=True)),
            label: 0.02 + signs[:, :5] @ (np.arange(5) / 1000) + residual,
            clock: pd.Timestamp("2024-01-03T02:00:00"),
            "forward_return_20d": 9.0,
            "forward_return_20d_available_at": pd.Timestamp("2029-01-01"),
        }
    )
    frame.loc[0, clock] = pd.Timestamp("2024-01-04T02:00:00")
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(
        market_id="key", asset_id="key", timestamp="timestamp", eligible="universe"
    )
    roles[label] = roles["forward_return_20d"] = "label"
    source = tmp_path / "source"
    write_dataset(
        source,
        frame,
        spec=ProcessedDataSpec(
            data_id="returns",
            market_id="a_share",
            kind="panel",
            frequency="tick" if label.startswith("future_mid") else "1d",
            grain="asset_timestamp",
            partition_columns=("year",),
            primary_key=("market_id", "asset_id", "timestamp"),
            column_roles=roles,
            column_units={
                label: "decimal_return",
                "forward_return_20d": "decimal_return",
            },
            label_availability_columns={
                label: clock,
                "forward_return_20d": "forward_return_20d_available_at",
            },
        ),
    )
    output = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "return_residuals",
            "inputs": {"returns": str(source)},
            "base_input_data_id": "returns",
            "factor_columns": [],
            "feature_columns": [],
            "feature_recipe": None,
            "label_columns": ["model_target"],
            "filter_columns": ["eligible"],
            "target_label_column": label,
            "target_transform": "risk_residual",
            "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
            "output_root": str(tmp_path / "residual"),
        }
    )
    result = read_dataset(output)
    np.testing.assert_allclose(result.model_target, residual, atol=2e-15)
    assert result.model_target_available_at.eq(
        pd.Timestamp("2024-01-04T02:00:00")
    ).all()
    saved = inspect_dataset(output).spec
    assert saved["target_label_column"] == label
    assert saved["column_units"]["model_target"] == "decimal_return"
    definition = _target_definition(saved, "model_target")
    assert definition["source_label"] == label
