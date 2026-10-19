"""Ordinary FMB coefficients and undefined inference survive native publication."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts import statistics
from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.research.statistics_production import (
    _fama_macbeth_period_records,
    _fama_macbeth_summary,
)


@pytest.mark.parametrize("unit", (0.001, 1.0, 100.0))
def test_fmb_identity_publishes_periods_and_undefined_summary(tmp_path, unit):
    values = np.arange(8, dtype=float) * unit
    records = []
    for timestamp in pd.bdate_range("2023-01-02", periods=21):
        frame = pd.DataFrame(
            dict(
                timestamp=timestamp,
                asset_id=[str(i) for i in range(8)],
                total_market_cap=1.0,
                factor=values,
                label=values,
            )
        )
        frame["eligible"] = True
        frame["label_available_at"] = timestamp + pd.offsets.BDay(6)
        mature = legal_training_rows(
            frame,
            fitting_cutoff="2023-03-31",
            availability_column="label_available_at",
            feature_columns=("factor",),
            target_column="label",
        )
        pd.testing.assert_frame_equal(mature, frame)
        records.extend(
            _fama_macbeth_period_records(
                frame, return_column="label", factors=("factor",), core_factors=()
            )
        )
    periods = pd.DataFrame(records)
    periods["timestamp"] = periods.timestamp.astype("datetime64[ns]")
    tables = {"fama_macbeth_periods": periods}
    tables["fama_macbeth"] = _fama_macbeth_summary(
        periods, factors=("factor",), core_factors=(), max_lags=19
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics",
        metadata={
            "sample_role": "validation",
            "window": {"start": "2023-01-01", "end": "2023-03-31"},
        },
        tables=tables,
        table_group="factor",
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        assert len(actual) == 84
        assert actual.status.eq("DEFINED").all()
        np.testing.assert_allclose(
            actual.loc[actual.coefficient_id.eq("factor"), "value"],
            1.0,
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual.loc[actual.coefficient_id.eq("intercept"), "value"],
            0.0,
            atol=1e-12 * unit,
        )
        assert actual.white_p_value.isna().all()
        assert actual.white_reason.eq("white_test_undefined").all()
    with artifact.table("fama_macbeth.parquet") as actual:
        assert (
            actual.regression_id.tolist()
            == ["equal:single:factor"] * 2 + ["total_market_cap:single:factor"] * 2
        )
        assert actual.period_count.tolist() == [21] * 4
        assert actual.status.eq("UNDEFINED").all()
        assert actual.reason.eq("hac_inference_undefined").all()
        assert actual.standard_error.isna().all()
        assert actual.white_undefined_period_count.tolist() == [
            int(
                periods.loc[
                    periods.coefficient_id.eq(row.coefficient_id)
                    & periods.regression_id.eq(row.regression_id),
                    "white_p_value",
                ]
                .isna()
                .sum()
            )
            for row in actual.itertuples(index=False)
        ]


def test_core_fmb_signed_factors_survive_publication(tmp_path):
    from scipy.linalg import hadamard

    from quant_research_platform.research.statistics import fama_macbeth_period

    factors = tuple(f"x{i}" for i in range(6))
    values = hadamard(16)[:, 1:7].astype(float)
    values[0] = 0.01
    frame = pd.DataFrame(values, columns=factors)
    frame["timestamp"] = pd.Timestamp("2023-01-02")
    frame["asset_id"] = [str(i) for i in range(len(frame))]
    frame["label"] = frame.x0
    frame["total_market_cap"] = np.arange(1.0, len(frame) + 1.0)
    for weight_column in (None, "total_market_cap"):
        actual = fama_macbeth_period(
            frame,
            return_column="label",
            factor_columns=factors,
            weight_column=weight_column,
        )
        np.testing.assert_allclose(
            actual.coefficients, [0, 1, 0, 0, 0, 0, 0], rtol=1e-12, atol=1e-14
        )
        assert actual.observation_count == 16
        assert actual.white_reason == "white_test_undefined"

    periods = pd.DataFrame(
        _fama_macbeth_period_records(
            frame, return_column="label", factors=(), core_factors=factors
        )
    )
    periods["timestamp"] = periods.timestamp.astype("datetime64[ns]")
    tables = {"fama_macbeth_periods": periods}
    tables["fama_macbeth"] = _fama_macbeth_summary(
        periods, factors=(), core_factors=factors, max_lags=19
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics",
        metadata={
            "sample_role": "validation",
            "window": {"start": "2023-01-01", "end": "2023-03-31"},
        },
        tables=tables,
        table_group="factor",
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        assert len(actual) == 14
        assert actual.status.eq("DEFINED").all()
        assert actual.observation_count.eq(16).all()
        np.testing.assert_allclose(
            actual.value,
            actual.coefficient_id.eq("x0").astype(float),
            rtol=1e-12,
            atol=1e-14,
        )


def test_native_fmb_varying_scale_identity_has_no_false_intercept_inference(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.statistics_production import factor_statistics
    from quant_research_platform.storage.datasets import write_dataset

    dates = pd.bdate_range("2020-01-01", periods=21)
    values = np.arange(1.0, 9.0)[None, :] * np.arange(1.0, 22.0)[:, None] * 0.001
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(8),
            "asset_id": list("abcdefgh") * len(dates),
            "x": values.ravel(),
            "label": values.ravel(),
            "total_market_cap": np.tile(np.arange(1.0, 9.0), len(dates)),
            "eligible": True,
            "label_available_at": dates.repeat(8),
        }
    )
    spec = DataSpec(
        data_id="identity",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "label": "label",
            "total_market_cap": "context",
            "eligible": "mask",
            "label_available_at": "context",
        },
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "label_available_at"},
    )
    write_dataset(tmp_path / "input", frame, spec=spec)
    output = factor_statistics(
        {
            "dataset": str(tmp_path / "input"),
            "output": str(tmp_path / "output"),
            "window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
            "analyses": ["fama_macbeth"],
            "evaluation": {"factors": ["x"], "primary_label": "label"},
            "statistics": {"hac_lag": 19, "fama_macbeth_core_factors": []},
        }
    )
    artifact = statistics.read_statistics_artifact(output)
    with artifact.table("fama_macbeth_periods.parquet") as periods:
        assert periods.status.eq("DEFINED").all()
        assert periods.loc[periods.coefficient_id.eq("intercept"), "value"].eq(0).all()
        assert periods.white_p_value.isna().all()
    with artifact.table("fama_macbeth.parquet") as summary:
        assert summary.period_count.eq(21).all()
        assert summary.status.eq("UNDEFINED").all()
        assert summary.reason.eq("hac_inference_undefined").all()
        assert summary.loc[summary.coefficient_id.eq("intercept"), "mean"].eq(0).all()
        assert summary.p_value.isna().all()
        assert summary.standard_error.isna().all()
