"""Finite large-scale FMB coefficients survive native statistics publication."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts import statistics
from quant_research_platform.contracts.experiments import legal_training_rows
from quant_research_platform.research.statistics_production import (
    _fama_macbeth_period_records,
    _fama_macbeth_summary,
)
from tests.artifacts.test_statistics_artifact import metadata as metadata
from tests.artifacts.test_statistics_artifact import tables as tables


@pytest.mark.parametrize("extreme", (False, True))
@pytest.mark.parametrize("factor_case", (True,))
def test_fmb_exact_identity_publishes_periods_and_summary(
    tmp_path, tables, metadata, extreme, factor_case
):
    values = (
        1e308 + np.arange(8) * np.spacing(1e308)
        if extreme
        else np.arange(8, dtype=float)
    )
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
    tables["fama_macbeth_periods"] = periods
    tables["fama_macbeth"] = _fama_macbeth_summary(
        periods, factors=("factor",), core_factors=(), max_lags=19
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics", metadata=metadata, tables=tables, table_group="factor"
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        assert len(actual) == 84
        assert actual.status.eq("DEFINED").all()
        assert actual.loc[actual.coefficient_id.eq("factor"), "value"].eq(1).all()
        assert actual.loc[actual.coefficient_id.eq("intercept"), "value"].eq(0).all()
        if extreme:
            assert actual.white_p_value.isna().all()
            assert actual.white_reason.eq("white_test_undefined").all()
    with artifact.table("fama_macbeth.parquet") as actual:
        assert (
            actual.regression_id.tolist()
            == ["equal:single:factor"] * 2 + ["total_market_cap:single:factor"] * 2
        )
        assert actual.period_count.tolist() == [21] * 4
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


@pytest.mark.parametrize("factor_case", (True,))
def test_core_fmb_mixed_signed_integer_coordinates_survive_publication(
    tmp_path, tables, metadata, factor_case
):
    from scipy.linalg import hadamard

    from quant_research_platform.research.statistics import fama_macbeth_period

    factors = tuple(f"x{i}" for i in range(6))
    values = hadamard(16)[:, 1:7].astype(float)
    # Exact coordinates contain -2**63, 2**63 and 1 in each full-rank column.
    values[0] = 2.0**-63
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
        np.testing.assert_array_equal(actual.coefficients, [0, 1, 0, 0, 0, 0, 0])
        assert actual.observation_count == 16
        assert actual.white_reason == "white_test_undefined"

    periods = pd.DataFrame(
        _fama_macbeth_period_records(
            frame, return_column="label", factors=(), core_factors=factors
        )
    )
    periods["timestamp"] = periods.timestamp.astype("datetime64[ns]")
    tables["fama_macbeth_periods"] = periods
    tables["fama_macbeth"] = _fama_macbeth_summary(
        periods, factors=(), core_factors=factors, max_lags=19
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics", metadata=metadata, tables=tables, table_group="factor"
    )
    artifact = statistics.read_statistics_artifact(root)
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        assert len(actual) == 14
        assert actual.status.eq("DEFINED").all()
        assert actual.observation_count.eq(16).all()
        np.testing.assert_array_equal(
            actual.value, actual.coefficient_id.eq("x0").astype(float)
        )
