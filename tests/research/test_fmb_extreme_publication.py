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
from tests.artifacts.test_statistics_artifact import envelope as envelope
from tests.artifacts.test_statistics_artifact import tables as tables


@pytest.mark.parametrize("extreme", (False, True))
def test_fmb_exact_identity_publishes_periods_and_summary(
    tmp_path, tables, envelope, extreme
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
        tmp_path / "statistics", envelope=envelope, tables=tables
    )
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("fama_macbeth_periods.parquet") as actual:
        assert len(actual) == 42
        assert actual.status.eq("DEFINED").all()
        assert actual.loc[actual.coefficient_id.eq("factor"), "value"].eq(1).all()
        assert actual.loc[actual.coefficient_id.eq("intercept"), "value"].eq(0).all()
        if extreme:
            assert actual.white_p_value.isna().all()
            assert actual.white_reason.eq("white_test_undefined").all()
    with artifact.table("fama_macbeth.parquet") as actual:
        assert actual.regression_id.tolist() == ["single:factor"] * 2
        assert actual.period_count.tolist() == [21, 21]
        assert actual.white_undefined_period_count.tolist() == [
            int(
                periods.loc[periods.coefficient_id.eq(name), "white_p_value"]
                .isna()
                .sum()
            )
            for name in ("intercept", "factor")
        ]
