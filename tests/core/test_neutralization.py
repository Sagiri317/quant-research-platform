from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.core import NeutralizationConfig, PanelSpec
from quant_research_platform.core.neutralization import apply_neutralization


def test_demean_by_group_neutralizes_feature_within_each_timestamp_group():
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 4 + ["2026-01-02"] * 4,
            "asset": ["A", "B", "C", "D"] * 2,
            "industry": ["tech", "tech", "bank", "bank"] * 2,
            "factor": [10.0, 12.0, 20.0, 24.0, 100.0, 104.0, 40.0, 46.0],
            "forward_return": [0.01, 0.02, -0.01, -0.02] * 2,
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    neutralized, diagnostics = apply_neutralization(
        panel,
        spec,
        NeutralizationConfig(
            method="demean_by_group",
            group_columns=("industry",),
        ),
    )

    grouped_mean = neutralized.groupby(["timestamp", "industry"])["factor"].mean()
    assert grouped_mean.abs().max() < 1e-12
    assert set(diagnostics["method"]) == {"demean_by_group"}
    assert set(diagnostics["feature"]) == {"factor"}


def test_cross_sectional_ols_neutralizes_feature_per_timestamp_only():
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 4 + ["2026-01-02"] * 4,
            "asset": ["A", "B", "C", "D"] * 2,
            "beta": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            "factor": [2.1, 4.0, 5.9, 8.2, -10.2, -19.9, -30.1, -39.8],
            "forward_return": [0.01, 0.02, -0.01, -0.02] * 2,
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    neutralized, diagnostics = apply_neutralization(
        panel,
        spec,
        NeutralizationConfig(
            method="cross_sectional_ols",
            exposure_columns=("beta",),
        ),
    )

    for _, timestamp_frame in neutralized.groupby("timestamp"):
        assert abs(timestamp_frame["factor"].mean()) < 1e-12
        assert abs(timestamp_frame["factor"].corr(timestamp_frame["beta"])) < 1e-12
    assert set(diagnostics["method"]) == {"cross_sectional_ols"}


def test_cross_sectional_ols_rejects_non_numeric_exposure():
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "asset": ["A", "B"],
            "sector": ["tech", "bank"],
            "factor": [1.0, 2.0],
            "forward_return": [0.01, 0.02],
        }
    )
    spec = PanelSpec(feature_columns=("factor",), label_column="forward_return")

    with pytest.raises(TypeError, match="exposure column 'sector' must be numeric"):
        apply_neutralization(
            panel,
            spec,
            NeutralizationConfig(
                method="cross_sectional_ols",
                exposure_columns=("sector",),
            ),
        )
