"""Redundancy diagnostics depend on the selected X pair, never unrelated X or Y."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.research.factors.statistics import factor_statistics
from quant_research_platform.storage.datasets import write_dataset


def test_pairwise_factor_correlations_require_no_label_or_maturity(tmp_path):
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "x": [1.0, 2.0, 3.0],
            "y": [3.0, 2.0, 1.0],
            "missing": np.nan,
        }
    )
    source = tmp_path / "source"
    write_dataset(
        source,
        frame,
        spec=ProcessedDataSpec(
            data_id="features",
            market_id="a_share",
            kind="factor",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles={
                "market_id": "key",
                "asset_id": "key",
                "timestamp": "timestamp",
                "eligible": "universe",
                "x": "factor",
                "y": "factor",
                "missing": "factor",
            },
        ),
    )
    root = factor_statistics(
        {
            "dataset": str(source),
            "output": str(tmp_path / "statistics"),
            "window": {"start": "2024-01-02", "end": "2024-01-02"},
            "evaluation": {"factors": ["x", "y", "missing"]},
            "statistics": {},
            "analyses": ["factor_correlation"],
        }
    )
    result = pd.read_parquet(root / "factor_correlation.parquet")
    valid = result.loc[result.factor_id.eq("x") & result.comparison_factor_id.eq("y")]
    assert valid.statistic.tolist() == ["pearson", "spearman"]
    assert valid["mean"].tolist() == pytest.approx([-1, -1])
    assert valid.observation_count.tolist() == [1, 1]
    assert valid.observation_pairs.tolist() == [3, 3]
    unknown = result.loc[result.comparison_factor_id.eq("missing")]
    assert unknown.status.eq("UNDEFINED").all()
    assert unknown.observation_count.eq(0).all()
