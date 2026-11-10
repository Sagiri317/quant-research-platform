"""Broadcast context amplitudes survive the stock feature assembly boundary."""

import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session


def test_market_and_industry_contexts_are_not_cross_sectionally_rescaled(tmp_path):
    factors = (
        "market_up_fraction",
        "industry_amount_hhi",
        "leader_breadth_fragility",
        "reversal_20d",
    )
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["A", "B", "C"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "eligible": True,
            "market_up_fraction": 0.7,
            "industry_amount_hhi": [0.2, 0.2, 0.8],
            "leader_breadth_fragility": [0.03, 0.06, 0.09],
            "reversal_20d": [-0.1, 0.0, 0.1],
        }
    )
    source = tmp_path / "source"
    write_dataset(
        source,
        frame,
        spec=ProcessedDataSpec(
            data_id="contexts",
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
                "eligible": "mask",
                **dict.fromkeys(factors, "factor"),
            },
        ),
    )
    config = {
        "market_id": "a_share",
        "dataset_id": "contexts",
        "inputs": {"context": str(source)},
        "factor_columns": list(factors),
        "label_columns": [],
        "feature_transform": {
            "industry_neutralize": False,
            "size_neutralize": False,
            "clip": "none",
        },
        "feature_groups": ["raw", "processed"],
        "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
        "output_root": str(tmp_path / "assembled"),
    }
    result = assemble(config)
    actual = read_dataset(result)
    pd.testing.assert_frame_equal(
        actual.loc[:, list(factors)], frame.loc[:, list(factors)]
    )
    assert open_dataset_session(result).spec["feature_columns"] == [
        "feature_reversal_20d"
    ]
    assert actual.feature_reversal_20d.mean() == pytest.approx(0)
    with pytest.raises(ValueError, match="amplitude_factors_require_raw_group"):
        assemble(
            {
                **config,
                "feature_groups": ["processed"],
                "output_root": str(tmp_path / "invalid"),
            }
        )
