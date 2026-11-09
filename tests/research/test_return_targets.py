"""One risk target owner binds every horizon to its actual source and clock."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.models import (
    _target_definition,
    _validate_target_definition,
)
from quant_research_platform.contracts.risk import RISK_STYLE_COLUMNS
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.inspection import inspect_dataset


@pytest.mark.parametrize(
    "label",
    [
        "forward_return_1d",
        "forward_return_5d",
        "next_session_intraday_return",
        "future_mid_return_30s",
    ],
)
def test_explicit_risk_target_survives_native_assembly_and_definition_validation(
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
    _validate_target_definition(definition, "model_target")
