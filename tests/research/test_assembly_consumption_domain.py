"""Only contributing inputs create a research key-coverage relationship."""

from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.storage.low_level_fixtures import make_processed_dataset_spec


def test_unused_input_does_not_add_coverage_requirement(tmp_path):
    base = pd.DataFrame(
        {
            "market_id": ["custom", "custom"],
            "asset_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]).astype(
                "datetime64[ns]"
            ),
            "factor": [1.0, 2.0],
        }
    )
    spec = make_processed_dataset_spec(
        "base", market_id="custom", value_column="factor"
    )
    write_dataset(tmp_path / "base", base, spec=spec)
    auxiliary = base.iloc[:1].rename(columns={"factor": "unselected"})
    write_dataset(
        tmp_path / "auxiliary",
        auxiliary,
        spec=replace(
            spec,
            data_id="auxiliary",
            column_roles={
                **{key: spec.column_roles[key] for key in spec.primary_key},
                "unselected": "factor",
            },
        ),
    )
    config = {
        "market_id": "custom",
        "dataset_id": "consumed",
        "inputs": {name: str(tmp_path / name) for name in ("base", "auxiliary")},
        "base_input_data_id": "base",
        "factor_columns": ["factor"],
        "label_columns": [],
        "feature_recipe": None,
        "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
        "output_root": str(tmp_path / "consumed"),
    }
    root = assemble(config)
    pd.testing.assert_frame_equal(read_dataset(root), base)
    session = open_dataset_session(root)
    assert session.spec["input_data_ids"] == ["base", "auxiliary"]
    assert session.manifest["parameters"]["input_columns"] == {
        "base": list(base.columns)
    }
    with pytest.raises(ValueError, match="missing_research_input_keys: auxiliary: 1"):
        assemble(
            config
            | {
                "factor_columns": ["factor", "unselected"],
                "output_root": str(tmp_path / "required"),
            }
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        assemble(
            config
            | {
                "inputs": config["inputs"] | {"auxiliary": str(tmp_path / "missing")},
                "output_root": str(tmp_path / "missing_input"),
            }
        )
