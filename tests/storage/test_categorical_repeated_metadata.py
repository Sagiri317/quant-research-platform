"""Repeated part metadata retains the complete logical categorical domain."""

from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.storage.low_level_fixtures import make_processed_dataset_spec


@pytest.mark.parametrize("kind", ("string", "integer", "named_timezone"))
@pytest.mark.parametrize("ordered", (False, True))
def test_repeated_then_expanded_category_metadata_roundtrip(tmp_path, kind, ordered):
    if kind == "string":
        domain = pd.Index(["b", "a", "unobserved", "new"])
    elif kind == "integer":
        domain = pd.Index([2, 1, 9, 4])
    else:
        domain = pd.date_range("2021-01-01", periods=4, tz="America/New_York").as_unit(
            "us"
        )
    spec = make_processed_dataset_spec("category_parts")
    spec = replace(
        spec, column_roles={**spec.column_roles, "category": ColumnRole.CONTEXT}
    )
    frames = []
    for index in range(3):
        categories = domain[:3] if ordered or index < 2 else domain
        frame = pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["a"],
                "timestamp": [pd.Timestamp("2021-01-01") + pd.Timedelta(days=index)],
                "value": [float(index)],
            }
        )
        frame["category"] = pd.Categorical(
            [domain[0] if ordered or index < 2 else domain[3]],
            categories=categories,
            ordered=ordered,
        )
        frames.append(frame)
    root = StorageLayout(tmp_path).processed_data_root(
        "a_share", "factor", spec.data_id
    )
    write_dataset(root, frames, spec=spec)
    actual = read_dataset(root)
    assert actual.category.cat.ordered is ordered
    assert set(actual.category.cat.categories) == set(domain[:3] if ordered else domain)
    if ordered:
        assert actual.category.cat.categories.tolist() == domain[:3].tolist()
    assert actual.category.tolist() == [
        domain[0],
        domain[0],
        domain[0] if ordered else domain[3],
    ]
    if kind == "named_timezone":
        assert str(actual.category.cat.categories.tz) == "America/New_York"
        assert actual.category.cat.categories.dtype.unit == "us"
