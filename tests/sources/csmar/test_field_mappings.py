from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.sources.csmar import field_mappings


def test_normalize_provider_stock_code_series_accepts_known_provider_forms() -> None:
    values = pd.Series([1, 1.0, "1", "000001", "000001.0"])

    result = field_mappings.normalize_provider_stock_code_series(values)

    assert result.tolist() == ["000001"] * 5
    assert field_mappings.normalize_stock_code(values).tolist() == result.tolist()


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        float("nan"),
        "",
        "nan",
        "None",
        True,
        -1,
        1.5,
        [],
        {},
        set(),
        1000000,
    ],
)
def test_normalize_provider_stock_code_series_rejects_bad_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_csmar_stock_code"):
        field_mappings.normalize_provider_stock_code_series(pd.Series([value]))
