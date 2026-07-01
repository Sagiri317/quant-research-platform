from __future__ import annotations

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.markets.implementations.a_share.processed import (
    diagnostics,
)


def test_processed_schema_inventory_columns_include_masks_and_labels() -> None:
    columns = diagnostics.schema_inventory_important_columns(
        label_horizons=(1, 20),
        label_frequency=DataFrequency.DAILY,
    )

    assert columns == (
        "eligible",
        "can_enter",
        "can_exit",
        "is_st",
        "is_pt",
        "limit_status",
        "forward_return_1d",
        "forward_return_20d",
    )


def test_processed_schema_inventory_important_columns_use_frequency_suffix() -> None:
    assert diagnostics.schema_inventory_important_columns(
        label_horizons=(5,),
        label_frequency=DataFrequency.MINUTE,
    ) == (
        "eligible",
        "can_enter",
        "can_exit",
        "is_st",
        "is_pt",
        "limit_status",
        "forward_return_5m",
    )
