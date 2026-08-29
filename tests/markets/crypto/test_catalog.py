from __future__ import annotations

from quant_research_platform.contracts.data import DataFrequency, ProcessedDataKind
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.crypto import ids
from quant_research_platform.markets.implementations.crypto.factors.definitions import (
    factor_catalog,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    label_catalog,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_datasets import (
    build_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    DEFAULT_SIGNAL_COLUMNS,
    build_evaluation_catalog,
)
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)


def test_crypto_processed_and_research_catalogs_use_approved_ids() -> None:
    processed = processed_catalog()
    research = build_research_dataset_catalog()

    assert [spec.data_id for spec in processed.list()] == [
        "forward_returns",
        "market_panel_1m",
        "microstructure",
    ]
    assert research.list()[0].data_id == "microstructure_forward_returns"
    for spec in processed.list():
        assert spec.market_id == MarketId("crypto")
        assert spec.primary_key == ("market_id", "asset_id", "timestamp")
        assert "source_asset_id" not in spec.required_columns
    assert processed.get("market_panel_1m").kind is ProcessedDataKind.PANEL
    assert processed.get("market_panel_1m").frequency is DataFrequency.MINUTE
    assert research.list()[0].context_columns == (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "trade_flow_observed",
        "quote_observed",
        "quote_age_seconds",
        "mid_price",
    )


def test_crypto_factors_and_labels_are_strictly_limited() -> None:
    factors = factor_catalog()
    labels = label_catalog()

    assert [definition.output_column for definition in factors.list()] == [
        "spread",
        "relative_spread",
        "depth_imbalance",
        "microprice",
        "microprice_deviation",
        "trade_intensity_5m",
        "signed_volume_5m",
        "flow_imbalance_5m",
    ]
    assert [definition.output_column for definition in labels.list()] == [
        forward_return_label_column(1, DataFrequency.MINUTE),
        forward_return_label_column(5, DataFrequency.MINUTE),
        forward_return_label_column(15, DataFrequency.MINUTE),
    ]
    assert ids.MARKET_ID == MarketId("crypto")


def test_crypto_evaluation_uses_microstructure_research_dataset() -> None:
    evaluation = build_evaluation_catalog().list()[0]

    assert evaluation.evaluation_id == "microstructure_forward_returns_evaluation"
    assert evaluation.market_id == MarketId("crypto")
    assert evaluation.research_dataset_id == "microstructure_forward_returns"
    assert tuple(signal.source_column for signal in evaluation.signals) == (
        "spread",
        "relative_spread",
        "depth_imbalance",
        "microprice",
        "microprice_deviation",
        "trade_intensity_5m",
        "signed_volume_5m",
        "flow_imbalance_5m",
    )


def test_default_signal_columns_match_microstructure_factor_columns() -> None:
    assert DEFAULT_SIGNAL_COLUMNS == MICROSTRUCTURE_FACTOR_COLUMNS
