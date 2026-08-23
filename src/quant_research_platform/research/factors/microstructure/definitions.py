"""Generic microstructure factor definition builders."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

MICROSTRUCTURE_FACTOR_FAMILY = "microstructure"

POINT_IN_TIME_MICROSTRUCTURE_FACTORS = (
    "spread",
    "relative_spread",
    "depth_imbalance",
    "microprice",
    "microprice_deviation",
)

ROLLING_5M_MICROSTRUCTURE_FACTORS = (
    "trade_intensity_5m",
    "signed_volume_5m",
    "flow_imbalance_5m",
)

MICROSTRUCTURE_INPUT_COLUMNS = (
    "best_bid",
    "best_ask",
    "bid_quantity",
    "ask_quantity",
    "bid_depth",
    "ask_depth",
    "mid_price",
    "trade_count",
    "trade_flow_observed",
    "signed_volume",
    "volume",
)


def microstructure_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
) -> tuple[FactorDefinition, ...]:
    definitions: list[FactorDefinition] = []
    for factor_id in POINT_IN_TIME_MICROSTRUCTURE_FACTORS:
        definitions.append(
            _definition(
                factor_id,
                market_id=market_id,
                required_input_data_id=required_input_data_id,
                required_windows=(),
                lookback_window=None,
                window=None,
                timing_semantics="bar_close_asof_quote",
            )
        )
    for factor_id in ROLLING_5M_MICROSTRUCTURE_FACTORS:
        definitions.append(
            _definition(
                factor_id,
                market_id=market_id,
                required_input_data_id=required_input_data_id,
                required_windows=(5,),
                lookback_window=5,
                window=5,
                timing_semantics="clock_contiguous_5m_after_bar_close",
            )
        )
    return tuple(definitions)


def _definition(
    factor_id: str,
    *,
    market_id: MarketId,
    required_input_data_id: str,
    required_windows: tuple[int, ...],
    lookback_window: int | None,
    window: int | None,
    timing_semantics: str,
) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        market_id=market_id,
        factor_set=FactorSet.CORE,
        input_columns=MICROSTRUCTURE_INPUT_COLUMNS,
        base_input_columns=MICROSTRUCTURE_INPUT_COLUMNS,
        derived_input_columns=(),
        output_column=factor_id,
        display_name=factor_id,
        description="Generic microstructure factor registered by QRP.",
        family=MICROSTRUCTURE_FACTOR_FAMILY,
        required_windows=required_windows,
        lookback_window=lookback_window,
        grain=DataGrain.ASSET_TIMESTAMP,
        timing_semantics=timing_semantics,
        null_policy="preserve_nulls",
        source_layer="processed",
        required_input_data_id=required_input_data_id,
        window=window,
    )
