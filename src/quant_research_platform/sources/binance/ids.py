"""Binance source ids."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.ids import SourceId

SOURCE_ID = SourceId("binance")
RAW_RECORDER_RUN_DATA_ID = "binance_recorder_run"
SPOT_MARKET_TYPE = "spot"
FUTURES_MARKET_TYPE = "usd_s_m_futures"
FUTURES_HFT_RECORDING_SCOPE = "futures-hft"
SPOT_L2_RECORDING_SCOPE = "spot-l2"
RESEARCH_SYMBOLS = ("BTCUSDT", "ETHUSDT")
RECORDING_SCOPE_MARKET_TYPES = {
    FUTURES_HFT_RECORDING_SCOPE: (FUTURES_MARKET_TYPE,),
    SPOT_L2_RECORDING_SCOPE: (SPOT_MARKET_TYPE,),
}


def market_bars_data_id(frequency: DataFrequency | str = DataFrequency.MINUTE) -> str:
    if DataFrequency(frequency) is not DataFrequency.MINUTE:
        raise ValueError("unsupported_binance_market_bars_frequency")
    return "market_bars_1m"


def aggregate_trades_data_id() -> str:
    return "aggregate_trades_tick"


def depth_levels_data_id() -> str:
    return "depth_levels_tick"
