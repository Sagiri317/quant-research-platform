"""Crypto HFT protocol constants and catalogs; market facts stay in crypto."""

from __future__ import annotations

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.research.randomness import ROOT_SEED

MARKET_ID = MarketId("crypto")
MICROSTRUCTURE_DATASET_ID = "microstructure_forward_returns"
HFT_RESEARCH_DATASET_ID = "binance_futures_hft_research_100ms"
HFT_FACTOR_COLUMNS = (
    "spread",
    "relative_spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice",
    "microprice_deviation",
    "ofi_100ms",
    "signed_trade_flow_100ms",
    "taker_buy_volume_100ms",
    "taker_sell_volume_100ms",
    "return_100ms",
    "realized_volatility_1s",
    "top10_depth_per_spread",
)
HFT_LABEL_HORIZONS_MS = (100, 500, 1_000, 5_000)
CRYPTO_FORWARD_RETURN_HORIZONS = (1, 5, 15)
CRYPTO_FORWARD_RETURN_COLUMNS = tuple(
    forward_return_label_column(horizon, DataFrequency.MINUTE)
    for horizon in CRYPTO_FORWARD_RETURN_HORIZONS
)
HFT_PRICE_LABEL_COLUMNS = tuple(
    column
    for horizon in HFT_LABEL_HORIZONS_MS
    for column in (
        f"future_mid_return_{horizon}ms",
        f"future_mid_direction_{horizon}ms",
    )
)
HFT_LABEL_COLUMNS = (
    *HFT_PRICE_LABEL_COLUMNS,
    "maker_bid_sell_pressure_proxy_500ms",
)
HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS = 500
HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN = "maker_bid_sell_pressure_proxy_500ms"
HFT_MODEL_LABEL_COLUMN = "future_mid_return_500ms"
HFT_MODEL_ALGORITHMS = (
    "lightgbm_regressor",
    "linear_regression",
    "xgboost_regressor",
    "gradient_boosting_regressor",
    "ridge_regression",
)
HFT_PRICE_MODEL_ALGORITHMS = HFT_MODEL_ALGORITHMS[:3]
HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID = "crypto_hft_maker_bid_sell_pressure_proxy"
HFT_MULTI_ASSET_SCOPES = {
    "btc_independent": (("btc_usdt",), ("btc_usdt",)),
    "eth_independent": (("eth_usdt",), ("eth_usdt",)),
    "pooled": (("btc_usdt", "eth_usdt"), ("btc_usdt", "eth_usdt")),
    "leave_btc_out": (("eth_usdt",), ("btc_usdt",)),
    "leave_eth_out": (("btc_usdt",), ("eth_usdt",)),
}
HFT_EVALUATION_ID = "crypto_hft_microstructure_evaluation"
EVALUATION_ID = "microstructure_forward_returns_evaluation"


def build_signal_specs(
    *, columns: tuple[str, ...] = MICROSTRUCTURE_FACTOR_COLUMNS
) -> tuple[SignalSpec, ...]:
    return tuple(
        SignalSpec(
            signal_id=column,
            source_kind="factor",
            source_column=column,
            direction="lower_is_better"
            if column in {"spread", "relative_spread"}
            else "higher_is_better",
        )
        for column in columns
    )


def build_evaluation_catalog(
    *, columns: tuple[str, ...] = MICROSTRUCTURE_FACTOR_COLUMNS
) -> EvaluationCatalog:
    return EvaluationCatalog(
        (
            EvaluationSpec(
                evaluation_id=EVALUATION_ID,
                market_id=MARKET_ID,
                research_dataset_id=MICROSTRUCTURE_DATASET_ID,
                signals=build_signal_specs(columns=columns),
                label_columns=tuple(
                    forward_return_label_column(horizon, DataFrequency.MINUTE)
                    for horizon in (1, 5, 15)
                ),
            ),
        )
    )


def build_hft_model_catalog(*, training_row_bound: int) -> ModelCatalog:
    if type(training_row_bound) is not int or training_row_bound <= 0:
        raise ValueError("hft_training_row_bound_invalid")
    price_models = tuple(
        ModelSpec(
            model_id=f"crypto_hft_{algorithm}",
            model_algorithm=algorithm,
            input_factor_columns=("ofi_100ms",)
            if algorithm == "linear_regression"
            else HFT_FACTOR_COLUMNS,
            label_column=HFT_MODEL_LABEL_COLUMN,
            random_seed=ROOT_SEED,
            max_train_rows=training_row_bound,
            hyperparameters={"n_jobs": -1}
            if algorithm in {"xgboost_regressor", "lightgbm_regressor"}
            else {},
        )
        for algorithm in HFT_MODEL_ALGORITHMS[:3]
    )
    return ModelCatalog(
        (
            *price_models,
            ModelSpec(
                model_id="crypto_hft_maker_bid_sell_pressure_proxy",
                model_algorithm="gradient_boosting_regressor",
                input_factor_columns=HFT_FACTOR_COLUMNS,
                label_column="maker_bid_sell_pressure_proxy_500ms",
                random_seed=ROOT_SEED,
                max_train_rows=training_row_bound,
            ),
            *(
                ModelSpec(
                    model_id=f"crypto_hft_{mode}",
                    model_algorithm="ridge_regression",
                    input_factor_columns=HFT_FACTOR_COLUMNS,
                    label_column=HFT_MODEL_LABEL_COLUMN,
                    random_seed=ROOT_SEED,
                    max_train_rows=training_row_bound,
                    hyperparameters={"alpha": 1.0},
                )
                for mode in HFT_MULTI_ASSET_SCOPES
            ),
        )
    )


def build_hft_prediction_catalog(
    *, train_end: str, test_start: str, test_end: str
) -> PredictionCatalog:
    model_ids = (
        *(f"crypto_hft_{algorithm}" for algorithm in HFT_PRICE_MODEL_ALGORITHMS),
        HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
        *(f"crypto_hft_{mode}" for mode in HFT_MULTI_ASSET_SCOPES),
    )
    return PredictionCatalog(
        tuple(
            PredictionSpec(
                prediction_id=f"{model_id}_prediction",
                model_id=model_id,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                training_asset_ids=_hft_asset_scope(model_id, 0),
                evaluation_asset_ids=_hft_asset_scope(model_id, 1),
            )
            for model_id in model_ids
        )
    )


def _hft_asset_scope(model_id: str, position: int) -> tuple[str, ...]:
    mode = model_id.removeprefix("crypto_hft_")
    return (
        HFT_MULTI_ASSET_SCOPES[mode][position] if mode in HFT_MULTI_ASSET_SCOPES else ()
    )


def build_hft_evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        (
            EvaluationSpec(
                evaluation_id=HFT_EVALUATION_ID,
                market_id=MARKET_ID,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                signals=build_signal_specs(columns=HFT_FACTOR_COLUMNS),
                label_columns=HFT_LABEL_COLUMNS,
            ),
        )
    )


def build_research_dataset_catalog() -> ResearchDatasetCatalog:
    factors = MICROSTRUCTURE_FACTOR_COLUMNS
    labels = tuple(
        forward_return_label_column(horizon, DataFrequency.MINUTE)
        for horizon in (1, 5, 15)
    )
    context = (
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
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        *factors,
        *labels,
        "eligible",
        *context,
    )
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id=MICROSTRUCTURE_DATASET_ID,
                market_id=MARKET_ID,
                input_data_ids=("market_panel_1m", "microstructure", "forward_returns"),
                base_input_data_id="market_panel_1m",
                factor_columns=factors,
                label_columns=labels,
                filter_columns=("eligible",),
                context_columns=context,
                frequency=DataFrequency.MINUTE,
                grain=DataGrain.RESEARCH_ROW,
                column_roles=_roles(columns, factors, labels, ("eligible",)),
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
            ),
        )
    )


def build_hft_research_dataset_catalog() -> ResearchDatasetCatalog:
    context = (
        "segment_id",
        "book_segment_id",
        "book_connection_id",
        "trade_connection_id",
        "book_capture_seq",
        "trade_capture_seq_min",
        "trade_capture_seq_max",
        "book_observed_time_ms",
        "exchange_event_time_ms",
        "transaction_time_ms",
        "local_receive_time",
        "book_receive_monotonic_ns",
        "update_id",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
        "mid",
        "bid_depth_1",
        "ask_depth_1",
        "bid_depth_5",
        "ask_depth_5",
        "bid_depth_10",
        "ask_depth_10",
        "book_ofi_100ms",
        "trade_count_100ms",
        "trade_volume_100ms",
        "observed_signed_trade_volume_100ms",
        "observed_taker_buy_volume_100ms",
        "observed_taker_sell_volume_100ms",
    )
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        *HFT_FACTOR_COLUMNS,
        *HFT_LABEL_COLUMNS,
        "eligible",
        *context,
    )
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id=HFT_RESEARCH_DATASET_ID,
                market_id=MARKET_ID,
                input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    "binance_futures_microstructure_factors_100ms",
                    "binance_futures_mid_labels_100ms",
                ),
                base_input_data_id="binance_futures_l2_panel_100ms",
                factor_columns=HFT_FACTOR_COLUMNS,
                label_columns=HFT_LABEL_COLUMNS,
                filter_columns=("eligible",),
                context_columns=context,
                frequency=DataFrequency.TICK,
                grain=DataGrain.RESEARCH_ROW,
                column_roles=_roles(
                    columns, HFT_FACTOR_COLUMNS, HFT_LABEL_COLUMNS, ("eligible",)
                ),
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
            ),
        )
    )


def _roles(
    columns: tuple[str, ...],
    factors: tuple[str, ...],
    labels: tuple[str, ...],
    filters: tuple[str, ...],
) -> dict[str, ColumnRole]:
    return {
        column: ColumnRole.FACTOR
        if column in factors
        else ColumnRole.LABEL
        if column in labels
        else ColumnRole.UNIVERSE
        if column in filters
        else ColumnRole.TIMESTAMP
        if column == "timestamp"
        else ColumnRole.KEY
        if column in {"market_id", "asset_id"}
        else ColumnRole.CONTEXT
        for column in columns
    }
