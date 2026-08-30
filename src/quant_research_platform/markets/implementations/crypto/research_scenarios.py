"""Crypto research scenario specs."""

from __future__ import annotations

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.crypto.factors.definitions import (
    HFT_FACTOR_COLUMNS,
)
from quant_research_platform.markets.implementations.crypto.hft import HFT_GRID_MS
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_RESEARCH_DATASET_ID,
    MARKET_ID,
    MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
    HFT_LABEL_COLUMNS,
    HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN,
    HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS,
)
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)

EVALUATION_ID = "microstructure_forward_returns_evaluation"
HFT_EVALUATION_ID = "crypto_hft_microstructure_evaluation"
DEFAULT_SIGNAL_COLUMNS = MICROSTRUCTURE_FACTOR_COLUMNS
HFT_PRICE_MODEL_ALGORITHMS = (
    "lightgbm_regressor",
    "linear_regression",
    "xgboost_regressor",
)
HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID = "crypto_hft_maker_bid_sell_pressure_proxy"
HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ALGORITHM = "gradient_boosting_regressor"
HFT_MODEL_ALGORITHMS = (
    *HFT_PRICE_MODEL_ALGORITHMS,
    HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ALGORITHM,
    "standardized_ridge_regression",
)
HFT_MODEL_LABEL_HORIZON_MS = 500
HFT_MODEL_LABEL_COLUMN = f"future_mid_return_{HFT_MODEL_LABEL_HORIZON_MS}ms"
HFT_MULTI_ASSET_SCOPES = {
    "btc_independent": (("btc_usdt",), ("btc_usdt",)),
    "eth_independent": (("eth_usdt",), ("eth_usdt",)),
    "pooled": (("btc_usdt", "eth_usdt"), ("btc_usdt", "eth_usdt")),
    "leave_btc_out": (("eth_usdt",), ("btc_usdt",)),
    "leave_eth_out": (("btc_usdt",), ("eth_usdt",)),
}


def build_signal_specs(
    *,
    columns: tuple[str, ...] = DEFAULT_SIGNAL_COLUMNS,
) -> tuple[SignalSpec, ...]:
    return tuple(
        SignalSpec(
            signal_id=column,
            source_kind="factor",
            source_column=column,
            direction=(
                "lower_is_better"
                if column in {"spread", "relative_spread"}
                else "higher_is_better"
            ),
        )
        for column in columns
    )


def build_evaluation_specs(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
) -> tuple[EvaluationSpec, ...]:
    return (
        EvaluationSpec(
            evaluation_id=EVALUATION_ID,
            market_id=MARKET_ID,
            research_dataset_id=MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
            signals=signals or build_signal_specs(),
            label_columns=CRYPTO_FORWARD_RETURN_COLUMNS,
        ),
    )


def build_evaluation_catalog(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
) -> EvaluationCatalog:
    return EvaluationCatalog(build_evaluation_specs(signals=signals))


def build_hft_evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        (
            EvaluationSpec(
                evaluation_id=HFT_EVALUATION_ID,
                market_id=MARKET_ID,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                signals=build_signal_specs(columns=HFT_FACTOR_COLUMNS),
                label_columns=HFT_LABEL_COLUMNS,
                evaluation_axis="cross_sectional",
            ),
        )
    )


def build_hft_model_catalog(*, training_row_bound: int) -> ModelCatalog:
    if (
        isinstance(training_row_bound, bool)
        or type(training_row_bound) is not int
        or training_row_bound <= 0
    ):
        raise ValueError("hft_training_row_bound_invalid")
    price_models = tuple(
        ModelSpec(
            model_id=f"crypto_hft_{algorithm}",
            model_algorithm=algorithm,
            input_factor_columns=(
                ("ofi_100ms",)
                if algorithm == "linear_regression"
                else HFT_FACTOR_COLUMNS
            ),
            label_column=HFT_MODEL_LABEL_COLUMN,
            random_seed=7,
            purge_gap_steps=HFT_MODEL_LABEL_HORIZON_MS // HFT_GRID_MS,
            max_train_rows=training_row_bound,
        )
        for algorithm in HFT_PRICE_MODEL_ALGORITHMS
    )
    return ModelCatalog(
        (
            *price_models,
            ModelSpec(
                model_id=HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
                model_algorithm=HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ALGORITHM,
                input_factor_columns=HFT_FACTOR_COLUMNS,
                label_column=HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN,
                random_seed=7,
                purge_gap_steps=(
                    HFT_MAKER_SELL_PRESSURE_PROXY_HORIZON_MS // HFT_GRID_MS
                ),
                max_train_rows=training_row_bound,
            ),
            *(
                ModelSpec(
                    model_id=f"crypto_hft_{mode}",
                    model_algorithm="standardized_ridge_regression",
                    input_factor_columns=HFT_FACTOR_COLUMNS,
                    label_column=HFT_MODEL_LABEL_COLUMN,
                    random_seed=7,
                    purge_gap_steps=HFT_MODEL_LABEL_HORIZON_MS // HFT_GRID_MS,
                    max_train_rows=training_row_bound,
                )
                for mode in HFT_MULTI_ASSET_SCOPES
            ),
        )
    )


def build_hft_prediction_catalog(
    *,
    train_end: str,
    test_start: str,
    test_end: str,
) -> PredictionCatalog:
    price_predictions = tuple(
        PredictionSpec(
            prediction_id=f"crypto_hft_{algorithm}_prediction",
            model_id=f"crypto_hft_{algorithm}",
            research_dataset_id=HFT_RESEARCH_DATASET_ID,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for algorithm in HFT_PRICE_MODEL_ALGORITHMS
    )
    return PredictionCatalog(
        (
            *price_predictions,
            PredictionSpec(
                prediction_id=f"{HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID}_prediction",
                model_id=HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
                research_dataset_id=HFT_RESEARCH_DATASET_ID,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ),
            *(
                PredictionSpec(
                    prediction_id=f"crypto_hft_{mode}_prediction",
                    model_id=f"crypto_hft_{mode}",
                    research_dataset_id=HFT_RESEARCH_DATASET_ID,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    training_asset_ids=train_assets,
                    evaluation_asset_ids=evaluation_assets,
                )
                for mode, (
                    train_assets,
                    evaluation_assets,
                ) in HFT_MULTI_ASSET_SCOPES.items()
            ),
        )
    )
