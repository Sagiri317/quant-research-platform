"""Crypto research scenario specs."""

from __future__ import annotations

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.crypto.hft import HFT_GRID_MS
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_COLUMNS,
    HFT_MODEL_LABEL_HORIZON_MS,
    HFT_RESEARCH_DATASET_ID,
    MARKET_ID,
    MICROSTRUCTURE_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
)
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)

EVALUATION_RUN_ID = "microstructure_forward_returns_evaluation"
DEFAULT_SIGNAL_COLUMNS = MICROSTRUCTURE_FACTOR_COLUMNS
HFT_MODEL_ALGORITHMS = (
    "lightgbm_regressor",
    "linear_regression",
    "xgboost_regressor",
)
HFT_MODEL_LABEL_COLUMN = f"future_mid_return_{HFT_MODEL_LABEL_HORIZON_MS}ms"


def build_signal_specs() -> tuple[SignalSpec, ...]:
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
        for column in DEFAULT_SIGNAL_COLUMNS
    )


def build_evaluation_specs(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
) -> tuple[EvaluationSpec, ...]:
    return (
        EvaluationSpec(
            run_id=EVALUATION_RUN_ID,
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


def build_hft_model_catalog(*, training_row_bound: int) -> ModelCatalog:
    if (
        isinstance(training_row_bound, bool)
        or type(training_row_bound) is not int
        or training_row_bound <= 0
    ):
        raise ValueError("hft_training_row_bound_invalid")
    return ModelCatalog(
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
        for algorithm in HFT_MODEL_ALGORITHMS
    )


def build_hft_prediction_catalog(
    *,
    train_end: str,
    test_start: str,
    test_end: str,
) -> PredictionCatalog:
    return PredictionCatalog(
        PredictionSpec(
            prediction_id=f"crypto_hft_{algorithm}_prediction",
            model_id=f"crypto_hft_{algorithm}",
            research_dataset_id=HFT_RESEARCH_DATASET_ID,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for algorithm in HFT_MODEL_ALGORITHMS
    )
