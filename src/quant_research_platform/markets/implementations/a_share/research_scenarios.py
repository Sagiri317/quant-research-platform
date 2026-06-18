"""A-share research scenario specs and catalog builders."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import MODEL_ALGORITHMS, ModelSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)

DEFAULT_RETURN_SOURCE_COLUMN = "forward_return_20d"
DEFAULT_MODEL_RANDOM_SEED = 7
DEFAULT_MAX_TRAIN_ROWS = 5000
DEFAULT_TRAIN_END = "2014-06-30"
DEFAULT_TEST_START = "2014-07-01"
DEFAULT_TEST_END: str | None = None
DEFAULT_PREDICTION_MODEL_ALGORITHM = "linear_regression"
DEFAULT_EVALUATION_PERIOD_END = "2014-12-31"


def build_a_share_signal_specs() -> tuple[SignalSpec, ...]:
    return (
        SignalSpec(
            signal_id="mom_20d",
            source_kind="factor",
            source_column="mom_20d",
            direction="higher_is_better",
        ),
        SignalSpec(
            signal_id="rev_1d",
            source_kind="factor",
            source_column="rev_1d",
            direction="lower_is_better",
        ),
        SignalSpec(
            signal_id="mom_20d_zscore",
            source_kind="feature",
            source_column="mom_20d_zscore",
            direction="higher_is_better",
        ),
    )


def build_a_share_model_specs(
    *,
    model_algorithms: Iterable[str] = tuple(sorted(MODEL_ALGORITHMS)),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    random_seed: int | None = DEFAULT_MODEL_RANDOM_SEED,
    hyperparameter_preset: str = "smoke",
) -> tuple[ModelSpec, ...]:
    return tuple(
        ModelSpec(
            model_id=_a_share_model_id(model_algorithm, return_source_column),
            model_algorithm=model_algorithm,
            input_factor_columns=_a_share_model_factor_inputs(model_algorithm),
            input_feature_columns=_a_share_model_feature_inputs(model_algorithm),
            label_column=return_source_column,
            hyperparameters=_a_share_model_hyperparameters(
                model_algorithm,
                hyperparameter_preset=hyperparameter_preset,
            ),
            random_seed=random_seed,
            max_train_rows=max_train_rows,
        )
        for model_algorithm in sorted(model_algorithms)
    )


def build_a_share_prediction_specs(
    *,
    model_algorithms: Iterable[str] = tuple(sorted(MODEL_ALGORITHMS)),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    train_end: str = DEFAULT_TRAIN_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str | None = DEFAULT_TEST_END,
) -> tuple[PredictionSpec, ...]:
    return tuple(
        PredictionSpec(
            prediction_id=_a_share_prediction_id(model_algorithm, test_start),
            model_id=_a_share_model_id(model_algorithm, return_source_column),
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for model_algorithm in sorted(model_algorithms)
    )


def build_a_share_evaluation_specs(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
) -> tuple[EvaluationSpec, ...]:
    return (
        EvaluationSpec(
            run_id=default_a_share_evaluation_run_id(
                return_source_column=return_source_column,
            ),
            market_id=A_SHARE_MARKET_ID,
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            signals=signals or build_a_share_signal_specs(),
            label_columns=(return_source_column,),
        ),
    )


def build_a_share_portfolio_specs(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    frequency: str = "daily",
    execution_lag_steps: int = 1,
) -> tuple[PortfolioSpec, ...]:
    return (
        PortfolioSpec(
            portfolio_id=f"a_share_signal_{return_source_column}_portfolio_2014",
            market_id=A_SHARE_MARKET_ID,
            research_dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            signals=signals or build_a_share_signal_specs(),
            return_source_column=return_source_column,
            frequency=frequency,
            execution_lag_steps=execution_lag_steps,
        ),
    )


def build_a_share_model_catalog(
    *,
    model_algorithms: Iterable[str] = tuple(sorted(MODEL_ALGORITHMS)),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    random_seed: int | None = DEFAULT_MODEL_RANDOM_SEED,
    hyperparameter_preset: str = "smoke",
) -> ModelCatalog:
    catalog = ModelCatalog()
    for spec in build_a_share_model_specs(
        model_algorithms=model_algorithms,
        return_source_column=return_source_column,
        max_train_rows=max_train_rows,
        random_seed=random_seed,
        hyperparameter_preset=hyperparameter_preset,
    ):
        catalog.register(spec)
    return catalog


def build_a_share_prediction_catalog(
    *,
    model_algorithms: Iterable[str] = tuple(sorted(MODEL_ALGORITHMS)),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    train_end: str = DEFAULT_TRAIN_END,
    test_start: str = DEFAULT_TEST_START,
    test_end: str | None = DEFAULT_TEST_END,
) -> PredictionCatalog:
    catalog = PredictionCatalog()
    for spec in build_a_share_prediction_specs(
        model_algorithms=model_algorithms,
        return_source_column=return_source_column,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    ):
        catalog.register(spec)
    return catalog


def build_a_share_evaluation_catalog(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
) -> EvaluationCatalog:
    catalog = EvaluationCatalog()
    for spec in build_a_share_evaluation_specs(
        signals=signals,
        return_source_column=return_source_column,
    ):
        catalog.register(spec)
    return catalog


def build_a_share_portfolio_catalog(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    frequency: str = "daily",
    execution_lag_steps: int = 1,
) -> PortfolioCatalog:
    catalog = PortfolioCatalog()
    for spec in build_a_share_portfolio_specs(
        signals=signals,
        return_source_column=return_source_column,
        frequency=frequency,
        execution_lag_steps=execution_lag_steps,
    ):
        catalog.register(spec)
    return catalog


def default_a_share_evaluation_run_id(
    *,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
) -> str:
    return f"a_share_signal_{return_source_column}_2014"


def default_a_share_prediction_smoke_id(
    *,
    model_algorithm: str = DEFAULT_PREDICTION_MODEL_ALGORITHM,
    test_start: str = DEFAULT_TEST_START,
) -> str:
    return _a_share_prediction_id(model_algorithm, test_start)


def default_a_share_prediction_oos_period(
    *,
    model_algorithm: str = DEFAULT_PREDICTION_MODEL_ALGORITHM,
    test_start: str = DEFAULT_TEST_START,
    test_end: str | None = DEFAULT_TEST_END,
) -> str:
    spec = build_a_share_prediction_specs(
        model_algorithms=(model_algorithm,),
        test_start=test_start,
        test_end=test_end,
    )[0]
    return f"{spec.test_start} through {spec.test_end or DEFAULT_EVALUATION_PERIOD_END}"


def default_a_share_evaluation_period() -> str:
    return f"2014-01-29 through {DEFAULT_EVALUATION_PERIOD_END}"


def _a_share_model_id(model_algorithm: str, return_source_column: str) -> str:
    return f"a_share_{model_algorithm}_mom_20d_{return_source_column}"


def _a_share_prediction_id(model_algorithm: str, test_start: str) -> str:
    year = test_start[:4]
    return f"a_share_{model_algorithm}_mom_20d_prediction_{year}"


def _a_share_model_factor_inputs(model_algorithm: str) -> tuple[str, ...]:
    if model_algorithm == "single_factor_baseline":
        return ("mom_20d",)
    return ("mom_20d", "rev_1d")


def _a_share_model_feature_inputs(model_algorithm: str) -> tuple[str, ...]:
    if model_algorithm == "single_factor_baseline":
        return ()
    return ("mom_20d_zscore",)


def _a_share_model_hyperparameters(
    model_algorithm: str,
    *,
    hyperparameter_preset: str,
) -> dict[str, object]:
    if hyperparameter_preset != "smoke":
        raise ValueError("unsupported_a_share_hyperparameter_preset")
    if model_algorithm in {"random_forest_regressor", "extra_trees_regressor"}:
        return {"n_estimators": 5, "max_depth": 3, "n_jobs": 1}
    if model_algorithm == "gradient_boosting_regressor":
        return {"n_estimators": 5, "max_depth": 2}
    if model_algorithm == "hist_gradient_boosting_regressor":
        return {"max_iter": 5, "max_leaf_nodes": 7}
    if model_algorithm == "xgboost_regressor":
        return {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}
    if model_algorithm == "lightgbm_regressor":
        return {"n_estimators": 5, "num_leaves": 7, "n_jobs": 1, "verbosity": -1}
    return {}


__all__ = [
    "DEFAULT_MAX_TRAIN_ROWS",
    "DEFAULT_MODEL_RANDOM_SEED",
    "DEFAULT_RETURN_SOURCE_COLUMN",
    "DEFAULT_TEST_END",
    "DEFAULT_TEST_START",
    "DEFAULT_TRAIN_END",
    "build_a_share_evaluation_catalog",
    "build_a_share_evaluation_specs",
    "build_a_share_model_catalog",
    "build_a_share_model_specs",
    "build_a_share_portfolio_catalog",
    "build_a_share_portfolio_specs",
    "build_a_share_prediction_catalog",
    "build_a_share_prediction_specs",
    "build_a_share_signal_specs",
]
