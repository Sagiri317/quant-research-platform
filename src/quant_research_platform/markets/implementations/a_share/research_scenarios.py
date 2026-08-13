"""A-share research scenario specs and catalog builders."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.data import (
    DataFrequency,
    data_frequency_slug,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
    TradabilityConfig,
)
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share.ids import (
    MARKET_ID,
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)
from quant_research_platform.research.models.algorithm_facts import (
    algorithm_facts,
    model_algorithm_ids,
    model_default_hyperparameters,
    model_smoke_hyperparameters,
)

DEFAULT_MOMENTUM_WINDOW = 20
DEFAULT_REVERSAL_WINDOW = 1
DEFAULT_MOMENTUM_COLUMN = price_volume_window_column(
    "mom",
    DEFAULT_MOMENTUM_WINDOW,
    DataFrequency.DAILY,
)
DEFAULT_REVERSAL_COLUMN = price_volume_window_column(
    "rev",
    DEFAULT_REVERSAL_WINDOW,
    DataFrequency.DAILY,
)
DEFAULT_MOMENTUM_FEATURE_COLUMN = f"{DEFAULT_MOMENTUM_COLUMN}_zscore"


@dataclass(frozen=True)
class _ScenarioDefaults:
    frequency: DataFrequency = DataFrequency.DAILY
    return_horizon: int = 20
    random_seed: int = 7
    max_train_rows: int = 5000
    rebalance_frequency: str = field(
        default_factory=lambda: data_frequency_slug(DataFrequency.DAILY)
    )
    execution_lag_steps: int = 0
    test_end: str | None = None
    signal_columns: tuple[str, ...] = (
        DEFAULT_MOMENTUM_COLUMN,
        DEFAULT_REVERSAL_COLUMN,
        DEFAULT_MOMENTUM_FEATURE_COLUMN,
    )
    model_factor_inputs: tuple[str, ...] = (
        DEFAULT_MOMENTUM_COLUMN,
        DEFAULT_REVERSAL_COLUMN,
    )
    model_feature_inputs: tuple[str, ...] = (DEFAULT_MOMENTUM_FEATURE_COLUMN,)
    baseline_factor_inputs: tuple[str, ...] = (DEFAULT_MOMENTUM_COLUMN,)


SCENARIO_DEFAULTS = _ScenarioDefaults()
DEFAULT_RETURN_SOURCE_COLUMN = forward_return_label_column(
    SCENARIO_DEFAULTS.return_horizon,
    SCENARIO_DEFAULTS.frequency,
)
_DEFAULT_PORTFOLIO_RETURN_SOURCE_COLUMN = forward_return_label_column(
    1,
    SCENARIO_DEFAULTS.frequency,
)
_DEFAULT_FACTOR_PORTFOLIO_ID = f"signal_{DEFAULT_RETURN_SOURCE_COLUMN}_portfolio"
DEFAULT_MODEL_RANDOM_SEED = SCENARIO_DEFAULTS.random_seed
DEFAULT_MAX_TRAIN_ROWS = SCENARIO_DEFAULTS.max_train_rows
DEFAULT_TEST_END: str | None = SCENARIO_DEFAULTS.test_end
DEFAULT_PORTFOLIO_DIAGNOSTICS_CONFIG = PortfolioDiagnosticsConfig(
    capacity_amount_column="amount",
    capacity_float_market_cap_column="float_market_cap",
    categorical_exposure_columns=("csrc_2012_industry_code",),
)
_DEFAULT_PORTFOLIO_TRADABILITY_CONFIG = TradabilityConfig(
    eligibility_column="eligible",
    can_enter_column="can_enter",
    can_exit_column="can_exit",
)


def build_signal_specs() -> tuple[SignalSpec, ...]:
    return (
        SignalSpec(
            signal_id=DEFAULT_MOMENTUM_COLUMN,
            source_kind="factor",
            source_column=DEFAULT_MOMENTUM_COLUMN,
            direction="higher_is_better",
        ),
        SignalSpec(
            signal_id=DEFAULT_REVERSAL_COLUMN,
            source_kind="factor",
            source_column=DEFAULT_REVERSAL_COLUMN,
            direction="lower_is_better",
        ),
        SignalSpec(
            signal_id=DEFAULT_MOMENTUM_FEATURE_COLUMN,
            source_kind="feature",
            source_column=DEFAULT_MOMENTUM_FEATURE_COLUMN,
            direction="higher_is_better",
        ),
    )


def build_model_specs(
    *,
    model_algorithm_ids: Iterable[str] = model_algorithm_ids(),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    random_seed: int | None = DEFAULT_MODEL_RANDOM_SEED,
    hyperparameter_preset: str = "default",
) -> tuple[ModelSpec, ...]:
    if hyperparameter_preset not in {"default", "smoke"}:
        raise ValueError("unsupported_hyperparameter_preset")
    purge_gap_steps = label_definitions.label_catalog().get(
        return_source_column
    ).lookahead
    specs: list[ModelSpec] = []
    for model_algorithm in sorted(model_algorithm_ids):
        facts = algorithm_facts(model_algorithm)
        single_factor = facts.input_kind == "single_factor_input"
        specs.append(
            ModelSpec(
                model_id=_model_id(model_algorithm, return_source_column),
                model_algorithm=model_algorithm,
                input_factor_columns=(
                    SCENARIO_DEFAULTS.baseline_factor_inputs
                    if single_factor
                    else SCENARIO_DEFAULTS.model_factor_inputs
                ),
                input_feature_columns=(
                    () if single_factor else SCENARIO_DEFAULTS.model_feature_inputs
                ),
                label_column=return_source_column,
                hyperparameters=(
                    model_smoke_hyperparameters(model_algorithm)
                    if hyperparameter_preset == "smoke"
                    else model_default_hyperparameters(model_algorithm)
                ),
                random_seed=random_seed,
                purge_gap_steps=purge_gap_steps,
                max_train_rows=max_train_rows,
            )
        )
    return tuple(specs)


def build_prediction_specs(
    *,
    train_end: str,
    test_start: str,
    model_algorithm_ids: Iterable[str] = model_algorithm_ids(),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    test_end: str | None = DEFAULT_TEST_END,
) -> tuple[PredictionSpec, ...]:
    return tuple(
        PredictionSpec(
            prediction_id=(
                f"{model_algorithm}_{DEFAULT_MOMENTUM_COLUMN}_prediction_"
                f"{test_start[:4]}"
            ),
            model_id=_model_id(model_algorithm, return_source_column),
            research_dataset_id=PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        for model_algorithm in sorted(model_algorithm_ids)
    )


def build_evaluation_specs(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
) -> tuple[EvaluationSpec, ...]:
    return (
        EvaluationSpec(
            run_id=f"signal_{return_source_column}",
            market_id=MARKET_ID,
            research_dataset_id=PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            signals=signals or build_signal_specs(),
            label_columns=(return_source_column,),
        ),
    )


def build_portfolio_specs(
    *,
    rebalance_frequency: str = SCENARIO_DEFAULTS.rebalance_frequency,
    execution_lag_steps: int = SCENARIO_DEFAULTS.execution_lag_steps,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = _DEFAULT_PORTFOLIO_RETURN_SOURCE_COLUMN,
    prediction_id: str | None = None,
) -> tuple[PortfolioSpec, ...]:
    specs = [
        PortfolioSpec(
            portfolio_id=(
                _DEFAULT_FACTOR_PORTFOLIO_ID
                if return_source_column == _DEFAULT_PORTFOLIO_RETURN_SOURCE_COLUMN
                else f"signal_{return_source_column}_portfolio"
            ),
            market_id=MARKET_ID,
            research_dataset_id=PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            signals=signals or build_signal_specs(),
            return_source_column=return_source_column,
            rebalance_frequency=rebalance_frequency,
            execution_lag_steps=execution_lag_steps,
            tradability_config=_DEFAULT_PORTFOLIO_TRADABILITY_CONFIG,
            diagnostics_config=DEFAULT_PORTFOLIO_DIAGNOSTICS_CONFIG,
        ),
    ]
    if prediction_id is not None:
        specs.append(
            PortfolioSpec(
                portfolio_id=f"{prediction_id}_portfolio",
                market_id=MARKET_ID,
                research_dataset_id=PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
                signals=(
                    SignalSpec(
                        signal_id=prediction_id,
                        source_kind="prediction",
                        source_column=PREDICTION_SCORE_COLUMN,
                    ),
                ),
                return_source_column=return_source_column,
                rebalance_frequency=rebalance_frequency,
                execution_lag_steps=execution_lag_steps,
                tradability_config=_DEFAULT_PORTFOLIO_TRADABILITY_CONFIG,
                diagnostics_config=DEFAULT_PORTFOLIO_DIAGNOSTICS_CONFIG,
            )
        )
    return tuple(specs)


def build_model_catalog(
    *,
    model_algorithm_ids: Iterable[str] = model_algorithm_ids(),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    random_seed: int | None = DEFAULT_MODEL_RANDOM_SEED,
    hyperparameter_preset: str = "default",
) -> ModelCatalog:
    catalog = ModelCatalog()
    for spec in build_model_specs(
        model_algorithm_ids=model_algorithm_ids,
        return_source_column=return_source_column,
        max_train_rows=max_train_rows,
        random_seed=random_seed,
        hyperparameter_preset=hyperparameter_preset,
    ):
        catalog.register(spec)
    return catalog


def build_prediction_catalog(
    *,
    train_end: str,
    test_start: str,
    model_algorithm_ids: Iterable[str] = model_algorithm_ids(),
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
    test_end: str | None = DEFAULT_TEST_END,
) -> PredictionCatalog:
    catalog = PredictionCatalog()
    for spec in build_prediction_specs(
        model_algorithm_ids=model_algorithm_ids,
        return_source_column=return_source_column,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    ):
        catalog.register(spec)
    return catalog


def build_evaluation_catalog(
    *,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = DEFAULT_RETURN_SOURCE_COLUMN,
) -> EvaluationCatalog:
    catalog = EvaluationCatalog()
    for spec in build_evaluation_specs(
        signals=signals,
        return_source_column=return_source_column,
    ):
        catalog.register(spec)
    return catalog


def build_portfolio_catalog(
    *,
    rebalance_frequency: str = SCENARIO_DEFAULTS.rebalance_frequency,
    execution_lag_steps: int = SCENARIO_DEFAULTS.execution_lag_steps,
    signals: tuple[SignalSpec, ...] | None = None,
    return_source_column: str = _DEFAULT_PORTFOLIO_RETURN_SOURCE_COLUMN,
    prediction_id: str | None = None,
) -> PortfolioCatalog:
    catalog = PortfolioCatalog()
    for spec in build_portfolio_specs(
        signals=signals,
        return_source_column=return_source_column,
        rebalance_frequency=rebalance_frequency,
        execution_lag_steps=execution_lag_steps,
        prediction_id=prediction_id,
    ):
        catalog.register(spec)
    return catalog


def _model_id(model_algorithm: str, return_source_column: str) -> str:
    return f"{model_algorithm}_{DEFAULT_MOMENTUM_COLUMN}_{return_source_column}"


__all__ = [
    "DEFAULT_MAX_TRAIN_ROWS",
    "DEFAULT_MODEL_RANDOM_SEED",
    "DEFAULT_PORTFOLIO_DIAGNOSTICS_CONFIG",
    "DEFAULT_RETURN_SOURCE_COLUMN",
    "DEFAULT_TEST_END",
    "build_evaluation_catalog",
    "build_evaluation_specs",
    "build_model_catalog",
    "build_model_specs",
    "build_portfolio_catalog",
    "build_portfolio_specs",
    "build_prediction_catalog",
    "build_prediction_specs",
    "build_signal_specs",
]
