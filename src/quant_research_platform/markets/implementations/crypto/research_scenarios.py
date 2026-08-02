"""Crypto research scenario specs."""

from __future__ import annotations

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.crypto.ids import (
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
    catalog = EvaluationCatalog()
    for spec in build_evaluation_specs(signals=signals):
        catalog.register(spec)
    return catalog


__all__ = [
    "DEFAULT_SIGNAL_COLUMNS",
    "EVALUATION_RUN_ID",
    "build_evaluation_catalog",
    "build_evaluation_specs",
    "build_signal_specs",
]
