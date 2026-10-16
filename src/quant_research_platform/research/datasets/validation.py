"""ResearchDataset validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_research_platform.contracts.columns import _validate_asset_time_key
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.policy.lifecycle import closing_iterator


def validate_research_dataset(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    require_all_columns: bool = True,
) -> None:
    _validate_asset_time_key(spec.primary_key, spec.column_roles)
    duplicate_columns = tuple(
        dict.fromkeys(
            str(column)
            for column in frame.columns[frame.columns.duplicated(keep=False)]
        )
    )
    if duplicate_columns:
        raise ValueError(
            "duplicate_research_dataset_columns: " + ", ".join(duplicate_columns)
        )
    required = set(spec.required_columns if require_all_columns else spec.primary_key)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_research_required_fields: {', '.join(missing)}")

    if (
        "market_id" in frame
        and not frame["market_id"].eq(str(spec.market_id)).fillna(False).all()
    ):
        raise ValueError("research_dataset_market_mismatch")

    null_key_columns = [
        column for column in spec.primary_key if bool(frame[column].isna().any())
    ]
    if null_key_columns:
        joined = ", ".join(null_key_columns)
        raise ValueError(f"null_research_dataset_key_columns: {joined}")

    duplicates = frame.duplicated(list(spec.primary_key))
    if bool(duplicates.any()):
        raise ValueError("duplicate_research_dataset_keys")


def validate_evaluation_input(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    signal_columns: tuple[str, ...] = (),
    signals: tuple[SignalSpec, ...] = (),
    label_columns: tuple[str, ...],
) -> None:
    validate_research_dataset(frame, spec, require_all_columns=False)
    if signals:
        _require_declared_signal_sources(
            signals,
            spec=spec,
            frame=frame,
            error_name="invalid_evaluation_signal_columns",
        )
    else:
        _require_declared_columns(
            signal_columns,
            declared=(*spec.factor_columns, *spec.feature_columns),
            frame=frame,
            error_name="invalid_evaluation_signal_columns",
        )
    _require_declared_columns(
        label_columns,
        declared=spec.label_columns,
        frame=frame,
        error_name="invalid_evaluation_label_columns",
    )


def validate_research_input(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    input_columns: tuple[str, ...],
) -> None:
    validate_research_dataset(frame, spec, require_all_columns=False)
    _require_declared_columns(
        input_columns,
        declared=(*spec.factor_columns, *spec.feature_columns),
        frame=frame,
        error_name="invalid_prediction_input_columns",
    )


def iter_validated_research_input_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    dataset_spec: ResearchDatasetSpec,
    input_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    """Validate dataset chunks at the dataset owner boundary."""

    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
            validate_research_input(
                chunk,
                dataset_spec,
                input_columns=input_columns,
            )
            try:
                yield chunk
            except GeneratorExit:
                return


def _require_declared_columns(
    columns: tuple[str, ...],
    *,
    declared: tuple[str, ...],
    frame: pd.DataFrame,
    error_name: str,
) -> None:
    declared_set = set(declared)
    missing_or_undeclared = [
        column
        for column in columns
        if column not in declared_set or column not in frame.columns
    ]
    if missing_or_undeclared:
        raise ValueError(f"{error_name}: {', '.join(missing_or_undeclared)}")


def _require_declared_signal_sources(
    signals: tuple[SignalSpec, ...],
    *,
    spec: ResearchDatasetSpec,
    frame: pd.DataFrame,
    error_name: str,
) -> None:
    declared_by_kind = {
        "factor": set(spec.factor_columns),
        "feature": set(spec.feature_columns),
        "prediction": set(spec.context_columns),
    }
    missing_or_undeclared = [
        signal.source_column
        for signal in signals
        if signal.source_column not in declared_by_kind[signal.source_kind]
        or signal.source_column not in frame.columns
    ]
    if missing_or_undeclared:
        raise ValueError(f"{error_name}: {', '.join(missing_or_undeclared)}")
