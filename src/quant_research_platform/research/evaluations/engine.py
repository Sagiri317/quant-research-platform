"""Spec-driven evaluation orchestration for compute kernels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec


@dataclass(frozen=True)
class EvaluationResult:
    spec: EvaluationSpec
    metrics: pd.DataFrame
    grouped_returns: pd.DataFrame
    metadata: dict[str, object]


def evaluate_factor_frame(
    frame: pd.DataFrame,
    spec: EvaluationSpec,
    *,
    n_groups: int = 5,
) -> EvaluationResult:
    if len(spec.feature_columns) != 1 or len(spec.label_columns) != 1:
        raise ValueError("minimal_evaluation_requires_one_feature_and_one_label")
    feature = spec.feature_columns[0]
    label = spec.label_columns[0]
    required = {"timestamp", feature, label}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_evaluation_columns: {', '.join(missing)}")

    from quant_research_platform.research.filtering import apply_filter_mode

    filtered, metadata = apply_filter_mode(
        frame,
        mode=spec.filter_mode,
        metadata_key="filter_mode",
    )
    usable_frame = filtered.dropna(subset=[feature, label]).copy()
    metadata["rows_after_feature_label_dropna"] = int(len(usable_frame))

    compute_spec = ComputePanelSpec(
        feature_columns=(feature,),
        label_column=label,
        timestamp_column="timestamp",
        asset_column="asset_id" if "asset_id" in usable_frame else "asset",
    )
    metrics, grouped_returns = compute_ic_and_grouped_label_values(
        usable_frame,
        compute_spec,
        n_groups=n_groups,
    )
    metrics.attrs["metadata"] = metadata
    grouped_returns.attrs["metadata"] = metadata
    return EvaluationResult(
        spec=spec,
        metrics=metrics,
        grouped_returns=grouped_returns,
        metadata=metadata,
    )


__all__ = ["EvaluationResult", "evaluate_factor_frame"]
