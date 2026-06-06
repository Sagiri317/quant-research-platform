"""Minimal IC, RankIC, and grouped-return evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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

    rows: list[dict[str, object]] = []
    grouped_rows: list[dict[str, object]] = []
    for timestamp, group in usable_frame.groupby("timestamp", sort=True):
        usable = group[[feature, label]]
        if len(usable) < 2:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "feature": feature,
                "label": label,
                "ic": usable[feature].corr(usable[label]),
                "rankic": usable[feature].rank().corr(usable[label].rank()),
            }
        )
        buckets = pd.qcut(
            usable[feature].rank(method="first"),
            q=min(n_groups, len(usable)),
            labels=False,
            duplicates="drop",
        )
        bucketed = usable.assign(group=buckets.astype(int) + 1)
        for bucket, bucket_frame in bucketed.groupby("group", sort=True):
            grouped_rows.append(
                {
                    "timestamp": timestamp,
                    "group": int(bucket),
                    "mean_return": float(bucket_frame[label].mean()),
                }
            )
    metrics = pd.DataFrame(rows)
    grouped_returns = pd.DataFrame(grouped_rows)
    metrics.attrs["metadata"] = metadata
    grouped_returns.attrs["metadata"] = metadata
    return EvaluationResult(
        spec=spec,
        metrics=metrics,
        grouped_returns=grouped_returns,
        metadata=metadata,
    )


__all__ = ["EvaluationResult", "evaluate_factor_frame"]
