from __future__ import annotations

import pandas as pd
import pytest

import quant_research_platform.research.evaluations.engine as evaluation_engine
import quant_research_platform.research.portfolios.engine as portfolio_engine
import quant_research_platform.research.signals.materialization as materialization
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.features.transforms import apply_feature_transform
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)


def test_winsorize_clips_by_timestamp_quantiles_instead_of_zscore() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "mom_20d": [0.0, 10.0, 100.0],
        }
    )
    spec = FeatureTransformSpec(
        transform_id="mom_20d_winsorized",
        source_columns=("mom_20d",),
        transform="winsorize",
        output_columns=("mom_20d_winsorized",),
        winsorize_quantiles=(0.2, 0.8),
    )

    result = apply_feature_transform(frame, spec)

    assert result["mom_20d_winsorized"].tolist() == pytest.approx([4.0, 10.0, 64.0])


def test_fillna_transform_names_apply_their_fill_policy() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "factor": [1.0, None, 3.0],
        }
    )
    zero_spec = FeatureTransformSpec(
        transform_id="factor_fillna_zero",
        source_columns=("factor",),
        transform="fillna_zero",
        output_columns=("factor_zero",),
    )
    median_spec = FeatureTransformSpec(
        transform_id="factor_fillna_median",
        source_columns=("factor",),
        transform="fillna_median",
        output_columns=("factor_median",),
    )

    zero_result = apply_feature_transform(frame, zero_spec)
    median_result = apply_feature_transform(frame, median_spec)

    assert zero_result["factor_zero"].tolist() == pytest.approx([1.0, 0.0, 3.0])
    assert median_result["factor_median"].tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_zero_variance_zscore_keeps_numeric_output() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 3,
            "factor": [1.0, 1.0, 1.0],
        }
    )
    result = apply_feature_transform(
        frame,
        FeatureTransformSpec(
            transform_id="factor_zscore",
            source_columns=("factor",),
            transform="cross_sectional_zscore",
            output_columns=("factor_zscore",),
        ),
    )

    assert result["factor_zscore"].dtype == "float64"
    assert result["factor_zscore"].isna().all()


def test_signal_materialization_normalizes_direction_and_transform() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "risk": [1.0, 2.0, 3.0],
        }
    )
    signal = SignalSpec(
        signal_id="risk_signal",
        source_kind="factor",
        source_column="risk",
        direction="lower_is_better",
        transform="rank",
    )

    result = normalize_signal_frame(frame, (signal,))

    assert result["risk_signal"].tolist() == pytest.approx([-1.0, -2.0, -3.0])
    assert "risk" not in result.columns


def test_signal_materialization_enforces_source_kind_column_ownership() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "raw_factor": [1.0],
            "model_feature": [2.0],
        }
    )
    signal = SignalSpec(
        signal_id="bad_feature_signal",
        source_kind="feature",
        source_column="raw_factor",
    )

    with pytest.raises(ValueError, match="invalid_signal_source_kind_column"):
        normalize_signal_frame(
            frame,
            (signal,),
            source_columns_by_kind={
                "factor": ("raw_factor",),
                "feature": ("model_feature",),
                "prediction": (),
            },
        )


def test_evaluation_and_portfolio_share_signal_normalization_owner() -> None:
    assert (
        evaluation_engine.normalize_signal_frame
        is materialization.normalize_signal_frame
    )
    assert (
        portfolio_engine.normalize_signal_frame
        is materialization.normalize_signal_frame
    )
