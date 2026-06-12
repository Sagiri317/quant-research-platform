from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_research_platform.compute.config import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.compute.dataframe import constraints
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _cost_components,
    _direct_signal,
    _direct_signal_array,
    _normalize_group_key,
    _risk_values,
    _select_feature,
    _select_features,
    _selection_count,
    _uses_detailed_cost_config,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)


def test_compute_panel_spec_and_frame_validation_edges() -> None:
    with pytest.raises(ValidationError, match="feature_columns"):
        ComputePanelSpec(feature_columns=(), label_column="label")
    with pytest.raises(ValidationError, match="duplicates"):
        ComputePanelSpec(feature_columns=("feature", "feature"), label_column="label")
    with pytest.raises(ValidationError, match="label_column"):
        ComputePanelSpec(feature_columns=("feature",), label_column="feature")
    with pytest.raises(ValidationError, match="timestamp or asset"):
        ComputePanelSpec(feature_columns=("timestamp",), label_column="label")

    spec = ComputePanelSpec(feature_columns=("feature",), label_column="label")
    frame = _compute_frame()
    assert validate_compute_panel(frame.iloc[::-1], spec)["asset"].tolist() == [
        "A",
        "B",
    ]
    with pytest.raises(TypeError, match="frame must be"):
        validate_compute_panel(object(), spec)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="spec must be"):
        validate_compute_panel(frame, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="panel is missing columns"):
        validate_compute_panel(frame.drop(columns=["feature"]), spec)
    with pytest.raises(ValueError, match="missing timestamps"):
        validate_compute_panel(
            frame.assign(timestamp=[pd.NaT, pd.Timestamp("2024-01-02")]),
            spec,
        )
    with pytest.raises(ValueError, match="missing assets"):
        validate_compute_panel(frame.assign(asset=["A", None]), spec)
    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(pd.concat([frame, frame.iloc[[0]]]), spec)
    with pytest.raises(TypeError, match="must be numeric"):
        validate_compute_panel(frame.assign(feature=["x", "y"]), spec)


def test_portfolio_helper_validation_and_selection_edges() -> None:
    panel = _compute_frame().assign(segment=["s1", "s2"], risk=[1.0, 2.0])
    spec = ComputePanelSpec(
        feature_columns=("feature", "feature_alt"),
        label_column="label",
    )

    _validate_segment_columns(panel, ("segment",))
    with pytest.raises(ValueError, match="segment columns are missing"):
        _validate_segment_columns(panel, ("missing",))
    _validate_weight_columns(
        panel,
        WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
    )
    with pytest.raises(ValueError, match="risk column 'missing' is missing"):
        _validate_weight_columns(
            panel,
            WeightConfig(
                method="inverse_vol_scaled_quantile",
                risk_column="missing",
            ),
        )
    with pytest.raises(TypeError, match="risk column 'risk' must be numeric"):
        _validate_weight_columns(
            panel.assign(risk=["x", "y"]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )
    with pytest.raises(ValueError, match="risk column 'risk' must be positive"):
        _validate_weight_columns(
            panel.assign(risk=[1.0, 0.0]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )

    assert _risk_values(panel, WeightConfig(method="equal_weight_quantile")) is None
    assert _risk_values(
        panel,
        WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
    ).tolist() == [1.0, 2.0]
    assert _select_features(spec, None) == ("feature", "feature_alt")
    assert _select_features(spec, ("feature_alt",)) == ("feature_alt",)
    with pytest.raises(ValueError, match="portfolio features are not in spec"):
        _select_features(spec, ("missing",))
    assert _select_feature(spec, None) == "feature"
    with pytest.raises(ValueError, match="feature_column"):
        _select_feature(spec, "missing")


def test_portfolio_helper_costs_signals_and_group_keys() -> None:
    simple = _cost_components(
        buy_turnover=1.0,
        sell_turnover=1.0,
        turnover=2.0,
        weight_config=WeightConfig(cost_bps=10),
        cost_config=CostConfig(),
    )
    detailed = _cost_components(
        buy_turnover=1.0,
        sell_turnover=2.0,
        turnover=3.0,
        weight_config=WeightConfig(cost_bps=99),
        cost_config=CostConfig(
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=1,
            spread_bps=2,
            impact_model="custom",
        ),
    )

    assert simple["cost_model"] == "scalar_cost_bps"
    assert simple["total_cost"] == pytest.approx(0.002)
    assert _uses_detailed_cost_config(CostConfig(impact_model="custom")) is True
    assert detailed["cost_model"] == "custom"
    assert detailed["total_cost"] == pytest.approx(0.001 + 0.004 + 0.0003 + 0.0006)
    assert _direct_signal(pd.Series([1.0, 2.0]), "long_low").tolist() == [-1.0, -2.0]
    assert _direct_signal_array(np.array([1.0]), "long_high").tolist() == [1.0]
    with pytest.raises(ValueError, match="direction"):
        _direct_signal(pd.Series([1.0]), "bad")
    with pytest.raises(ValueError, match="direction"):
        _direct_signal_array(np.array([1.0]), "bad")
    assert _selection_count(3, 0.1) == 1
    assert _normalize_group_key("feature", ["feature"]) == {"feature": "feature"}
    assert _normalize_group_key(("s1", "feature"), ["segment", "feature"]) == {
        "segment": "s1",
        "feature": "feature",
    }


def test_dataframe_tradability_helper_edges() -> None:
    spec = ComputePanelSpec(feature_columns=("feature",), label_column="label")
    frame = _compute_frame().assign(
        eligible=[True, False],
        can_enter=[True, False],
        can_exit=[False, True],
    )
    tradability = TradabilityConfig(
        eligibility_column="eligible",
        can_enter_column="can_enter",
        can_exit_column="can_exit",
    )

    constraints._validate_tradability_columns(frame, tradability)
    with pytest.raises(ValueError, match="tradability columns are missing"):
        constraints._validate_tradability_columns(
            frame,
            TradabilityConfig(can_enter_column="missing"),
        )
    with pytest.raises(TypeError, match="must be boolean"):
        constraints._validate_tradability_columns(
            frame.assign(can_enter=[1, 0]),
            TradabilityConfig(can_enter_column="can_enter"),
        )

    assert len(constraints._filter_eligible_assets(frame, tradability)) == 1
    assert len(
        constraints._filter_new_entry_blocked_assets(
            frame,
            spec=spec,
            tradability_config=tradability,
            previous_positions={"B": 1.0},
        )
    ) == 2
    counts = constraints._tradability_diagnostic_counts(
        timestamp_frame=frame,
        feature="feature",
        spec=spec,
        tradability_config=tradability,
    )
    assert counts["entry_blocked_count"] == 0
    context = constraints._build_tradability_context(
        timestamp_frame=frame,
        spec=spec,
        tradability_config=tradability,
    )
    assert context["entry_checks_required"] is True
    assert context["exit_checks_required"] is True
    assert constraints._is_exit_blocked(
        asset="A",
        target_position=0.0,
        previous_position=1.0,
        can_exit_flags={"A": False},
        tradability_config=tradability,
    )
    assert constraints._is_entry_blocked(
        asset="B",
        target_position=1.0,
        previous_position=0.0,
        can_enter_flags={"B": False},
        tradability_config=tradability,
    )


def _compute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["A", "B"],
            "feature": [1.0, 2.0],
            "feature_alt": [2.0, 1.0],
            "label": [0.1, 0.2],
        }
    )
