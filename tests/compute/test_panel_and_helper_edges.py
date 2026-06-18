from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_research_platform.compute.dataframe import constraints
from quant_research_platform.compute.dataframe.construction import (
    build_weighted_positions,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _cost_components,
    _normalize_group_key,
    _risk_values,
    _select_signal,
    _select_signals,
    _selection_count,
    _uses_detailed_cost_config,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


def test_compute_panel_spec_and_frame_validation_edges() -> None:
    with pytest.raises(ValidationError, match="signal_columns"):
        ComputePanelSpec(signal_columns=(), label_column="label")
    with pytest.raises(ValidationError, match="duplicates"):
        ComputePanelSpec(signal_columns=("signal", "signal"), label_column="label")
    with pytest.raises(ValidationError, match="label_column"):
        ComputePanelSpec(signal_columns=("signal",), label_column="signal")
    with pytest.raises(ValidationError, match="timestamp or asset"):
        ComputePanelSpec(signal_columns=("timestamp",), label_column="label")

    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")
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
        validate_compute_panel(frame.drop(columns=["signal"]), spec)
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
        validate_compute_panel(frame.assign(signal=["x", "y"]), spec)


def test_validate_compute_panel_uses_segment_aware_uniqueness() -> None:
    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")
    segmented = pd.DataFrame(
        {
            "segment": ["s1", "s2"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["A", "A"],
            "signal": [1.0, 2.0],
            "label": [0.1, 0.2],
        }
    )

    result = validate_compute_panel(segmented, spec, segment_columns=("segment",))

    assert result["segment"].tolist() == ["s1", "s2"]
    with pytest.raises(ValueError, match="unique segment-timestamp-asset keys"):
        validate_compute_panel(
            segmented.assign(segment=["s1", "s1"]),
            spec,
            segment_columns=("segment",),
        )
    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(segmented, spec)


def test_portfolio_helper_validation_and_selection_edges() -> None:
    panel = _compute_frame().assign(segment=["s1", "s2"], risk=[1.0, 2.0])
    spec = ComputePanelSpec(
        signal_columns=("signal", "signal_alt"),
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
    assert _select_signals(spec, None) == ("signal", "signal_alt")
    assert _select_signals(spec, ("signal_alt",)) == ("signal_alt",)
    with pytest.raises(ValueError, match="portfolio signals are not in spec"):
        _select_signals(spec, ("missing",))
    assert _select_signal(spec, None) == "signal"
    with pytest.raises(ValueError, match="signal_column"):
        _select_signal(spec, "missing")


def test_portfolio_helper_costs_signals_and_group_keys() -> None:
    simple = _cost_components(
        buy_turnover=1.0,
        sell_turnover=1.0,
        turnover=2.0,
        cost_config=CostConfig(),
    )
    detailed = _cost_components(
        buy_turnover=1.0,
        sell_turnover=2.0,
        turnover=3.0,
        cost_config=CostConfig(
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=1,
            spread_bps=2,
            impact_model="custom",
        ),
    )

    assert simple["cost_model"] == "zero_cost"
    assert simple["total_cost"] == pytest.approx(0.0)
    assert _uses_detailed_cost_config(CostConfig(impact_model="custom")) is True
    assert detailed["cost_model"] == "custom"
    assert detailed["total_cost"] == pytest.approx(0.001 + 0.004 + 0.0003 + 0.0006)
    assert _selection_count(3, 0.1) == 1
    assert _normalize_group_key("signal", ["signal"]) == {"signal": "signal"}
    assert _normalize_group_key(("s1", "signal"), ["segment", "signal"]) == {
        "segment": "s1",
        "signal": "signal",
    }


def test_rank_weighted_short_leg_weights_more_extreme_signals_more_heavily() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"] * 4),
            "asset": ["A", "B", "C", "D"],
            "signal": [0.0, 1.0, 2.0, 3.0],
            "label": [0.0, 0.0, 0.0, 0.0],
        }
    )
    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(
            method="rank_weighted_quantile",
            top_quantile=0.5,
            bottom_quantile=0.5,
        ),
    )
    weights = positions.set_index("asset")["position"]
    assert abs(weights["A"]) > abs(weights["B"])
    assert abs(weights["D"]) > abs(weights["C"])


def test_dataframe_tradability_helper_edges() -> None:
    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")
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
        signal="signal",
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
            "signal": [1.0, 2.0],
            "signal_alt": [2.0, 1.0],
            "label": [0.1, 0.2],
        }
    )
