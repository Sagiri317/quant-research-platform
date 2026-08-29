from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from quant_research_platform.compute._portfolio_math import (
    _uses_detailed_cost_config,
    cost_components,
    select_signals,
    selection_count,
    validate_tradability_columns,
    validate_weight_columns,
)
from quant_research_platform.compute.dataframe import constraints, position_math
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    risk_values,
)
from quant_research_platform.compute.grouping import normalize_group_key
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
    with pytest.raises(TypeError, match="must be numeric"):
        validate_compute_panel(frame.assign(signal=["1.0", "2.0"]), spec)
    with pytest.raises(ValueError, match="finite numeric"):
        validate_compute_panel(frame.assign(signal=[float("inf"), 2.0]), spec)


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
    with pytest.raises(ValueError, match="segment column 'segment'.*invalid keys"):
        validate_compute_panel(
            segmented.assign(segment=["s1", None]),
            spec,
            segment_columns=("segment",),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        validate_compute_panel(segmented, spec, segment_columns=("timestamp",))
    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        validate_compute_panel(segmented, spec)


def test_portfolio_helper_validation_and_selection_edges() -> None:
    panel = _compute_frame().assign(segment=["s1", "s2"], risk=[1.0, 2.0])
    spec = ComputePanelSpec(
        signal_columns=("signal", "signal_alt"),
        label_column="label",
    )

    validate_weight_columns(
        panel,
        WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
    )
    with pytest.raises(ValueError, match="risk column 'missing' is missing"):
        validate_weight_columns(
            panel,
            WeightConfig(
                method="inverse_vol_scaled_quantile",
                risk_column="missing",
            ),
        )
    with pytest.raises(TypeError, match="risk column 'risk' must be numeric"):
        validate_weight_columns(
            panel.assign(risk=["x", "y"]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )
    with pytest.raises(TypeError, match="risk column 'risk' must be numeric"):
        validate_weight_columns(
            panel.assign(risk=["1.0", "2.0"]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )
    with pytest.raises(ValueError, match="risk column 'risk' must be finite numeric"):
        validate_weight_columns(
            panel.assign(risk=[1.0, float("inf")]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )
    with pytest.raises(ValueError, match="risk column 'risk' must be positive"):
        validate_weight_columns(
            panel.assign(risk=[1.0, 0.0]),
            WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
        )

    assert risk_values(panel, WeightConfig(method="equal_weight_quantile")) is None
    assert risk_values(
        panel,
        WeightConfig(method="inverse_vol_scaled_quantile", risk_column="risk"),
    ).tolist() == [1.0, 2.0]
    assert select_signals(spec, None) == ("signal", "signal_alt")
    assert select_signals(spec, ("signal_alt",)) == ("signal_alt",)
    with pytest.raises(ValueError, match="must not contain duplicates"):
        select_signals(spec, ("signal", "signal"))
    with pytest.raises(ValueError, match="portfolio signals are not in spec"):
        select_signals(spec, ("missing",))


def test_portfolio_helper_costs_signals_and_group_keys() -> None:
    simple = cost_components(
        buy_turnover=1.0,
        sell_turnover=1.0,
        turnover=2.0,
        cost_config=CostConfig(),
    )
    detailed = cost_components(
        buy_turnover=1.0,
        sell_turnover=2.0,
        turnover=3.0,
        cost_config=CostConfig(
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=1,
            spread_bps=2,
        ),
    )

    assert simple["cost_model"] == "zero_cost"
    assert simple["total_cost"] == pytest.approx(0.0)
    assert _uses_detailed_cost_config(CostConfig(spread_bps=2)) is True
    assert detailed["cost_model"] == "detailed_bps"
    assert detailed["total_cost"] == pytest.approx(0.001 + 0.004 + 0.0003 + 0.0006)
    assert selection_count(3, 0.1) == 1
    assert normalize_group_key("signal", ["signal"]) == {"signal": "signal"}
    assert normalize_group_key(("s1", "signal"), ["segment", "signal"]) == {
        "segment": "s1",
        "signal": "signal",
    }


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

    validate_tradability_columns(frame, tradability)
    with pytest.raises(ValueError, match="tradability columns are missing"):
        validate_tradability_columns(
            frame,
            TradabilityConfig(can_enter_column="missing"),
        )
    with pytest.raises(TypeError, match="must be boolean"):
        validate_tradability_columns(
            frame.assign(can_enter=[1, 0]),
            TradabilityConfig(can_enter_column="can_enter"),
        )
    with pytest.raises(ValueError, match="can_enter.*null"):
        validate_tradability_columns(
            frame.assign(can_enter=pd.Series([True, pd.NA], dtype="boolean")),
            TradabilityConfig(can_enter_column="can_enter"),
        )

    counts = constraints.tradability_diagnostic_counts(
        timestamp_frame=frame,
        signal="signal",
        spec=spec,
        tradability_config=tradability,
    )
    assert counts["entry_blocked_count"] == 0
    context = constraints.build_tradability_context(
        timestamp_frame=frame,
        spec=spec,
        tradability_config=tradability,
    )
    assert context["entry_checks_required"] is True
    assert context["exit_checks_required"] is True


@pytest.mark.parametrize(
    ("previous", "target", "can_buy", "can_sell", "expected"),
    (
        (0.0, 1.0, False, True, 0.0),
        (0.0, 1.0, True, False, 1.0),
        (0.5, 1.0, False, True, 0.5),
        (1.0, 0.5, True, False, 1.0),
        (1.0, 0.0, True, False, 1.0),
        (1.0, 0.0, False, True, 0.0),
        (0.0, -1.0, True, False, 0.0),
        (0.0, -1.0, False, True, -1.0),
        (-0.5, -1.0, True, False, -0.5),
        (-1.0, -0.5, False, True, -1.0),
        (-1.0, 0.0, False, True, -1.0),
        (-1.0, 0.0, True, False, 0.0),
        (1.0, -1.0, True, False, 1.0),
        (1.0, -1.0, False, True, -1.0),
        (-1.0, 1.0, False, True, -1.0),
        (-1.0, 1.0, True, False, 1.0),
    ),
)
def test_tradability_constraints_apply_real_buy_sell_permissions(
    previous: float,
    target: float,
    can_buy: bool,
    can_sell: bool,
    expected: float,
) -> None:
    frame = _compute_frame().iloc[[0]].assign(
        can_enter=can_buy,
        can_exit=can_sell,
    )
    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")
    config = TradabilityConfig(
        can_enter_column="can_enter",
        can_exit_column="can_exit",
        blocked_exit_policy="carry_forward",
    )

    adjusted = constraints.apply_tradability_constraints(
        positions={"A": target},
        timestamp_frame=frame,
        spec=spec,
        tradability_config=config,
        previous_positions={"A": previous},
        tradability_context={
            "entry_checks_required": True,
            "exit_checks_required": True,
            "can_enter_flags": {"A": can_buy},
            "can_exit_flags": {"A": can_sell},
        },
    )

    assert adjusted == {"A": expected}


def test_tradability_constraints_preserve_blocked_positions_and_reject_exit() -> None:
    frame = _compute_frame().assign(can_enter=[False, False], can_exit=[False, True])
    spec = ComputePanelSpec(signal_columns=("signal",), label_column="label")
    carry = TradabilityConfig(
        can_enter_column="can_enter",
        can_exit_column="can_exit",
        blocked_exit_policy="carry_forward",
    )
    context = {
        "entry_checks_required": True,
        "exit_checks_required": True,
        "can_enter_flags": {"A": False, "B": False, "C": False, "D": False},
        "can_exit_flags": {"A": False, "B": True, "C": True, "D": False},
    }

    adjusted = constraints.apply_tradability_constraints(
        positions={"A": 0.0, "B": 1.0, "C": -1.0},
        timestamp_frame=frame,
        spec=spec,
        tradability_config=carry,
        previous_positions={"A": 1.0, "C": 1.0, "D": 1.0},
        tradability_context=context,
    )

    assert adjusted == {"A": 1.0, "B": 0.0, "C": -1.0, "D": 1.0}
    with pytest.raises(ValueError, match="exit is blocked"):
        constraints.apply_tradability_constraints(
            positions={"A": 0.0},
            timestamp_frame=frame,
            spec=spec,
            tradability_config=TradabilityConfig(
                can_exit_column="can_exit",
                blocked_exit_policy="error",
            ),
            previous_positions={"A": 1.0},
            tradability_context={
                "entry_checks_required": False,
                "exit_checks_required": True,
                "can_enter_flags": None,
                "can_exit_flags": {"A": False},
            },
        )


@pytest.mark.parametrize(
    ("method", "risk", "sign"),
    (
        ("equal_weight_quantile", None, 1.0),
        ("rank_weighted_quantile", None, -1.0),
        ("zscore_weighted_quantile", None, 1.0),
        ("signal_abs_weighted_quantile", None, 1.0),
        ("inverse_vol_scaled_quantile", np.array([1.0, 2.0]), 1.0),
    ),
)
def test_leg_weight_methods_normalize_one_signed_leg(
    method: str,
    risk: np.ndarray | None,
    sign: float,
) -> None:
    weights, fallback = position_math._leg_weights(
        asset_values=np.array(["A", "B"]),
        signal_values=np.array([1.0, 3.0]),
        risk_values=risk,
        signal="signal",
        weight_config=WeightConfig(
            method=method,
            risk_column="risk" if risk is not None else None,
        ),
        sign=sign,
    )

    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)
    assert all(np.sign(value) == sign for value in weights.values())
    assert fallback == ""


def test_zero_zscore_leg_uses_equal_weight_fallback() -> None:
    weights, fallback = position_math._leg_weights(
        asset_values=np.array(["A", "B"]),
        signal_values=np.array([1.0, 1.0]),
        risk_values=None,
        signal="signal",
        weight_config=WeightConfig(method="zscore_weighted_quantile"),
        sign=1.0,
    )

    assert weights == {"A": 0.5, "B": 0.5}
    assert fallback == "equal_weight_fallback"


def test_tiny_nonconstant_zscore_leg_does_not_falsely_fallback() -> None:
    weights, fallback = position_math._leg_weights(
        asset_values=np.array(["A", "B", "C"]),
        signal_values=np.array([1e-320, 2e-320, 3e-320]),
        risk_values=None,
        signal="signal",
        weight_config=WeightConfig(method="zscore_weighted_quantile"),
        sign=1.0,
    )

    assert weights == {"A": 0.5, "B": 0.0, "C": 0.5}
    assert fallback == ""


def test_position_validation_rejects_shape_and_key_ambiguity() -> None:
    valid = pd.DataFrame(
        {
            "timestamp": ["2026-01-02", "2026-01-01"],
            "asset": ["B", "A"],
            "signal": ["factor", "factor"],
            "position": [0.5, -0.5],
        }
    )
    assert position_math.validate_positions(valid)["asset"].tolist() == ["A", "B"]
    with pytest.raises(TypeError, match="pandas DataFrame"):
        position_math.validate_positions(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing columns"):
        position_math.validate_positions(valid.drop(columns="signal"))
    with pytest.raises(TypeError, match="position must be numeric"):
        position_math.validate_positions(valid.assign(position=["x", "y"]))
    with pytest.raises(TypeError, match="position must be numeric"):
        position_math.validate_positions(valid.assign(position=["0.5", "-0.5"]))
    with pytest.raises(ValueError, match="position must be finite numeric"):
        position_math.validate_positions(valid.assign(position=[0.5, float("inf")]))
    with pytest.raises(ValueError, match="unique"):
        position_math.validate_positions(pd.concat([valid, valid.iloc[[0]]]))


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
