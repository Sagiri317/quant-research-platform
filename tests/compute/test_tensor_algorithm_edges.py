from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.compute.tensor import constraints
from quant_research_platform.compute.tensor.conversion import (
    panel_to_tensor,
    panel_to_tensor_batch,
)
from quant_research_platform.compute.tensor.cross_section import (
    compute_grouped_label_values_tensor,
    compute_ic_and_grouped_label_values_tensor,
    compute_ic_by_timestamp_tensor,
    summarize_group_spread_tensor,
)
from quant_research_platform.compute.tensor.exposure import (
    exposure_record_from_arrays,
    summarize_tensor_exposure_from_path,
)
from quant_research_platform.compute.tensor.portfolio_helpers import (
    filter_panel_group,
    leg_weight_values,
)
from quant_research_platform.compute.tensor.portfolio_returns import (
    _long_short_record_from_arrays,
    build_path_records_from_arrays,
    calculate_tensor_long_short,
)
from quant_research_platform.compute.tensor.turnover import (
    _current_position_state,
    _path_turnover_and_cost_records,
    calculate_tensor_turnover_and_cost,
    path_current_state,
    path_turnover_and_cost_records_from_arrays,
    position_map_from_arrays,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


def test_tensor_conversion_and_cross_section_outputs() -> None:
    panel = _panel()
    spec = ComputePanelSpec(
        signal_columns=("factor", "factor_alt"),
        label_column="forward_return",
    )

    tensor = panel_to_tensor(panel, spec, value_dtype="float32")
    assert tensor.signal_values.shape == (2, 3, 2)
    assert tensor.label_values.shape == (2, 3)
    assert tensor.valid_mask.shape == (2, 3, 2)

    ic = compute_ic_by_timestamp_tensor(tensor)
    combined_ic, grouped = compute_ic_and_grouped_label_values_tensor(
        tensor,
        n_groups=3,
    )
    grouped_only = compute_grouped_label_values_tensor(tensor, n_groups=3)
    spread = summarize_group_spread_tensor(grouped)

    assert len(ic) == 4
    assert len(combined_ic) == 4
    assert not grouped.empty
    assert grouped_only.equals(grouped)
    assert {
        "signal",
        "top_group_mean_label_value",
        "bottom_group_mean_label_value",
    }.issubset(spread.columns)


def test_tensor_batch_conversion_validates_columns_and_dtype() -> None:
    batch = panel_to_tensor_batch(
        _panel(),
        timestamp_column="timestamp",
        asset_column="asset",
        signal_columns=("factor",),
        label_columns=("forward_return",),
        value_dtype="float64",
    )
    assert batch.signal_values.shape == (2, 3, 1)
    assert batch.label_values.shape == (2, 3, 1)

    with pytest.raises(ValueError, match="value_dtype"):
        panel_to_tensor_batch(
            _panel(),
            timestamp_column="timestamp",
            asset_column="asset",
            signal_columns=("factor",),
            label_columns=("forward_return",),
            value_dtype="int64",
        )
    with pytest.raises(ValueError, match="signal_columns must contain"):
        panel_to_tensor_batch(
            _panel(),
            timestamp_column="timestamp",
            asset_column="asset",
            signal_columns=(),
            label_columns=("forward_return",),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        panel_to_tensor_batch(
            _panel(),
            timestamp_column="timestamp",
            asset_column="asset",
            signal_columns=("factor",),
            label_columns=("factor",),
        )
    with pytest.raises(TypeError, match="must be numeric"):
        panel_to_tensor_batch(
            _panel().assign(text="x"),
            timestamp_column="timestamp",
            asset_column="asset",
            signal_columns=("text",),
            label_columns=("forward_return",),
        )


def test_tensor_portfolio_helpers_cover_weight_methods_and_errors() -> None:
    assets = np.array([0, 1, 2])
    values = np.array([1.0, 2.0, 4.0])

    equal, reason = leg_weight_values(
        asset_indices=assets,
        signal_values=values,
        risk_values=None,
        signal="factor",
        weight_config=WeightConfig(method="equal_weight_quantile"),
        sign=1.0,
    )
    assert equal.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert reason == ""

    ranked, _ = leg_weight_values(
        asset_indices=assets,
        signal_values=values,
        risk_values=None,
        signal="factor",
        weight_config=WeightConfig(method="rank_weighted_quantile"),
        sign=-1.0,
    )
    assert ranked.sum() == pytest.approx(-1.0)
    assert abs(ranked[0]) > abs(ranked[1]) > abs(ranked[2])

    zscore, fallback = leg_weight_values(
        asset_indices=assets,
        signal_values=np.array([1.0, 1.0, 1.0]),
        risk_values=None,
        signal="factor",
        weight_config=WeightConfig(method="zscore_weighted_quantile"),
        sign=1.0,
    )
    assert zscore.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert fallback == "equal_weight_fallback"

    inverse, _ = leg_weight_values(
        asset_indices=assets,
        signal_values=values,
        risk_values=np.array([1.0, 2.0, 4.0]),
        signal="factor",
        weight_config=WeightConfig(
            method="inverse_vol_scaled_quantile",
            risk_column="risk",
        ),
        sign=1.0,
    )
    assert inverse.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="risk_column values"):
        leg_weight_values(
            asset_indices=assets,
            signal_values=values,
            risk_values=np.array([1.0, 0.0, 1.0]),
            signal="factor",
            weight_config=WeightConfig(
                method="inverse_vol_scaled_quantile",
                risk_column="risk",
            ),
            sign=1.0,
        )


def test_tensor_constraints_rank_and_apply_tradability() -> None:
    timestamp_input = SimpleNamespace(
        eligible=np.array([True, True, False]),
        can_enter=np.array([True, False, True]),
        can_exit=np.array([False, True, True]),
    )
    ranked = constraints.ranked_asset_indices(
        assets=("A", "B", "C"),
        signal_values=np.array([1.0, 2.0, np.nan]),
        weight_config=WeightConfig(),
        tradability_config=TradabilityConfig(blocked_entry_policy="redistribute"),
        timestamp_input=timestamp_input,
        previous_positions={"B": -0.5},
    )
    assert ranked.tolist() == [0, 1]

    adjusted = constraints.apply_array_tradability_constraints(
        positions=np.array([1.0, -1.0, 0.5]),
        assets=("A", "B", "C"),
        timestamp_input=timestamp_input,
        tradability_config=TradabilityConfig(blocked_exit_policy="carry_forward"),
        previous_positions={"A": -0.5, "B": -0.5},
    )
    assert adjusted.tolist() == pytest.approx([-0.5, -1.0, 0.5])

    with pytest.raises(ValueError, match="exit is blocked"):
        constraints.apply_array_tradability_constraints(
            positions=np.array([1.0, -1.0, 0.0]),
            assets=("A", "B", "C"),
            timestamp_input=timestamp_input,
            tradability_config=TradabilityConfig(blocked_exit_policy="error"),
            previous_positions={"A": -0.5},
        )


def test_tensor_turnover_cost_and_state_helpers() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "asset": ["A", "A"],
            "signal": ["factor", "factor"],
            "position": [1.0, -0.5],
            "segment": ["s1", "s1"],
        }
    )
    turnover, cost = calculate_tensor_turnover_and_cost(
        positions,
        WeightConfig(),
        CostConfig(slippage_bps=10),
        segment_columns=("segment",),
    )
    assert turnover["turnover"].tolist() == [1.0, 1.5]
    assert cost["total_cost"].round(6).tolist() == [0.001, 0.0015]

    empty_turnover, empty_cost = calculate_tensor_turnover_and_cost(
        positions.iloc[0:0],
        WeightConfig(),
    )
    assert empty_turnover.empty
    assert empty_cost.empty

    turnover_record, cost_record = _path_turnover_and_cost_records(
        context={"segment": "s1"},
        timestamp=pd.Timestamp("2024-01-02"),
        signal="factor",
        previous_positions={"A": -0.5},
        current_positions={"A": 1.0, "B": -0.25},
        weight_config=WeightConfig(),
        cost_config=CostConfig(slippage_bps=10),
        segment_columns=("segment",),
    )
    assert turnover_record["turnover"] == pytest.approx(1.75)
    assert cost_record["total_cost"] == pytest.approx(0.00175)

    array_turnover, _array_cost = path_turnover_and_cost_records_from_arrays(
        context={},
        timestamp=pd.Timestamp("2024-01-03"),
        signal="factor",
        previous_values=np.array([0.5, 0.0]),
        current_values=np.array([0.0, 1.0]),
        previous_positions={"C": 0.25},
        asset_to_offset={"A": 0, "B": 1},
        weight_config=WeightConfig(),
        cost_config=CostConfig(),
    )
    assert array_turnover["turnover"] == pytest.approx(1.75)

    offsets, state = path_current_state(
        assets=("A", "B"),
        position_values=np.array([0.0, -1.0]),
    )
    assert offsets.tolist() == [1]
    assert state["position_count"] == 1
    current, current_state = _current_position_state(
        assets=("A", "B"),
        positions=np.array([0.5, 0.0]),
        nonzero_offsets=np.array([0]),
    )
    assert current == {"A": 0.5}
    assert current_state["net_exposure"] == pytest.approx(0.5)
    assert position_map_from_arrays(
        assets=("A", "B"),
        positions=np.array([0.0, -0.25]),
    ) == {"B": -0.25}


def test_tensor_portfolio_returns_and_exposure_helpers() -> None:
    spec = ComputePanelSpec(
        signal_columns=("factor",),
        label_column="forward_return",
    )
    panel = _panel()
    positions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset": ["A", "C", "A", "C"],
            "signal": ["factor"] * 4,
            "position": [1.0, -1.0, 0.5, -0.5],
        }
    )
    turnover = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "signal": ["factor", "factor"],
            "turnover": [2.0, 1.0],
        }
    )
    cost = turnover.assign(total_cost=[0.002, 0.001])

    returns = calculate_tensor_long_short(
        panel=panel,
        spec=spec,
        positions=positions,
        turnover=turnover,
        cost_breakdown=cost,
        return_source_column="forward_return",
    )
    assert returns["net_portfolio_return"].round(6).tolist() == [-0.022, 0.009]
    assert calculate_tensor_long_short(
        panel=panel,
        spec=spec,
        positions=positions.iloc[0:0],
        turnover=turnover,
        cost_breakdown=cost,
        return_source_column="forward_return",
    ).empty

    assert (
        _long_short_record_from_arrays(
            context={},
            timestamp=pd.Timestamp("2024-01-02"),
            signal="factor",
            label="forward_return",
            portfolio_mode="long_short",
            position_values=np.array([1.0, 0.0]),
            label_values=np.array([0.01, 0.02]),
            turnover=1.0,
            cost=0.0,
            segment_columns=(),
        )
        is None
    )
    path = build_path_records_from_arrays(
        context={"segment": "s1"},
        timestamp=pd.Timestamp("2024-01-02"),
        signal="factor",
        label="forward_return",
        assets=("A", "B"),
        position_values=np.array([1.0, -1.0]),
        label_values=np.array([0.01, -0.01]),
        previous_values=np.array([0.0, 0.0]),
        previous_positions={},
        asset_to_offset={"A": 0, "B": 1},
        weight_config=WeightConfig(),
        cost_config=CostConfig(slippage_bps=10),
        segment_columns=("segment",),
    )
    assert path[1]["turnover"] == pytest.approx(2.0)
    assert path[3] is not None
    assert path[4]["gross_exposure"] == pytest.approx(2.0)

    exposure = summarize_tensor_exposure_from_path(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "signal": ["factor"],
                "gross_exposure": [2.0],
                "net_exposure": [0.0],
                "position_count": [2],
            }
        ),
        returns,
    )
    assert exposure.loc[0, "mean_position_count"] == pytest.approx(2.0)
    assert summarize_tensor_exposure_from_path(
        pd.DataFrame(),
        returns,
    ).empty
    assert exposure_record_from_arrays(
        context={"segment": "s1"},
        timestamp="2024-01-02",
        signal="factor",
        exposure_values={"gross_exposure": 1.0},
        segment_columns=("segment",),
    )["segment"] == "s1"


def test_tensorfilter_panel_group_uses_segments() -> None:
    panel = _panel().assign(segment=["s1", "s2", "s1", "s1", "s2", "s1"])

    filtered = filter_panel_group(
        panel=panel,
        segment_columns=("segment",),
        group_values={"segment": "s1"},
    )

    assert set(filtered["segment"]) == {"s1"}
    assert filter_panel_group(
        panel=panel,
        segment_columns=(),
        group_values={},
    ).equals(panel)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "factor_alt": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )
