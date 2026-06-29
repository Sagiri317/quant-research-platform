from __future__ import annotations

import pandas as pd
import pytest

import quant_research_platform.compute.tensor.construction as tensor_construction
import quant_research_platform.compute.tensor.exposure as tensor_exposure
import quant_research_platform.compute.tensor.turnover as tensor_turnover
from quant_research_platform.compute.dataframe import (
    constraints as dataframe_constraints,
)
from quant_research_platform.compute.dataframe import (
    construction as dataframe_construction,
)
from quant_research_platform.compute.dataframe import (
    exposure as dataframe_exposure,
)
from quant_research_platform.compute.dataframe import (
    performance as dataframe_performance,
)
from quant_research_platform.compute.dataframe import (
    portfolio_returns as dataframe_portfolio_returns,
)
from quant_research_platform.compute.dataframe import (
    position_math as dataframe_position_math,
)
from quant_research_platform.compute.dataframe import (
    turnover as dataframe_turnover,
)
from quant_research_platform.compute.dataframe.construction import (
    build_long_short_positions,
    build_weighted_positions,
)
from quant_research_platform.compute.dataframe.exposure import summarize_exposure
from quant_research_platform.compute.dataframe.performance import (
    calculate_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    simulate_long_short_returns,
    simulate_weighted_long_short_returns,
)
from quant_research_platform.compute.dataframe.turnover import (
    calculate_cost_breakdown,
    calculate_turnover,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.compute.tensor import (
    portfolio_returns as tensor_portfolio_returns,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


def test_portfolio_split_modules_are_direct_owners() -> None:
    assert dataframe_construction.build_weighted_positions is build_weighted_positions
    assert dataframe_turnover.calculate_turnover is calculate_turnover
    assert dataframe_exposure.summarize_exposure is summarize_exposure
    assert dataframe_performance.summarize_performance is summarize_performance
    assert hasattr(tensor_construction, "build_tensor_weighted_positions")
    assert hasattr(tensor_turnover, "calculate_tensor_turnover_and_cost")
    assert hasattr(tensor_exposure, "summarize_tensor_exposure_from_path")
    assert hasattr(tensor_portfolio_returns, "calculate_tensor_long_short")


def _ranking_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-02",
            ],
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.03, 0.02, 0.01],
        }
    )


def test_build_long_short_positions_creates_equal_weight_legs() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions = build_long_short_positions(_ranking_panel(), spec, top_quantile=1 / 3)

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }


def test_build_weighted_positions_outputs_nonzero_rows_only() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        _ranking_panel(),
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    assert (positions["position"] != 0).all()
    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }


def test_tensor_weighted_positions_builds_artifacts_from_execution_path() -> None:
    panel = _ranking_panel().assign(
        timestamp=lambda frame: pd.to_datetime(frame["timestamp"])
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    artifacts = tensor_construction.build_tensor_portfolio_artifacts(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )
    positions, diagnostics = tensor_construction.build_tensor_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )

    assert isinstance(artifacts, tensor_construction.TensorPortfolioArtifacts)
    assert list(artifacts.positions.columns) == [
        "timestamp",
        "asset",
        "signal",
        "position",
    ]
    assert list(artifacts.diagnostics["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    assert len(artifacts.turnover) == 2
    assert {"timestamp", "signal", "turnover"}.issubset(artifacts.turnover.columns)
    assert len(artifacts.cost_breakdown) == 2
    assert {"timestamp", "signal", "total_cost"}.issubset(
        artifacts.cost_breakdown.columns
    )
    assert len(artifacts.returns) == 2
    assert {"timestamp", "signal", "net_portfolio_return"}.issubset(
        artifacts.returns.columns
    )
    assert len(artifacts.exposure_path) == 2
    assert {"timestamp", "signal", "gross_exposure", "net_exposure"}.issubset(
        artifacts.exposure_path.columns
    )
    first = artifacts.positions[
        artifacts.positions["timestamp"] == pd.Timestamp("2026-01-01")
    ]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }
    assert positions.equals(artifacts.positions)
    assert diagnostics.equals(artifacts.diagnostics)


def test_tensor_weighted_positions_position_only_skips_return_source() -> None:
    panel = _ranking_panel().assign(
        timestamp=lambda frame: pd.to_datetime(frame["timestamp"])
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, diagnostics = tensor_construction.build_tensor_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    assert not positions.empty
    assert diagnostics["timestamp"].tolist() == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]


def test_tensor_weighted_positions_accepts_string_timestamps_like_dataframe() -> None:
    panel = _ranking_panel()
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    weight_config = WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3)

    dataframe_positions, _ = build_weighted_positions(panel, spec, weight_config)
    tensor_positions, _ = tensor_construction.build_tensor_weighted_positions(
        panel,
        spec,
        weight_config,
        return_source_column="forward_return",
    )

    assert dataframe_positions["timestamp"].map(type).eq(pd.Timestamp).all()
    assert tensor_positions["timestamp"].map(type).eq(pd.Timestamp).all()


def test_tensor_weighted_positions_rejects_duplicate_keys_like_dataframe() -> None:
    panel = pd.concat([_ranking_panel(), _ranking_panel().iloc[[0]]], ignore_index=True)
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    weight_config = WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3)

    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        build_weighted_positions(panel, spec, weight_config)
    with pytest.raises(ValueError, match="unique timestamp-asset keys"):
        tensor_construction.build_tensor_weighted_positions(
            panel,
            spec,
            weight_config,
            return_source_column="forward_return",
        )


def test_build_weighted_positions_supports_long_only_top_leg() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, diagnostics = build_weighted_positions(
        _ranking_panel(),
        spec,
        WeightConfig(
            portfolio_mode="long_only",
            top_quantile=1 / 3,
            bottom_quantile=1 / 3,
        ),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {"C": 1.0}
    assert (positions["position"] >= 0).all()
    assert diagnostics["short_count"].tolist() == [0, 0]


def test_build_weighted_positions_long_only_allows_full_top_quantile() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, diagnostics = build_weighted_positions(
        _ranking_panel(),
        spec,
        WeightConfig(
            portfolio_mode="long_only",
            top_quantile=1.0,
            bottom_quantile=1.0,
        ),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert set(first["asset"]) == {"A", "B", "C"}
    assert (positions["position"] > 0).all()
    assert diagnostics["short_count"].tolist() == [0, 0]


def test_build_weighted_positions_keeps_stable_tie_order() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 3,
            "asset": ["A", "B", "C"],
            "factor": [1.0, 1.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03],
        }
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    assert positions.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }


def test_weighted_position_ranking_avoids_dataframe_signal_sort(
    monkeypatch,
) -> None:
    original_sort_values = pd.DataFrame.sort_values

    def fail_directed_signal_sort(frame, by=None, *args, **kwargs):
        if by == "_directed_signal":
            raise AssertionError("ranking path should avoid DataFrame signal sort")
        return original_sort_values(frame, by, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sort_values", fail_directed_signal_sort)
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        _ranking_panel(),
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }


def test_weighted_positions_pass_sparse_targets_to_tradability(monkeypatch) -> None:
    original_apply = dataframe_position_math._apply_tradability_constraints

    def assert_sparse_positions(*, positions, **kwargs):
        assert len(positions) == 2
        return original_apply(positions=positions, **kwargs)

    monkeypatch.setattr(
        dataframe_position_math,
        "_apply_tradability_constraints",
        assert_sparse_positions,
    )
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"] * 6,
            "asset": ["A", "B", "C", "D", "E", "F"],
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "can_exit": [True] * 6,
        }
    )

    positions, _diagnostics = build_weighted_positions(
        panel,
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 6, bottom_quantile=1 / 6),
        tradability_config=TradabilityConfig(can_exit_column="can_exit"),
    )

    assert set(positions["asset"]) == {"A", "F"}


def test_build_weighted_positions_keeps_flat_timestamp_marker() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        _ranking_panel().iloc[:1],
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    assert len(positions) == 1
    assert positions.iloc[0]["timestamp"] == pd.Timestamp("2026-01-01")
    assert positions.iloc[0]["position"] == 0.0


def test_eligible_column_excludes_asset_from_selection_universe() -> None:
    panel = _ranking_panel().assign(
        eligible=[True, True, False, True, True, True],
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(eligibility_column="eligible"),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "B": 1.0,
    }


def test_weight_diagnostics_report_filter_counts() -> None:
    panel = _ranking_panel().assign(
        eligible=[True, True, False, True, True, True],
        can_enter=[True, False, False, True, True, True],
        can_exit=[True, True, True, True, False, True],
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    _positions, diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(
            eligibility_column="eligible",
            can_enter_column="can_enter",
            can_exit_column="can_exit",
        ),
    )

    first = diagnostics[diagnostics["timestamp"] == pd.Timestamp("2026-01-01")].iloc[0]
    assert first["universe_count"] == 3
    assert first["eligible_count"] == 2
    assert first["can_enter_count"] == 1
    assert first["can_exit_count"] == 3
    assert first["entry_blocked_count"] == 1
    assert first["exit_blocked_count"] == 0


def test_can_enter_column_blocks_new_positions_without_redistribution() -> None:
    panel = _ranking_panel().assign(
        can_enter=[True, True, False, True, True, True],
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(
            can_enter_column="can_enter",
            blocked_entry_policy="hold_cash",
        ),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
    }


def test_can_exit_column_carries_forward_blocked_old_positions() -> None:
    panel = _ranking_panel().assign(
        can_exit=[True, True, True, True, True, False],
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(can_exit_column="can_exit"),
    )

    second = positions[positions["timestamp"] == pd.Timestamp("2026-01-02")]
    assert dict(zip(second["asset"], second["position"], strict=True)) == {
        "A": 1.0,
        "C": 1.0,
    }


def test_tradability_ignores_previous_assets_missing_from_timestamp() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-02", "2026-01-02"],
            "asset": ["A", "B"],
            "factor": [1.0, 0.0],
            "forward_return": [0.01, -0.01],
            "can_exit": [False, False],
        }
    )
    positions = {"A": 1.0}

    adjusted = dataframe_constraints._apply_tradability_constraints(
        positions=positions,
        timestamp_frame=panel,
        spec=ComputePanelSpec(
            signal_columns=("factor",),
            label_column="forward_return",
        ),
        tradability_config=TradabilityConfig(can_exit_column="can_exit"),
        previous_positions={"C": -1.0},
    )

    assert adjusted == {"A": 1.0}


def test_tradability_constraints_avoid_asset_set_index(monkeypatch) -> None:
    original_set_index = pd.DataFrame.set_index

    def fail_asset_set_index(frame, keys, *args, **kwargs):
        normalized_keys = [keys] if isinstance(keys, str) else list(keys)
        if normalized_keys == ["asset"]:
            raise AssertionError("tradability path should avoid asset set_index")
        return original_set_index(frame, keys, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "set_index", fail_asset_set_index)
    panel = _ranking_panel().assign(
        can_enter=[True, True, False, True, True, True],
        can_exit=[True, True, True, True, True, False],
    )
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(
            can_enter_column="can_enter",
            can_exit_column="can_exit",
            blocked_entry_policy="hold_cash",
        ),
    )

    second = positions[positions["timestamp"] == pd.Timestamp("2026-01-02")]
    assert dict(zip(second["asset"], second["position"], strict=True)) == {
        "A": 1.0,
        "C": -1.0,
    }


def test_tradability_constraints_skip_asset_loop_when_all_flags_true(
    monkeypatch,
) -> None:
    def fail_block_check(*args, **kwargs):
        raise AssertionError("all-true tradability flags should skip asset checks")

    monkeypatch.setattr(dataframe_constraints, "_is_entry_blocked", fail_block_check)
    monkeypatch.setattr(dataframe_constraints, "_is_exit_blocked", fail_block_check)
    panel = _ranking_panel().assign(can_enter=True, can_exit=True)
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    positions, _diagnostics = build_weighted_positions(
        panel,
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        tradability_config=TradabilityConfig(
            can_enter_column="can_enter",
            can_exit_column="can_exit",
        ),
    )

    first = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    assert first.set_index("asset")["position"].to_dict() == {
        "A": -1.0,
        "C": 1.0,
    }


def test_tradability_constraints_adjust_positions_in_place() -> None:
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "asset": ["A", "B"],
            "factor": [1.0, 0.0],
            "forward_return": [0.01, -0.01],
            "can_enter": [True, False],
        }
    )
    positions = {"A": 1.0, "B": -1.0}

    adjusted = dataframe_constraints._apply_tradability_constraints(
        positions=positions,
        timestamp_frame=panel,
        spec=ComputePanelSpec(
            signal_columns=("factor",),
            label_column="forward_return",
        ),
        tradability_config=TradabilityConfig(can_enter_column="can_enter"),
        previous_positions={},
    )

    assert adjusted is positions
    assert adjusted == {"A": 1.0, "B": 0.0}


def test_tradability_constraints_skip_entry_check_for_zero_targets(
    monkeypatch,
) -> None:
    def fail_entry_check(*args, **kwargs):
        raise AssertionError("zero target positions should skip entry checks")

    monkeypatch.setattr(dataframe_constraints, "_is_entry_blocked", fail_entry_check)
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "asset": ["A", "B"],
            "factor": [1.0, 0.0],
            "forward_return": [0.01, -0.01],
            "can_enter": [False, False],
        }
    )
    positions = {"A": 0.0, "B": 0.0}

    adjusted = dataframe_constraints._apply_tradability_constraints(
        positions=positions,
        timestamp_frame=panel,
        spec=ComputePanelSpec(
            signal_columns=("factor",),
            label_column="forward_return",
        ),
        tradability_config=TradabilityConfig(can_enter_column="can_enter"),
        previous_positions={},
    )

    assert adjusted == {"A": 0.0, "B": 0.0}


def test_tradability_constraints_skip_entry_check_for_same_sign_positions(
    monkeypatch,
) -> None:
    def fail_entry_check(*args, **kwargs):
        raise AssertionError("same-sign existing positions should skip entry checks")

    monkeypatch.setattr(dataframe_constraints, "_is_entry_blocked", fail_entry_check)
    panel = pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "asset": ["A"],
            "factor": [1.0],
            "forward_return": [0.01],
            "can_enter": [False],
        }
    )
    positions = {"A": 1.0}

    adjusted = dataframe_constraints._apply_tradability_constraints(
        positions=positions,
        timestamp_frame=panel,
        spec=ComputePanelSpec(
            signal_columns=("factor",),
            label_column="forward_return",
        ),
        tradability_config=TradabilityConfig(can_enter_column="can_enter"),
        previous_positions={"A": 0.5},
    )

    assert adjusted == {"A": 1.0}


def test_calculate_turnover_uses_absolute_position_changes() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "asset": ["A", "B", "A", "B"],
            "signal": ["factor", "factor", "factor", "factor"],
            "position": [1.0, -1.0, -1.0, 1.0],
        }
    )

    turnover = calculate_turnover(positions)

    assert turnover["turnover"].tolist() == [2.0, 4.0]


def test_simulate_long_short_returns_applies_turnover_costs() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    result = simulate_long_short_returns(
        _ranking_panel(),
        spec,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        cost_config=CostConfig(slippage_bps=10),
        return_source_column="forward_return",
    )

    assert result["gross_portfolio_return"].round(6).tolist() == [0.02, 0.02]
    assert result["turnover"].tolist() == [2.0, 4.0]
    assert result["cost"].round(6).tolist() == [0.002, 0.004]
    assert result["net_portfolio_return"].round(6).tolist() == [0.018, 0.016]


def test_portfolio_construction_requires_explicit_return_source_column() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    with pytest.raises(ValueError, match="portfolio_return_source_column_required"):
        simulate_weighted_long_short_returns(
            _ranking_panel(),
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        )
    with pytest.raises(ValueError, match="portfolio_return_source_column_required"):
        tensor_construction.build_tensor_portfolio_artifacts(
            _ranking_panel(),
            spec,
            WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        )


def test_simulate_weighted_returns_keeps_long_only_rows_without_short_leg() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    _positions, turnover, returns, diagnostics = simulate_weighted_long_short_returns(
        _ranking_panel(),
        spec,
        WeightConfig(
            portfolio_mode="long_only",
            top_quantile=1 / 3,
            bottom_quantile=1 / 3,
        ),
        cost_config=CostConfig(slippage_bps=10),
        return_source_column="forward_return",
    )

    assert returns["gross_portfolio_return"].round(6).tolist() == [0.03, 0.03]
    assert returns["net_portfolio_return"].round(6).tolist() == [0.029, 0.028]
    assert returns["short_count"].tolist() == [0, 0]
    assert turnover["turnover"].tolist() == [1.0, 2.0]
    assert diagnostics["short_count"].tolist() == [0, 0]


def test_detailed_cost_config_splits_buy_sell_slippage_and_spread_costs() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    positions, _diagnostics = build_weighted_positions(
        _ranking_panel(),
        spec,
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    breakdown = calculate_cost_breakdown(
        positions,
        WeightConfig(),
        CostConfig(
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=2,
            spread_bps=1,
        ),
    )

    assert breakdown["buy_turnover"].tolist() == [1.0, 2.0]
    assert breakdown["sell_turnover"].tolist() == [1.0, 2.0]
    assert breakdown["buy_cost"].round(6).tolist() == [0.001, 0.002]
    assert breakdown["sell_cost"].round(6).tolist() == [0.002, 0.004]
    assert breakdown["slippage_cost"].round(6).tolist() == [0.0004, 0.0008]
    assert breakdown["spread_cost"].round(6).tolist() == [0.0002, 0.0004]
    assert breakdown["total_cost"].round(6).tolist() == [0.0036, 0.0072]


def test_simulation_uses_detailed_cost_config_when_provided() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")

    _positions, _turnover, returns, _diagnostics = simulate_weighted_long_short_returns(
        _ranking_panel(),
        spec,
        WeightConfig(
            top_quantile=1 / 3,
            bottom_quantile=1 / 3,
        ),
        cost_config=CostConfig(
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=2,
            spread_bps=1,
        ),
        return_source_column="forward_return",
    )

    assert returns["cost"].round(6).tolist() == [0.0036, 0.0072]
    assert returns["net_portfolio_return"].round(6).tolist() == [0.0164, 0.0128]


def test_cost_bps_monotonic_and_zero_cost_identity() -> None:
    spec = ComputePanelSpec(signal_columns=("factor",), label_column="forward_return")
    zero_cost = simulate_long_short_returns(
        _ranking_panel(),
        spec,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        cost_config=CostConfig(slippage_bps=0),
        return_source_column="forward_return",
    )
    lower_cost = simulate_long_short_returns(
        _ranking_panel(),
        spec,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        cost_config=CostConfig(slippage_bps=10),
        return_source_column="forward_return",
    )
    higher_cost = simulate_long_short_returns(
        _ranking_panel(),
        spec,
        top_quantile=1 / 3,
        bottom_quantile=1 / 3,
        cost_config=CostConfig(slippage_bps=20),
        return_source_column="forward_return",
    )

    assert zero_cost["cost"].eq(0.0).all()
    pd.testing.assert_series_equal(
        zero_cost["net_portfolio_return"],
        zero_cost["gross_portfolio_return"],
        check_names=False,
    )
    assert (
        higher_cost["net_portfolio_return"] <= lower_cost["net_portfolio_return"]
    ).all()
    assert (
        lower_cost["net_portfolio_return"] <= zero_cost["net_portfolio_return"]
    ).all()
    assert (higher_cost["cost"] >= lower_cost["cost"]).all()


def test_zero_turnover_rows_have_zero_cost() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "asset": ["A", "A"],
            "signal": ["factor", "factor"],
            "position": [1.0, 1.0],
        }
    )

    costs = calculate_cost_breakdown(
        positions,
        WeightConfig(),
        CostConfig(slippage_bps=10, spread_bps=5),
    )

    second = costs.loc[costs["timestamp"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    assert second["turnover"] == 0.0
    assert second["total_cost"] == 0.0


def test_long_short_positions_are_market_neutral_by_timestamp() -> None:
    positions, _diagnostics = build_weighted_positions(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    net_weights = positions.groupby(["timestamp", "signal"])["position"].sum()

    assert net_weights.abs().max() == pytest.approx(0.0)


def test_simulation_does_not_build_timestamp_asset_label_lookup(monkeypatch) -> None:
    set_index_calls = 0
    original_set_index = pd.DataFrame.set_index

    def counting_set_index(frame, keys, *args, **kwargs):
        nonlocal set_index_calls
        normalized_keys = [keys] if isinstance(keys, str) else list(keys)
        if normalized_keys == ["timestamp", "asset"]:
            set_index_calls += 1
        return original_set_index(frame, keys, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "set_index", counting_set_index)

    simulate_weighted_long_short_returns(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )

    assert set_index_calls == 0


def test_simulation_does_not_join_timestamp_asset_labels(monkeypatch) -> None:
    join_calls = 0
    original_join = pd.DataFrame.join

    def counting_join(frame, other, *args, **kwargs):
        nonlocal join_calls
        if kwargs.get("on") == ["timestamp", "asset"]:
            join_calls += 1
        return original_join(frame, other, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "join", counting_join)

    simulate_weighted_long_short_returns(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )

    assert join_calls == 0


def test_simulation_does_not_scan_turnover_and_cost_per_return_group(
    monkeypatch,
) -> None:
    assert not hasattr(dataframe_portfolio_returns, "_lookup_turnover")
    assert not hasattr(dataframe_portfolio_returns, "_lookup_cost")

    simulate_weighted_long_short_returns(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )


def test_simulation_streams_portfolio_path_without_rebuilding_from_positions(
    monkeypatch,
) -> None:
    def fail_rebuild(*args, **kwargs):
        raise AssertionError("simulation should stream turnover/cost/returns")

    monkeypatch.setattr(dataframe_turnover, "calculate_turnover", fail_rebuild)
    monkeypatch.setattr(dataframe_turnover, "calculate_cost_breakdown", fail_rebuild)
    monkeypatch.setattr(
        dataframe_portfolio_returns,
        "_calculate_long_short_from_positions",
        fail_rebuild,
    )

    _positions, turnover, returns, _diagnostics = simulate_weighted_long_short_returns(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )

    assert turnover["turnover"].tolist() == [2.0, 4.0]
    assert returns["gross_portfolio_return"].round(6).tolist() == [0.02, 0.02]


def test_simulation_calculates_long_short_legs_without_dataframe_dropna(
    monkeypatch,
) -> None:
    def fail_dropna(*args, **kwargs):
        if kwargs.get("subset") == ["forward_return"]:
            raise AssertionError("portfolio leg math should use array masks")
        return original_dropna(*args, **kwargs)

    original_dropna = pd.DataFrame.dropna
    monkeypatch.setattr(pd.DataFrame, "dropna", fail_dropna)

    _positions, _turnover, returns, _diagnostics = simulate_weighted_long_short_returns(
        _ranking_panel(),
        ComputePanelSpec(signal_columns=("factor",), label_column="forward_return"),
        WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
        return_source_column="forward_return",
    )

    assert returns["gross_portfolio_return"].round(6).tolist() == [0.02, 0.02]


def test_calculate_cumulative_returns_accumulates_by_signal() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-02",
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
            ],
            "signal": ["factor", "factor", "alt_factor", "alt_factor"],
            "gross_portfolio_return": [0.03, 0.02, -0.01, 0.04],
            "cost": [0.001, 0.002, 0.0, 0.001],
            "net_portfolio_return": [0.03, 0.02, -0.01, 0.04],
        }
    )

    cumulative_returns = calculate_cumulative_returns(returns)

    ordered = cumulative_returns.sort_values(["signal", "timestamp"]).reset_index(
        drop=True
    )
    assert ordered["cumulative_net_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["cumulative_gross_return"].round(6).tolist() == [
        -0.01,
        0.03,
        0.02,
        0.05,
    ]
    assert ordered["cumulative_cost"].round(6).tolist() == [0.0, 0.001, 0.002, 0.003]
    assert ordered["drawdown"].round(6).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_summarize_performance_reports_common_portfolio_diagnostics() -> None:
    returns = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "signal": ["factor", "factor", "factor"],
            "gross_portfolio_return": [0.03, -0.02, 0.04],
            "turnover": [1.0, 2.0, 1.5],
            "cost": [0.001, 0.002, 0.0015],
            "net_portfolio_return": [0.029, -0.022, 0.0385],
        }
    )
    cumulative_returns = calculate_cumulative_returns(returns)

    summary = summarize_performance(returns, cumulative_returns)

    row = summary.iloc[0]
    assert row["periods"] == 3
    assert row["final_cumulative_net_return"] == pytest.approx(0.0455)
    assert row["win_rate"] == pytest.approx(2 / 3)
    assert row["loss_rate"] == pytest.approx(1 / 3)
    assert row["sortino_like"] != 0
    assert row["volatility_net_portfolio_return"] >= 0.0
    assert row["downside_volatility"] >= 0.0
    assert row["profit_factor"] > 1
    assert row["average_win"] == pytest.approx((0.029 + 0.0385) / 2)
    assert row["average_loss"] == pytest.approx(-0.022)
    assert row["max_consecutive_wins"] == 1
    assert row["max_consecutive_losses"] == 1
    assert row["best_period_return"] == pytest.approx(0.0385)
    assert row["worst_period_return"] == pytest.approx(-0.022)
    assert row["positive_return_rate"] == pytest.approx(2 / 3)
    assert row["negative_return_rate"] == pytest.approx(1 / 3)
    assert row["max_drawdown"] == pytest.approx(-0.022)
    assert row["drawdown_duration"] >= 1
    assert row["mean_turnover"] == pytest.approx(1.5)
    assert row["max_turnover"] == pytest.approx(2.0)
    assert row["total_cost"] == pytest.approx(0.0045)
    assert row["mean_cost"] == pytest.approx(0.0015)
    assert row["cost_to_gross_return"] == pytest.approx(0.0045 / 0.05)


def test_summarize_exposure_reports_position_diagnostics() -> None:
    positions = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "asset": ["A", "B", "A", "B"],
            "signal": ["factor", "factor", "factor", "factor"],
            "position": [0.6, -0.4, 0.5, -0.5],
        }
    )
    long_short = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02"],
            "signal": ["factor", "factor"],
            "long_count": [1, 1],
            "short_count": [1, 1],
        }
    )

    exposure = summarize_exposure(positions, long_short)

    row = exposure.iloc[0]
    assert row["mean_gross_exposure"] == pytest.approx(1.0)
    assert row["mean_net_exposure"] == pytest.approx(0.1)
    assert row["max_gross_exposure"] == pytest.approx(1.0)
    assert row["max_abs_net_exposure"] == pytest.approx(0.2)
    assert row["mean_long_count"] == pytest.approx(1.0)
    assert row["mean_short_count"] == pytest.approx(1.0)
    assert row["mean_position_count"] == pytest.approx(2.0)
