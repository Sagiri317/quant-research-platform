from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as a_share_label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    label_catalog,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
    forward_return_label_column,
    forward_return_valid_bar_read_window,
)
from tests.helpers.synthetic_a_share import make_synthetic_market_panel

pytestmark = pytest.mark.guardrails


def test_forward_return_label_column_uses_frequency_suffix_contract() -> None:
    assert forward_return_label_column(5, DataFrequency.DAILY) == "forward_return_5d"
    assert forward_return_label_column(5, DataFrequency.MINUTE) == "forward_return_5m"
    assert (
        forward_return_label_column(5, DataFrequency.TICK)
        == "forward_return_5tick"
    )


@pytest.mark.parametrize("horizons", [(), [1], (True,), (0,), (1, 1)])
def test_forward_return_labels_reject_invalid_horizons(horizons: object) -> None:
    with pytest.raises(ValueError, match="invalid_forward_return_horizons"):
        add_forward_return_labels(
            make_synthetic_market_panel(sessions=3, assets=("asset_a",)),
            horizons=horizons,  # type: ignore[arg-type]
        )


def test_forward_return_label_column_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="invalid_forward_return_horizon"):
        forward_return_label_column(True)  # type: ignore[arg-type]


def test_forward_return_labels_group_sort_horizons_and_invalid_entry_price() -> None:
    panel = make_synthetic_market_panel(sessions=130)
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    asset_a_dates = ordered.loc[ordered["asset_id"].eq("asset_a"), "timestamp"].tolist()
    asset_b_dates = ordered.loc[ordered["asset_id"].eq("asset_b"), "timestamp"].tolist()
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[1]),
        "adj_open",
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_b") & panel["timestamp"].eq(asset_b_dates[1]),
        "adj_open",
    ] = pd.NA
    shuffled = panel.sample(frac=1.0, random_state=11).reset_index(drop=True)

    horizons = (1, 5, 10, 20, 60, 120)
    result = add_forward_return_labels(shuffled, horizons=horizons)
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    asset_a = result[result["asset_id"].eq("asset_a")].reset_index(drop=True)
    asset_b = result[result["asset_id"].eq("asset_b")].reset_index(drop=True)

    for horizon in horizons:
        column = forward_return_label_column(horizon)
        assert asset_a.tail(horizon + 1)[column].isna().all()
        assert asset_b.tail(horizon + 1)[column].isna().all()

    expected_asset_a_second = (
        asset_a.loc[3, "adj_open"] / asset_a.loc[2, "adj_open"] - 1.0
    )
    expected_asset_b_second = (
        asset_b.loc[3, "adj_open"] / asset_b.loc[2, "adj_open"] - 1.0
    )

    one_step_label = forward_return_label_column(1)
    assert pd.isna(asset_a.loc[0, one_step_label])
    assert pd.isna(asset_b.loc[0, one_step_label])
    assert asset_a.loc[1, one_step_label] == pytest.approx(
        expected_asset_a_second
    )
    assert asset_b.loc[1, one_step_label] == pytest.approx(
        expected_asset_b_second
    )
    label_columns = [forward_return_label_column(horizon) for horizon in horizons]
    assert not np.isinf(result.loc[:, label_columns].to_numpy(dtype="float64")).any()


def test_forward_return_short_history_long_horizon_is_all_nan() -> None:
    panel = make_synthetic_market_panel(sessions=3, assets=("asset_a",))

    result = add_forward_return_labels(panel, horizons=(5,))

    assert result[forward_return_label_column(5)].isna().all()


def test_forward_return_labels_treat_non_finite_prices_as_invalid() -> None:
    panel = make_synthetic_market_panel(sessions=6, assets=("asset_a",))
    ordered_dates = panel.sort_values("timestamp")["timestamp"].tolist()
    panel.loc[panel["timestamp"].eq(ordered_dates[1]), "adj_open"] = np.inf
    panel.loc[panel["timestamp"].eq(ordered_dates[3]), "adj_open"] = np.inf

    result = add_forward_return_labels(
        panel.sample(frac=1.0, random_state=23),
        horizons=(1,),
    ).sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    one_step_label = forward_return_label_column(1)
    assert pd.isna(result.loc[0, one_step_label])
    assert pd.isna(result.loc[1, one_step_label])
    assert not np.isinf(result[[one_step_label]].to_numpy(dtype="float64")).any()


def test_forward_return_minute_contiguous_labels_do_not_cross_gap() -> None:
    panel = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a", "asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00",
                    "2026-07-06 00:01",
                    "2026-07-06 00:03",
                    "2026-07-06 00:04",
                    "2026-07-06 00:05",
                ]
            ),
            "close": [100.0, 101.0, 103.0, 104.0, 105.0],
        }
    )

    result = add_forward_return_labels(
        panel,
        horizons=(1,),
        frequency=DataFrequency.MINUTE,
        execution_price_column="close",
        expected_interval=pd.Timedelta(minutes=1),
        require_contiguous=True,
    )

    column = forward_return_label_column(1, DataFrequency.MINUTE)
    assert pd.isna(result.loc[0, column])
    assert pd.isna(result.loc[1, column])
    assert result.loc[2, column] == pytest.approx(105.0 / 104.0 - 1.0)


def test_forward_return_labels_reject_duplicate_asset_timestamp() -> None:
    panel = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:00"]),
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate_forward_return_asset_timestamp"):
        add_forward_return_labels(
            panel,
            horizons=(1,),
            frequency=DataFrequency.MINUTE,
            execution_price_column="close",
        )


@pytest.mark.parametrize("require_contiguous", [False, True])
def test_forward_return_labels_reject_invalid_timestamp(
    require_contiguous: bool,
) -> None:
    panel = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": ["bad", "2026-07-06 00:01"],
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_forward_return_timestamp"):
        add_forward_return_labels(
            panel,
            horizons=(1,),
            frequency=DataFrequency.MINUTE,
            execution_price_column="close",
            expected_interval=pd.Timedelta(minutes=1),
            require_contiguous=require_contiguous,
        )


@pytest.mark.parametrize("asset_id", [None, pd.NA, [], {}, set()])
@pytest.mark.parametrize("require_contiguous", [False, True])
def test_forward_return_labels_reject_invalid_asset_key(
    asset_id: object,
    require_contiguous: bool,
) -> None:
    panel = pd.DataFrame(
        {
            "asset_id": [asset_id, "asset_a"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00", "2026-07-06 00:01"]
            ),
            "close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_forward_return_asset_key"):
        add_forward_return_labels(
            panel,
            horizons=(1,),
            frequency=DataFrequency.MINUTE,
            execution_price_column="close",
            expected_interval=pd.Timedelta(minutes=1),
            require_contiguous=require_contiguous,
        )


@pytest.mark.parametrize(
    ("drop_column", "error_column"),
    [
        ("close", "close"),
        ("asset_id", "asset_id"),
        ("timestamp", "timestamp"),
    ],
)
def test_forward_return_labels_reject_missing_required_columns(
    drop_column: str,
    error_column: str,
) -> None:
    panel = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "close": [100.0, 101.0],
        }
    ).drop(columns=[drop_column])

    with pytest.raises(
        ValueError,
        match=f"missing_forward_return_columns: {error_column}",
    ):
        add_forward_return_labels(
            panel,
            horizons=(1,),
            frequency=DataFrequency.MINUTE,
            execution_price_column="close",
        )


def test_forward_return_valid_bar_read_window_uses_sparse_asset_keys() -> None:
    panel_keys = pd.DataFrame(
        [
            ("asset_a", "2024-01-02"),
            ("asset_a", "2024-01-03"),
            ("asset_a", "2024-01-05"),
            ("asset_a", "2024-01-09"),
            ("asset_b", "2024-01-03"),
            ("asset_b", "2024-01-04"),
            ("asset_b", "2024-01-08"),
            ("asset_c", "2024-01-10"),
        ],
        columns=["asset_id", "timestamp"],
    )

    read_window = forward_return_valid_bar_read_window(
        panel_keys,
        read_window=TimeWindow("2024-01-03", "2024-01-03"),
        core_sessions=(pd.Timestamp("2024-01-03"),),
        lookahead_bars=2,
    )

    assert read_window == TimeWindow("2024-01-03", "2024-01-09")


def test_forward_return_read_window_rejects_null_group_key() -> None:
    panel_keys = pd.DataFrame(
        {
            "asset_id": [None],
            "timestamp": pd.to_datetime(["2024-01-03"]),
        }
    )

    with pytest.raises(ValueError, match="invalid_grouped_timestamp_keys"):
        forward_return_valid_bar_read_window(
            panel_keys,
            read_window=TimeWindow("2024-01-03", "2024-01-03"),
            core_sessions=(pd.Timestamp("2024-01-03"),),
            lookahead_bars=1,
        )


@pytest.mark.parametrize("lookahead_bars", [True, -1, 1.5])
def test_forward_return_read_window_rejects_invalid_lookahead_bars(
    lookahead_bars: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_forward_return_lookahead_bars"):
        forward_return_valid_bar_read_window(
            pd.DataFrame({"asset_id": ["asset_a"], "timestamp": ["2024-01-02"]}),
            read_window=TimeWindow("2024-01-02", "2024-01-02"),
            core_sessions=(pd.Timestamp("2024-01-02"),),
            lookahead_bars=lookahead_bars,  # type: ignore[arg-type]
        )


def test_forward_return_labels_multi_horizon_do_not_read_past_exit_bar() -> None:
    panel = make_synthetic_market_panel(sessions=14, assets=("asset_a",))
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    horizons = (1, 5, 10)
    decision_index = 1
    max_exit_index = decision_index + max(horizons) + 1

    baseline = add_forward_return_labels(ordered, horizons=horizons)
    after_exit_noise = ordered.copy()
    after_exit_noise.loc[max_exit_index + 1 :, "adj_open"] = (
        after_exit_noise.loc[max_exit_index + 1 :, "adj_open"] * 17.0
    )
    with_after_exit_noise = add_forward_return_labels(
        after_exit_noise,
        horizons=horizons,
    )
    five_day_exit_noise = ordered.copy()
    five_day_exit_index = decision_index + 5 + 1
    five_day_exit_noise.loc[five_day_exit_index, "adj_open"] = (
        five_day_exit_noise.loc[five_day_exit_index, "adj_open"] * 3.0
    )
    with_five_day_exit_noise = add_forward_return_labels(
        five_day_exit_noise,
        horizons=horizons,
    )

    for horizon in horizons:
        column = forward_return_label_column(horizon)
        assert with_after_exit_noise.loc[decision_index, column] == pytest.approx(
            baseline.loc[decision_index, column]
        )

    one_day_column = forward_return_label_column(1)
    five_day_column = forward_return_label_column(5)
    ten_day_column = forward_return_label_column(10)
    assert with_five_day_exit_noise.loc[
        decision_index,
        one_day_column,
    ] == pytest.approx(baseline.loc[decision_index, one_day_column])
    assert with_five_day_exit_noise.loc[decision_index, five_day_column] != (
        pytest.approx(baseline.loc[decision_index, five_day_column])
    )
    assert with_five_day_exit_noise.loc[decision_index, ten_day_column] == (
        pytest.approx(baseline.loc[decision_index, ten_day_column])
    )


def test_label_catalog_horizons_mask_invalid_entry() -> None:
    panel = make_synthetic_market_panel(sessions=130, assets=("asset_a",))
    ordered_dates = panel.sort_values("timestamp")["timestamp"].tolist()
    panel.loc[panel["timestamp"].eq(ordered_dates[1]), "adj_open"] = 0.0

    horizons = tuple(
        definition.horizon for definition in label_catalog().list()
    )
    result = add_forward_return_labels(
        panel.sample(frac=1.0, random_state=3),
        horizons=horizons,
    )
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    expected = result.loc[3, "adj_open"] / result.loc[2, "adj_open"] - 1.0
    label_columns = [forward_return_label_column(horizon) for horizon in horizons]
    one_step_label = forward_return_label_column(1)

    assert set(label_columns).issubset(result.columns)
    assert pd.isna(result.loc[0, one_step_label])
    assert result.loc[1, one_step_label] == pytest.approx(expected)
    assert not np.isinf(result.loc[:, label_columns].to_numpy(dtype="float64")).any()


@pytest.mark.parametrize(
    "definition",
    sorted(label_catalog().list(), key=lambda item: item.horizon),
    ids=lambda definition: definition.output_column,
)
def test_label_definition_window_uses_future_rows_without_later_leakage(
    definition,
) -> None:
    sessions = definition.lookahead + 4
    decision_index = 2
    entry_index = decision_index + 1
    exit_index = decision_index + definition.lookahead
    after_exit_index = exit_index + 1
    panel = make_synthetic_market_panel(sessions=sessions, assets=("asset_a",))
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    label_column = definition.output_column

    baseline = a_share_label_calculations.calculate_labels_for_panel(
        ordered,
        definitions=(definition,),
    )
    after_exit_noise = ordered.copy()
    after_exit_noise.loc[after_exit_index:, "adj_open"] = (
        after_exit_noise.loc[after_exit_index:, "adj_open"] * 11.0
    )
    exit_noise = ordered.copy()
    exit_noise.loc[exit_index, "adj_open"] = (
        exit_noise.loc[exit_index, "adj_open"] * 3.0
    )
    later = a_share_label_calculations.calculate_labels_for_panel(
        after_exit_noise,
        definitions=(definition,),
    )
    changed = a_share_label_calculations.calculate_labels_for_panel(
        exit_noise,
        definitions=(definition,),
    )

    decision_timestamp = ordered.loc[decision_index, "timestamp"]
    entry_timestamp = ordered.loc[entry_index, "timestamp"]
    exit_timestamp = ordered.loc[exit_index, "timestamp"]
    assert entry_timestamp > decision_timestamp
    assert exit_timestamp > decision_timestamp
    assert baseline.loc[decision_index, "timestamp"] == decision_timestamp
    assert later.loc[decision_index, label_column] == pytest.approx(
        baseline.loc[decision_index, label_column]
    )
    assert changed.loc[decision_index, label_column] != pytest.approx(
        baseline.loc[decision_index, label_column]
    )
