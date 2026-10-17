"""Saved signals and unrelated source anomalies cannot change independent accounts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios import accounting, production
from tests.research.test_enhanced_decision_semantics import _decision_inputs
from tests.research.test_portfolio_production import _build_producer, _inputs


@pytest.mark.parametrize("feature", ["missing", "removed"])
def test_saved_prediction_account_does_not_depend_on_historical_x(
    tmp_path, monkeypatch, feature
):
    frames, fees, window = _inputs()
    frames["dataset"]["feature_x"] = 1.0
    baseline, _ = _build_producer(
        tmp_path / "baseline",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    expected = pd.concat(list(baseline), ignore_index=True)
    frames["dataset"] = frames["dataset"].copy()
    if feature == "missing":
        frames["dataset"]["feature_x"] = np.nan
    else:
        frames["dataset"] = frames["dataset"].drop(columns="feature_x")
    changed, calls = _build_producer(
        tmp_path / "changed",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    actual = pd.concat(list(changed), ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "column,invalid",
    [
        ("amount", -1.0),
        ("amount", "unavailable"),
        ("amount", True),
        ("amount", 1 + 2j),
        ("open", 0.0),
        ("open", "unavailable"),
        ("open", True),
        ("open", 1 + 2j),
        ("close", np.inf),
    ],
)
def test_unselected_unheld_bad_cells_preserve_independent_account(
    tmp_path, monkeypatch, column, invalid
):
    frames, fees, window = _inputs()
    baseline, _ = _build_producer(
        tmp_path / "baseline",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    expected = pd.concat(list(baseline), ignore_index=True)
    frames["panel"][column] = frames["panel"][column].astype(object)
    frames["panel"].loc[frames["panel"].asset_id.eq("a"), column] = invalid
    changed, _ = _build_producer(
        tmp_path / "changed",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    actual = pd.concat(list(changed), ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    "cap",
    [
        -1.0,
        0.0,
        np.inf,
        np.nan,
        "unavailable",
        True,
        np.bool_(False),
        1 + 2j,
        pd.Timestamp("2024-01-01"),
        pd.Timedelta(days=1),
    ],
)
def test_unavailable_selected_cap_preserves_independent_proven_exit(cap):
    args = _decision_inputs()
    args["enhanced_decision"] = None
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_float_cap",
        benchmark_id=None,
        top_quantile=1.0,
        risk_style_columns=(),
    )
    args["dataset"]["float_market_cap"] = args["dataset"]["float_market_cap"].astype(
        object
    )
    args["dataset"].loc[args["dataset"].asset_id.eq("a"), "float_market_cap"] = cap
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.executed_delta_notional.tolist() == [0.0, -50.0]
    assert account.post_trade_cash == 50.0
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "target_float_market_cap_missing"
    )


def test_alpha_window_uses_complete_252_session_clock_and_mature_labels():
    sessions = pd.bdate_range("2023-01-02", "2024-06-28").difference(
        pd.bdate_range("2023-10-02", "2023-10-06")
    )
    intervals = accounting.alpha_calibration_intervals(sessions, "daily")
    dates = intervals.timestamp
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(2),
            "prediction_rank_score": [-1.0, 1.0] * len(dates),
            "holding_return": [-0.01, 0.01] * len(dates),
        }
    )
    # A missing observation does not compress the complete market clock.
    absent = sessions[-100]
    frame = frame.loc[frame.timestamp.ne(absent)]
    result = accounting.calibrate_expected_alpha(
        frame,
        as_of=sessions[-1],
        scheduled_intervals=intervals,
        trading_sessions=sessions,
    )
    assert (
        accounting.alpha_calibration_start(sessions, as_of=sessions[-1])
        == sessions[-252]
    )
    assert result.scheduled_rebalance_dates == 250
    assert result.valid_rebalance_dates == 249
    assert result.coverage == pytest.approx(249 / 250)
    assert result.slope == pytest.approx(0.01)


def test_partial_saved_prediction_matches_context_without_requiring_extra_scores():
    import pyarrow as pa

    from quant_research_platform.artifacts.outputs.prediction import _PredictionLookup
    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    context = pd.DataFrame(
        {
            "asset_id": ["a", "b", "excluded"],
            "timestamp": pd.Timestamp("2023-01-03"),
            "eligible": [True, True, False],
        }
    )
    saved = context.loc[context.asset_id.ne("a"), ["asset_id", "timestamp"]].assign(
        prediction_rank_score=[0.5, 0.7]
    )
    index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
    index.add_float_value_table(
        pa.Table.from_pandas(saved, preserve_index=False),
        namespace="prediction",
        value_column="prediction_rank_score",
    )
    lookup = _PredictionLookup(
        index=index,
        primary_key=("timestamp", "asset_id"),
        prediction_column="prediction_rank_score",
    )
    result = production._prediction_alpha(context, lookup)
    assert pd.isna(result.loc["a"])
    assert result.loc["b"] == 0.5
    assert pd.isna(result.loc["excluded"])
    assert lookup.coverage_counts() == (0, 0)


@pytest.mark.parametrize("asset", ["a", "b"])
@pytest.mark.parametrize("unknown", ["amount", "settlement", "announcement"])
def test_unknown_corporate_terms_block_only_real_entitlement(
    tmp_path, monkeypatch, asset, unknown
):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    frames, fees, window = _inputs()
    baseline, _ = _build_producer(
        tmp_path / "baseline",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    expected = pd.concat(list(baseline), ignore_index=True)
    event = dict(
        source_asset_id=asset,
        timestamp=pd.Timestamp("2023-01-05"),
        kind="CA",
        market_type=1,
        announcement_date=pd.Timestamp("2023-01-02"),
        settlement_date=pd.NaT,
        settlement_date_unresolved=False,
        amount=1.0,
    )
    if unknown == "amount":
        event["amount"] = np.nan
    elif unknown == "settlement":
        event["settlement_date_unresolved"] = True
    else:
        event["announcement_date"] = pd.NaT
    frames["corporate_actions"] = pd.DataFrame([event])
    changed, _ = _build_producer(
        tmp_path / "changed",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    if asset == "a":
        pd.testing.assert_frame_equal(
            pd.concat(list(changed), ignore_index=True), expected
        )
    else:
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_held_rights_unproved"
        ):
            list(changed)


@pytest.mark.parametrize("column,invalid", [("amount", -1.0), ("open", 0.0)])
def test_required_bad_input_has_missing_fact_semantics_without_invented_trade(
    tmp_path, monkeypatch, column, invalid
):
    outputs = []
    for label, value in (("missing", np.nan), ("invalid", invalid)):
        frames, fees, window = _inputs()
        frames["panel"].loc[frames["panel"].asset_id.eq("b"), column] = value
        chunks, _ = _build_producer(
            tmp_path / label,
            monkeypatch,
            frames=frames,
            fee_frame=fees,
            output_window=window,
        )
        outputs.append(pd.concat(list(chunks), ignore_index=True))
    pd.testing.assert_frame_equal(*outputs)
    assert outputs[1].executed_delta_notional.eq(0.0).all()
    assert (
        outputs[1]
        .loc[outputs[1].asset_id.eq("__cash__"), "end_notional"]
        .eq(100.0)
        .all()
    )


@pytest.mark.parametrize("asset", ["a", "b"])
def test_unresolved_delisting_status_blocks_only_dependent_holding(
    tmp_path, monkeypatch, asset
):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    frames, fees, window = _inputs()
    baseline, _ = _build_producer(
        tmp_path / "baseline",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    expected = pd.concat(list(baseline), ignore_index=True)
    first = frames["panel"].timestamp.min()
    frames["stock_master"] = pd.DataFrame(
        [
            dict(
                source_asset_id=name,
                timestamp=first,
                listing_date=first,
                delisting_date=pd.NaT,
                delisting_date_unresolved=False,
            )
            for name in ("a", "b")
        ]
        + [
            dict(
                source_asset_id=asset,
                timestamp=pd.Timestamp("2023-01-05"),
                listing_date=first,
                delisting_date=pd.NaT,
                delisting_date_unresolved=True,
            )
        ]
    )
    changed, _ = _build_producer(
        tmp_path / "changed",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
    )
    if asset == "a":
        pd.testing.assert_frame_equal(
            pd.concat(list(changed), ignore_index=True), expected
        )
    else:
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_held_return_missing"
        ):
            list(changed)


@pytest.mark.parametrize("cap", [True, 1 + 2j])
def test_unselected_typed_bad_cap_is_not_a_target_dependency(cap):
    args = _decision_inputs()
    args["enhanced_decision"] = None
    args["alpha"] = pd.Series({"a": 0.0, "b": 1.0})
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_float_cap",
        benchmark_id=None,
        top_quantile=0.5,
        risk_style_columns=(),
    )
    args["dataset"]["float_market_cap"] = args["dataset"]["float_market_cap"].astype(
        object
    )
    args["dataset"].loc[args["dataset"].asset_id.eq("a"), "float_market_cap"] = cap
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.executed_delta_notional.tolist() == [-50.0, 50.0]
    assert account.post_trade_cash == 0.0
