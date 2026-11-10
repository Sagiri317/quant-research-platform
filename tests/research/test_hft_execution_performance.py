"""Clock and selected-input semantics for the native high-frequency account."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios.high_frequency import _decisions
from tests.helpers.high_frequency import parameters, quotes, trade_input
from tests.research.test_etf_inventory_policy import run_stream
from tests.research.test_passive_execution import replay


@pytest.mark.parametrize("cancelled,borrower", [("a", "z"), ("z", "a")])
@pytest.mark.parametrize("extra_quote", [False, True])
def test_cancelled_cash_is_available_without_another_quote_for_that_asset(
    tmp_path, cancelled, borrower, extra_quote
):
    panel = quotes(
        "2024-01-02",
        assets=(cancelled, borrower),
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:08", "01:30:09"),
    )
    seconds = [0, 3, 9] + ([6] if extra_quote else [])
    panel = panel.loc[
        panel.asset_id.eq(borrower) | panel.timestamp.dt.second.isin(seconds)
    ].assign(bid_vol1=100.0)
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T01:30:00", "2024-01-02T01:30:04", "2024-01-02T01:30:05.5"],
                format="mixed",
            ),
            "asset_id": [cancelled, cancelled, borrower],
            "prediction_score": [1.0, -1.0, 1.0],
        }
    )
    batches, summary = replay(
        tmp_path,
        panel,
        signals,
        [(7, borrower, "sell", 1000, 9.99)],
        cancel_ms=1000,
        capital=3998,
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert [(trade.asset_id, trade.quantity) for trade in trades] == [(borrower, 200)]
    assert pd.Timestamp(trades[0].timestamp_ns).second == 7
    # 1998 released cash buys 200 shares at 9.99, marked at 10.
    assert summary["nav"] == pytest.approx(4000)
    acknowledgements = [
        batch.valuations[-1].timestamp_ns
        for batch in batches
        if any(row.name == "passive_order_cancelled" for row in batch.diagnostics)
    ]
    assert acknowledgements == [pd.Timestamp("2024-01-02T01:30:05.5").value]


def test_off_quote_policy_observes_released_cash_before_its_next_decision(tmp_path):
    panel = quotes(
        "2024-01-02",
        clocks=("01:30:00", "01:30:03", "01:30:06", "01:30:09"),
    ).assign(instrument_type="gold_etf", bid_vol1=100.0)
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T01:30:00.001",
                    "2024-01-02T01:30:04",
                    "2024-01-02T01:30:05.5",
                ],
                format="mixed",
            ),
            "asset_id": ["a", "a", "b"],
            "prediction_score": [1.0, -1.0, 1.0],
        }
    )
    params = {
        **parameters(),
        "initial_capital": 3998,
        "passive": {"queue_model": "conservative", "cancel_latency_ms": 1000},
    }
    events = trade_input(tmp_path / "trades", [(7, "b", "sell", 1000, 9.99)])
    decisions, batches = run_stream(
        panel,
        signals,
        params,
        events=events,
        policy=lambda decision: decision.features.prediction_score.map(
            lambda score: 300.0 if score > 0 else 0.0
        ),
    )
    assert decisions[-1].cash == pytest.approx(1998)
    assert decisions[-1].reserved_cash == 0
    assert decisions[-1].pending_quantities.eq(0).all()
    assert [(t.asset_id, t.quantity) for b in batches for t in b.trades] == [("b", 200)]


@pytest.mark.parametrize("latency_ns", [0, 1_000_000_000, 4_000_000_000])
def test_decision_mapping_matches_direct_clock_and_segment_rules(latency_ns):
    start = pd.Timestamp("2024-01-02T01:30:00")
    panel = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta([0, 3, 6, 9, 0, 4, 8], unit="s"),
            "asset_id": ["a"] * 4 + ["b"] * 3,
            "segment_id": [0, 0, 1, 1, 0, 0, 0],
        }
    )
    signals = pd.DataFrame(
        [
            (start + pd.Timedelta(seconds=second), asset, score)
            for asset in ("a", "b", "absent")
            for second, score in [
                (9, 1.0),
                (0, 2.0),
                (-1, 3.0),
                (2.5, 4.0),
                (4, 5.0),
                (6, np.nan),
                (10, 6.0),
            ]
        ],
        columns=["timestamp", "asset_id", "prediction_score"],
    )
    expected = {}
    for asset, book in panel.groupby("asset_id"):
        rows = list(book.itertuples(index=False))
        for signal in (
            signals.loc[signals.asset_id.eq(asset)]
            .sort_values("timestamp")
            .itertuples(index=False)
        ):
            if pd.isna(signal.prediction_score):
                continue
            previous = [row for row in rows if row.timestamp <= signal.timestamp]
            if not previous or signal.timestamp > rows[-1].timestamp:
                continue
            later = [
                row
                for row in rows
                if row.timestamp > signal.timestamp
                and row.timestamp.value >= signal.timestamp.value + latency_ns
            ]
            destination = (
                later[0].timestamp.value
                if later and later[0].segment_id == previous[-1].segment_id
                else None
            )
            expected[signal.timestamp.value, asset] = (
                destination,
                signal.prediction_score,
            )
    assert _decisions(panel, signals, latency_ns) == expected


def test_selected_prediction_chunks_preserve_unsorted_days_and_relationships(tmp_path):
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.high_frequency import (
        inventory_inputs,
    )
    from tests.helpers.output_frames import make_price_volume_prediction_output_frame
    from tests.research.test_policy_learning import hft_episode

    episode = hft_episode(tmp_path / "inputs")
    paths = []
    expected = []
    for number, second in enumerate((0, 3)):
        metadata = make_price_volume_prediction_output_frame().attrs[
            "prediction_metadata"
        ]
        metadata.update(prediction_window={"start": "2024-01-02", "end": "2024-01-03"})
        metadata.pop("prediction_rows")
        metadata.pop("effective_prediction_end")
        # Native PRED has no global timestamp ordering requirement. The unselected
        # asset repeats across paths and must not enter the account's key domain.
        chunks = []
        for day in ("2024-01-03", "2024-01-02"):
            frame = pd.DataFrame(
                {
                    "timestamp": [
                        pd.Timestamp(day)
                        + pd.Timedelta(hours=1, minutes=30, seconds=second),
                        pd.Timestamp(day),
                    ],
                    "asset_id": ["a", "unselected"],
                    "prediction_score": [float(number + 1), 99.0],
                    "prediction_rank_score": [0.0, 0.0],
                }
            )
            frame["timestamp"] = frame.timestamp.astype("datetime64[ns]")
            chunks.append(frame)
            expected.append(frame.iloc[:1, :3])
        paths.append(
            write_prediction_outputs(
                output_path=tmp_path / f"pred{number}",
                output_chunks=chunks,
                metadata=metadata,
            )
        )
    validated = []
    sessions, _, _ = inventory_inputs(
        episode["account"],
        TimeWindow("2024-01-02", "2024-01-03"),
        prediction_paths=paths,
        prediction_validator=validated.append,
    )
    actual = pd.concat([frame for _, _, frame in sessions], ignore_index=True)

    def sort(frame):
        return frame.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        sort(actual), sort(pd.concat(expected, ignore_index=True))
    )
    assert len(validated) == 2
    with pytest.raises(ValueError, match="hft_prediction_keys_overlap"):
        inventory_inputs(
            episode["account"],
            TimeWindow("2024-01-02", "2024-01-03"),
            prediction_paths=paths + paths[:1],
        )


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_prediction_asset_filter_preserves_native_strings_and_empty_results(
    tmp_path, dtype
):
    from quant_research_platform.artifacts.outputs.prediction import (
        _iter_prediction_member,
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from tests.helpers.output_frames import make_price_volume_prediction_output_frame

    frame = pd.concat(
        [make_price_volume_prediction_output_frame()] * 3, ignore_index=True
    )
    frame["asset_id"] = pd.Series(["001", "\u8bc1\u5238", "3"], dtype=dtype)
    metadata = frame.attrs["prediction_metadata"]
    metadata["prediction_rows"] = 3
    path = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    for selected in (("001", "\u8bc1\u5238"), ("absent",), ()):
        chunks = list(
            _iter_prediction_member(
                path,
                metadata,
                time_window=TimeWindow("2014-01-03", "2014-01-03"),
                asset_ids=selected,
            )
        )
        if selected == ("001", "\u8bc1\u5238"):
            result = pd.concat(chunks, ignore_index=True)
            assert result.asset_id.tolist() == list(selected)
            assert str(result.timestamp.dtype) == "datetime64[ns]"
            assert result.prediction_score.tolist() == [0.25, 0.25]
        else:
            assert chunks == []


def test_public_prediction_asset_projection_keeps_empty_schema(tmp_path):
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from tests.helpers.output_frames import native_bundle

    root, _, _ = native_bundle(tmp_path)
    all_rows, facts = read_predictions(root)
    selected, selected_facts = read_predictions(root, asset_ids=["asset-a"])
    pd.testing.assert_frame_equal(selected, all_rows)
    assert selected_facts == facts
    for assets in ([], ["absent"]):
        empty, empty_facts = read_predictions(root, asset_ids=assets)
        pd.testing.assert_frame_equal(empty, all_rows.iloc[:0])
        assert empty_facts == facts
        projected, _ = read_predictions(
            root, columns=("prediction_score",), asset_ids=assets
        )
        pd.testing.assert_frame_equal(
            projected, all_rows.iloc[:0][["prediction_score"]]
        )
