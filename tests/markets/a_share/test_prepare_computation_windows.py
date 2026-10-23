"""Output slices retain actual supplied history and own-market future endpoints."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)


@pytest.mark.parametrize("history_start", ("2009-11-02", "2019-11-01"))
def test_native_narrow_outputs_equal_full_with_actual_history_and_lookahead(
    tmp_path, history_start
):
    dates = pd.bdate_range(history_start, periods=180).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    other = bars.source_asset_id.ne("000001")
    bars.loc[other, "market_type"] = 4
    closed = dates[np.arange(len(dates)) % 3 != 0]
    bars.loc[~other & bars.timestamp.isin(closed), "filling_flag"] = 2
    calendar = source["trading_calendar"]
    source["trading_calendar"] = pd.concat(
        (
            calendar.assign(is_trading_day=calendar.calendar_date.isin(dates[::3])),
            calendar.assign(source_market_type=4),
        ),
        ignore_index=True,
    )
    inputs = _write_inputs(tmp_path / "source", source)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "full", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    references = {}
    for name in ("price_volume_facts", "execution_mask", "forward_returns"):
        spec = processed_catalog().get(name)
        config = {
            "market_id": "a_share",
            "data_id": name,
            "inputs": {
                key: str(inputs[key])
                for key in (*spec.processed_input_data_ids, *spec.source_input_data_ids)
            },
            **(
                {"factor_columns": ["reversal_5d"]}
                if name == "price_volume_facts"
                else {}
            ),
        }
        inputs[name] = prepare(
            {
                **config,
                "time_window": {
                    "start": str(dates[0].date()),
                    "end": str(dates[-1].date()),
                },
                "output_root": str(tmp_path / "full" / name),
            }
        )
        references[name] = config, read_dataset(inputs[name])
    start, end = dates[45], dates[49]
    # Precision changes only the requested projection, not the full reference.
    for precise in (False, True):
        output_root = tmp_path / ("precise" if precise else "daily")
        window = {
            "start": (start + pd.Timedelta(1, "ns")).isoformat()
            if precise
            else str(start.date()),
            "end": (end - pd.Timedelta(1, "ns")).isoformat()
            if precise
            else str(end.date()),
        }
        outputs = {}
        for name, (config, full) in references.items():
            narrowed = read_dataset(
                prepare(
                    {
                        **config,
                        "time_window": window,
                        "output_root": str(output_root / "narrow" / name),
                    }
                )
            )
            expected = full.loc[
                full.timestamp.between(
                    pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
                )
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                narrowed.select_dtypes(exclude="floating"),
                expected.select_dtypes(exclude="floating"),
                check_exact=True,
            )
            pd.testing.assert_frame_equal(narrowed.isna(), expected.isna())
            pd.testing.assert_frame_equal(
                narrowed, expected, check_exact=False, rtol=1e-12, atol=0
            )
            assert not narrowed.empty
            outputs[name] = narrowed
        assert outputs["price_volume_facts"].reversal_5d.notna().all()
        assert outputs["execution_mask"][["can_enter", "can_exit"]].all().all()
        labels = outputs["forward_returns"]
        assert labels.forward_return_5d.notna().all()
        assert labels.forward_return_5d_available_at.gt(end).all()
        for row in labels.itertuples():
            axis = dates[::3] if row.asset_id == "000001" else dates
            assert row.forward_return_5d_available_at == axis[
                axis.get_loc(row.timestamp) + 6
            ] + pd.Timedelta(hours=7)

        # Future prices exist for labels; they must not influence causal factors.
        panel = read_dataset(inputs["market_panel_1d"])
        panel.loc[panel.timestamp.gt(end), "adj_close"] *= 100.0
        future_changed = output_root / "future-changed-panel"
        write_dataset(
            future_changed, panel, spec=processed_catalog().get("market_panel_1d")
        )
        unchanged = read_dataset(
            prepare(
                {
                    "market_id": "a_share",
                    "data_id": "price_volume_facts",
                    "inputs": {
                        "market_panel_1d": str(future_changed),
                        "trading_calendar": str(inputs["trading_calendar"]),
                    },
                    "factor_columns": ["reversal_5d"],
                    "time_window": window,
                    "output_root": str(output_root / "causal-factors"),
                }
            )
        )
        expected = outputs["price_volume_facts"]
        pd.testing.assert_frame_equal(
            unchanged.select_dtypes(exclude="floating"),
            expected.select_dtypes(exclude="floating"),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(unchanged.isna(), expected.isna())
        pd.testing.assert_frame_equal(
            unchanged, expected, check_exact=False, rtol=1e-12, atol=0
        )


def test_native_panel_request_before_source_start_keeps_daily_axis(tmp_path):
    dates = pd.bdate_range("2009-11-02", periods=8).as_unit("ns")
    inputs = _write_inputs(tmp_path / "source", _synthetic_interim_frames(dates))
    config = {
        "market_id": "a_share",
        "data_id": "market_panel_1d",
        "inputs": {
            key: str(inputs[key])
            for key in processed_catalog().get("market_panel_1d").source_input_data_ids
        },
    }
    result = read_dataset(
        prepare(
            {
                **config,
                "time_window": {
                    "start": (dates[0] - pd.Timedelta(hours=12)).isoformat(),
                    "end": (dates[-1] - pd.Timedelta(1, "ns")).isoformat(),
                },
                "output_root": str(tmp_path / "panel"),
            }
        )
    )
    assert result.timestamp.drop_duplicates().tolist() == list(dates[:-1])
