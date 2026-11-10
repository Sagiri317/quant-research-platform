"""Independent likelihood/gradient, timestamp ties, and native fit-to-factor flow."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.hawkes import (
    fit_hawkes,
    hawkes_parameters,
    intensities,
    load_hawkes,
    negative_log_likelihood,
)
from quant_research_platform.research.protocols.a_share_hft import (
    HAWKES_FACTORS,
    processed_specs,
)
from quant_research_platform.sources.venvoo import interim_spec, normalize
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)


def test_likelihood_matches_direct_sum_and_analytic_gradient():
    times = np.array([0.2, 0.8, 1.7])
    counts = np.array([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0]])
    theta = np.array([0.3, 0.4, 0.2, 0.1, 0.05, 0.25, 1.2])
    duration = 3.0
    mu, alpha, beta = theta[:2], theta[2:6].reshape(2, 2), theta[-1]
    expected = duration * mu.sum()
    for index, time in enumerate(times):
        history = sum(
            (counts[j] * np.exp(-beta * (time - times[j])) for j in range(index)),
            np.zeros(2),
        )
        expected -= (counts[index] * np.log(mu + beta * alpha @ history)).sum()
        expected += alpha.sum(axis=0) @ (
            counts[index] * (1 - np.exp(-beta * (duration - time)))
        )
    loss, gradient = negative_log_likelihood(theta, [(times, counts, duration)])
    assert loss == pytest.approx(expected)
    numeric = []
    for i in range(7):
        step = np.eye(7)[i] * 1e-6
        numeric.append(
            (
                negative_log_likelihood(theta + step, [(times, counts, duration)])[0]
                - negative_log_likelihood(theta - step, [(times, counts, duration)])[0]
            )
            / 2e-6
        )
    np.testing.assert_allclose(gradient, numeric, rtol=1e-7, atol=1e-8)


def test_same_timestamp_pre_jump_intensity_and_prefix_invariance():
    params = hawkes_parameters(
        {"baseline": [1.0, 2.0], "branching": [[0.2, 0.1], [0.3, 0.1]], "decay": 2.0}
    )
    times, counts = np.array([1.0, 2.0]), np.array([[2.0, 1.0], [1.0, 0.0]])
    actual = intensities(times, counts, np.array([0.0, 1.0, 1.5, 2.0]), params)
    np.testing.assert_allclose(actual[:2], [[1.0, 2.0], [1.0, 2.0]])
    expected = params[0] + 2 * np.exp(-1) * params[1] @ counts[0]
    np.testing.assert_allclose(actual[2], expected)
    np.testing.assert_allclose(
        actual,
        intensities(times[:1], counts[:1], np.array([0.0, 1.0, 1.5, 2.0]), params),
    )
    with pytest.raises(ValueError, match="parameters_invalid"):
        hawkes_parameters(
            {
                "baseline": [1.0, 2.0],
                "branching": [[1.0, 0.0], [0.0, 0.2]],
                "decay": 1.0,
            }
        )


@pytest.mark.parametrize("unknown_kind", [False, True])
def test_fit_save_prepare_and_assemble_only_after_cutoff(tmp_path, unknown_kind):
    panels, all_trades = [], []
    for day in (20180214, 20180215):
        raw = raw_events("hft_trades")
        raw["trade_code"], raw["bs_flag"], raw["date"] = "T", ["B", "S", "B", "S"], day
        if unknown_kind and day == 20180214:
            raw.loc[raw.index[-1], "trade_code"] = "?"
        trades = normalize(raw, "hft_trades", INSTRUMENTS)
        orders = normalize(
            raw_events("hft_orders").assign(date=day), "hft_orders", INSTRUMENTS
        )
        quotes = normalize(
            raw_quotes((93000000, 93003000, 93006000, 93009000, 93012000)).assign(
                date=day
            ),
            "hft_quotes",
            INSTRUMENTS,
        )
        panels.append(build_panel(quotes, orders, trades, max_gap_seconds=6))
        all_trades.append(trades)
    paths = {name: str(tmp_path / name) for name in ("hft_panel", "hft_trades")}
    write_dataset(
        paths["hft_panel"],
        pd.concat(
            [*panels, *(panel.assign(asset_id="excluded") for panel in panels)],
            ignore_index=True,
        ),
        spec=next(item for item in processed_specs() if item.data_id == "hft_panel"),
    )
    write_dataset(
        paths["hft_trades"],
        pd.concat(
            [
                *all_trades,
                *(trades.assign(asset_id="excluded") for trades in all_trades),
            ],
            ignore_index=True,
        ),
        spec=interim_spec("hft_trades"),
    )
    model = fit_hawkes(
        {
            "inputs": paths,
            "asset_ids": ["000001.SZ"],
            "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
            "fit_cutoff": "2018-02-15T00:00:00",
            "initial_parameters": {
                "baseline": [0.2, 0.2],
                "branching": [[0.1, 0.1], [0.1, 0.1]],
                "decay": 1.0,
            },
            "max_branching": 0.9,
            "max_iterations": 200,
            "tolerance": 1e-8,
            "output": str(tmp_path / "model"),
        }
    )
    params, _ = load_hawkes(model)
    theta = np.r_[params[0], params[1].ravel(), params[2]]
    expected_loss, _ = negative_log_likelihood(
        theta,
        [
            (
                np.array([0.0, 2.0, 5.0] if unknown_kind else [0.0, 2.0, 5.0, 10.0]),
                np.array(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
                    if unknown_kind
                    else [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
                ),
                9.0 if unknown_kind else 12.0,
            )
        ],
    )
    facts = json.loads((model / "model.json").read_text())
    assert facts["negative_log_likelihood"] == pytest.approx(expected_loss)
    assert facts["unknown_kind_events"] == int(unknown_kind)
    assert facts["unknown_kind_truncated_segments"] == int(unknown_kind)
    window = {"start": "2018-02-15T01:30:00", "end": "2018-02-15T01:30:13"}
    config = {
        "market_id": "a_share",
        "data_id": "hft_hawkes_factors",
        "inputs": paths,
        "parameters": {"model": str(model)},
        "time_window": window,
        "output_root": str(tmp_path / "factors"),
    }
    factors = prepare(config)
    values = read_dataset(factors)
    np.testing.assert_allclose(
        values.loc[0, ["hawkes_buy_intensity", "hawkes_sell_intensity"]].to_numpy(
            dtype=float
        ),
        params[0],
    )
    assembled = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "hawkes_test",
            "base_input_data_id": "hft_panel",
            "inputs": {
                "hft_panel": paths["hft_panel"],
                "hft_hawkes_factors": str(factors),
            },
            "factor_columns": list(HAWKES_FACTORS),
            "feature_recipe": None,
            "time_window": window,
            "output_root": str(tmp_path / "dataset"),
        }
    )
    assert len(read_dataset(assembled)) == len(values)
    with pytest.raises(ValueError, match="not_out_of_sample"):
        prepare(
            {
                **config,
                "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
                "output_root": str(tmp_path / "invalid"),
            }
        )


def test_factor_unknown_barrier_preserves_independent_pre_jump_intensity():
    from quant_research_platform.research.factors.hawkes import compute_hawkes_factors

    origin = pd.Timestamp("2024-01-02T01:30:00")
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": origin + pd.to_timedelta(np.arange(5), unit="s"),
            "session_date": origin.normalize(),
            "segment_id": "am",
        }
    )
    trades = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": origin + pd.to_timedelta([0, 1, 2, 2, 3], unit="s"),
            "source_row": np.arange(5),
            "event_kind": ["trade", "trade", "unknown", "trade", "trade"],
            "side": ["buy", "sell", "buy", "buy", "sell"],
        }
    )
    mu = np.array([0.5, 0.7])
    alpha = np.array([[0.2, 0.1], [0.1, 0.2]])
    beta = 2.0
    result = compute_hawkes_factors(
        panel, trades, ((mu, alpha, beta), origin - pd.Timedelta(days=1))
    )
    expected = np.array(
        [
            mu,
            mu + beta * alpha @ [np.exp(-beta), 0],
            mu + beta * alpha @ [np.exp(-2 * beta), np.exp(-beta)],
        ]
    )
    np.testing.assert_allclose(
        result.loc[:2, ["hawkes_buy_intensity", "hawkes_sell_intensity"]], expected
    )
    total = expected.sum(axis=1)
    np.testing.assert_allclose(
        result.hawkes_intensity_imbalance.iloc[:3],
        (expected[:, 0] - expected[:, 1]) / total,
    )
    np.testing.assert_allclose(
        result.hawkes_excitation_fraction.iloc[:3], 1 - mu.sum() / total
    )
    assert result.loc[3:, list(HAWKES_FACTORS)].isna().all().all()
