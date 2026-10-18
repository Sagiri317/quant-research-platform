from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.a_share_catalogs import scientific_catalogs
from tests.helpers.a_share_sources import _calendar_input
from tests.helpers.paths import StorageLayout


def _produce_factors(root, data_id, inputs, columns, window):
    profile = scientific_catalogs()
    layout = StorageLayout(root)
    input_paths = {}
    for input_id, frame in inputs.items():
        if input_id == "market_panel_1d":
            frame = frame.assign(filling_flag=0, market_type=1)
            input_paths["trading_calendar"] = _calendar_input(
                root / "calendar", {1: pd.DatetimeIndex(frame.timestamp.unique())}
            )
        base = profile.processed.get(input_id)
        spec = replace(
            base,
            column_roles={
                name: role for name, role in base.column_roles.items() if name in frame
            },
        )
        path = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(path, frame.loc[:, spec.required_columns], spec=spec)
        input_paths[input_id] = path
    base = profile.processed.get(data_id)
    spec = replace(
        base,
        column_roles={
            name: role
            for name, role in base.column_roles.items()
            if name in (*base.primary_key, *columns)
        },
    )
    path = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=input_paths,
        time_window=window,
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata):
        write_dataset(path, frames, spec=spec, metadata=metadata)
    return read_dataset(path)


@pytest.mark.parametrize("batch_rows", (6, 1000))
def test_abnormal_turnover_observed_windows_persist_across_chunks(
    tmp_path, monkeypatch, batch_rows
):
    dates = pd.bdate_range("2020-01-02", periods=252).as_unit("ns")
    sequences = {
        "lower_turnover": [0.2] * 230 + [0.03] * 22,
        "ordinary": list(np.arange(1, 253) * 0.001),
        "small": [0.00001] * 252,
        "zero": [0.0] * 252,
        "negative": [-1.0] * 252,
        "missing": [0.2] * 230 + [0.03] * 22,
    }
    sequences["missing"][240] = np.nan
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=date,
                listing_date=dates[0],
                turnover=value,
            )
            for asset, values in sequences.items()
            for date, value in zip(dates, values, strict=True)
        ]
    )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = _produce_factors(
        tmp_path,
        "price_volume_facts",
        {"market_panel_1d": panel},
        ("abnormal_turnover_20_250",),
        TimeWindow(dates[248], dates[-1]),
    )
    assert len(saved) == 4 * len(sequences)
    expected = []
    for row in saved.itertuples():
        stop = dates.get_loc(row.timestamp) + 1
        values = sequences[row.asset_id][stop - 250 : stop]
        value = np.nan
        if (
            stop >= 250
            and np.isfinite(values).sum() >= 188
            and np.isfinite(values[-20:]).sum() >= 15
        ):
            observed = [value for value in values if np.isfinite(value)]
            short = [value for value in values[-20:] if np.isfinite(value)]
            total = sum(map(Fraction, observed))
            if total > 0:
                value = float(
                    len(observed) * sum(map(Fraction, short)) / (len(short) * total)
                )
        expected.append(value)
    np.testing.assert_allclose(
        saved.abnormal_turnover_20_250, expected, rtol=1e-12, atol=1e-14
    )
    assert saved.loc[
        saved.asset_id.eq("lower_turnover") & saved.timestamp.eq(dates[249]),
        "abnormal_turnover_20_250",
    ].iloc[0] == pytest.approx(0.03 / ((230 * 0.2 + 20 * 0.03) / 250))


@pytest.mark.parametrize("batch_rows", (1, 1000))
@pytest.mark.parametrize("include_cap", (False, True))
def test_fundamental_growth_stable_differences_persist_across_chunks(
    tmp_path, monkeypatch, batch_rows, include_cap
):
    next_value = np.nextafter(3.0, np.inf)
    cases = [
        (next_value, 3.0),
        (np.nextafter(next_value, np.inf), 3.0),
        (3.0, 3.0),
        (np.nan, 3.0),
        (3.0, np.nan),
        (3.0, 0.0),
        (3.0, -1.0),
        (-1e308, 1e308),
        (np.nextafter(1e308, np.inf), 1e308),
        (1e308, 5e-324),
        (1e-323, 5e-324),
    ]
    dates = pd.bdate_range("2024-01-02", periods=len(cases)).as_unit("ns")
    facts = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="asset",
            timestamp=dates,
            has_legal_fundamental_pit=True,
        )
    )
    formulas = {
        "asset_growth_yoy": ("total_assets", "prior_year_total_assets"),
        "revenue_growth_ttm_yoy": (
            "operating_revenue_ttm",
            "prior_year_operating_revenue_ttm",
        ),
        "parent_equity_growth_yoy": ("parent_equity", "prior_year_parent_equity"),
    }
    for numerator, denominator in formulas.values():
        facts[numerator], facts[denominator] = zip(*cases, strict=True)
    panel = facts.loc[:, ["market_id", "asset_id", "timestamp"]].assign(
        total_market_cap=1.0, listing_date=dates[0]
    )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = _produce_factors(
        tmp_path,
        "fundamental_factors",
        {"fundamental_pit": facts, "market_panel_1d": panel},
        (
            *formulas,
            *(("book_to_market",) if include_cap else ()),
            "has_legal_fundamental_pit",
        ),
        TimeWindow(dates[0], dates[-1]),
    )
    expected = []
    for numerator, denominator in cases:
        value = np.nan
        if np.isfinite(numerator) and denominator > 0.0:
            try:
                value = float(
                    (Fraction(numerator) - Fraction(denominator))
                    / Fraction(denominator)
                )
            except OverflowError:
                pass
        expected.append(value)
    for column in formulas:
        np.testing.assert_array_equal(saved[column], expected)
        assert saved[column].iloc[0] < saved[column].iloc[1]
    assert saved.has_legal_fundamental_pit.all()


@pytest.mark.parametrize("first", [0.5, 1.0])
def test_abnormal_turnover_requires_positive_long_mean(first):
    values = [first] + [0.0] * 229 + [-1.0] + [0.0] * 19
    panel = pd.DataFrame(
        dict(
            asset_id="asset",
            timestamp=pd.bdate_range("2020-01-02", periods=250),
            turnover=values,
        )
    )
    output = calculate_price_volume_factors(
        panel, factor_columns=("abnormal_turnover_20_250", "turnover_20d")
    )
    assert pd.isna(output.abnormal_turnover_20_250.iloc[-1])
    assert output.turnover_20d.iloc[-1] == pytest.approx(-0.05)


def test_signed_rolling_baselines_keep_zero_masks_without_erasing_small_values():
    from math import fsum

    values = np.array(
        [
            100.1,
            100.3,
            -0.1,
            0.2,
            0.3,
            -0.2,
            -0.2,
            -100.3,
            -100.3,
            -1.3,
            -100.3,
            100.3,
            -0.3,
            -0.1,
            0.1,
            0.1,
            -0.1,
            100.3,
            0.1,
            0.2,
            1.3,
            1.0,
        ]
    )
    assert fsum(values[1:21]) == 0.0
    columns = (
        "avg_amount_20d",
        "log_avg_amount_20d",
        "amount_shock_20d",
        "volume_shock_20d",
    )
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=len(values)),
            "amount": values,
            "volume": values,
        }
    )
    actual = calculate_price_volume_factors(frame, factor_columns=columns)
    assert actual.avg_amount_20d.iloc[20] == 0.0
    assert np.isnan(actual.log_avg_amount_20d.iloc[20])
    assert actual.loc[21, list(columns[2:])].isna().all()
    clipped = calculate_price_volume_factors(frame.iloc[1:], factor_columns=columns)
    np.testing.assert_allclose(
        actual.loc[21, list(columns)].to_numpy(dtype=float),
        clipped.loc[20, list(columns)].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
    )

    turnover = np.r_[100.1, np.zeros(230), values[1:21]]
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=len(turnover)),
            "turnover": turnover,
        }
    )
    for current in (frame, frame.iloc[1:]):
        result = calculate_price_volume_factors(
            current, factor_columns=("turnover_250d", "abnormal_turnover_20_250")
        ).iloc[-1]
        assert result.turnover_250d == 0.0
        assert np.isnan(result.abnormal_turnover_20_250)

    # Small same-sign observations are not numerical zero; ordinary signed
    # nonzero and negative means retain their original ratio/log domains.
    signed = values.copy()
    signed[20] += 0.125
    for amounts in (np.full(22, 1e-8), signed, -np.ones(22)):
        frame = pd.DataFrame(
            {
                "asset_id": "a",
                "timestamp": pd.bdate_range("2024-01-01", periods=22),
                "amount": amounts,
                "volume": amounts,
            }
        )
        result = calculate_price_volume_factors(frame, factor_columns=columns)
        mean = fsum(amounts[1:21]) / 20
        assert result.avg_amount_20d.iloc[20] == pytest.approx(mean, rel=1e-12, abs=0)
        assert result.amount_shock_20d.iloc[21] == pytest.approx(
            (amounts[21] - mean) / mean, rel=1e-12, abs=1e-14
        )
        if mean > 0:
            assert result.log_avg_amount_20d.iloc[20] == pytest.approx(np.log(mean))
        else:
            assert np.isnan(result.log_avg_amount_20d.iloc[20])


@pytest.mark.parametrize("scale", (1e16, 1e308))
@pytest.mark.parametrize("batch_rows", (6, 1000))
def test_signed_amount_cancellation_persists_rank_direction(
    tmp_path, monkeypatch, scale, batch_rows
):
    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
    )

    dates = pd.bdate_range("2020-01-02", periods=20).as_unit("ns")
    returns = [0.01, 0.01, -0.01] + [0.0] * 17
    amounts = [scale, 1.0, scale] + [1.0] * 17
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=day,
                listing_date=dates[0],
                period_return=direction * value,
                amount=amount,
            )
            for asset, direction in (("negative", -1.0), ("positive", 1.0))
            for day, value, amount in zip(dates, returns, amounts, strict=True)
        ]
    )
    columns = ("signed_amount_flow_20d", "return_amount_imbalance_20d")
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = _produce_factors(
        tmp_path,
        "price_volume_facts",
        {"market_panel_1d": panel},
        columns,
        TimeWindow(dates[-1], dates[-1]),
    )
    exact_amounts = tuple(map(Fraction, amounts))
    exact_returns = tuple(map(Fraction, returns))
    expected = (
        float(Fraction(1) / sum(exact_amounts)),
        float(
            sum(
                value * amount
                for value, amount in zip(exact_returns, exact_amounts, strict=True)
            )
            / sum(
                abs(value) * amount
                for value, amount in zip(exact_returns, exact_amounts, strict=True)
            )
        ),
    )
    saved = saved.sort_values("asset_id")
    for column, value in zip(columns, expected, strict=True):
        assert value > 0.0
        np.testing.assert_array_equal(saved[column], [-value, value])
    saved["label"] = [-1.0, 1.0]
    model = EqualWeightRankEnsembleFactory(columns, "label").fit(
        saved, saved.label.to_numpy()
    )
    assert model.directions == (1.0, 1.0)
    assert model.predict(saved)[0] < model.predict(saved)[1]


@pytest.mark.parametrize("batch_rows", [1, 1000])
def test_trend_efficiency_cancellation_survives_producer_persistence_and_rank_model(
    tmp_path, monkeypatch, batch_rows
):
    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
    )

    dates = pd.bdate_range("2024-01-02", periods=60).as_unit("ns")
    sequences = {
        "small": [0.1, 1e-18, -0.1] + [0.0] * 57,
        "larger": [0.1, 2e-18, -0.1] + [0.0] * 57,
        "ordinary": [0.01] * 60,
        "missing": [0.01] * 59 + [np.nan],
        "zero": [0.0] * 60,
    }
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                timestamp=date,
                listing_date=dates[0],
                period_return=value,
            )
            for asset, values in sequences.items()
            for date, value in zip(dates, values, strict=True)
        ]
    )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = _produce_factors(
        tmp_path,
        "price_volume_facts",
        {"market_panel_1d": panel},
        ("trend_efficiency_60d",),
        TimeWindow(dates[-1], dates[-1]),
    )
    values = saved.set_index("asset_id").trend_efficiency_60d
    for asset in ("small", "larger", "ordinary"):
        terms = list(map(Fraction, sequences[asset]))
        expected = float(abs(sum(terms)) / sum(map(abs, terms)))
        assert values[asset] == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert values[["missing", "zero"]].isna().all()
    training = saved.loc[saved.asset_id.isin(("small", "larger"))].copy()
    training["label"] = training.asset_id.map({"small": 1.0, "larger": 2.0})
    model = EqualWeightRankEnsembleFactory(("trend_efficiency_60d",), "label").fit(
        training, training.label.to_numpy()
    )
    assert model.directions == (1.0,)
    scores = pd.Series(model.predict(training), index=training.asset_id)
    assert scores["small"] < scores["larger"]


@pytest.mark.parametrize("case", ("ordinary", "adjacent", "extreme", "cancellation"))
def test_partial_correlation_matches_independent_exact_binary_oracle(case):
    from decimal import Decimal, localcontext

    rng = np.random.default_rng(719)
    x, y = rng.normal(size=40), rng.normal(size=40)
    if case == "adjacent":
        x = 1e200 + np.arange(40) * np.spacing(1e200)
    elif case == "extreme":
        x[0], x[1], y[2] = 1e308, -1e308, np.nextafter(0.0, 1.0)
    elif case == "cancellation":
        x = np.tile([1.0, 0.0, -1.0, 0.0], 10)
        y = np.tile([1e-200, 1.0, 0.0, -1.0], 10)
    x[5::8] = np.nan
    y[7::8] = np.nan
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.date_range("2020-01-01", periods=40),
            "period_return": x,
            "amount": y,
            "filling_flag": 0,
        }
    )
    actual = calculate_price_volume_factors(
        frame, factor_columns=("return_amount_corr_20d",)
    )
    for end in range(19, 40):
        left, right = x[end - 19 : end + 1], y[end - 19 : end + 1]
        paired = np.isfinite(left) & np.isfinite(right)
        if paired.sum() < 15:
            assert np.isnan(actual.return_amount_corr_20d.iloc[end])
            continue
        a, b = list(map(Fraction, left[paired])), list(map(Fraction, right[paired]))
        n = len(a)
        cov = n * sum(u * v for u, v in zip(a, b, strict=True)) - sum(a) * sum(b)
        aa = n * sum(u * u for u in a) - sum(a) ** 2
        bb = n * sum(v * v for v in b) - sum(b) ** 2
        squared = cov * cov / (aa * bb)
        with localcontext() as context:
            context.prec = 90
            expected = float(
                (Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt()
            )
        expected = -expected if cov < 0 else expected
        assert actual.return_amount_corr_20d.iloc[end] == pytest.approx(
            expected, rel=2e-14, abs=0
        )
