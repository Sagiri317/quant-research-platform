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
from tests.helpers.paths import StorageLayout


def _produce_factors(root, data_id, inputs, columns, window):
    profile = scientific_catalogs()
    layout = StorageLayout(root)
    input_paths = {}
    for input_id, frame in inputs.items():
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
def test_abnormal_turnover_exact_windows_persist_across_chunks(
    tmp_path, monkeypatch, batch_rows
):
    dates = pd.bdate_range("2020-01-02", periods=252).as_unit("ns")
    sequences = {
        "extreme": [1e308] * 230 + [2.3e-16] * 22,
        "ordinary": list(map(float, range(1, 253))),
        "subnormal": [5e-324] * 252,
        "zero": [0.0] * 252,
        "negative": [-1.0] * 252,
        "missing": [1e308] * 230 + [2.3e-16] * 22,
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
        if stop >= 250 and np.isfinite(values).all():
            total = sum(map(Fraction, values))
            if total > 0:
                value = float(250 * sum(map(Fraction, values[-20:])) / (20 * total))
        expected.append(value)
    np.testing.assert_array_equal(saved.abnormal_turnover_20_250, expected)
    assert saved.loc[
        saved.asset_id.eq("extreme") & saved.timestamp.eq(dates[249]),
        "abnormal_turnover_20_250",
    ].iloc[0] == np.nextafter(0.0, 1.0)


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


def test_abnormal_turnover_keeps_unrepresentable_signed_result():
    values = [1e308] + [0.0] * 228 + [5e-324, -1e308] + [0.0] * 19
    panel = pd.DataFrame(
        dict(
            asset_id="asset",
            timestamp=pd.bdate_range("2020-01-02", periods=250),
            turnover=values,
        )
    )
    output = calculate_price_volume_factors(
        panel, factor_columns=("abnormal_turnover_20_250",)
    )
    assert output.abnormal_turnover_20_250.iloc[-1] == -np.inf


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
