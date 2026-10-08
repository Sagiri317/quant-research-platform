"""Adversarial contracts and exact kernel comparisons against independent references."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import (
    ModelSpec,
    TrainingWeighting,
    model_spec_payload,
)
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    TradeCostComponents,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeRule,
    FeeSchedule,
    prepare_trade_cost,
    trade_cost,
)
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.features.transforms import (
    _industry_size_residuals,
    apply_feature_transforms,
    canonical_feature_pipeline,
    canonical_feature_transforms,
)
from quant_research_platform.research.labels.forward_returns import (
    _session_axis,
    _validate_horizons,
    _validate_label_input_frame,
    add_forward_return_labels,
    forward_return_label_column,
)
from quant_research_platform.research.models.rank_ensemble import (
    EqualWeightRankEnsembleFactory,
    ICWeightedRankEnsembleFactory,
)
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)
from quant_research_platform.research.protocols import a_share, crypto_hft
from quant_research_platform.research.statistics import (
    cscv_pbo,
    probabilistic_and_deflated_sharpe,
)


def _reference_forward_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    frequency: DataFrequency | str = DataFrequency.DAILY,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
    sessions: Sequence[object],
    can_enter_column: str = "can_enter",
) -> pd.DataFrame:
    """Add entry-open to exit-open returns on the supplied trading-session axis."""

    horizons = _validate_horizons(horizons)
    validated = _validate_label_input_frame(
        frame,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )
    if can_enter_column not in validated.columns:
        raise ValueError(f"missing_forward_return_columns: {can_enter_column}")
    axis = _session_axis(sessions)
    result = validated.sort_values([asset_column, date_column], kind="mergesort").copy()
    result[date_column] = strict_timestamp_series(
        result[date_column], error="invalid_forward_return_timestamp"
    )
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    row_lookup = {
        (asset, _timestamp): position
        for position, (asset, _timestamp) in enumerate(
            zip(result[asset_column], result[date_column], strict=True)
        )
    }
    position_by_session = {timestamp: index for index, timestamp in enumerate(axis)}
    positions = result[date_column].map(position_by_session).to_numpy()
    prices = price.to_numpy(dtype="float64", copy=False)
    can_enter = strict_bool_series(
        result[can_enter_column],
        error="invalid_forward_return_can_enter",
        allow_null=True,
    ).to_numpy(copy=False)
    for horizon in horizons:
        labels = np.full(len(result), np.nan, dtype="float64")
        available = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
        for position, (asset, _timestamp) in enumerate(
            zip(result[asset_column], result[date_column], strict=True)
        ):
            session_position = positions[position]
            if pd.isna(session_position):
                continue
            entry_position = int(session_position) + 1
            exit_position = entry_position + horizon
            if exit_position >= len(axis):
                continue
            entry_date = axis[entry_position]
            exit_date = axis[exit_position]
            entry_row = row_lookup.get((asset, entry_date))
            exit_row = row_lookup.get((asset, exit_date))
            if entry_row is None or exit_row is None:
                continue
            if pd.isna(can_enter[position]) or not bool(can_enter[position]):
                continue
            entry_price = prices[entry_row]
            exit_price = prices[exit_row]
            if (
                not np.isfinite(entry_price)
                or entry_price <= 0.0
                or not np.isfinite(exit_price)
                or exit_price <= 0.0
            ):
                continue
            with np.errstate(over="ignore"):
                value = exit_price / entry_price - 1.0
            if not np.isfinite(value):
                continue
            labels[position] = value
            available[position] = exit_date.to_datetime64()
        label_column = forward_return_label_column(horizon, frequency)
        result[label_column] = labels
        result[f"{label_column}_available_at"] = available
    return result.reset_index(drop=True)


def _reference_trade_cost(
    execution_date: date,
    market: str,
    side: str,
    notional: float,
    adv20: float,
    schedule: FeeSchedule,
    assumptions: ExecutionAssumptions,
    *,
    face_value_notional: float | None = None,
) -> TradeCostComponents:
    """Calculate statutory and modeled costs for one executed trade."""

    if not isinstance(schedule, FeeSchedule) or not isinstance(
        assumptions, ExecutionAssumptions
    ):
        raise ValueError("a_share_trade_cost_inputs_invalid")
    try:
        notional_value = float(notional)
        adv_value = float(adv20)
    except (TypeError, ValueError) as exc:
        raise ValueError("a_share_trade_cost_inputs_invalid") from exc
    if (
        isinstance(notional, bool)
        or isinstance(adv20, bool)
        or not np.isfinite(notional_value)
        or not np.isfinite(adv_value)
        or notional_value < 0.0
        or adv_value <= 0.0
    ):
        raise ValueError("a_share_trade_cost_inputs_invalid")
    if face_value_notional is not None:
        try:
            face_value = float(face_value_notional)
        except (TypeError, ValueError) as exc:
            raise ValueError("a_share_face_value_notional_invalid") from exc
        if (
            isinstance(face_value_notional, bool)
            or not np.isfinite(face_value)
            or face_value <= 0.0
        ):
            raise ValueError("a_share_face_value_notional_invalid")
    else:
        face_value = None
    rules = schedule.rules_for(execution_date, market=market, side=side)
    required_codes = {"BROKER_COMMISSION", "TRANSFER_FEE"}
    if side == "sell":
        required_codes.add("STAMP_DUTY")
    by_code: dict[str, FeeRule] = {rule.fee_code: rule for rule in rules}
    if not required_codes.issubset(by_code):
        raise ValueError("a_share_fee_schedule_missing")
    fee_values: dict[str, float] = {}
    for code in required_codes:
        rule = by_code[code]
        if rule.basis == "face_value":
            if face_value is None:
                raise ValueError("a_share_face_value_notional_required")
            base = face_value
        else:
            base = notional_value
        fee_values[code] = base * rule.rate
    amount, amount_exponent = np.frexp(notional_value)
    adv_amount, adv_exponent = np.frexp(adv_value)
    slippage_rate, slippage_exponent = np.frexp(assumptions.slippage_bps)
    impact_rate, impact_exponent = np.frexp(assumptions.impact_bps)
    root_exponent, odd_exponent = divmod(amount_exponent - adv_exponent, 2)
    participation_root = np.sqrt(np.ldexp(amount / adv_amount, odd_exponent))
    slippage = np.ldexp(
        amount * slippage_rate / 10_000.0, amount_exponent + slippage_exponent
    )
    impact = np.ldexp(
        amount * impact_rate / 10_000.0 * participation_root,
        amount_exponent + impact_exponent + root_exponent,
    )
    commission = fee_values["BROKER_COMMISSION"]
    stamp_duty = fee_values.get("STAMP_DUTY", 0.0)
    transfer_fee = fee_values["TRANSFER_FEE"]
    return TradeCostComponents(
        notional=notional_value,
        commission_cost=float(commission),
        stamp_duty_cost=float(stamp_duty),
        transfer_fee_cost=float(transfer_fee),
        slippage_cost=float(slippage),
        impact_cost=float(impact),
        source_identity=schedule.content_identity,
    )


def _reference_residuals(target, industries, sizes):
    counts = industries.value_counts(sort=False)
    sums = {
        group: [Fraction(), Fraction(), Fraction(), Fraction()]
        for group in counts.index
    }
    for i, (group, value) in enumerate(zip(industries, target, strict=True)):
        y, x = (
            Fraction(float(value)),
            Fraction() if sizes is None else Fraction(float(sizes[i])),
        )
        row = sums[group]
        row[0] += y
        row[1] += x
        row[2] += x * x
        row[3] += x * y
    slope = Fraction()
    if sizes is not None:
        xx = sum(
            (row[2] - row[1] ** 2 / int(counts[g]) for g, row in sums.items()),
            Fraction(),
        )
        xy = sum(
            (row[3] - row[0] * row[1] / int(counts[g]) for g, row in sums.items()),
            Fraction(),
        )
        slope = xy / xx
    return np.array(
        [
            float(
                Fraction(float(y))
                - sums[g][0] / int(counts[g])
                - slope
                * (
                    Fraction()
                    if sizes is None
                    else Fraction(float(sizes[i])) - sums[g][1] / int(counts[g])
                )
            )
            for i, (g, y) in enumerate(zip(industries, target, strict=True))
        ]
    )


@pytest.mark.parametrize(
    "name,count",
    [
        ("core6", 6),
        ("pv25", 6),
        ("pv_expanded", 6),
        ("fusion_nf_core", 12),
        ("fusion_nf_expectations", 16),
    ],
)
def test_selected_core_intersection_does_not_shrink_research(name, count):
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/research/a_share"
        / (name + ".json")
    )
    configuration = json.loads(path.read_text(encoding="utf-8"))
    bound = bind_research_configuration(research_profile(), configuration)
    protocol = bound.protocol(configuration["scenario_id"])
    assert len(protocol.manifest()["statistics"]["fama_macbeth_core_factors"]) == count
    assert protocol.factor_columns == tuple(configuration["evaluation"]["factors"])
    if name == "fusion_nf_core":
        assert len(protocol.factor_columns) == 44


def test_equal_timestamp_decay_is_one_frozen_model_contract():
    stamps = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"])
    weighting = TrainingWeighting(half_life=1)
    weights = equal_timestamp_sample_weights(stamps, weighting)
    np.testing.assert_allclose(weights, np.array([2, 2, 8, 16]) / 7)
    assert weights.sum() == pytest.approx(4)
    spec = ModelSpec(
        model_id="x",
        model_algorithm="linear_regression",
        input_feature_columns=("x",),
        label_column="y",
    )
    decayed = ModelSpec(
        model_id="x",
        model_algorithm="linear_regression",
        input_feature_columns=("x",),
        label_column="y",
        training_weighting={"method": "equal_timestamp", "half_life": 756},
    )
    assert canonical_json_hash(model_spec_payload(spec)) != canonical_json_hash(
        model_spec_payload(decayed)
    )
    assert model_spec_payload(decayed)["training_weighting"]["half_life"] == 756
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                np.repeat(pd.date_range("2020-01-01", periods=3), 3)
            ),
            "x": [1.0, 2.0, 3.0] * 3,
            "y": [-1.0, -2.0, -3.0] * 2 + [1.0, 2.0, 3.0],
        }
    )
    weights = equal_timestamp_sample_weights(frame.timestamp, weighting)
    for factory in (EqualWeightRankEnsembleFactory, ICWeightedRankEnsembleFactory):
        plain = factory(("x",), "y").fit(frame, frame.y.to_numpy())
        decayed = factory(("x",), "y").fit(
            frame, frame.y.to_numpy(), sample_weight=weights
        )
        assert plain.predict(frame)[0] > 0
        assert decayed.predict(frame)[0] < 0


def test_canonical_pipeline_is_executable_without_a_scenario_bypass():
    n = 160
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": np.where(np.arange(n) % 2, "A", "B"),
            "log_float_market_cap": np.sin(np.arange(n) / 11) + 10,
            "x": np.cos(np.arange(n) / 7),
            "eligible": np.arange(n) > 4,
        }
    )
    frame.loc[10:15, "x"] = np.nan
    columns = ("x", "log_float_market_cap")
    transforms = canonical_feature_transforms(columns)
    assert len(transforms) == 1 and transforms[0].transform == "a_share_canonical"
    pd.testing.assert_frame_equal(
        apply_feature_transforms(frame, transforms),
        canonical_feature_pipeline(frame, factor_columns=columns),
    )


def test_crypto_labels_have_one_horizon_and_maturity_catalog():
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_LABEL_DATA_ID,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        catalog as processed_catalog,
    )

    catalog = crypto_hft.label_catalog()
    assert len(catalog.list()) == 9
    profile = research_profile("crypto_binance_hft")
    assert profile.labels.list() == catalog.list()
    spec = processed_catalog.hft_processed_catalog().get(HFT_LABEL_DATA_ID)
    assert json.loads(spec.parameters_json)["labels"] == [
        item.to_manifest() for item in catalog.list()
    ]
    for item in catalog.list():
        assert item.horizon_unit == "milliseconds"
        assert crypto_hft.label_horizon_ms(item.output_column) == item.horizon
        assert item.available_at_column in spec.required_columns
    assert all(
        item.horizon_unit == "market_sessions"
        for item in a_share.label_catalog().list()
    )


def test_single_candidate_keeps_psr_and_explains_dsr_pbo():
    returns = np.array([0.01, -0.02, 0.03, 0.005, -0.01] * 4)
    result = probabilistic_and_deflated_sharpe(
        returns, benchmark_sharpe=0.0, trial_sharpes=[0.2]
    )
    assert 0 <= result.psr <= 1
    assert result.dsr is None and result.dsr_reason == "insufficient_candidate_count"
    pbo = cscv_pbo(returns[:, None], slice_count=10, max_combinations=comb(10, 5))
    assert pbo.status == "UNDEFINED" and pbo.reason == "insufficient_candidate_count"


def test_training_row_count_cache_is_exact_and_cutoff_scoped():
    from types import SimpleNamespace

    from quant_research_platform.contracts.experiments import (
        training_availability_columns,
    )
    from quant_research_platform.research.models.production import _count_training_rows

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "asset_id": ["a", "a"],
            "eligible": [True, True],
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            training_availability_columns("y")[0]: pd.to_datetime(
                ["2020-01-01", "2020-01-02"]
            ),
        }
    )
    calls = []

    def chunks(columns, **kwargs):
        calls.append(kwargs)
        yield frame.loc[:, list(columns)]

    dataset = SimpleNamespace(primary_key=("timestamp", "asset_id"))
    cache = {}
    for algorithm in ("linear_regression", "ridge_regression"):
        model = ModelSpec(
            model_id=algorithm,
            model_algorithm=algorithm,
            input_feature_columns=("x",),
            label_column="y",
        )
        assert (
            _count_training_rows(
                dataset, model, chunks=chunks, train_end="2020-01-02", cache=cache
            )
            == 2
        )
    assert len(calls) == 1
    assert (
        _count_training_rows(
            dataset, model, chunks=chunks, train_end="2020-01-01", cache=cache
        )
        == 1
    )
    assert len(calls) == 2


def test_training_count_failure_closes_borrowed_dataset_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from quant_research_platform.contracts.experiments import (
        training_availability_columns,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.models import fitting, production
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from tests.storage.low_level_fixtures import write_low_level_parquet_artifact

    dates = pd.to_datetime(["2020-01-01", "2020-01-02"]).as_unit("ns")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 2,
            "asset_id": ["a"] * 2,
            "timestamp": dates,
            "eligible": [True] * 2,
            "x": [1.0, 2.0],
            "y": [2.0, 3.0],
            training_availability_columns("y")[0]: dates,
        }
    )
    root = tmp_path / "artifact"
    primary_key = ("market_id", "asset_id", "timestamp")
    write_low_level_parquet_artifact(
        root, frame, primary_key=primary_key, partition_columns=("year",)
    )
    dataset = SimpleNamespace(primary_key=primary_key, market_id="a_share")
    model = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_feature_columns=("x",),
        label_column="y",
    )
    primary = MemoryError("training_filter_failure")
    sources = []
    cache = {}

    def fail_filter(*args, **kwargs):
        raise primary

    monkeypatch.setattr(fitting, "legal_training_rows", fail_filter)
    session = open_dataset_session(root, columns=None, time_window=None)

    def chunks(columns, **kwargs):
        source = production._model_input_chunks(
            session,
            dataset_spec=dataset,
            columns=columns,
            time_window=TimeWindow("2020-01-01", "2020-01-02"),
        )
        sources.append(source)
        return source

    try:
        with pytest.raises(MemoryError) as raised:
            production._count_training_rows(
                dataset,
                model,
                chunks=chunks,
                train_end="2020-01-02",
                cache=cache,
            )
        assert raised.value is primary
        assert not cache
        assert sources and all(source.gi_frame is None for source in sources)
    finally:
        for source in sources:
            source.close()


def test_risk_manifest_serializes_the_same_runtime_config():
    config = FactorRiskConfig(
        factor_window=15,
        specific_window=8,
        minimum_factor_dates=3,
        minimum_specific_dates=2,
        factor_half_life=7,
        specific_half_life=3,
        diagonal_shrinkage=0.4,
        annualization=250,
    )
    protocol = a_share.AShareProtocol("pv_all", risk_config=config)
    assert protocol.manifest()["risk"] == asdict(config)
    assert FactorRiskState(style_columns=("x",), config=config).config is config


def test_forward_labels_exact_on_sparse_sessions_and_extreme_prices():
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", periods=80)
    frame = pd.DataFrame(
        [
            (asset, timestamp, float(rng.lognormal()), bool(rng.integers(0, 2)))
            for asset in ("a", "b", "c", "d", "e")
            for timestamp in dates
            if rng.random() > 0.05
        ],
        columns=("asset_id", "timestamp", "adj_open", "can_enter"),
    )
    frame.loc[:3, "adj_open"] = [0.0, np.nan, 1e-300, 1e300]
    frame = frame.sample(frac=1, random_state=9)
    options = dict(horizons=(1, 5, 20), sessions=dates)
    expected = _reference_forward_labels(frame, **options)
    actual = add_forward_return_labels(frame, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("with_size", [False, True])
def test_neutralization_is_bit_exact_for_dyadic_inputs(with_size):
    rng = np.random.default_rng(27)
    target = np.ldexp(rng.normal(size=700), rng.integers(-100, 100, size=700))
    industry = pd.Series(np.arange(700) % 7)
    sizes = rng.normal(size=700) if with_size else None
    expected = _reference_residuals(target, industry, sizes)
    actual = _industry_size_residuals(target, industry, sizes)
    np.testing.assert_array_equal(actual, expected)


def test_prepared_costs_are_bit_exact_for_201_random_cases():
    rng = np.random.default_rng(61)
    schedule = FeeSchedule(
        tuple(
            FeeRule(
                code,
                date(2020, 1, 1),
                date(2030, 1, 1),
                "shanghai",
                side,
                "amount",
                rate,
                True,
            )
            for side in ("buy", "sell")
            for code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.00001),
                ("STAMP_DUTY", 0.001),
            )
        ),
        "a" * 64,
    )
    assumptions = ExecutionAssumptions(1e9, 5.0, 25.0, 0.05, (0.05,))
    notionals = np.exp2(rng.uniform(-300, 300, 201))
    adv = np.exp2(rng.uniform(-300, 300, 201))
    for side in ("buy", "sell"):
        prepared = prepare_trade_cost(
            date(2026, 1, 1), "shanghai", side, schedule, assumptions
        )
        expected = [
            _reference_trade_cost(
                date(2026, 1, 1),
                "shanghai",
                side,
                float(n),
                float(a),
                schedule,
                assumptions,
            )
            for n, a in zip(notionals, adv, strict=True)
        ]
        totals = prepared.totals(notionals, adv)
        np.testing.assert_array_equal(totals, [item.total_cost for item in expected])
        for n, a, item in zip(notionals, adv, expected, strict=True):
            assert (
                trade_cost(
                    date(2026, 1, 1),
                    "shanghai",
                    side,
                    float(n),
                    float(a),
                    schedule,
                    assumptions,
                )
                == item
            )


@pytest.mark.parametrize("count", [32, 1080])
def test_decay_normalizes_before_float64_underflow(count):
    from quant_research_platform.research.models.scaler import ModelMatrixScaler

    timestamps = pd.date_range("2020-01-01", periods=count)
    actual = equal_timestamp_sample_weights(timestamps, TrainingWeighting(half_life=1))
    normalizer = 2 - Fraction(1, 2 ** (count - 1))
    expected = np.array(
        [
            float(count * Fraction(1, 2**age) / normalizer)
            for age in range(count - 1, -1, -1)
        ]
    )
    assert (actual > 0).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)
    assert actual.sum() == pytest.approx(count)
    frame = pd.DataFrame({"x": np.arange(count, dtype="float64")})
    scaler = ModelMatrixScaler.fit(frame, sample_weight=actual)
    assert np.isfinite(scaler.transform(frame).to_numpy()).all()


def test_decay_rejects_truly_unrepresentable_positive_weights():
    with pytest.raises(ValueError, match="sample_weights_unrepresentable"):
        equal_timestamp_sample_weights(
            pd.date_range("2020-01-01", periods=2000),
            TrainingWeighting(half_life=1),
        )
