from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import statistics
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_read_bytes,
    filesystem_stat,
    filesystem_write_bytes,
    native_filesystem_path,
)
from quant_research_platform.storage.semantic_artifacts import write_semantic_artifact
from tests.helpers.research_configuration import research_configuration

MAKER_MODEL_IDS = next(
    (item["price_model_id"], item["proxy_model_id"])
    for item in research_configuration("crypto", "baseline")["portfolios"]
    if item.get("price_model_id") is not None
)


@pytest.fixture
def tables(request):
    parameters = getattr(request.node, "callspec", None)
    parameters = {} if parameters is None else parameters.params
    selected = parameters.get(
        "member", parameters.get("name", parameters.get("table", ""))
    ).removesuffix(".parquet")
    group = (
        "factor"
        if "factor_case" in parameters
        or selected
        in {
            "factor_diagnostics",
            "fama_macbeth",
            "fama_macbeth_periods",
        }
        else "research"
    )
    row = dict(
        family_id="family",
        factor_id="factor",
        prediction_id="prediction",
        quarter_id="2023Q1",
        selection_group="global",
        decision_id="final",
        config_id="config",
        metric_id="validation_rankic",
        test_id="spa_reality_check",
        regression_id="single:factor",
        coefficient_id="factor",
        access_window="development",
        benchmark_id="benchmark",
        statistic_id="p_value",
        selection_status="CHAMPION",
        status="DEFINED",
        reason="",
        white_reason="",
        timestamp=pd.Timestamp("2023-01-02"),
        observation_count=32,
        period_count=32,
        mean=0.2,
        standard_error=0.1,
        t_statistic=2.0,
        p_value=0.05,
        ci_lower=0.0,
        ci_upper=0.4,
        bootstrap_lower=0.0,
        bootstrap_upper=0.4,
        fdr_method="BH",
        fdr_alpha=0.05,
        fdr_adjusted_p_value=0.05,
        planned_hypothesis_count=1,
        defined_hypothesis_count=1,
        missing_hypothesis_count=0,
        bootstrap_block_size=20,
        bootstrap_repetitions=10000,
        bootstrap_seed=42,
        value=0.2,
        loss=0.2,
        white_p_value=0.05,
        white_defined_period_count=32,
        white_undefined_period_count=0,
        common_date=1,
        common_observation_count=32,
        asset_coverage_id="d" * 64,
        block_size=20,
        repetitions=10000,
        alpha=0.05,
        seed=42,
    )
    result = {
        member.removesuffix(".parquet"): pd.DataFrame(
            [
                {
                    column: row[column]
                    for column in statistics.STATISTICS_TABLE_COLUMNS[member]
                }
            ]
        )
        for member in statistics.STATISTICS_TABLE_GROUPS[group]
        if member != "model_group_comparison.parquet"
    }
    if group == "research":
        result["model_group_comparison"] = pa.Table.from_pylist(
            [], schema=statistics._ARROW_SCHEMAS["model_group_comparison.parquet"]
        ).to_pandas()
    for frame in result.values():
        if "timestamp" in frame:
            frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    return result


@pytest.fixture
def envelope():
    return build_business_manifest_envelope(
        logical_artifact_id="statistics",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-02",
            "end": "2023-03-31",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )


@pytest.fixture
def maker_calibration_tables():
    from quant_research_platform.markets.implementations.crypto.execution import (
        replay_maker_calibration,
    )
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_RESEARCH_DATASET_ID,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        CryptoMakerCandidate,
        protocol,
    )
    from tests.helpers.binance_hft import _quote, hft_research_plan
    from tests.markets.crypto.calibration_test_support import maker_event_rows
    from tests.markets.crypto.test_hft_execution import _state, _trade

    plan = hft_research_plan()
    plan["folds"][0]["validation"] = {
        "start": "1970-01-01T00:00:02.100000",
        "end": "1970-01-01T00:00:03.500000",
    }
    research_protocol = replace(
        protocol(HFT_RESEARCH_DATASET_ID, research_plan=plan),
        configuration_json=json.dumps(
            research_configuration("crypto", "baseline", plan=plan)
        ),
    )
    candidate = next(
        candidate
        for candidate in research_protocol.portfolio_candidates()
        if isinstance(candidate, CryptoMakerCandidate)
        and candidate.asset_id == "btc_usdt"
        and (candidate.model_ids == MAKER_MODEL_IDS)
    )
    manifest = research_protocol.target_manifest(
        candidate.calibration_target_id, "STATISTIC"
    )
    reference = manifest["research_plan"]["execution"]["maker"]
    samples, events = ([], [])

    def on_sample(**sample):
        samples.append(
            {
                "fold_id": "fold_1",
                "side": sample["side"],
                "sample_id": sample["sample_id"],
                "decision_at": sample["decision_ns"],
                "prediction_end": sample["prediction_end_ns"],
                "label_available_at": sample["available_ns"],
                "price_prediction": sample["prediction"],
                "observed_return": sample["observed_return"],
                "forecast_error": sample["observed_return"] - sample["prediction"],
                "filled_quantity": sample["filled"],
                "quote_quantity": sample["quote_quantity"],
                "terminal_reason": sample["terminal_reason"],
                "book_segment_id": sample["book_segment_id"],
                "trade_observation_id": sample["trade_observation_id"],
                "initial_book_event_id": sample["initial_book_event_id"],
                "initial_bid_index": sample["initial_bid_index"],
                "initial_ask_index": sample["initial_ask_index"],
                "event_start": sample["event_start"],
                "event_end": sample["event_end"],
            }
        )

    def on_event(*, event_id, event):
        events.extend(
            {"fold_id": "fold_1", **row} for row in maker_event_rows(event_id, event)
        )

    replay_maker_calibration(
        [_state(stamp) for stamp in (2100, 2150, 2650, 2700)],
        [
            _trade(stamp, price="99.5", quantity="3", maker_buyer=True)
            for stamp in (2180, 2400)
        ],
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "price_prediction": 0.0,
                "price_label": 0.0,
                "proxy_prediction": 1.0,
                "label_available_at_ns": (stamp + 500) * 1000000,
            }
            for stamp in (2100, 2200)
        ],
        order_quantity=reference["order_quantity"],
        latency_ms=reference["latency_ms"],
        observed_coverage={"s1": (2100000000, 2800000000)},
        trade_coverage={"trade-connection-1": (2100000000, 2800000000)},
        cutoff_ns=3500000000,
        on_sample=on_sample,
        on_event=on_event,
    )
    frames = {
        "crypto_maker_samples": pd.DataFrame(
            samples,
            columns=statistics.STATISTICS_TABLE_COLUMNS["crypto_maker_samples.parquet"],
        ),
        "crypto_maker_events": pd.DataFrame(
            events,
            columns=statistics.STATISTICS_TABLE_COLUMNS["crypto_maker_events.parquet"],
        ),
        "crypto_maker_calibration": pd.DataFrame(
            [
                {
                    "fold_id": "fold_1",
                    "side": side,
                    "price_unit": "0.1",
                    "sample_count": 2,
                    "zero_fill_count": 0 if side == "buy" else 2,
                    "post_only_rejection_count": 0,
                    "excluded_decisions": 0,
                    "minimum_samples": 2,
                    "status": "DEFINED",
                    "reason": "",
                }
                for side in ("buy", "sell")
            ],
            columns=statistics.STATISTICS_TABLE_COLUMNS[
                "crypto_maker_calibration.parquet"
            ],
        ),
    }
    for column in ("decision_at", "prediction_end", "label_available_at"):
        frames["crypto_maker_samples"][column] = pd.to_datetime(
            frames["crypto_maker_samples"][column], unit="ns"
        )
    frames["crypto_maker_events"]["event_at"] = pd.to_datetime(
        frames["crypto_maker_events"]["event_at"], unit="ns"
    )
    assert frames["crypto_maker_events"].event_id.nunique() == 5
    assert len(frames["crypto_maker_samples"]) == 4
    assert frames["crypto_maker_samples"].loc[
        lambda frame: frame.side.eq("buy"), "filled_quantity"
    ].tolist() == [1.0, 1.0]
    envelope = build_business_manifest_envelope(
        logical_artifact_id=candidate.calibration_target_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01T00:00:02.100000",
            "end": "1970-01-01T00:00:03.500000",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"target_contract": {"protocol": manifest}},
    )
    return (frames, envelope)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "truncate",
        "immature",
        "coordinate",
        "book_zero",
        "level_order",
        "reference",
        "pair",
        "unit_invalid",
        "unit_mixed",
        "tail",
        "empty",
    ],
)
def test_mature_shared_maker_tape_round_trip_and_rejects_corruption(
    tmp_path, maker_calibration_tables, mutation
):
    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        validate_maker_calibration,
    )
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
    )

    root = statistics.write_statistics_artifact(
        tmp_path / "valid",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )
    original = read_semantic_artifact_record(root)
    artifact = statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=validate_maker_calibration,
    )
    for member in statistics.STATISTICS_TABLE_GROUPS["crypto_maker_calibration"]:
        with artifact.table(member) as actual:
            pd.testing.assert_frame_equal(
                actual, tables[member.removesuffix(".parquet")]
            )
    assert "crypto_maker_fills.parquet" not in artifact._members
    if mutation is None:
        for missing_validator in (None, object()):
            with pytest.raises(
                ValueError, match="statistics_business_validator_required"
            ):
                statistics.write_statistics_artifact(
                    tmp_path / "missing-validator",
                    envelope=envelope,
                    tables=produce,
                    table_group="crypto_maker_calibration",
                    validate_tables=missing_validator,
                )
            with pytest.raises(
                ValueError, match="statistics_business_validator_required"
            ):
                statistics.read_statistics_artifact(
                    root,
                    expected_identity=envelope.instance_identity,
                    validate_tables=missing_validator,
                )
        assert not filesystem_exists(tmp_path / "missing-validator")
        return
    if mutation == "truncate":
        tables["crypto_maker_events"] = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.lt(4)].copy()
        )
    elif mutation == "tail":
        extra = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.eq(4)].copy()
        )
        extra.event_id = 5
        extra.event_at += pd.Timedelta(milliseconds=100)
        tables["crypto_maker_events"] = pd.concat(
            [tables["crypto_maker_events"], extra], ignore_index=True
        )
    elif mutation == "empty":
        tables["crypto_maker_samples"] = tables["crypto_maker_samples"].iloc[:0]
    elif mutation == "immature":
        tables["crypto_maker_samples"].loc[0, "label_available_at"] = pd.Timestamp(
            "1970-01-01T00:00:04"
        )
    elif mutation == "coordinate":
        tables["crypto_maker_samples"].loc[0, "initial_book_event_id"] = 2
    elif mutation == "book_zero":
        tables["crypto_maker_events"].loc[0, "quantity"] = 0.0
    elif mutation == "level_order":
        tables["crypto_maker_events"].loc[1, "price_index"] = 1000
    elif mutation == "reference":
        tables["crypto_maker_samples"].loc[0, "filled_quantity"] = 0.5
    elif mutation == "unit_invalid":
        tables["crypto_maker_calibration"]["price_unit"] = "invalid"
    elif mutation == "unit_mixed":
        tables["crypto_maker_calibration"].loc[0, "price_unit"] = "0.01"
    else:
        tables["crypto_maker_samples"].loc[1, "price_prediction"] = 0.001
        tables["crypto_maker_samples"].loc[1, "forecast_error"] = -0.001
    with pytest.raises(
        ValueError,
        match="order_book_price_unit_invalid"
        if mutation == "unit_invalid"
        else "crypto_calibration_",
    ):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            envelope=envelope,
            tables=produce,
            table_group="crypto_maker_calibration",
            validate_tables=validate_maker_calibration,
        )
    assert not tuple(filesystem_glob(tmp_path / "rejected" / "instances", "*"))

    def write_corruption(staging):
        for name, frame in tables.items():
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[f"{name}.parquet"],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / f"{name}.parquet"),
            )

    bad = write_semantic_artifact(tmp_path / "corrupted", envelope, write_corruption)
    with pytest.raises(
        ValueError,
        match="order_book_price_unit_invalid"
        if mutation == "unit_invalid"
        else "crypto_calibration_",
    ):
        statistics.read_statistics_artifact(
            bad,
            expected_identity=envelope.instance_identity,
            validate_tables=validate_maker_calibration,
        )
    assert read_semantic_artifact_record(root) == original


@pytest.mark.parametrize(
    "member",
    (
        *statistics.STATISTICS_TABLE_GROUPS["research"],
        *statistics.STATISTICS_TABLE_GROUPS["factor"],
    ),
)
@pytest.mark.parametrize("mutation", ["status", "reason", "count", "undefined"])
def test_research_table_rejects_invalid_values_at_write_and_read(
    tmp_path, tables, envelope, member, mutation
):
    name = member.removesuffix(".parquet")
    if name == "model_group_comparison":
        tables[name] = pd.DataFrame([_defined_group_comparison_row()])
    if mutation == "status":
        tables[name].loc[0, "status"] = "UNKNOWN"
    elif mutation == "reason":
        tables[name].loc[0, "reason"] = "contradicts_defined_status"
    elif mutation == "undefined":
        tables[name].loc[0, "status"] = "UNDEFINED"
    else:
        count = "period_count" if name == "fama_macbeth" else "observation_count"
        tables[name].loc[0, count] = -1
    error = (
        "statistics_table_count_invalid"
        if mutation == "count"
        else "statistics_group_comparison_invalid"
        if name == "model_group_comparison"
        else "statistics_table_status_invalid"
        if mutation == "status"
        else "statistics_table_status_value_invalid"
    )
    _assert_rejected_at_write_and_read(
        tmp_path, tables, envelope, member, error=f"^{error}$"
    )


def _assert_rejected_at_write_and_read(
    tmp_path,
    tables,
    envelope,
    member,
    *,
    table_group=None,
    error="statistics_table_",
):
    if table_group is None:
        table_group = (
            "factor"
            if member in statistics.STATISTICS_TABLE_GROUPS["factor"]
            else "research"
        )
    with pytest.raises(ValueError, match=error):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            envelope=envelope,
            tables=tables,
            table_group=table_group,
        )

    def write_unvalidated(staging):
        for table_name, frame in tables.items():
            table_member = f"{table_name}.parquet"
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[table_member],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / table_member),
            )

    root = write_semantic_artifact(tmp_path / "persisted", envelope, write_unvalidated)
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with pytest.raises(ValueError, match=error):
        with artifact.table(member):
            pytest.fail("invalid persisted values must not reach the consumer")


@pytest.mark.parametrize(
    "name,column",
    [
        ("factor_diagnostics", "mean"),
        ("prediction_diagnostics", "mean"),
        ("model_trials", "value"),
        ("fama_macbeth", "mean"),
        ("fama_macbeth_periods", "value"),
        ("model_losses", "loss"),
        ("model_comparison", "mean"),
        ("model_reality_checks", "value"),
    ],
)
def test_defined_research_measurements_cannot_be_missing(
    tmp_path, tables, envelope, name, column
):
    tables[name].loc[0, column] = float("nan")
    _assert_rejected_at_write_and_read(tmp_path, tables, envelope, f"{name}.parquet")


@pytest.mark.parametrize(
    "member",
    (
        *statistics.STATISTICS_TABLE_GROUPS["research"],
        *statistics.STATISTICS_TABLE_GROUPS["factor"],
    ),
)
@pytest.mark.parametrize("value", [True, 1.5, 2**63])
def test_research_table_counts_reject_lossy_coercion(
    tmp_path, tables, envelope, member, value
):
    name = member.removesuffix(".parquet")
    if name == "model_group_comparison":
        tables[name] = pd.DataFrame([_defined_group_comparison_row()])
    count = "period_count" if name == "fama_macbeth" else "observation_count"
    tables[name][count] = pd.Series([value], dtype=object)
    with pytest.raises(ValueError, match="statistics_table_count_invalid"):
        statistics.write_statistics_artifact(
            tmp_path / "rejected",
            envelope=envelope,
            tables=tables,
            table_group="factor" if "factor_diagnostics" in tables else "research",
        )


@pytest.mark.parametrize("count", (32, 64))
def test_group_comparison_integral_counts_round_trip(tmp_path, tables, envelope, count):
    tables["model_group_comparison"] = pd.DataFrame([_defined_group_comparison_row()])
    tables["model_group_comparison"]["observation_count"] = pd.Series(
        [count], dtype=object
    )
    root = statistics.write_statistics_artifact(
        tmp_path / "valid",
        envelope=envelope,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("model_group_comparison.parquet") as frame:
        assert frame.observation_count.tolist() == [count]
        assert frame.calendar_count.tolist() == [64]


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("factor_diagnostics", "p_value", 1.1),
        ("factor_diagnostics", "fdr_adjusted_p_value", -0.1),
        ("factor_diagnostics", "fdr_alpha", 0.0),
        ("prediction_diagnostics", "standard_error", -1.0),
        ("prediction_diagnostics", "ci_lower", 1.0),
        ("prediction_diagnostics", "bootstrap_lower", 1.0),
        ("fama_macbeth_periods", "white_p_value", 1.1),
        ("model_reality_checks", "alpha", 1.0),
        ("model_reality_checks", "value", 1.1),
        ("model_reality_checks", "statistic_id", "superior"),
    ],
)
def test_research_measurement_domains(tmp_path, tables, envelope, table, column, value):
    tables[table].loc[0, column] = value
    _assert_rejected_at_write_and_read(tmp_path, tables, envelope, f"{table}.parquet")


@pytest.mark.parametrize(
    "factor_case", ["formal", "diagnostic_only", "no_factor_columns"]
)
def test_research_statistics_tables_round_trip_by_factor_case(
    tmp_path, tables, envelope, factor_case
):
    factor = tables["factor_diagnostics"]
    if factor_case != "formal":
        factor["fdr_method"] = "NOT_APPLICABLE"
        factor["fdr_alpha"] = factor["fdr_adjusted_p_value"] = float("nan")
        factor["family_id"] = factor_case
    if factor_case == "no_factor_columns":
        factor["family_id"] = factor["factor_id"] = "none"
        factor["status"] = "UNDEFINED"
        factor["reason"] = factor_case
        factor.loc[
            :,
            [
                "mean",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
        ] = float("nan")
        factor.loc[
            :,
            [
                "observation_count",
                "planned_hypothesis_count",
                "defined_hypothesis_count",
                "missing_hypothesis_count",
            ],
        ] = 0
    root = statistics.write_statistics_artifact(
        tmp_path / "statistics",
        envelope=envelope,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    for member in statistics.STATISTICS_TABLE_GROUPS["factor"]:
        with artifact.table(member) as actual:
            pd.testing.assert_frame_equal(
                actual, tables[member.removesuffix(".parquet")]
            )


def test_writer_cleanup_preserves_primary(tmp_path, tables, envelope, monkeypatch):
    primary = KeyboardInterrupt("write_interrupted")
    original = pq.ParquetWriter.close

    def cleanup_failure(self):
        original(self)
        raise RuntimeError("writer_close_failed")

    def interrupt(*args, **kwargs):
        raise primary

    monkeypatch.setattr(pq.ParquetWriter, "close", cleanup_failure)
    monkeypatch.setattr(pq.ParquetWriter, "write_table", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        statistics.write_statistics_artifact(
            tmp_path / "statistics",
            envelope=envelope,
            tables=tables,
            table_group="factor" if "factor_diagnostics" in tables else "research",
        )
    assert caught.value is primary
    assert any("RuntimeError" in note for note in primary.__notes__)


@pytest.fixture
def crypto_tables():
    row = dict(
        candidate_id="candidate",
        execution_mode="continuous",
        episode_id="continuous",
        kind="oos",
        fold_id="fold_1",
        metric_id="net_pnl",
        start_at=pd.Timestamp("2023-01-02"),
        end_at=pd.Timestamp("2023-01-03"),
        stop_at=pd.Timestamp("2023-01-03"),
        start_nav=100.0,
        end_nav=110.0,
        gross_pnl=10.0,
        net_pnl=10.0,
        fees=0.0,
        funding=0.0,
        turnover=100.0,
        fill_events=1,
        net_return=0.1,
        benchmark_return=0.0,
        status="DEFINED",
        reason="",
        value=0.2,
        observation_count=1,
        text="",
        fee_bps=1.0,
        additional_spread_bps=0.0,
        cost_delta=0.0,
        method="fixed_turnover_rescaling",
    )
    result = {
        member.removesuffix(".parquet"): pd.DataFrame(
            [
                {
                    column: row[column]
                    for column in statistics.STATISTICS_TABLE_COLUMNS[member]
                }
            ]
        )
        for member in statistics.STATISTICS_TABLE_GROUPS["crypto_portfolio"]
    }
    result["crypto_portfolio_coverage"].loc[0, "kind"] = "covered"
    result["crypto_execution_diagnostics"].loc[0, ["value", "text"]] = [
        float("nan"),
        "estimated_queue_ahead",
    ]
    return result


@pytest.mark.parametrize(
    "member",
    (
        "crypto_portfolio_daily.parquet",
        "crypto_portfolio_statistics.parquet",
        "crypto_execution_diagnostics.parquet",
    ),
)
@pytest.mark.parametrize(
    "mutation", ("defined_null", "undefined_value", "reason", "status")
)
def test_crypto_status_measurements_are_rejected_at_write_and_read(
    tmp_path, crypto_tables, envelope, member, mutation
):
    frame = crypto_tables[member.removesuffix(".parquet")]
    if mutation == "defined_null":
        frame.loc[
            0, "net_return" if member == "crypto_portfolio_daily.parquet" else "value"
        ] = float("nan")
        if "text" in frame:
            frame.loc[0, "text"] = ""
    elif mutation == "undefined_value":
        frame.loc[0, ["status", "reason"]] = ["UNDEFINED", "missing_measurement"]
    elif mutation == "reason":
        frame.loc[0, "reason"] = "contradicts_defined_status"
    else:
        frame.loc[0, "status"] = "UNKNOWN"
    _assert_rejected_at_write_and_read(
        tmp_path, crypto_tables, envelope, member, table_group="crypto_portfolio"
    )


def test_crypto_text_diagnostics_and_undefined_daily_are_preserved(
    tmp_path, crypto_tables, envelope
):
    daily = crypto_tables["crypto_portfolio_daily"]
    daily.loc[0, ["end_nav", "net_return", "status", "reason"]] = [
        float("nan"),
        float("nan"),
        "UNDEFINED",
        "utc_boundary_valuation_missing",
    ]
    root = statistics.write_statistics_artifact(
        tmp_path / "crypto",
        envelope=envelope,
        tables=crypto_tables,
        table_group="crypto_portfolio",
    )
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("crypto_execution_diagnostics.parquet") as frame:
        assert frame.loc[0, "status"] == "DEFINED"
        assert pd.isna(frame.loc[0, "value"])
        assert frame.loc[0, "text"] == "estimated_queue_ahead"
    with artifact.table("crypto_portfolio_daily.parquet") as frame:
        assert frame.loc[0, "status"] == "UNDEFINED"
        assert pd.isna(frame.loc[0, "net_return"])
        assert frame.loc[0, "start_nav"] == 100.0
        assert frame.loc[0, "benchmark_return"] == 0.0


@pytest.mark.parametrize("finish", ["normal", "writer_failure"])
def test_maker_producer_physical_batches_preserve_tables_and_rebuild(
    tmp_path, monkeypatch, maker_calibration_tables, finish
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import portfolio
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )
    from tests.markets.crypto.test_hft_execution import _state, _trade

    expected, envelope = maker_calibration_tables
    manifest = envelope.payload()["semantic_metadata"]["target_contract"]["protocol"]
    source_identity = "d" * 64
    coverage = {"s1": (2100000000, 2800000000)}
    run = SimpleNamespace(
        source_identity=source_identity,
        market_metadata=lambda *_: {"price_unit": "0.1"},
        observation_coverage=lambda **kwargs: {
            "trade-connection-1": (2100000000, 2800000000)
        },
    )
    monkeypatch.setattr(
        portfolio,
        "_panel",
        lambda *_: {
            "source_artifact_identity": source_identity,
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": coverage}}
            },
        },
    )

    def validation_chunks(*, fold_id, model_id):
        assert fold_id == "fold_1"
        try:
            yield pd.DataFrame(
                {
                    "asset_id": ["btc_usdt"] * 2,
                    "timestamp": pd.to_datetime([2100, 2200], unit="ms"),
                    "label_available_at": pd.to_datetime([2600, 2700], unit="ms"),
                    "book_segment_id": ["s1"] * 2,
                    "trade_observation_id": ["trade-connection-1"] * 2,
                    "segment_id": ["segment"] * 2,
                    "prediction_score": [0.0 if model_id == MAKER_MODEL_IDS[0] else 1.0]
                    * 2,
                    "label": [0.0] * 2,
                }
            )
        finally:
            closed.append(model_id)

    models = SimpleNamespace(
        models=tuple(
            {
                "fold_id": "fold_1",
                "model_id": model_id,
                "model_metadata": {
                    "model_algorithm": manifest["model_scopes"][model_id][
                        "model_algorithm"
                    ]
                },
            }
            for model_id in MAKER_MODEL_IDS
        ),
        validation_chunks=validation_chunks,
    )
    primary = KeyboardInterrupt("synthetic_maker_writer_failure")
    roots, group_counts = ([], [])
    for build in (1, 2) if finish == "normal" else (1,):
        closed, emitted, frames = ([], [], [])

        def books(*_args, closed=closed, **_kwargs):
            try:
                yield from (_state(stamp) for stamp in (2100, 2150, 2650, 2700))
            finally:
                closed.append("books")

        def trades(*_args, closed=closed, **_kwargs):
            try:
                yield from (
                    _trade(stamp, price="99.5", quantity="3", maker_buyer=True)
                    for stamp in (2180, 2400)
                )
            finally:
                closed.append("trades")

        def produce(on_table, emitted=emitted, frames=frames):

            def append(table_name, output_frame):
                emitted.append((table_name, len(output_frame)))
                if finish == "writer_failure" and len(emitted) == 2:
                    raise primary
                on_table(table_name, output_frame)

            portfolio.produce_maker_calibration(
                asset_id="btc_usdt",
                model_ids=MAKER_MODEL_IDS,
                on_table=append,
                layout=StorageLayout(tmp_path),
                protocol_manifest=manifest,
                time_window=TimeWindow(
                    "1970-01-01T00:00:02.100000", "1970-01-01T00:00:03.500000"
                ),
                upstream_identities={f"{manifest['scenario_id']}_models": "e" * 64},
                raw_root=tmp_path / "raw",
                open_run=lambda *_: nullcontext(run),
                reconstruct_depth=books,
                read_trades=trades,
                read_model_artifact=lambda *_args, **_kwargs: models,
                symbol="BTCUSDT",
                market_type="futures",
            )

        logical = tmp_path / f"build_{build}"

        def write(logical=logical):
            return statistics.write_statistics_artifact(
                logical,
                envelope=envelope,
                tables=produce,
                table_group="crypto_maker_calibration",
                validate_tables=portfolio.validate_maker_calibration,
            )

        if finish != "normal":
            with pytest.raises(KeyboardInterrupt) as caught:
                write()
            assert caught.value is primary and len(emitted) == 2
            assert sorted(closed) == sorted(["books", "trades", *MAKER_MODEL_IDS])
            assert not tuple(filesystem_glob(logical / "instances", "*"))
            return
        root = write()
        artifact = statistics.read_statistics_artifact(
            root,
            expected_identity=envelope.instance_identity,
            validate_tables=portfolio.validate_maker_calibration,
        )
        for name, frame in expected.items():
            with artifact.table(f"{name}.parquet") as actual:
                pd.testing.assert_frame_equal(actual, frame)
            with pq.ParquetFile(
                native_filesystem_path(root / f"{name}.parquet")
            ) as parquet:
                assert parquet.schema_arrow.equals(
                    statistics._ARROW_SCHEMAS[f"{name}.parquet"]
                )
        assert sorted(closed) == sorted(["books", "trades", *MAKER_MODEL_IDS])
        roots.append(root)
        with pq.ParquetFile(
            native_filesystem_path(root / "crypto_maker_events.parquet")
        ) as parquet:
            group_counts.append(parquet.num_row_groups)
    assert group_counts[0] == group_counts[1]
    assert all(root.name == envelope.instance_identity for root in roots)
    for root in roots[1:]:
        assert compare_semantic_artifacts(roots[0], root).status == "MATCH"


@pytest.mark.parametrize("failure", ("disk_full", "interrupted", "writer_failed"))
def test_incremental_maker_publication_preserves_old_good_and_primary(
    tmp_path, monkeypatch, maker_calibration_tables, failure
):
    import errno

    from quant_research_platform.markets.implementations.crypto import portfolio
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
    )

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    primary = {
        "disk_full": OSError(errno.ENOSPC, "synthetic_disk_full"),
        "interrupted": KeyboardInterrupt("synthetic_interruption"),
        "writer_failed": RuntimeError("synthetic_writer_failure"),
    }[failure]
    good = statistics.write_statistics_artifact(
        tmp_path / "good",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    before = read_semantic_artifact_record(good)
    native = pq.ParquetWriter.write_table
    written = 0

    def fail_after_progress(writer, *args, **kwargs):
        nonlocal written
        written += 1
        if written == 2:
            raise primary
        return native(writer, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", fail_after_progress)
    with pytest.raises(type(primary)) as caught:
        statistics.write_statistics_artifact(
            tmp_path / "candidate",
            envelope=envelope,
            tables=produce,
            table_group="crypto_maker_calibration",
            validate_tables=portfolio.validate_maker_calibration,
        )
    assert caught.value is primary and written == 2
    assert read_semantic_artifact_record(good) == before
    assert not tuple(filesystem_glob(tmp_path / "candidate" / "instances", "*"))
    statistics.read_statistics_artifact(
        good,
        expected_identity=envelope.instance_identity,
        validate_tables=portfolio.validate_maker_calibration,
    )


def test_maker_calibration_streams_many_events_without_full_table_reads(
    tmp_path, monkeypatch, record_testsuite_property
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import portfolio
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_RESEARCH_DATASET_ID,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        CryptoMakerCandidate,
        protocol,
    )
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.binance_hft import hft_research_plan
    from tests.markets.crypto.test_hft_execution import _state

    count, depth = (1800, 512)
    first, last = (2100, 2100 + (count - 1) * 1000)

    def iso(stamp):
        return pd.Timestamp(stamp, unit="ms").isoformat()

    plan = hft_research_plan()
    fold = plan["folds"][0]
    fold["validation"] = {"start": iso(first), "end": iso(last + 600)}
    plan["folds"] = [fold]
    plan["test"] = {"start": iso(last + 2100), "end": iso(last + 3000)}
    research_protocol = replace(
        protocol(HFT_RESEARCH_DATASET_ID, research_plan=plan),
        configuration_json=json.dumps(
            research_configuration("crypto", "baseline", plan=plan)
        ),
    )
    candidate = next(
        candidate
        for candidate in research_protocol.portfolio_candidates()
        if isinstance(candidate, CryptoMakerCandidate)
        and candidate.asset_id == "btc_usdt"
        and (candidate.model_ids == MAKER_MODEL_IDS)
    )
    manifest = research_protocol.target_manifest(
        candidate.calibration_target_id, "STATISTIC"
    )
    window = fold["validation"]
    envelope = build_business_manifest_envelope(
        logical_artifact_id=candidate.calibration_target_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={**window, "access": "development"},
        upstream_identities={},
        semantic_metadata={"target_contract": {"protocol": manifest}},
    )
    coverage = {"s1": (first * 1000000, (last + 600) * 1000000)}
    run = SimpleNamespace(
        source_identity="d" * 64,
        market_metadata=lambda *_: {"price_unit": "0.1"},
        observation_coverage=lambda **kwargs: {
            "trade-connection-1": (first * 1000000, (last + 600) * 1000000)
        },
    )
    monkeypatch.setattr(
        portfolio,
        "_panel",
        lambda *_: {
            "source_artifact_identity": run.source_identity,
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": coverage}}
            },
        },
    )

    def validation_chunks(*, fold_id, model_id):
        assert fold_id == "fold_1"
        for start in range(0, count, 64):
            stamps = [
                first + index * 1000 for index in range(start, min(count, start + 64))
            ]
            yield pd.DataFrame(
                {
                    "asset_id": ["btc_usdt"] * len(stamps),
                    "timestamp": pd.to_datetime(stamps, unit="ms"),
                    "label_available_at": pd.to_datetime(
                        [stamp + 500 for stamp in stamps], unit="ms"
                    ),
                    "book_segment_id": ["s1"] * len(stamps),
                    "trade_observation_id": ["trade-connection-1"] * len(stamps),
                    "segment_id": ["segment"] * len(stamps),
                    "prediction_score": [0.0 if model_id == MAKER_MODEL_IDS[0] else 1.0]
                    * len(stamps),
                    "label": [0.0] * len(stamps),
                }
            )

    models = SimpleNamespace(
        models=tuple(
            {
                "fold_id": "fold_1",
                "model_id": model_id,
                "model_metadata": {
                    "model_algorithm": manifest["model_scopes"][model_id][
                        "model_algorithm"
                    ]
                },
            }
            for model_id in MAKER_MODEL_IDS
        ),
        validation_chunks=validation_chunks,
    )
    bids = tuple((995 - index, 1.0) for index in range(depth))
    asks = tuple((1005 + index, 1.0) for index in range(depth))
    table_read = statistics.StatisticsArtifact.table

    def summary_only(artifact, member):
        assert member == "crypto_maker_calibration.parquet"
        return table_read(artifact, member)

    monkeypatch.setattr(statistics.StatisticsArtifact, "table", summary_only)
    emitted = dict.fromkeys(
        ("crypto_maker_samples", "crypto_maker_events", "crypto_maker_calibration"), 0
    )

    def produce(on_table):

        def append(name, frame):
            emitted[name] += len(frame)
            on_table(name, frame)

        portfolio.produce_maker_calibration(
            asset_id="btc_usdt",
            model_ids=MAKER_MODEL_IDS,
            on_table=append,
            layout=StorageLayout(tmp_path),
            protocol_manifest=manifest,
            time_window=TimeWindow(window["start"], window["end"]),
            upstream_identities={f"{HFT_RESEARCH_DATASET_ID}_models": "e" * 64},
            raw_root=tmp_path / "raw",
            open_run=lambda *_: nullcontext(run),
            reconstruct_depth=lambda *_args, **_kwargs: (
                _state(first + index * 1000, bid_levels=bids, ask_levels=asks)
                for index in range(count)
            ),
            read_trades=lambda *_args, **_kwargs: iter(()),
            read_model_artifact=lambda *_args, **_kwargs: models,
            symbol="BTCUSDT",
            market_type="futures",
        )

    root = statistics.write_statistics_artifact(
        tmp_path / "bounded",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    artifact = statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=portfolio.validate_maker_calibration,
    )
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        assert summary.sample_count.tolist() == [count, count]
        assert summary.zero_fill_count.tolist() == [count, count]
    assert artifact._row_counts["crypto_maker_samples.parquet"] == count * 2
    assert artifact._row_counts["crypto_maker_events.parquet"] == count * depth * 2
    assert emitted == {
        "crypto_maker_samples": count * 2,
        "crypto_maker_events": count * depth * 2,
        "crypto_maker_calibration": 2,
    }
    record_testsuite_property("maker_event_rows", emitted["crypto_maker_events"])


def test_streamed_calibration_rechecks_member_bytes_after_consumption(
    tmp_path, monkeypatch, maker_calibration_tables
):
    import os

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "calibration",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    original = portfolio._calibration_headers
    changed = False

    def mutate_after_headers(*args):
        nonlocal changed
        yield from original(*args)
        if not changed:
            changed = True
            path = root / "crypto_maker_calibration.parquet"
            before = filesystem_stat(path)
            content = bytearray(filesystem_read_bytes(path))
            content[4] ^= 1
            filesystem_write_bytes(path, content)
            os.utime(
                native_filesystem_path(path),
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            assert filesystem_stat(path).st_mtime_ns == before.st_mtime_ns

    monkeypatch.setattr(portfolio, "_calibration_headers", mutate_after_headers)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        statistics.read_statistics_artifact(
            root,
            expected_identity=envelope.instance_identity,
            validate_tables=portfolio.validate_maker_calibration,
        )


def test_maker_calibration_uses_declared_asset_without_btc_limits(
    tmp_path, maker_calibration_tables
):
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        validate_maker_calibration,
    )

    tables, original = maker_calibration_tables
    payload = original.payload()
    metadata = payload["semantic_metadata"]
    contract = metadata["target_contract"]["protocol"]
    candidate = contract["portfolio_candidates"][0]
    candidate["asset_id"] = "eth_usdt"
    candidate["candidate_id"] = "eth_fixture_maker"
    envelope = build_business_manifest_envelope(
        logical_artifact_id="eth_fixture_maker_calibration",
        schema_identity=payload["schema_identity"],
        protocol_or_spec_identity=payload["protocol_or_spec_identity"],
        software_identity=payload["software_identity"],
        research_window=payload["research_window"],
        upstream_identities=payload["upstream_identities"],
        semantic_metadata=metadata,
    )

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "eth",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )
    statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=validate_maker_calibration,
    )


@pytest.mark.parametrize("finish", ["normal", "source_changed", "cancel"])
def test_maker_history_reads_real_artifact_once_and_rechecks_source(
    tmp_path, monkeypatch, maker_calibration_tables, finish
):
    import os
    from collections import Counter
    from contextlib import ExitStack
    from functools import partial

    from quant_research_platform.markets.implementations.crypto import (
        execution,
        portfolio,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _maker_config, _state

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "calibration",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    artifact = statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=portfolio.validate_maker_calibration,
    )
    batches = statistics.StatisticsArtifact._batches
    reads = Counter()

    def counted_batches(owner, member, **kwargs):
        reads[member] += 1
        yield from batches(owner, member, **kwargs)

    monkeypatch.setattr(statistics.StatisticsArtifact, "_batches", counted_batches)
    model = execution.MakerCalibration(
        partial(portfolio._calibration_samples, artifact, "fold_1"),
        partial(portfolio._calibration_events, artifact, "fold_1"),
        price_unit="0.1",
    )
    primary = KeyboardInterrupt("consumer_cancel")
    caught = None
    try:
        with closing_iterator(ExitStack()) as resources:
            resources.callback(artifact._verify)
            state = _state(0)
            answers = []
            for side in ("buy", "sell", "buy"):
                answers.append(
                    model.select(
                        state=state,
                        decision_ns=0,
                        new_order=execution._maker_new_order(
                            0,
                            side,
                            1.0,
                            state,
                            _quote(0, book_segment_id="s1"),
                            _maker_config().latency_ms,
                        ),
                        active_order=None,
                        commands=(),
                        lifetime_ns=500000000,
                        prediction=0.0,
                        fee_bps=0.0,
                        funding=[],
                    )
                )
            assert answers[0] == answers[2]
            assert reads == {
                "crypto_maker_samples.parquet": 1,
                "crypto_maker_events.parquet": 1,
            }
            if finish != "normal":
                path = root / "crypto_maker_events.parquet"
                before = filesystem_stat(path)
                payload = bytearray(filesystem_read_bytes(path))
                payload[4] ^= 1
                filesystem_write_bytes(path, payload)
                os.utime(
                    native_filesystem_path(path),
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                assert filesystem_stat(path).st_mtime_ns == before.st_mtime_ns
            if finish == "cancel":
                raise primary
    except (ValueError, KeyboardInterrupt) as error:
        caught = error
    if finish == "normal":
        assert caught is None
    elif finish == "source_changed":
        assert str(caught) == "semantic_artifact_member_integrity_invalid"
    else:
        assert caught is primary
        assert any("ValueError" in note for note in primary.__notes__)


@pytest.mark.parametrize("name", ("prediction_diagnostics", "model_comparison"))
def test_undefined_model_inference_preserves_descriptive_mean(
    tmp_path, tables, envelope, name
):
    frame = tables[name]
    frame.loc[0, "status"] = "UNDEFINED"
    frame.loc[0, "reason"] = "global_development_selection_reuses_validation"
    inference_columns = [
        "standard_error",
        "t_statistic",
        "p_value",
        "ci_lower",
        "ci_upper",
        "bootstrap_lower",
        "bootstrap_upper",
    ]
    frame.loc[0, inference_columns] = float("nan")
    root = statistics.write_statistics_artifact(
        tmp_path / "descriptive",
        envelope=envelope,
        tables=tables,
        table_group="factor" if "factor_diagnostics" in tables else "research",
    )
    artifact = statistics.read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table(f"{name}.parquet") as result:
        assert result["mean"].iloc[0] == 0.2
        assert result.observation_count.iloc[0] == 32
        assert result.status.iloc[0] == "UNDEFINED"
        assert result.loc[:, inference_columns].isna().all().all()


@pytest.mark.parametrize("name", ("prediction_diagnostics", "model_comparison"))
@pytest.mark.parametrize("corruption", ("p_value", "mean_missing", "empty_mean"))
def test_undefined_model_inference_rejects_inferential_values_and_false_means(
    tmp_path, tables, envelope, name, corruption
):
    frame = tables[name]
    frame.loc[0, "status"] = "UNDEFINED"
    frame.loc[0, "reason"] = "global_development_selection_reuses_validation"
    frame.loc[
        0,
        [
            "standard_error",
            "t_statistic",
            "p_value",
            "ci_lower",
            "ci_upper",
            "bootstrap_lower",
            "bootstrap_upper",
        ],
    ] = float("nan")
    if corruption == "p_value":
        frame.loc[0, "p_value"] = 0.01
    elif corruption == "mean_missing":
        frame.loc[0, "mean"] = float("nan")
    else:
        frame.loc[0, "observation_count"] = 0
    _assert_rejected_at_write_and_read(tmp_path, tables, envelope, f"{name}.parquet")


@pytest.mark.parametrize(
    "corruption",
    (
        "none",
        "global_mean",
        "quarterly_mean",
        "mean_difference",
        "observation_count",
        "access_window",
    ),
)
@pytest.mark.parametrize("fdr_method", ("BY", "NOT_APPLICABLE"))
def test_defined_group_comparison_requires_descriptive_facts(
    tmp_path, tables, envelope, corruption, fdr_method
):
    envelope = build_business_manifest_envelope(
        logical_artifact_id=envelope.logical_artifact_id,
        schema_identity=envelope.schema_identity,
        protocol_or_spec_identity=envelope.protocol_or_spec_identity,
        research_window={**envelope.research_window, "access": "holdout"},
        upstream_identities=envelope.upstream_identities,
        software_identity=envelope.software_identity,
        semantic_metadata=envelope.semantic_metadata,
    )
    row = _defined_group_comparison_row()
    row["fdr_method"] = fdr_method
    if fdr_method == "NOT_APPLICABLE":
        row["fdr_adjusted_p_value"] = float("nan")
    if corruption == "access_window":
        row[corruption] = "development"
    elif corruption != "none":
        row[corruption] = 0 if corruption == "observation_count" else float("nan")
    tables["model_group_comparison"] = pd.DataFrame([row])
    if corruption != "none":
        _assert_rejected_at_write_and_read(
            tmp_path,
            tables,
            envelope,
            "model_group_comparison.parquet",
            error="^statistics_group_comparison_invalid$",
        )
    else:
        root = statistics.write_statistics_artifact(
            tmp_path / "group",
            envelope=envelope,
            tables=tables,
            table_group="factor" if "factor_diagnostics" in tables else "research",
        )
        artifact = statistics.read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with artifact.table("model_group_comparison.parquet") as frame:
            assert frame.global_mean.iloc[0] == 0.2
            assert frame.observation_count.iloc[0] == 64
            assert frame.fdr_method.iloc[0] == fdr_method
            assert frame.fdr_adjusted_p_value.isna().all() == (
                fdr_method == "NOT_APPLICABLE"
            )


def _defined_group_comparison_row():
    return dict(
        family_id="family",
        access_window="holdout",
        status="DEFINED",
        reason="",
        fdr_method="BY",
        bootstrap_status="UNDEFINED",
        bootstrap_reason="insufficient_blocks",
        observation_count=64,
        calendar_count=64,
        fdr_family_size=1,
        hac_lag=19,
        bootstrap_block_size=20,
        bootstrap_repetitions=10000,
        bootstrap_seed=42,
        global_mean=0.2,
        quarterly_mean=0.3,
        mean_difference=0.1,
        standard_error=0.1,
        t_statistic=1.0,
        p_value=0.3,
        ci_lower=-0.1,
        ci_upper=0.3,
        fdr_adjusted_p_value=0.3,
        fdr_alpha=0.05,
        confidence=0.95,
        bootstrap_lower=float("nan"),
        bootstrap_upper=float("nan"),
    )


@pytest.mark.parametrize("boundary", [False, True])
def test_maker_validator_reuses_overlapping_and_adjacent_native_paths(
    tmp_path, monkeypatch, maker_calibration_tables, boundary
):
    from collections import deque

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables
    if boundary:
        from tests.markets.crypto.calibration_test_support import maker_event_rows
        from tests.markets.crypto.test_hft_execution import _boundary

        events = tables["crypto_maker_events"]
        rows = pd.DataFrame(
            [
                {"fold_id": "fold_1", **row}
                for row in maker_event_rows(3, _boundary(2250))
            ],
            columns=events.columns,
        )
        rows.event_at = pd.to_datetime(rows.event_at, unit="ns")
        events.loc[events.event_id.ge(3), "event_id"] += 1
        tables["crypto_maker_events"] = pd.concat(
            [
                events.loc[events.event_id.lt(3)],
                rows,
                events.loc[events.event_id.gt(3)],
            ],
            ignore_index=True,
        )
        samples = tables["crypto_maker_samples"]
        samples.event_end += 1
        samples.loc[
            samples.sample_id.eq(1) & samples.side.eq("buy"), "filled_quantity"
        ] = 0.0
        samples.loc[
            samples.sample_id.eq(1) | samples.side.eq("sell"), "terminal_reason"
        ] = "book_boundary"
        summary = tables["crypto_maker_calibration"]
        summary.loc[summary.side.eq("buy"), "zero_fill_count"] += 1
    event_count = tables["crypto_maker_events"].event_id.nunique()
    for name in ("crypto_maker_samples", "crypto_maker_events"):
        frame = tables[name]
        later = frame.copy()
        for column in (
            "decision_at",
            "prediction_end",
            "label_available_at",
            "event_at",
        ):
            if column in later:
                later[column] += pd.Timedelta(milliseconds=600)
        for column in ("initial_book_event_id", "event_start", "event_end", "event_id"):
            if column in later:
                later[column] += event_count
        if "sample_id" in later:
            later.sample_id += 2
        tables[name] = pd.concat([frame, later], ignore_index=True)
    summary = tables["crypto_maker_calibration"]
    for column in ("sample_count", "zero_fill_count", "post_only_rejection_count"):
        summary[column] *= 2

    paths, decoded, closed, maxima = [], [], [], []
    native_events = portfolio._calibration_events
    native_response = portfolio.MakerCalibration._response

    def events(*args):
        count = 0
        try:
            for event in native_events(*args):
                count += 1
                yield event
        finally:
            decoded.append(count)
            closed.append(True)

    def response(model, sample, **kwargs):
        path = tuple(kwargs.pop("events"))
        paths.append((sample["sample_id"], sample["side"], path))
        return (yield from native_response(model, sample, events=iter(path), **kwargs))

    class Window(deque):
        def append(self, event):
            super().append(event)
            maxima.append(len(self))

    def no_full_history(_model):
        pytest.fail("validator materialized a full fold history")

    monkeypatch.setattr(portfolio, "deque", Window, raising=False)
    monkeypatch.setattr(portfolio, "_calibration_events", events)
    monkeypatch.setattr(portfolio.MakerCalibration, "_response", response)
    monkeypatch.setattr(portfolio.MakerCalibration, "_load", no_full_history)

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "paths",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=portfolio.validate_maker_calibration,
    )
    assert decoded == [event_count * 2] * 2 and len(closed) == 2
    assert max(maxima) < event_count * 2
    assert len(paths) == 16
    for first in (0, 8):
        for index in range(first, first + 8, 2):
            buy, sell = paths[index : index + 2]
            assert (buy[1], sell[1]) == ("buy", "sell")
            assert len(buy[2]) == len(sell[2])
            assert all(
                left is right for left, right in zip(buy[2], sell[2], strict=True)
            )
        previous = {event.event_id: event for event in paths[first][2]}
        overlap = [event for event in paths[first + 2][2] if event.event_id in previous]
        assert overlap and all(event is previous[event.event_id] for event in overlap)
        assert paths[first + 4][2][0].event_id > paths[first + 2][2][-1].event_id


@pytest.mark.parametrize("finish", ["close_failed", "cancel"])
def test_maker_validator_closes_native_stream_and_preserves_primary(
    tmp_path, monkeypatch, maker_calibration_tables, finish
):
    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = statistics.write_statistics_artifact(
        tmp_path / "good",
        envelope=envelope,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=portfolio.validate_maker_calibration,
    )
    before = {
        path.name: filesystem_read_bytes(path)
        for path in root.iterdir()
        if path.is_file()
    }
    native_events = portfolio._calibration_events
    primary = KeyboardInterrupt("synthetic_reference_cancel")
    cleanup = RuntimeError("synthetic_native_close_failure")
    closed, opened = [], 0

    def events(*args):
        nonlocal opened
        opened += 1
        number = opened
        try:
            yield from native_events(*args)
        finally:
            closed.append(number)
            if number == 1:
                raise cleanup

    def cancel(*_args, **_kwargs):
        raise primary

    with monkeypatch.context() as patch:
        patch.setattr(portfolio, "_calibration_events", events)
        if finish == "cancel":
            patch.setattr(portfolio.MakerCalibration, "_response", cancel)
        with pytest.raises(
            KeyboardInterrupt if finish == "cancel" else RuntimeError
        ) as caught:
            statistics.read_statistics_artifact(
                root,
                expected_identity=envelope.instance_identity,
                validate_tables=portfolio.validate_maker_calibration,
            )
        assert caught.value is (primary if finish == "cancel" else cleanup)
    assert closed == [1]
    if finish == "cancel":
        assert any("RuntimeError" in note for note in primary.__notes__)
    assert {
        path.name: filesystem_read_bytes(path)
        for path in root.iterdir()
        if path.is_file()
    } == before
    statistics.read_statistics_artifact(
        root,
        expected_identity=envelope.instance_identity,
        validate_tables=portfolio.validate_maker_calibration,
    )


@pytest.mark.parametrize(
    "damage",
    [
        "side",
        "events_side",
        "ordinal",
        "generic_before_ordinal",
        "book_cross_batch",
        "trade_cross_batch",
    ],
)
def test_maker_validator_rejects_compound_damage_without_error_priority(
    tmp_path, monkeypatch, maker_calibration_tables, damage
):
    from functools import partial

    from quant_research_platform.markets.implementations.crypto import portfolio

    tables, envelope = maker_calibration_tables
    tables["crypto_maker_samples"].loc[0, "filled_quantity"] = 0.5
    if damage in ("side", "events_side", "generic_before_ordinal"):
        extra = tables["crypto_maker_samples"].iloc[[0]].copy()
        extra.side = "events" if damage == "events_side" else "unknown"
        if damage == "generic_before_ordinal":
            extra.sample_id = -1
        tables["crypto_maker_samples"] = pd.concat(
            [tables["crypto_maker_samples"], extra], ignore_index=True
        )
    if damage in ("ordinal", "generic_before_ordinal"):
        extra = (
            tables["crypto_maker_events"].loc[lambda frame: frame.event_id.eq(4)].copy()
        )
        extra.event_id = 6
        extra.event_at += pd.Timedelta(milliseconds=500)
        tables["crypto_maker_events"] = pd.concat(
            [tables["crypto_maker_events"], extra], ignore_index=True
        )

    if damage in ("book_cross_batch", "trade_cross_batch"):
        events = tables["crypto_maker_events"]
        extra = events.loc[
            events.event_id.eq(4 if damage == "book_cross_batch" else 2)
        ].copy()
        if damage == "trade_cross_batch":
            extra = pd.concat([extra, extra], ignore_index=True)
        extra.event_id = 6
        extra.event_at = pd.Timestamp("1970-01-01T00:00:03.150000")
        extra.iloc[-1, extra.columns.get_loc("quantity")] = float("inf")
        tables["crypto_maker_events"] = pd.concat([events, extra], ignore_index=True)
        monkeypatch.setattr(
            statistics,
            "iter_declared_parquet_member_batches",
            partial(statistics.iter_declared_parquet_member_batches, batch_rows=1),
        )

    def write(staging):
        for name, frame in tables.items():
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[f"{name}.parquet"],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / f"{name}.parquet"),
            )

    root = write_semantic_artifact(tmp_path / "compound", envelope, write)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    with pytest.raises(ValueError):
        statistics.read_statistics_artifact(
            root,
            expected_identity=envelope.instance_identity,
            validate_tables=portfolio.validate_maker_calibration,
        )
    assert {
        path.name: path.read_bytes() for path in root.iterdir() if path.is_file()
    } == before
