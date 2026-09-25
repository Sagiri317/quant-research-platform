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
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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
from tests.helpers.runtime_resources import capture_test_policy

MAKER_MODEL_IDS = next(
    (item["price_model_id"], item["proxy_model_id"])
    for item in research_configuration("crypto", "baseline")["portfolios"]
    if item.get("price_model_id") is not None
)


@pytest.fixture
def tables():
    row = dict(
        family_id="family",
        factor_id="factor",
        prediction_id="prediction",
        quarter_id="2023Q1",
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
        for member in statistics.STATISTICS_TABLE_GROUPS["research"]
    }
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
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _maker_config,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        CryptoMakerCandidate,
        protocol,
    )
    from tests.helpers.binance_hft import _quote, hft_research_plan
    from tests.markets.crypto.calibration_test_support import maker_event_rows
    from tests.markets.crypto.test_hft_execution import _state, _trade

    plan = hft_research_plan()
    samples, events = [], []

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

    with runtime_resource_policy_scope() as policy:
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
            config=_maker_config(plan, "btc_usdt", calibration=True),
            observed_coverage={"s1": (2100000000, 2800000000)},
            trade_coverage={"trade-connection-1": (2100000000, 2800000000)},
            cutoff_ns=3500000000,
            resource_policy=policy,
            on_sample=on_sample,
            on_event=on_event,
        )
        assert policy.used_memory_bytes == 0
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
        and candidate.model_ids == MAKER_MODEL_IDS
    )
    manifest = research_protocol.target_manifest(
        candidate.calibration_target_id, "STATISTIC"
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id=candidate.calibration_target_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01T00:00:04",
            "end": "1970-01-01T00:00:05",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"target_contract": {"protocol": manifest}},
    )
    return frames, envelope


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

    with runtime_resource_policy_scope() as policy:
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
            assert policy.used_memory_bytes == 0
            return
        if mutation == "truncate":
            tables["crypto_maker_events"] = (
                tables["crypto_maker_events"]
                .loc[lambda frame: frame.event_id.lt(4)]
                .copy()
            )
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
        else:
            tables["crypto_maker_samples"].loc[1, "price_prediction"] = 0.001
            tables["crypto_maker_samples"].loc[1, "forecast_error"] = -0.001
        with pytest.raises(ValueError, match="crypto_calibration_"):
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

        bad = write_semantic_artifact(
            tmp_path / "corrupted", envelope, write_corruption
        )
        with pytest.raises(ValueError, match="crypto_calibration_"):
            statistics.read_statistics_artifact(
                bad,
                expected_identity=envelope.instance_identity,
                validate_tables=validate_maker_calibration,
            )
        assert read_semantic_artifact_record(root) == original
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("member", statistics.STATISTICS_TABLE_GROUPS["research"])
@pytest.mark.parametrize("mutation", ["status", "reason", "count", "undefined"])
def test_research_table_rejects_invalid_values_at_write_and_read(
    tmp_path,
    tables,
    envelope,
    member,
    mutation,
):
    name = member.removesuffix(".parquet")
    if mutation == "status":
        tables[name].loc[0, "status"] = "UNKNOWN"
    elif mutation == "reason":
        tables[name].loc[0, "reason"] = "contradicts_defined_status"
    elif mutation == "undefined":
        tables[name].loc[0, "status"] = "UNDEFINED"
    else:
        count = "period_count" if name == "fama_macbeth" else "observation_count"
        tables[name].loc[0, count] = -1
    _assert_rejected_at_write_and_read(tmp_path, tables, envelope, member)


def _assert_rejected_at_write_and_read(
    tmp_path, tables, envelope, member, *, table_group="research"
):
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="statistics_table_"):
            statistics.write_statistics_artifact(
                tmp_path / "rejected",
                envelope=envelope,
                tables=tables,
                table_group=table_group,
            )
        assert policy.used_memory_bytes == 0

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

        root = write_semantic_artifact(
            tmp_path / "persisted", envelope, write_unvalidated
        )
        artifact = statistics.read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with pytest.raises(ValueError, match="statistics_table_"):
            with artifact.table(member):
                pytest.fail("invalid persisted values must not reach the consumer")
        assert policy.used_memory_bytes == 0


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
    tmp_path,
    tables,
    envelope,
    name,
    column,
):
    tables[name].loc[0, column] = float("nan")
    _assert_rejected_at_write_and_read(tmp_path, tables, envelope, f"{name}.parquet")


@pytest.mark.parametrize("member", statistics.STATISTICS_TABLE_GROUPS["research"])
@pytest.mark.parametrize("value", [True, 1.5, 2**63])
def test_research_table_counts_reject_lossy_coercion(
    tmp_path, tables, envelope, member, value
):
    name = member.removesuffix(".parquet")
    count = "period_count" if name == "fama_macbeth" else "observation_count"
    tables[name][count] = pd.Series([value], dtype=object)
    with runtime_resource_policy_scope():
        with pytest.raises(ValueError, match="statistics_table_count_invalid"):
            statistics.write_statistics_artifact(
                tmp_path / "rejected", envelope=envelope, tables=tables
            )


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
def test_nine_tables_complete_sequentially_under_16_mib(
    tmp_path,
    tables,
    envelope,
    factor_case,
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
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2, explicit_cap_bytes=16 * 1024**2
    )
    with runtime_resource_policy_scope(policy):
        root = statistics.write_statistics_artifact(
            tmp_path / "statistics", envelope=envelope, tables=tables
        )
        assert policy.used_memory_bytes == 0
        artifact = statistics.read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        for member in statistics.STATISTICS_TABLE_GROUPS["research"]:
            with artifact.table(member) as actual:
                pd.testing.assert_frame_equal(
                    actual, tables[member.removesuffix(".parquet")]
                )
        assert policy.used_memory_bytes == 0


def test_validation_is_admitted_before_copying(tmp_path, tables, envelope, monkeypatch):
    first = next(iter(tables))
    tables[first] = pd.concat([tables[first]] * 10000, ignore_index=True)

    def not_admitted(*args, **kwargs):
        pytest.fail("validation copy must not run without working memory")

    monkeypatch.setattr(statistics, "_validated_frame", not_admitted)
    policy = capture_test_policy(available_memory_bytes=8 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="statistics_validation_exceeds_task_memory_budget"
        ):
            statistics.write_statistics_artifact(
                tmp_path / "statistics", envelope=envelope, tables=tables
            )
        assert policy.used_memory_bytes == 0


def test_reader_admits_complete_working_set_before_decode(
    tmp_path,
    tables,
    envelope,
    monkeypatch,
):
    tables["factor_diagnostics"] = pd.concat(
        [tables["factor_diagnostics"]] * 10000, ignore_index=True
    )
    writer_policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(writer_policy):
        root = statistics.write_statistics_artifact(
            tmp_path / "statistics", envelope=envelope, tables=tables
        )
        assert writer_policy.used_memory_bytes == 0

    def not_admitted(*args, **kwargs):
        pytest.fail("full Arrow decode must not run without working memory")

    monkeypatch.setattr(pq.ParquetFile, "read", not_admitted)
    reader_policy = capture_test_policy(available_memory_bytes=8 * 1024**2)
    with runtime_resource_policy_scope(reader_policy):
        artifact = statistics.read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with pytest.raises(
            ValueError, match="statistics_table_exceeds_task_memory_budget"
        ):
            with artifact.table("factor_diagnostics.parquet"):
                pytest.fail("oversized full table must not reach the consumer")
        assert reader_policy.used_memory_bytes == 0


def test_statistics_plain_dictionary_reader_admits_expansion_before_read(
    tmp_path, tables, envelope, monkeypatch
):
    tables["factor_diagnostics"] = pd.concat(
        [tables["factor_diagnostics"]] * 256, ignore_index=True
    )
    tables["factor_diagnostics"]["factor_id"] = "x" * (32 * 1024)

    def write_dictionary_members(staging):
        for name, frame in tables.items():
            member = f"{name}.parquet"
            pq.write_table(
                pa.Table.from_pandas(
                    frame,
                    schema=statistics._ARROW_SCHEMAS[member],
                    preserve_index=False,
                ),
                native_filesystem_path(staging / member),
                use_dictionary=True,
                write_statistics=False,
            )

    root = write_semantic_artifact(
        tmp_path / "dictionary-statistics", envelope, write_dictionary_members
    )

    def no_full_read(*args, **kwargs):
        pytest.fail("plain dictionary expansion started before admission")

    monkeypatch.setattr(pq.ParquetFile, "read", no_full_read)
    with runtime_resource_policy_scope(explicit_cap_bytes=4 * 1024**2) as policy:
        artifact = statistics.read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with pytest.raises(
            ValueError, match="statistics_table_exceeds_task_memory_budget"
        ):
            with artifact.table("factor_diagnostics.parquet"):
                pytest.fail("oversized full table reached the consumer")
        assert policy.used_memory_bytes == 0


def test_writer_cleanup_preserves_primary(tmp_path, tables, envelope, monkeypatch):
    from quant_research_platform.storage import parquet

    primary = KeyboardInterrupt("write_interrupted")
    original = parquet._parquet_table_write_state

    def cleanup_failure():
        raise RuntimeError("release_failed")

    def native_state(*args, **kwargs):
        table, state, rows, retained = original(*args, **kwargs)
        state.callback(cleanup_failure)
        return table, state, rows, retained

    def interrupt(*args, **kwargs):
        raise primary

    monkeypatch.setattr(parquet, "_parquet_table_write_state", native_state)
    monkeypatch.setattr(pq.ParquetWriter, "write_table", interrupt)
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt) as caught:
            statistics.write_statistics_artifact(
                tmp_path / "statistics", envelope=envelope, tables=tables
            )
        assert caught.value is primary
        assert any("RuntimeError" in note for note in primary.__notes__)
        assert policy.used_memory_bytes == 0


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
    with runtime_resource_policy_scope() as policy:
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
        assert policy.used_memory_bytes == 0


def test_maker_producer_physical_batches_preserve_tables_and_rebuild(
    tmp_path, monkeypatch, maker_calibration_tables
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

    models = SimpleNamespace(validation_chunks=validation_chunks)
    roots, group_counts = [], []
    for budget in (4096, 65536):
        monkeypatch.setattr(portfolio, "_CALIBRATION_BATCH_BYTES", budget)
        with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024**2) as policy:

            def produce(on_table):
                portfolio.produce_maker_calibration(
                    asset_id="btc_usdt",
                    model_ids=MAKER_MODEL_IDS,
                    on_table=on_table,
                    layout=StorageLayout(tmp_path),
                    protocol_manifest=manifest,
                    time_window=TimeWindow(
                        "1970-01-01T00:00:04", "1970-01-01T00:00:05"
                    ),
                    upstream_identities={f"{manifest['scenario_id']}_models": "e" * 64},
                    raw_root=tmp_path / "raw",
                    resource_policy=policy,
                    open_run=lambda *_: nullcontext(run),
                    reconstruct_depth=lambda *_args, **_kwargs: iter(
                        [_state(stamp) for stamp in (2100, 2150, 2650, 2700)]
                    ),
                    read_trades=lambda *_args, **_kwargs: iter(
                        [
                            _trade(stamp, price="99.5", quantity="3", maker_buyer=True)
                            for stamp in (2180, 2400)
                        ]
                    ),
                    read_model_artifact=lambda *_args, **_kwargs: models,
                    symbol="BTCUSDT",
                    market_type="futures",
                )

            root = statistics.write_statistics_artifact(
                tmp_path / f"batch_{budget}",
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
            for name, frame in expected.items():
                with artifact.table(f"{name}.parquet") as actual:
                    pd.testing.assert_frame_equal(actual, frame)
            roots.append(root)
            with pq.ParquetFile(
                native_filesystem_path(root / "crypto_maker_events.parquet")
            ) as parquet:
                group_counts.append(parquet.num_row_groups)
            assert policy.used_memory_bytes == 0
    assert group_counts[0] > group_counts[1]
    assert compare_semantic_artifacts(*roots).status == "MATCH"


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
    with runtime_resource_policy_scope(explicit_cap_bytes=64 * 1024**2) as policy:
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
        assert policy.used_memory_bytes == 0


def test_maker_calibration_completes_payload_larger_than_cap_without_full_reads(
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
    from quant_research_platform.runtime.resources import pandas_frame_payload_bytes
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.binance_hft import hft_research_plan
    from tests.markets.crypto.test_hft_execution import _state

    count, depth = 1800, 512
    cap = 64 * 1024**2
    first, last = 2100, 2100 + (count - 1) * 1000

    def iso(stamp):
        return pd.Timestamp(stamp, unit="ms").isoformat()

    plan = hft_research_plan()
    fold = plan["folds"][0]
    fold["validation"]["end"] = iso(last + 600)
    fold["development_test"] = {"start": iso(last + 1000), "end": iso(last + 2000)}
    plan["folds"] = [fold]
    plan["final_holdout"] = {"start": iso(last + 2100), "end": iso(last + 3000)}
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
        and candidate.model_ids == MAKER_MODEL_IDS
    )
    manifest = research_protocol.target_manifest(
        candidate.calibration_target_id, "STATISTIC"
    )
    window = fold["development_test"]
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

    models = SimpleNamespace(validation_chunks=validation_chunks)
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
    payload = peak_batch = peak_reserved = 0
    with runtime_resource_policy_scope(explicit_cap_bytes=cap) as policy:

        def produce(on_table):
            def append(name, frame):
                nonlocal payload, peak_batch, peak_reserved
                size = pandas_frame_payload_bytes(frame)
                payload += size
                peak_batch = max(peak_batch, size)
                peak_reserved = max(peak_reserved, policy.used_memory_bytes)
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
                resource_policy=policy,
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
        assert policy.used_memory_bytes == 0
    assert emitted == {
        "crypto_maker_samples": count * 2,
        "crypto_maker_events": count * depth * 2,
        "crypto_maker_calibration": 2,
    }
    assert payload > cap * 1.5
    assert peak_batch < cap // 16
    assert peak_reserved <= cap

    record_testsuite_property("maker_cap_bytes", cap)
    record_testsuite_property("maker_output_payload_bytes", payload)
    record_testsuite_property("maker_max_emitted_batch_bytes", peak_batch)
    record_testsuite_property("maker_producer_sampled_reserved_bytes", peak_reserved)
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

    with runtime_resource_policy_scope() as policy:
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
        with pytest.raises(
            ValueError, match="semantic_artifact_member_integrity_invalid"
        ):
            statistics.read_statistics_artifact(
                root,
                expected_identity=envelope.instance_identity,
                validate_tables=portfolio.validate_maker_calibration,
            )
        assert changed and policy.used_memory_bytes == 0


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
    limits = contract["research_plan"]["exposure_limits"]
    limits["eth_usdt"] = limits.pop("btc_usdt")
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

    with runtime_resource_policy_scope():
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
