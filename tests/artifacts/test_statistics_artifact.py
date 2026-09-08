from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import statistics
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.semantic_artifacts import write_semantic_artifact
from tests.helpers.runtime_resources import capture_test_policy


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
    primary = KeyboardInterrupt("write_interrupted")
    original = statistics._artifact_table_write_state

    def cleanup_failure():
        raise RuntimeError("release_failed")

    def native_state(*args, **kwargs):
        table, state, rows, retained = original(*args, **kwargs)
        state.callback(cleanup_failure)
        return table, state, rows, retained

    def interrupt(*args, **kwargs):
        raise primary

    monkeypatch.setattr(statistics, "_artifact_table_write_state", native_state)
    monkeypatch.setattr(statistics.pq, "write_table", interrupt)
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt) as caught:
            statistics.write_statistics_artifact(
                tmp_path / "statistics", envelope=envelope, tables=tables
            )
        assert caught.value is primary
        assert any("release also failed" in note for note in primary.__notes__)
        assert policy.used_memory_bytes == 0


@pytest.fixture
def crypto_tables():
    row = dict(
        candidate_id="candidate",
        kind="oos",
        fold_id="fold_1",
        metric_id="net_pnl",
        start_at=pd.Timestamp("2023-01-02"),
        end_at=pd.Timestamp("2023-01-03"),
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
