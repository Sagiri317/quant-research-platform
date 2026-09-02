from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.artifacts.portfolio_ledger as ledger_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import PORTFOLIO_LEDGER_COLUMNS
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.runtime_resources import capture_test_policy


def _envelope() -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="portfolio-ledger",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-01-03",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"ledger_schema": list(PORTFOLIO_LEDGER_COLUMNS)},
    )


def _session(
    decision: str,
    execution: str,
    *,
    pre_asset: float,
    target_asset: float,
    post_asset: float,
    pre_cash: float,
    target_cash: float,
    post_cash: float,
) -> pd.DataFrame:
    rows = [
        {
            "decision_timestamp": decision,
            "execution_date": execution,
            "asset_id": "asset",
            "pre_trade_notional": pre_asset,
            "target_notional": target_asset,
            "executed_delta_notional": post_asset - pre_asset,
            "post_trade_notional": post_asset,
            "holding_return": 0.0,
            "end_notional": post_asset,
            "block_reason": None,
            "adv20": 1000.0,
            "alpha_score": 1.0,
            "expected_alpha": np.nan,
            "benchmark_weight": np.nan,
            "asset_specific_variance": np.nan,
            "decision_reason": None,
        },
        {
            "decision_timestamp": decision,
            "execution_date": execution,
            "asset_id": "__cash__",
            "pre_trade_notional": pre_cash,
            "target_notional": target_cash,
            "executed_delta_notional": post_cash - pre_cash,
            "post_trade_notional": post_cash,
            "holding_return": 0.0,
            "end_notional": post_cash,
            "block_reason": None,
            "adv20": np.nan,
            "alpha_score": np.nan,
            "expected_alpha": np.nan,
            "benchmark_weight": np.nan,
            "asset_specific_variance": np.nan,
            "decision_reason": "simple_rule",
        },
    ]
    frame = pd.DataFrame.from_records(rows)
    for column in PORTFOLIO_LEDGER_COLUMNS:
        if column not in frame:
            frame[column] = 0.0 if column.endswith("_cost") else np.nan
    for column in (
        "commission_cost",
        "stamp_duty_cost",
        "transfer_fee_cost",
        "slippage_cost",
        "impact_cost",
    ):
        frame[column] = 0.0
    for column in (
        "alpha_slope",
        "alpha_slope_se",
        "alpha_slope_t",
        "alpha_slope_pvalue",
        "alpha_slope_ci_low",
        "alpha_slope_ci_high",
        "alpha_weighted_r2",
        "alpha_coverage",
        "alpha_valid_dates",
        "alpha_scheduled_dates",
        "predicted_tracking_error",
        "portfolio_factor_variance",
        "portfolio_specific_variance",
        "expected_transaction_cost",
    ):
        frame.loc[frame["asset_id"] != "__cash__", column] = np.nan
    return frame.loc[:, list(PORTFOLIO_LEDGER_COLUMNS)]


def test_portfolio_ledger_roundtrip_is_one_row_group_per_session(
    tmp_path: Path,
) -> None:
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    second = _session(
        "2019-01-02",
        "2019-01-03",
        pre_asset=100.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=0.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame for frame in (first, second)),
    )

    artifact = read_portfolio_ledger_artifact(
        root,
        expected_identity=root.name,
    )
    chunks = list(artifact.iter_chunks())
    assert len(chunks) == 2
    assert [chunk["asset_id"].tolist() for chunk in chunks] == [
        ["asset", "__cash__"],
        ["asset", "__cash__"],
    ]


def test_portfolio_ledger_member_mutation_fails_closed(tmp_path: Path) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    with (root / "ledger.parquet").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_rejects_an_extra_persisted_member(tmp_path: Path) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    (root / "extra.parquet").write_bytes(b"not-a-ledger")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_reconciles_cash_rotation_costs_and_zero_asset_carry(
    tmp_path: Path,
) -> None:
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=100.0,
        target_asset=0.0,
        post_asset=0.0,
        pre_cash=0.0,
        target_cash=0.0,
        post_cash=1.0,
    )
    first.loc[first["asset_id"].eq("asset"), "commission_cost"] = 1.0
    first.loc[first["asset_id"].eq("__cash__"), "executed_delta_notional"] = 1.0
    buyer = first.loc[first["asset_id"].eq("asset")].copy()
    buyer["asset_id"] = "buyer"
    buyer["pre_trade_notional"] = 0.0
    buyer["target_notional"] = 100.0
    buyer["executed_delta_notional"] = 98.0
    buyer["post_trade_notional"] = 98.0
    buyer["end_notional"] = 98.0
    buyer["commission_cost"] = 0.0
    buyer["block_reason"] = "cash_pro_rata"
    seller = first.loc[first["asset_id"].eq("asset")].copy()
    cash_row = first.loc[first["asset_id"].eq("__cash__")].copy()
    first = pd.concat([seller, buyer, cash_row], ignore_index=True)
    second = first.loc[first["asset_id"].isin(("buyer", "__cash__"))].copy()
    second["decision_timestamp"] = "2019-01-02"
    second["execution_date"] = "2019-01-03"
    second.loc[second["asset_id"].eq("buyer"), "pre_trade_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "target_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "post_trade_notional"] = 98.0
    second.loc[second["asset_id"].eq("buyer"), "end_notional"] = 98.0
    second.loc[second["asset_id"].eq("__cash__"), "pre_trade_notional"] = 1.0
    second.loc[second["asset_id"].eq("__cash__"), "target_notional"] = 1.0
    second["executed_delta_notional"] = 0.0
    second["post_trade_notional"] = second["target_notional"]
    second["end_notional"] = second["post_trade_notional"]
    second["block_reason"] = None
    root = write_portfolio_ledger_artifact(
        tmp_path / "rotation",
        envelope=_envelope(),
        ledger_chunks=(first, second),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    chunks = list(artifact.iter_chunks())
    assert float(chunks[0]["post_trade_notional"].sum()) == 99.0
    assert chunks[0]["target_notional"].sum() == 100.0
    assert (
        chunks[0]
        .loc[chunks[0]["asset_id"].eq("__cash__"), "executed_delta_notional"]
        .iloc[0]
        == 1.0
    )
    assert chunks[1]["asset_id"].tolist() == ["buyer", "__cash__"]


def test_portfolio_ledger_diagnostic_counts_are_nullable_integers(
    tmp_path: Path,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 3
    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_scheduled_dates"] = 4
    root = write_portfolio_ledger_artifact(
        tmp_path / "valid",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    read = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    result = next(read.iter_chunks())
    assert str(result["alpha_valid_dates"].dtype) == "Int64"
    assert (
        result.loc[result["asset_id"].eq("__cash__"), "alpha_valid_dates"].iloc[0] == 3
    )

    frame.loc[frame["asset_id"].eq("__cash__"), "alpha_valid_dates"] = 1.5
    with pytest.raises(ValueError, match="portfolio_ledger_count_invalid"):
        write_portfolio_ledger_artifact(
            tmp_path / "invalid",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )


def test_portfolio_ledger_writer_keeps_native_state_until_close(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    real_writer = ledger_artifact.pq.ParquetWriter
    close_usage: list[int] = []

    class ObservedWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self._writer.write_table(*args, **kwargs)

        def close(self):  # noqa: ANN201
            close_usage.append(policy.used_memory_bytes)
            return self._writer.close()

    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", ObservedWriter)
    with runtime_resource_policy_scope(policy):
        root = write_portfolio_ledger_artifact(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )
    assert policy.used_memory_bytes == 0
    assert close_usage and close_usage[0] > 0
    read_portfolio_ledger_artifact(root, expected_identity=root.name)


def test_portfolio_ledger_frame_lease_covers_validation_and_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    active = False
    observed = {"validated": False, "written": False}
    real_reservation = ledger_artifact._artifact_frame_reservation
    real_validate = ledger_artifact._validated_frame

    @contextmanager
    def reservation(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal active
        with real_reservation(*args, **kwargs):
            active = True
            try:
                yield
            finally:
                active = False

    def validate(*args, **kwargs):  # noqa: ANN002, ANN003
        assert active
        observed["validated"] = True
        return real_validate(*args, **kwargs)

    real_writer = ledger_artifact.pq.ParquetWriter

    class ObservedWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            assert active
            observed["written"] = True
            return self._writer.write_table(*args, **kwargs)

        def close(self):  # noqa: ANN201
            return self._writer.close()

    monkeypatch.setattr(ledger_artifact, "_artifact_frame_reservation", reservation)
    monkeypatch.setattr(ledger_artifact, "_validated_frame", validate)
    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", ObservedWriter)
    write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    assert observed == {"validated": True, "written": True}


def test_portfolio_ledger_small_session_budget_releases_table_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    real_table_state = ledger_artifact._artifact_table_write_state

    def force_indivisible_session(*args, **kwargs):  # noqa: ANN002, ANN003
        persisted, state, _batch_rows, retained = real_table_state(*args, **kwargs)
        return persisted, state, 1, retained

    monkeypatch.setattr(
        ledger_artifact,
        "_artifact_table_write_state",
        force_indivisible_session,
    )
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            MemoryError,
            match="portfolio_session_exceeds_task_memory_budget",
        ):
            write_portfolio_ledger_artifact(
                tmp_path / "portfolio",
                envelope=_envelope(),
                ledger_chunks=(frame,),
            )
    assert policy.used_memory_bytes == 0
    instances = tmp_path / "portfolio" / "instances"
    assert not instances.exists() or not any(instances.iterdir())


def test_portfolio_ledger_primary_error_survives_writer_close_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    second = first.drop(columns=["decision_reason"])
    real_writer = ledger_artifact.pq.ParquetWriter

    class FailingCloseWriter:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self._writer.write_table(*args, **kwargs)

        def close(self):  # noqa: ANN201
            self._writer.close()
            raise OSError("secondary_close_failure")

    monkeypatch.setattr(ledger_artifact.pq, "ParquetWriter", FailingCloseWriter)
    with pytest.raises(ValueError, match="portfolio_ledger_schema_invalid") as raised:
        write_portfolio_ledger_artifact(
            tmp_path / "portfolio",
            envelope=_envelope(),
            ledger_chunks=(first, second),
        )
    assert any("close also failed" in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("negative_adv", "portfolio_ledger_adv_invalid"),
        ("zero_trade_adv", "portfolio_ledger_adv_invalid"),
        ("bool_numeric", "portfolio_ledger_numeric_invalid"),
        ("block_reason", "portfolio_ledger_block_reason_invalid"),
        ("decision_reason", "portfolio_ledger_decision_reason_invalid"),
    ),
)
def test_portfolio_ledger_trust_boundary_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    asset = frame["asset_id"].eq("asset")
    if mutation == "negative_adv":
        frame.loc[asset, "adv20"] = -1.0
    elif mutation == "zero_trade_adv":
        frame.loc[asset, "adv20"] = 0.0
    elif mutation == "bool_numeric":
        frame["adv20"] = frame["adv20"].astype(object)
        frame.loc[asset, "adv20"] = True
    elif mutation == "block_reason":
        frame.loc[asset, "block_reason"] = "unknown_reason"
    else:
        frame.loc[asset, "decision_reason"] = "unknown_reason"
    with pytest.raises(ValueError, match=error):
        write_portfolio_ledger_artifact(
            tmp_path / mutation,
            envelope=_envelope(),
            ledger_chunks=(frame,),
        )


def test_portfolio_ledger_rechecks_outer_manifest_and_member_set_each_iteration(
    tmp_path: Path,
) -> None:
    frame = _session(
        "2019-01-01",
        "2019-01-02",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    root = write_portfolio_ledger_artifact(
        tmp_path / "portfolio",
        envelope=_envelope(),
        ledger_chunks=(frame,),
    )
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"].append({"path": "extra.parquet", "size": 0, "sha256": "0" * 64})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        next(artifact.iter_chunks())
