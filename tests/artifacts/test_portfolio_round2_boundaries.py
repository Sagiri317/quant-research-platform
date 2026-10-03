from __future__ import annotations

import math
import sys
from dataclasses import asdict, replace
from fractions import Fraction

import pytest

import quant_research_platform.artifacts.portfolio_ledger as ledger
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.runtime.resources import (
    MemoryLease,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.identity import filesystem_exists
from tests.artifacts.test_portfolio_ledger_artifact import _file_bytes


def _envelope(capital):
    return build_business_manifest_envelope(
        logical_artifact_id=capital.candidate_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01",
            "end": "1970-01-02",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )


@pytest.mark.parametrize("capital_amount", (1700.0, 1.7e308))
def test_finite_offset_cash_reconciliation_survives_real_publication(
    tmp_path, capital_amount
):
    capital = PortfolioCapital(
        "finite_cash", capital_amount, "USDT", "linear_perpetual"
    )
    normal = []
    economics = PortfolioEconomics(capital, "btc_usdt", normal.append)
    economics.mark(0, capital_amount)
    economics.fill(
        timestamp_ns=1,
        order_id="open",
        quantity=1.0,
        price=capital_amount,
        mark_price=capital_amount,
        fee=0.0,
    )
    mark = capital_amount * (7.0 / 17.0)
    economics.mark(2, mark)
    metrics = economics.mark(3, mark)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["nav"] == pytest.approx(mark)
    assert metrics["nav"] > 0
    changed_cash = capital_amount * (16.0 / 17.0)
    assert Fraction(changed_cash) - Fraction(capital_amount) != 0
    assert not normal[-1].cashflows
    envelope = _envelope(capital)

    def produce(emit):
        for batch in normal:
            emit(batch)

    with runtime_resource_policy_scope() as policy:
        old_root = ledger.write_portfolio_ledger_artifact(
            tmp_path / "old", envelope=envelope, produce=produce
        )
        original = _file_bytes(old_root)
        with policy.lease("test.portfolio.metadata", 0) as lease:
            artifact = ledger.read_portfolio_ledger_artifact(
                old_root, expected_identity=envelope.instance_identity, lease=lease
            )
            batches = tuple(artifact.iter_chunks())
            assert len(batches) == 4
            values = {
                row.asset_id: row.value
                for row in batches[-1].valuations
                if row.phase == "end"
            }
            assert values["__cash__"] == capital_amount
            assert sum(values.values()) == pytest.approx(mark)
            artifact = batches = None

        def corrupt(emit):
            for batch in normal[:-1]:
                emit(batch)
            emit(
                replace(
                    normal[-1],
                    valuations=tuple(
                        replace(row, value=changed_cash, quantity=changed_cash)
                        if row.asset_id == "__cash__" and row.phase in {"after", "end"}
                        else row
                        for row in normal[-1].valuations
                    ),
                )
            )

        with pytest.raises(
            ValueError, match="^portfolio_ledger_cash_reconciliation_invalid$"
        ):
            ledger.write_portfolio_ledger_artifact(
                tmp_path / "invalid", envelope=envelope, produce=corrupt
            )
        assert not filesystem_exists(
            tmp_path / "invalid" / "instances" / envelope.instance_identity
        )
        assert _file_bytes(old_root) == original
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("assets", (64, 1024))
@pytest.mark.parametrize("boundary", ("normal", "memory", "early_close"))
def test_reader_continuity_released_before_lease_close(
    tmp_path, monkeypatch, assets, boundary
):
    capital = PortfolioCapital("reader_continuity", 100.0, "USDT", "linear_perpetual")
    envelope = _envelope(capital)
    cap = 256 * 1024**2
    original_close = MemoryLease.close
    observed = []
    errors_at_close = []

    def close(lease):
        if lease.owner == "artifacts.portfolio.continuity" and lease.active:
            cursor = sys._getframe(1)
            try:
                while cursor is not None:
                    if cursor.f_code is ledger._iter_batches.__code__:
                        previous = cursor.f_locals.get("previous")
                        observed.append(0 if previous is None else len(previous))
                        errors_at_close.append(sys.exception())
                        break
                    cursor = cursor.f_back
            finally:
                cursor = previous = None
        original_close(lease)

    def produce(emit):
        emit(_batch(0, assets))
        emit(_batch(1, 2 * assets))

    with runtime_resource_policy_scope(explicit_cap_bytes=cap) as policy:
        root = ledger.write_portfolio_ledger_artifact(
            tmp_path / "ledger", envelope=envelope, produce=produce
        )
        original = _file_bytes(root)
        monkeypatch.setattr(MemoryLease, "close", close)
        with policy.lease("test.reader.metadata", 0) as metadata:
            artifact = ledger.read_portfolio_ledger_artifact(
                root, lease=metadata, expected_identity=envelope.instance_identity
            )
            assert observed == [0]
            assert errors_at_close == [None]
            observed.clear()
            errors_at_close.clear()
            pressure = None
            try:
                with closing_iterator(artifact.iter_chunks()) as batches:
                    first = next(batches)
                    assert first.sequence == 0
                    first = None
                    if boundary == "memory":
                        pressure = policy.lease(
                            "test.concurrent_owner", cap - policy.used_memory_bytes
                        )
                        with pytest.raises(MemoryError) as caught:
                            next(batches)
                        assert errors_at_close == [caught.value]
                        cursor = caught.value.__traceback__
                        reader_frames = 0
                        try:
                            while cursor is not None:
                                if (
                                    cursor.tb_frame.f_code
                                    is ledger._iter_batches.__code__
                                ):
                                    reader_frames += 1
                                    assert cursor.tb_frame.f_locals["previous"] is None
                                cursor = cursor.tb_next
                        finally:
                            cursor = None
                        assert reader_frames == 1
                    elif boundary == "normal":
                        following = next(batches)
                        assert following.sequence == 1
                        assert len(following.valuations) == 3 * (2 * assets + 1)
                        following = None
                        with pytest.raises(StopIteration):
                            next(batches)
                    else:
                        batches.close()
            finally:
                if pressure is not None:
                    pressure.close()
                pressure = artifact = batches = None
            assert observed == [0]
            assert policy.used_memory_bytes == metadata.reserved_bytes
        assert policy.used_memory_bytes == 0
        assert _file_bytes(root) == original


def _batch(sequence, assets, *, cash=100.0):
    return PortfolioLedgerBatch(
        sequence,
        (),
        (),
        tuple(
            row
            for phase in ("before", "after", "end")
            for row in (
                PortfolioValuation(sequence, "__cash__", phase, cash, cash),
                *(
                    PortfolioValuation(sequence, f"asset_{index}", phase, 0.0, 0.0)
                    for index in range(assets)
                ),
            )
        ),
        (),
    )


@pytest.mark.parametrize("assets", (64, 1024))
@pytest.mark.parametrize(
    "boundary", ("normal", "producer", "transition", "writer_close")
)
def test_writer_continuity_released_before_lease_close_and_readback(
    tmp_path, monkeypatch, assets, boundary
):
    capital = PortfolioCapital("continuity", 100.0, "USDT", "linear_perpetual")
    envelope = _envelope(capital)
    error = RuntimeError("controlled_portfolio_failure")
    original_close = MemoryLease.close
    original_grow = MemoryLease.grow
    original_validate = ledger.validate_portfolio_transition
    closed = []
    read_admissions = []
    map_ids = set()

    def observe_writer():
        cursor = sys._getframe(1)
        try:
            while cursor is not None:
                if (
                    cursor.f_code.co_name == "write"
                    and cursor.f_code.co_filename == ledger.__file__
                ):
                    previous = cursor.f_locals.get("previous")
                    return 0 if previous is None else len(previous)
                cursor = cursor.f_back
        finally:
            previous = cursor = None
        return None

    def close(lease):
        original_close(lease)
        if lease.owner == "artifacts.portfolio.write_continuity":
            closed.append(observe_writer())

    def grow(lease, amount):
        if lease.owner == "artifacts.portfolio.transition" and not read_admissions:
            read_admissions.append(observe_writer())
        return original_grow(lease, amount)

    def validate(batch, capital, previous):
        try:
            if previous is not None:
                map_ids.add(id(previous))
            return original_validate(batch, capital, previous)
        finally:
            previous = None

    def produce(emit):
        emit(_batch(0, 0))
        emit(_batch(1, assets))
        if boundary == "producer":
            raise error
        if boundary == "transition":
            emit(_batch(2, assets, cash=99.0))

    with runtime_resource_policy_scope() as policy:
        old_root = ledger.write_portfolio_ledger_artifact(
            tmp_path / "old", envelope=envelope, produce=lambda emit: emit(_batch(0, 0))
        )
        original = _file_bytes(old_root)
        monkeypatch.setattr(MemoryLease, "close", close)
        monkeypatch.setattr(MemoryLease, "grow", grow)
        monkeypatch.setattr(ledger, "validate_portfolio_transition", validate)
        if boundary == "writer_close":
            original_writer = ledger.pq.ParquetWriter

            class FailingCloseWriter:
                def __init__(self, *args, **kwargs):
                    self.writer = original_writer(*args, **kwargs)

                def write_table(self, *args, **kwargs):
                    return self.writer.write_table(*args, **kwargs)

                def add_key_value_metadata(self, *args, **kwargs):
                    return self.writer.add_key_value_metadata(*args, **kwargs)

                def close(self):
                    self.writer.close()
                    raise error

            monkeypatch.setattr(ledger.pq, "ParquetWriter", FailingCloseWriter)

        if boundary == "normal":
            root = ledger.write_portfolio_ledger_artifact(
                tmp_path / "candidate", envelope=envelope, produce=produce
            )
            assert root.name == envelope.instance_identity
            assert read_admissions == [0]
        else:
            expected = ValueError if boundary == "transition" else RuntimeError
            with pytest.raises(expected) as raised:
                ledger.write_portfolio_ledger_artifact(
                    tmp_path / "candidate", envelope=envelope, produce=produce
                )
            if boundary == "transition":
                assert (
                    str(raised.value) == "portfolio_ledger_cross_session_state_invalid"
                )
            else:
                assert raised.value is error
            assert raised.value.__traceback__ is not None
            traceback = raised.value.__traceback__
            try:
                while traceback is not None:
                    assert not any(
                        isinstance(value, dict)
                        and id(value) in map_ids
                        and "__cash__" in value
                        for value in traceback.tb_frame.f_locals.values()
                    )
                    traceback = traceback.tb_next
            finally:
                traceback = None
            assert not filesystem_exists(
                tmp_path / "candidate" / "instances" / envelope.instance_identity
            )
        assert closed == [0]
        assert policy.used_memory_bytes == 0
        assert _file_bytes(old_root) == original
