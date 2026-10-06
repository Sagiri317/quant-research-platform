from __future__ import annotations

import math
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
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
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

    old_root = ledger.write_portfolio_ledger_artifact(
        tmp_path / "old", envelope=envelope, produce=produce
    )
    original = _file_bytes(old_root)
    artifact = ledger.read_portfolio_ledger_artifact(
        old_root, expected_identity=envelope.instance_identity
    )
    batches = tuple(artifact.iter_chunks())
    assert len(batches) == 4
    values = {
        row.asset_id: row.value for row in batches[-1].valuations if row.phase == "end"
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
