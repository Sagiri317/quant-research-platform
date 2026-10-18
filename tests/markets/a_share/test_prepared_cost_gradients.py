"""Analytic fee and impact derivatives independently match currency economics."""

from datetime import date

import numpy as np
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.markets.implementations.a_share.execution import (
    prepare_trade_cost,
)
from tests.research.test_portfolio_ledger import _schedule


@pytest.mark.parametrize("side, fee", (("buy", 0.0004), ("sell", 0.0009)))
def test_prepared_cost_value_and_marginal_rate_match_closed_form(side, fee):
    prepared = prepare_trade_cost(
        date(2023, 1, 4),
        "shanghai",
        side,
        _schedule(),
        ExecutionAssumptions(100.0, 2.0, 10.0, 1.0, (1.0,)),
    )
    amount = np.array([-25.0, 0.0, 25.0, 100.0])
    adv = np.full(4, 400.0)
    values, gradients = prepared.totals_and_gradients(amount, adv)
    linear = fee + 0.0002
    assert values == pytest.approx(
        linear * amount + 0.001 * np.maximum(amount, 0.0) ** 1.5 / 20.0,
        rel=1e-14,
        abs=0,
    )
    assert gradients == pytest.approx(
        linear + 0.0015 * np.sqrt(np.maximum(amount, 0.0) / 400.0), rel=1e-14, abs=0
    )
    for value, q in zip(values[2:], amount[2:], strict=True):
        assert prepared.trade(float(q), 400.0).total_cost == pytest.approx(
            value, rel=1e-14, abs=1e-14
        )
    with pytest.raises(ValueError, match="a_share_trade_cost_inputs_invalid"):
        prepared.trade(-25.0, 400.0)
