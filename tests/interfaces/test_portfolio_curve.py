"""Native publication-to-query curve projection without run or identity gates."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.interfaces.queries import portfolio_curve


def _published_curve(tmp_path):
    capital = PortfolioCapital("candidate", 100.0, "CNY", "adjusted_notional")

    def produce(emit):
        for sequence in range(3):
            stamp = (pd.Timestamp("2020-01-01") + pd.Timedelta(days=sequence)).value
            emit(
                PortfolioLedgerBatch(
                    sequence,
                    (),
                    (),
                    tuple(
                        PortfolioValuation(stamp, "__cash__", phase, 100.0, 100.0)
                        for phase in ("before", "after", "end")
                    ),
                    (
                        PortfolioDiagnostic(
                            "__cash__", None, "decision_reason", text="simple_rule"
                        ),
                        PortfolioDiagnostic(
                            "__cash__", None, "target_notional", number=100.0
                        ),
                        *(
                            (
                                PortfolioDiagnostic(
                                    "__cash__", None, "oos_window_start", number=0.0
                                ),
                            )
                            if sequence == 0
                            else ()
                        ),
                    ),
                )
            )

    root = tmp_path / "ledger"
    write_portfolio_ledger_artifact(
        root,
        capital=capital,
        metadata={
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
            "sample_role": "validation",
        },
        produce=produce,
    )
    return root


def test_native_curve_retains_markers_and_terminal_points(tmp_path):
    root = _published_curve(tmp_path)
    curve = portfolio_curve(root, stride=5)
    assert curve["status"] == "DEFINED"
    assert curve["sample_role"] == "validation"
    assert curve["source_point_count"] == 9
    assert curve["displayed_point_count"] == 6
    assert {point["sequence"] for point in curve["points"]} == {0, 2}
    assert len(curve["boundaries"]) == 1
    assert all(point["nav"] == 100 for point in curve["points"])
    assert portfolio_curve(root, stride=5) == curve


def test_native_curve_rejects_actual_missing_member(tmp_path):
    root = _published_curve(tmp_path)
    (root / "valuations.parquet").unlink()
    with pytest.raises((ValueError, OSError)):
        portfolio_curve(root)


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_curve_stride_is_positive_integer(stride, tmp_path):
    with pytest.raises(ValueError, match="portfolio_curve_stride_invalid"):
        portfolio_curve(tmp_path, stride=stride)
