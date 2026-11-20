"""Native publication-to-query curve projection without run or identity gates."""

import pandas as pd
import pytest

import quant_research_platform.artifacts.portfolio_ledger as ledger
from quant_research_platform.artifacts.portfolio_curve import portfolio_curve
from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)


def _published_curve(tmp_path):
    capital = PortfolioCapital("candidate", 100.0, "CNY", "cash_equity")

    def produce(emit):
        for sequence in range(3):
            stamp = (pd.Timestamp("2020-01-01") + pd.Timedelta(days=sequence)).value
            emit(
                PortfolioLedgerBatch(
                    sequence,
                    (),
                    (),
                    tuple(
                        row
                        for phase in ("before", "after", "end")
                        for row in (
                            PortfolioValuation(stamp, "__cash__", phase, 90.0, 90.0),
                            PortfolioValuation(
                                stamp + int(phase == "end"),
                                "stock",
                                phase,
                                1.0,
                                10.0 + sequence + int(phase == "end"),
                                mark_price=10.0 + sequence + int(phase == "end"),
                            ),
                        )
                    ),
                    (
                        PortfolioDiagnostic(
                            "__cash__", None, "decision_reason", text="simple_rule"
                        ),
                        PortfolioDiagnostic(
                            "__cash__", None, "target_notional", number=100.0
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


def test_native_curve_retains_first_and_terminal_points(tmp_path):
    root = _published_curve(tmp_path)
    curve = portfolio_curve(root, stride=5)
    assert curve["status"] == "DEFINED"
    assert curve["sample_role"] == "validation"
    assert curve["source_point_count"] == 9
    assert curve["displayed_point_count"] == 6
    assert {point["sequence"] for point in curve["points"]} == {0, 2}
    assert [point["nav"] for point in curve["points"]] == [100, 100, 101, 102, 102, 103]
    assert portfolio_curve(root, stride=5) == curve
    from quant_research_platform.reports.plots import plot

    image = plot(
        {
            "kind": "curve",
            "inputs": str(root),
            "output": str(tmp_path / "curve.png"),
            "sample_role": "validation",
            "timezone": "Asia/Shanghai",
        }
    )
    assert image.stat().st_size > 1000


def test_native_curve_reads_only_valuations_across_physical_batches(
    tmp_path, monkeypatch
):
    root = _published_curve(tmp_path)
    read = ledger.iter_dataset_part_batches
    reads = []

    def observed(path, **options):
        reads.append((path.name, options.get("columns")))
        return read(path, **options, batch_rows=2)

    monkeypatch.setattr(ledger, "iter_dataset_part_batches", observed)
    for stride, sequences in ((1, [0, 1, 2]), (2, [0, 2])):
        result = portfolio_curve(root, stride=stride)
        assert result["points"] == [
            {
                "sequence": day,
                "phase": phase,
                "timestamp_ns": (
                    pd.Timestamp("2020-01-01") + pd.Timedelta(days=day)
                ).value
                + int(phase == "end"),
                "nav": 100 + day + int(phase == "end"),
            }
            for day in sequences
            for phase in ("before", "after", "end")
        ]
        assert result["observed_end_ns"] == result["points"][-1]["timestamp_ns"]
        assert result["source_point_count"] == 9
    assert len(reads) == 2
    assert all(
        name == "valuations.parquet"
        and set(columns or ()) == {"sequence", "phase", "timestamp_ns", "value"}
        for name, columns in reads
    )


def test_native_curve_rejects_actual_missing_member(tmp_path):
    root = _published_curve(tmp_path)
    (root / "valuations.parquet").unlink()
    with pytest.raises((ValueError, OSError)):
        portfolio_curve(root)


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_curve_stride_is_positive_integer(stride, tmp_path):
    with pytest.raises(ValueError, match="portfolio_curve_stride_invalid"):
        portfolio_curve(tmp_path, stride=stride)
