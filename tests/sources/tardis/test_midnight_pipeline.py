import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.binance.replay import ReconstructionStats
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from tests.helpers.tardis_midnight import write_tardis_midnight_case


@pytest.mark.parametrize("gap_kind", (None, "depth", "agg_trades"))
def test_native_midnight_panel_preserves_real_book_and_trade_gaps(tmp_path, gap_kind):
    raw, origin = write_tardis_midnight_case(tmp_path, gap_kind)
    root = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "source": {"root": str(raw)},
            "inputs": {},
            "time_window": {
                "start": origin.tz_localize(None).isoformat(),
                "end": (
                    origin.tz_localize(None) + pd.Timedelta(seconds=11)
                ).isoformat(),
            },
            "output_root": str(tmp_path / "panel"),
        }
    )
    panel = read_dataset(root)
    assert panel.timestamp.dt.date.nunique() == 2
    metadata = read_dataset_manifest_evidence(root)
    at_gap = origin.value + 8_800_000_000
    for symbol, stats_payload in metadata["producer_diagnostics"][
        "reconstruction"
    ].items():
        stats = ReconstructionStats.from_payload(stats_payload)
        if gap_kind == "depth":
            # The native source stops proven coverage at the ID gap.
            assert not any(
                start <= at_gap < end for start, end in stats.coverage_segments.values()
            )
        else:
            assert stats.gaps == 0
            assert any(
                start <= at_gap < end for start, end in stats.coverage_segments.values()
            )
        with open_market_source(raw) as source:
            coverage = source.observation_coverage(
                market_type="usd_s_m_futures", data_kind="agg_trades", symbol=symbol
            ).covering
            assert bool(coverage(at_gap - 100_000_000, at_gap + 100_000_000)) is (
                gap_kind != "agg_trades"
            )
