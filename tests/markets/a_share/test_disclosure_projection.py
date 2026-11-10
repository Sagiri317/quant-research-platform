"""Extra published disclosure context does not invalidate financial preparation."""

from dataclasses import replace

import pandas as pd

from quant_research_platform.markets.prepare import prepare
from quant_research_platform.sources.choice.interim import first_disclosure_spec
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.a_share_catalogs import scientific_catalogs
from tests.markets.a_share.test_calendar_dependency_scope import _calendar
from tests.markets.a_share.test_financial_pit import _financial_frame


def test_prepare_financial_pit_accepts_disclosure_context(tmp_path):
    catalogs = scientific_catalogs()
    balance = _financial_frame(
        "financial_balance",
        (
            {
                "asset": "asset",
                "period": "2019-12-31",
                "corrected": False,
                "correction_date": pd.NaT,
                "version": "original",
                "value": 100.0,
            },
        ),
    )
    balance["timestamp"] = balance.report_period
    calendar = _calendar({1: pd.DatetimeIndex(["2020-01-31", "2020-02-03"])})
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["asset"],
            "timestamp": pd.to_datetime(["2020-02-03"]),
        }
    )
    for data_id, frame, spec in (
        ("financial_balance", balance, catalogs.interim.get("financial_balance")),
        ("trading_calendar", calendar, catalogs.interim.get("trading_calendar")),
        ("market_panel_1d", panel, catalogs.processed.get("market_panel_1d")),
    ):
        spec = replace(
            spec,
            column_roles={name: spec.column_roles[name] for name in frame},
            column_units={
                name: unit for name, unit in spec.column_units.items() if name in frame
            },
            partition_columns=(),
        )
        write_dataset(tmp_path / data_id, frame, spec=spec)
    disclosure = pd.DataFrame(
        {
            "asset_id": ["asset"],
            "report_period": pd.to_datetime(["2019-12-31"]),
            "first_disclosure_date": pd.to_datetime(["2020-01-30"]),
        }
    )
    results = []
    for extra in (False, True):
        frame = (
            disclosure.assign(provenance_note="vendor context") if extra else disclosure
        )
        spec = first_disclosure_spec()
        spec = replace(
            spec,
            column_roles={
                **spec.column_roles,
                **({"provenance_note": "context"} if extra else {}),
            },
        )
        path = tmp_path / f"disclosure-{extra}"
        write_dataset(path, frame, spec=spec)
        root = prepare(
            {
                "market_id": "a_share",
                "data_id": "fundamental_pit",
                "inputs": {
                    **{
                        name: str(tmp_path / name)
                        for name in (
                            "financial_balance",
                            "trading_calendar",
                            "market_panel_1d",
                        )
                    },
                    "financial_first_disclosure": str(path),
                },
                "factor_columns": ["book_to_market"],
                "time_window": {"start": "2020-02-03", "end": "2020-02-03"},
                "output_root": str(tmp_path / f"prepared-{extra}"),
            }
        )
        results.append(read_dataset(root))
    pd.testing.assert_frame_equal(*results)
