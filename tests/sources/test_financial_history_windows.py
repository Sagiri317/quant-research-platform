from pathlib import Path

import pandas as pd

from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
from quant_research_platform.sources.choice.raw_exports import choice_session
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)
from tests.helpers.choice_raw import SDK, response, write_css
from tests.helpers.paths import StorageLayout


def test_explicit_financial_windows_keep_choice_history_and_point_in_time_revisions(
    tmp_path,
):
    year = 2010
    previous = year - 1
    raw_layout = StorageLayout(tmp_path / "raw-project")
    fields = tuple(field for mapping in FINANCIAL_FIELDS.values() for field in mapping)
    with choice_session(SDK(response())) as session:
        for period, first, value in (
            (f"{previous}-03-31", f"{previous}-04-20", 1.0),
            (f"{previous}-12-31", f"{year}-04-20", 4.0),
            (f"{year}-03-31", f"{year}-04-30", 5.0),
        ):
            codes = (
                ("000001.SZ",)
                if period == f"{previous}-03-31"
                else ("000001.SZ", "000002.SZ")
            )
            correction = f"{year}-05-05" if period == f"{previous}-12-31" else None
            write_css(
                session,
                raw_layout,
                period=period,
                end_date="2025-12-31",
                indicators=("STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE"),
                data={code: [first, None, correction] for code in codes},
            )
            snapshots = [(first, value)]
            if correction:
                snapshots.append((correction, 6.0))
            for day, amount in snapshots:
                for branch in ("1", "3"):
                    write_css(
                        session,
                        raw_layout,
                        period=period,
                        end_date=day,
                        indicators=fields,
                        data={
                            code: [amount if branch == "1" else None] * len(fields)
                            for code in codes
                        },
                        branch=branch,
                    )
    sessions = pd.bdate_range(f"{previous}-01-02", f"{year}-05-08")
    frames = _synthetic_interim_frames(sessions)

    inputs = _write_inputs(tmp_path / "source", frames)
    financial_ids = (
        "financial_income",
        "financial_balance",
        "financial_cashflow",
        "financial_first_disclosure",
    )
    financial_paths = {name: str(tmp_path / "choice" / name) for name in financial_ids}
    prepare_source(
        {
            "source_id": "choice",
            "raw_root": str(raw_layout.raw_source_root("choice")),
            "outputs": financial_paths,
            "start_date": f"{previous}-01-01",
            "end_date": f"{year}-05-08",
        }
    )
    inputs.update({key: Path(value) for key, value in financial_paths.items()})
    income = read_dataset(inputs["financial_income"])
    assert income.report_period.min() == pd.Timestamp(f"{previous}-03-31")
    assert income.is_corrected.any()
    observations = []
    common_outputs = []
    for index, start in enumerate((f"{previous}-01-01", f"{year}-01-01")):
        prepared = dict(inputs)
        for name in ("market_panel_1d", "fundamental_pit"):
            prepared[name] = _prepare(
                tmp_path / f"prepared-{index}",
                name,
                prepared,
                pd.Timestamp(start),
                sessions[-1],
            )
        saved = read_dataset(prepared["fundamental_pit"])
        assert saved.timestamp.ge(start).all()
        assert saved.timestamp.le(f"{year}-05-08").all()
        common_outputs.append(
            saved.loc[saved.timestamp.ge(f"{year}-01-01")].reset_index(drop=True)
        )
        selected = saved.loc[saved.asset_id.eq("000001")].set_index("timestamp")
        observations.append(
            selected.loc[
                pd.to_datetime([f"{year}-05-04", f"{year}-05-05", f"{year}-05-06"]),
                "operating_revenue_ttm",
            ].tolist()
        )
        incomplete = saved.loc[
            saved.asset_id.eq("000002") & saved.timestamp.ge(f"{year}-05-04")
        ]
        assert not incomplete.empty
        assert incomplete.operating_revenue_ttm.isna().all()
    pd.testing.assert_frame_equal(*common_outputs, check_exact=True)
    assert observations == [[8.0, 8.0, 10.0], [8.0, 8.0, 10.0]]
    early = read_dataset(
        _prepare(
            tmp_path / "before-correction",
            "fundamental_pit",
            prepared,
            pd.Timestamp(f"{year}-05-04"),
            pd.Timestamp(f"{year}-05-04"),
        )
    )
    expected = saved.loc[saved.timestamp.eq(f"{year}-05-04")].reset_index(drop=True)
    pd.testing.assert_frame_equal(early, expected, check_exact=True)
    assert early.loc[early.asset_id.eq("000001"), "operating_revenue_ttm"].eq(8.0).all()
