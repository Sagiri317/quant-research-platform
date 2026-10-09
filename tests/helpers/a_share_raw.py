"""Deterministic provider-shaped inputs for the actual A-share production chain."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.ids import (
    FORMAL_BENCHMARKS,
)
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
from quant_research_platform.sources.choice.raw_exports import choice_session
from quant_research_platform.sources.csmar.field_mappings import RAW_TABLE_CONTRACTS
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FEE_CSV_COLUMNS,
)
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_stat,
    filesystem_write_text,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.choice_raw import SDK, response, write_css


def _write_csmar(root, data_id, rows, *, extra_columns=()):
    columns = (*RAW_TABLE_CONTRACTS[data_id].required_columns, *extra_columns)
    path = root / f"csmar__{data_id}__synthetic__full__export-20260101.csv"
    with open_filesystem_file(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_a_share_raw_fixture(
    layout: StorageLayout,
    *,
    benchmark_total_return_indices: Mapping[str, str],
) -> None:
    """Keep real source, processed, dataset, model, and portfolio bindings intact."""
    raw_root = layout.raw_source_root("csmar").parent
    root = Path(raw_root) / "csmar"
    filesystem_mkdir(root, parents=True)
    # Preserve the production factor owner's 312-session warm-up.
    history_end = a_share.HOLDOUT_END
    sessions = pd.bdate_range(
        pd.Timestamp(a_share.FORMAL_DATASET_START) - pd.offsets.BDay(312),
        history_end,
    )
    benchmark_dates = pd.bdate_range(
        pd.Timestamp(a_share.DEVELOPMENT_START) - pd.offsets.BDay(1), history_end
    )
    # Active enhancement requires the risk owner's 100-asset cross-section floor.
    assets = tuple(f"{index:06}" for index in range(1, 101))
    rng = np.random.default_rng(91027)
    drift = np.linspace(-0.0006, 0.0014, len(assets))
    returns = drift[None, :] + rng.normal(0, 0.002, (len(sessions), len(assets)))
    close = np.linspace(10.0, 30.0, len(assets))[None, :] * np.cumprod(
        1 + returns, axis=0
    )
    previous = close / (1 + returns)
    opened = previous * (1 + rng.normal(0, 0.0005, close.shape))
    shares = np.linspace(1.0e9, 2.0e9, len(assets))
    volume = shares[None, :] * rng.uniform(0.03, 0.07, close.shape)

    def bars():
        for column, asset in enumerate(assets):
            for row, timestamp in enumerate(sessions):
                price = float(close[row, column])
                opening = float(opened[row, column])
                yield {
                    "Stkcd": asset,
                    "Trddt": timestamp.date().isoformat(),
                    "Markettype": 4,
                    "Trdsta": 0,
                    "Filling": 0,
                    "Opnprc": opening,
                    "Hiprc": max(price, opening) * 1.002,
                    "Loprc": min(price, opening) * 0.998,
                    "Clsprc": price,
                    "Dnshrtrd": float(volume[row, column]),
                    "Dnvaltrd": float(volume[row, column]) * price,
                    "Ahshrtrd_D": 0.0,
                    "Ahvaltrd_D": 0.0,
                    "Dsmvosd": price * shares[column] / 1000,
                    "Dsmvtl3": price * shares[column] * 1.2 / 1000,
                    "Dretwd": float(returns[row, column]),
                    "Adjprcwd": price,
                    "PreClosePrice": float(previous[row, column]),
                    "LimitStatus": 0,
                }

    _write_csmar(root, "market_bars_1d", bars())
    _write_csmar(
        root,
        "stock_master",
        (
            {
                "Stkcd": asset,
                "Stknme": f"synthetic_{asset}",
                "Listdt": "2011-01-03" if index == 0 else "2000-01-03",
                "DelistedDate": "",
                "Markettype": 4,
                "Nnindcd": "A" if index % 2 else "C",
                "Nnindnme": "synthetic industry",
                "Statco": "normal",
                "Statdt": "2011-01-03",
            }
            for index, asset in enumerate(assets)
        ),
    )
    _write_csmar(
        root,
        "market_cap",
        (
            {
                "Stkcd": asset,
                "Shrchgdt": "2011-01-03",
                "Nshrttl": shares[index] * 1.2,
                "Nshra": shares[index],
            }
            for index, asset in enumerate(assets)
        ),
    )
    _write_csmar(
        root,
        "trading_calendar",
        (
            {
                "Markettype": 4,
                "Clddt": timestamp.date().isoformat(),
                "Daywk": timestamp.weekday() + 1,
                "State": "O",
            }
            for timestamp in sessions
        ),
    )
    _write_csmar(
        root,
        "industry_classification",
        (
            {
                "Symbol": asset,
                "IndustryClassificationID": "P0207",
                "ImplementDate": "2011-01-03",
                "IndustryCode": "A01" if index % 2 else "C10",
                "IndustryName": "synthetic industry",
            }
            for index, asset in enumerate(assets)
        ),
    )
    _write_csmar(
        root,
        "stock_info_changes",
        (
            {
                "Symbol": asset,
                "AnnouncementDate": "2011-01-02",
                "ImplementDate": "2011-01-03",
                "ChangedItem": "P0207行业代码",
                "SecurityID": asset,
                "ListedCoID": asset,
                "Value_Before": "",
                "Value_After": "synthetic industry",
            }
            for asset in assets
        ),
        extra_columns=("SecurityID", "ListedCoID"),
    )
    _write_csmar(
        root,
        "industry_classification_sets",
        (
            {
                "INDUSTRYCODE": code,
                "INDCLASSIFYSYSTEMCODE": "P0207",
                "INDUSTRYNAME": f"synthetic_{code}",
                "P_INDUSTRYCODE": "" if len(code) == 1 else code[0],
                "INDUSTRYID": code,
                "RANK": "1" if len(code) == 1 else "2",
            }
            for code in ("A", "C", "A01", "C10")
        ),
    )
    _write_csmar(
        root,
        "industry_classification_versions",
        (
            {
                "IndClassifySystemCode": "P0207",
                "IndClassifySystem": "synthetic CSRC2012",
                "ImplementDate": "2011-01-03",
                "AbandonDate": "",
                "InstitutionName": "synthetic provider",
            },
        ),
    )
    _write_csmar(
        root,
        "trading_no_limit",
        (
            {
                "TradingDate": "2011-01-03",
                "Symbol": assets[0],
                "Reason": "新股上市首日",
            },
        ),
    )
    _write_csmar(
        root,
        "special_treatment_changes",
        (
            {"Symbol": assets[1], "Chgtype": code, "Annoudt": stamp, "Execudt": stamp}
            for code, stamp in (("AB", "2011-01-04"), ("BA", "2011-01-06"))
        ),
    )
    total_return_codes = benchmark_total_return_indices
    info = []
    for code, (benchmark, name) in FORMAL_BENCHMARKS.items():
        sources = [(code, name)]
        if benchmark in total_return_codes:
            sources.append((total_return_codes[benchmark], f"{name}全收益"))
        for source_code, label in sources:
            info.append(
                {
                    "Indexcd": source_code,
                    "Idxinfo01": label,
                    "Idxinfo11": "2000-01-03",
                    "Idxinfo02": "2000-01-03",
                    "Idxinfo03": 1000.0,
                    "Idxinfo04": "synthetic_a_share",
                    "Idxinfo05": "market_cap",
                    "Idxinfo06": "synthetic_provider",
                    "Idxinfo07": "synthetic_scale_index",
                    "Idxinfo08": "total_return"
                    if source_code.startswith("H")
                    else "price",
                    "Idxinfo09": "a_share",
                    "Idxinfo10": "weighted_index",
                }
            )
    _write_csmar(
        root,
        "index_info",
        info,
        extra_columns=tuple(
            f"Idxinfo{index:02}" for index in range(1, 11) if index not in (1, 2)
        ),
    )
    _write_csmar(
        root,
        "index_weights",
        (
            {
                "Indexcd": code,
                "Enddt": timestamp.date().isoformat(),
                "Stkcd": asset,
                "Constdnme": f"synthetic_{asset}",
                "Weight": 100 / len(assets),
            }
            for code in sorted(FORMAL_BENCHMARKS)
            for timestamp in benchmark_dates
            for asset in assets
        ),
    )

    def index_rows():
        for code in sorted(info, key=lambda item: item["Indexcd"]):
            for index, timestamp in enumerate(sessions):
                opening = 1000 * (1.0002**index)
                yield {
                    "Indexcd": code["Indexcd"],
                    "Idxtrd01": timestamp.date().isoformat(),
                    "Idxtrd02": opening,
                    "Idxtrd03": opening * 1.001,
                    "Idxtrd04": opening * 0.999,
                    "Idxtrd05": opening * 1.0001,
                    "Idxtrd06": 10000.0,
                    "Idxtrd07": 100000.0,
                    "Idxtrd08": 0.02,
                    "Idxtrd09": code["Idxinfo01"],
                }

    _write_csmar(root, "index_market", index_rows())
    write_industry_reference_fixture(
        raw_root,
        observations=(
            (
                "initial",
                "CSRC2012",
                "2011-01-03",
                "2011-01-04",
                [
                    {
                        "code": asset,
                        "name": "synthetic issuer",
                        "category": "A" if index % 2 else "C",
                        "major": "",
                        "industry": "01" if index % 2 else "10",
                        "source_locator": "page:1",
                    }
                    for index, asset in enumerate(assets)
                ],
            ),
        ),
    )
    schedule_root = Path(raw_root) / "reference/a_share_fee_schedule"
    filesystem_mkdir(schedule_root, parents=True)
    schedule = schedule_root / A_SHARE_FEE_CSV_NAME
    rows = []
    for fee_code, direction, rate in (
        ("BROKER_COMMISSION", "买入+卖出", 0.0003),
        ("TRANSFER_FEE", "买入+卖出", 0.00001),
        ("STAMP_DUTY", "卖出", 0.001),
    ):
        row = dict.fromkeys(FEE_CSV_COLUMNS, "")
        row.update(
            {
                "费用代码": fee_code,
                "生效日": "2011-01-01",
                "截止日": "2025-12-31",
                "市场": "沪深",
                "方向": direction,
                "计费基数": "成交金额",
                "十进制费率": str(rate),
                "费率(‰)": str(rate * 1000),
                "费率(bps)": str(rate * 10000),
                "进入正式账本": "是",
            }
        )
        rows.append(row)
    with open_filesystem_file(
        schedule, "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FEE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    filesystem_write_text(
        schedule_root / A_SHARE_FEE_SIDECAR_NAME,
        json.dumps(
            {
                "file_sha256": hashlib.sha256(
                    filesystem_read_bytes(schedule)
                ).hexdigest(),
                "file_bytes": filesystem_stat(schedule).st_size,
                "row_count": len(rows),
                "fields_raw": list(FEE_CSV_COLUMNS),
            }
        ),
        encoding="utf-8",
    )
    fields = tuple(field for mapping in FINANCIAL_FIELDS.values() for field in mapping)
    names = tuple(
        name for mapping in FINANCIAL_FIELDS.values() for name in mapping.values()
    )
    profitability = np.random.default_rng(563).uniform(0.03, 0.08, len(assets))
    with choice_session(SDK(response())) as session:
        for period in pd.period_range(
            "2010Q1", pd.Timestamp(history_end).to_period("Q") - 1, freq="Q"
        ):
            quarter = period.quarter
            first = pd.Timestamp(
                year=period.year + int(quarter == 4),
                month={1: 5, 2: 8, 3: 11, 4: 3}[quarter],
                day=15,
            )
            report_date = period.end_time.date().isoformat()
            first_text = first.date().isoformat()
            total_assets = (
                shares
                * np.linspace(8.0, 13.0, len(assets))
                * (1 + 0.05 * (period.year - 2010))
            )
            fraction = quarter / 4
            values = {
                "total_assets": total_assets,
                "total_liabilities": total_assets * 0.4,
                "parent_equity": total_assets * 0.6,
                "cash": total_assets * 0.12,
                "short_term_borrowings": total_assets * 0.03,
                "current_portion_noncurrent_liabilities": total_assets * 0.02,
                "long_term_borrowings": total_assets * 0.1,
                "bonds_payable": total_assets * 0.02,
                "operating_revenue": total_assets * 0.9 * fraction,
                "operating_cost": total_assets * 0.6 * fraction,
                "consolidated_net_income": total_assets * profitability * fraction,
                "parent_net_income": total_assets * profitability * 0.9 * fraction,
                "operating_cash_flow": total_assets * 0.06 * fraction,
                "capex_cash_paid": total_assets * 0.025 * fraction,
            }
            write_css(
                session,
                layout,
                period=report_date,
                end_date="2025-12-31",
                indicators=("STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE"),
                data={
                    f"{asset}.SZ": [first_text, first_text, None] for asset in assets
                },
            )
            write_css(
                session,
                layout,
                period=report_date,
                end_date=first_text,
                indicators=fields,
                data={
                    f"{asset}.SZ": [float(values[name][index]) for name in names]
                    for index, asset in enumerate(assets)
                },
            )


def write_industry_reference_fixture(raw_root, *, observations, document_suffix=".pdf"):
    """Write reviewed-source-shaped snapshots for the real reference reader."""
    root = Path(raw_root) / "reference/a_share_industry_snapshots"
    filesystem_mkdir(root, parents=True, exist_ok=True)
    files = {}
    snapshots = []
    for identifier, standard, observed, published, rows in observations:
        document_name = identifier + document_suffix
        page_name = identifier + ".html"
        candidate_name = identifier + ".json"
        filesystem_write_text(
            root / document_name, "synthetic public source", encoding="utf-8"
        )
        filesystem_write_text(
            root / page_name, "<html>" + published + "</html>", encoding="utf-8"
        )
        candidate = {
            "source": document_name,
            "source_sha256": hashlib.sha256(
                filesystem_read_bytes(root / document_name)
            ).hexdigest(),
            "records": rows,
        }
        filesystem_write_text(
            root / candidate_name,
            json.dumps(candidate, ensure_ascii=False),
            encoding="utf-8",
        )
        for name in (document_name, page_name, candidate_name):
            files[name] = hashlib.sha256(filesystem_read_bytes(root / name)).hexdigest()
        host = "www.csrc.gov.cn" if standard == "CSRC2012" else "www.capco.org.cn"
        snapshots.append(
            {
                "snapshot_id": identifier,
                "industry_standard_id": standard,
                "observation_date": observed,
                "publication_date": published,
                "source_url": "https://" + host + "/synthetic.html",
                "candidate_file": candidate_name,
                "document_file": document_name,
                "page_file": page_name,
            }
        )
    metadata = {
        "schema_id": "a_share_industry_snapshots/0.1",
        "files": files,
        "snapshots": snapshots,
    }
    filesystem_write_text(
        root / "metadata.json",
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )
    return root
