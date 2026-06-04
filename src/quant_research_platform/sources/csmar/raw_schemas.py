"""Internal provider raw file, table, and schema contracts.

本模块不读取真实 CSMAR 文件，只定义预期 dataset contracts 和小型 validators，
使真实私有 vendor export 检查前可以先用小样本测试。

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    """定义一个 CSMAR 派生 dataset 的内部 schema。"""

    name: str
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]


@dataclass(frozen=True)
class RawFileMetadata:
    """记录从原始 CSMAR export 文件名解析出的 metadata。"""

    dataset: str
    market_scope: str
    start_date: str
    end_date: str
    export_date: str
    extension: str
    part: str | None = None


@dataclass(frozen=True)
class RawTableContract:
    """定义内部标准化前的原始 CSMAR export contract。"""

    dataset: str
    table_code: str
    table_name_zh: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class RawExportInspection:
    """记录原始 CSMAR export 文件的 header-level 检查结果。"""

    path: Path
    dataset: str
    table_code: str
    columns: tuple[str, ...]
    missing_required_columns: list[str]
    has_metadata: bool


DATASET_CONTRACTS: dict[str, DatasetContract] = {
    "stock_master": DatasetContract(
        name="stock_master",
        required_columns=(
            "stock_code",
            "stock_name",
            "exchange",
            "board",
            "share_class",
            "listing_date",
            "delisting_date",
            "company_status",
        ),
        primary_key=("stock_code",),
    ),
    "daily_market": DatasetContract(
        name="daily_market",
        required_columns=(
            "stock_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "amount",
            "turnover",
            "return_1d",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "adjustment": DatasetContract(
        name="adjustment",
        required_columns=("stock_code", "trade_date", "adj_factor"),
        primary_key=("stock_code", "trade_date"),
    ),
    "market_cap": DatasetContract(
        name="market_cap",
        required_columns=(
            "stock_code",
            "trade_date",
            "total_market_cap",
            "float_market_cap",
            "total_shares",
            "float_shares",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "trading_status": DatasetContract(
        name="trading_status",
        required_columns=(
            "stock_code",
            "trade_date",
            "is_trading",
            "is_suspended",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "st_status": DatasetContract(
        name="st_status",
        required_columns=("stock_code", "st_type", "start_date", "end_date"),
        primary_key=("stock_code", "start_date"),
    ),
    "industry": DatasetContract(
        name="industry",
        required_columns=(
            "stock_code",
            "industry_system",
            "industry_level",
            "industry_code",
            "industry_name",
            "start_date",
            "end_date",
        ),
        primary_key=("stock_code", "industry_system", "industry_level", "start_date"),
    ),
    "index_market": DatasetContract(
        name="index_market",
        required_columns=("index_code", "trade_date", "open", "close", "return_1d"),
        primary_key=("index_code", "trade_date"),
    ),
    "index_info": DatasetContract(
        name="index_info",
        required_columns=("index_code",),
        primary_key=("index_code",),
    ),
}

RAW_TABLE_CONTRACTS: dict[str, RawTableContract] = {
    "stock_master": RawTableContract(
        dataset="stock_master",
        table_code="TRDNEW_Co",
        table_name_zh="公司文件",
        required_columns=(
            "Stkcd",
            "Stknme",
            "Listdt",
            "DelistedDate",
            "Markettype",
            "Nnindcd",
            "Nnindnme",
            "Statco",
            "Statdt",
        ),
    ),
    "daily_market": RawTableContract(
        dataset="daily_market",
        table_code="TRDNEW_Dalyr",
        table_name_zh="日个股回报率文件",
        required_columns=(
            "Stkcd",
            "Trddt",
            "Markettype",
            "Trdsta",
            "Filling",
            "Opnprc",
            "Hiprc",
            "Loprc",
            "Clsprc",
            "Dnshrtrd",
            "Dnvaltrd",
            "Dsmvosd",
            "Dsmvtll",
            "Dretwd",
            "Dretnd",
            "Adjprcwd",
            "Adjprcnd",
            "PreClosePrice",
            "ChangeRatio",
            "LimitDown",
            "LimitUp",
            "LimitStatus",
        ),
    ),
    "market_cap": RawTableContract(
        dataset="market_cap",
        table_code="TRDNEW_Capchg",
        table_name_zh="股本变动文件",
        required_columns=("Stkcd", "Shrchgdt", "Nshrttl", "Nshra", "Nshrb", "Nshrh"),
    ),
    "adjustment": RawTableContract(
        dataset="adjustment",
        table_code="TRD_AdjustFactor",
        table_name_zh="股票价格复权因子表(日)",
        required_columns=(
            "Symbol",
            "TradingDate",
            "FwardFactor",
            "BwardFactor",
            "CumulateFwardFactor",
            "CumulateBwardFactor",
        ),
    ),
    "index_market": RawTableContract(
        dataset="index_market",
        table_code="IDX_Idxtrd",
        table_name_zh="国内指数日行情文件",
        required_columns=(
            "Indexcd",
            "Idxtrd01",
            "Idxtrd02",
            "Idxtrd03",
            "Idxtrd04",
            "Idxtrd05",
            "Idxtrd06",
            "Idxtrd07",
            "Idxtrd08",
            "Idxtrd09",
        ),
    ),
    "index_info": RawTableContract(
        dataset="index_info",
        table_code="IDX_Idxinfo",
        table_name_zh="指数基本信息文件",
        required_columns=("Indexcd",),
    ),
}


FILENAME_RE = re.compile(
    r"^csmar__(?P<dataset>[a-z0-9_]+)__"
    r"(?P<market_scope>[a-z0-9_]+)__"
    r"(?:(?P<start>\d{8})_(?P<end>\d{8})|(?P<full>full))__"
    r"(?:(?:part-(?P<part>\d{2})__)?)"
    r"export-(?P<export>\d{8})\.(?P<extension>csv|xlsx|txt)$"
)

NATIVE_TABLE_DATASETS: dict[str, str] = {
    "TRDNEW_Co": "stock_master",
    "TRDNEW_Dalyr": "daily_market",
    "TRDNEW_Capchg": "market_cap",
    "TRD_AdjustFactor": "adjustment",
    "IDX_Idxinfo": "index_info",
    "IDX_Idxtrd": "index_market",
}


def parse_raw_filename(path: str | Path) -> RawFileMetadata:
    """解析推荐的原始 CSMAR 文件名格式。"""

    name = Path(path).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid CSMAR raw filename: {name}")

    dataset = match.group("dataset")
    if dataset not in DATASET_CONTRACTS:
        raise ValueError(f"Unknown CSMAR dataset in filename: {dataset}")

    return RawFileMetadata(
        dataset=dataset,
        market_scope=match.group("market_scope"),
        start_date="full" if match.group("full") else match.group("start"),
        end_date="full" if match.group("full") else match.group("end"),
        export_date=match.group("export"),
        extension=match.group("extension"),
        part=match.group("part"),
    )


def infer_dataset_from_native_export(path: str | Path) -> str:
    """从 CSMAR 原生表文件名或父目录推断 dataset。"""

    export_path = Path(path)
    stem = re.sub(r"\d+$", "", export_path.stem)
    if stem in NATIVE_TABLE_DATASETS:
        return NATIVE_TABLE_DATASETS[stem]

    for part in reversed(export_path.parts):
        if part in DATASET_CONTRACTS:
            return part

    raise ValueError(f"Cannot infer CSMAR dataset from native filename: {export_path}")


def missing_required_columns(frame: pd.DataFrame, dataset: str) -> list[str]:
    """返回 DataFrame 中缺失的 required columns。"""

    contract = DATASET_CONTRACTS[dataset]
    return [
        column for column in contract.required_columns if column not in frame.columns
    ]


def read_export_header(path: str | Path) -> tuple[str, ...]:
    """只读取 CSV、TXT 或 XLSX CSMAR export 的 header row。"""

    export_path = Path(path)
    suffix = export_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(export_path, nrows=0)
    elif suffix == ".txt":
        frame = pd.read_csv(export_path, nrows=0, sep="\t")
    elif suffix == ".xlsx":
        frame = pd.read_excel(export_path, nrows=0)
    else:
        raise ValueError(f"Unsupported CSMAR export extension: {suffix}")
    return tuple(str(column) for column in frame.columns)


def default_metadata_path(path: str | Path) -> Path:
    """返回原始 export 推荐的 sidecar metadata path。"""

    export_path = Path(path)
    return export_path.with_suffix(".metadata.json")



def inspect_raw_export_file(path: str | Path) -> RawExportInspection:
    """在不读取完整数据的情况下检查文件名、header 和 metadata sidecar。"""

    export_path = Path(path)
    try:
        metadata = parse_raw_filename(export_path)
        dataset = metadata.dataset
    except ValueError:
        dataset = infer_dataset_from_native_export(export_path)
    columns = read_export_header(export_path)
    contract = RAW_TABLE_CONTRACTS[dataset]
    from quant_research_platform.sources.csmar.field_mappings import (
        missing_raw_column_names,
    )

    missing = missing_raw_column_names(columns, dataset)
    return RawExportInspection(
        path=export_path,
        dataset=dataset,
        table_code=contract.table_code,
        columns=columns,
        missing_required_columns=missing,
        has_metadata=default_metadata_path(export_path).exists(),
    )


def assert_required_columns(frame: pd.DataFrame, dataset: str) -> None:
    """当 DataFrame 不满足 dataset required columns 时抛出异常。"""

    missing = missing_required_columns(frame, dataset)
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def duplicate_primary_keys(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """返回 dataset 中 primary key 重复的 rows。"""

    contract = DATASET_CONTRACTS[dataset]
    assert_required_columns(frame, dataset)
    return frame[frame.duplicated(list(contract.primary_key), keep=False)]


def ohlc_violations(frame: pd.DataFrame) -> pd.DataFrame:
    """返回日行情中 OHLC 关系不可能成立的 rows。"""

    assert_required_columns(frame, "daily_market")
    invalid = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    return frame[invalid]


def negative_liquidity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """返回日行情中 volume、amount 或 turnover 为负的 rows。"""

    assert_required_columns(frame, "daily_market")
    invalid = (frame[["volume", "amount", "turnover"]] < 0).any(axis=1)
    return frame[invalid]
