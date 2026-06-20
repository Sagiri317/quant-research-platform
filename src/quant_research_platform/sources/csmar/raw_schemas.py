"""Internal provider raw file, table, and export-inspection contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
    "market_bars_1d": RawTableContract(
        dataset="market_bars_1d",
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
        required_columns=("Indexcd", "Idxinfo11", "Idxinfo02"),
    ),
    "trading_calendar": RawTableContract(
        dataset="trading_calendar",
        table_code="TRDNEW_Cale",
        table_name_zh="交易日历",
        required_columns=("Markettype", "Clddt", "Daywk", "State"),
    ),
}


FILENAME_RE = re.compile(
    r"^csmar__(?P<dataset>[a-z0-9_]+)__"
    r"(?P<market_scope>[a-z0-9_]+)__"
    r"(?:(?P<start>\d{8})_(?P<end>\d{8})|(?P<full>full))__"
    r"(?:(?:part-(?P<part>\d{2})__)?)"
    r"export-(?P<export>\d{8})\.(?P<extension>csv)$"
)

NATIVE_TABLE_DATASETS: dict[str, str] = {
    "TRDNEW_Co": "stock_master",
    "TRDNEW_Dalyr": "market_bars_1d",
    "TRDNEW_Capchg": "market_cap",
    "TRD_AdjustFactor": "adjustment",
    "IDX_Idxinfo": "index_info",
    "IDX_Idxtrd": "index_market",
    "TRDNEW_Cale": "trading_calendar",
}


def parse_raw_filename(path: str | Path) -> RawFileMetadata:
    """解析推荐的原始 CSMAR 文件名格式。"""

    name = Path(path).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid CSMAR raw filename: {name}")

    dataset = match.group("dataset")
    if dataset not in RAW_TABLE_CONTRACTS:
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
        if part in RAW_TABLE_CONTRACTS:
            return part

    raise ValueError(f"Cannot infer CSMAR dataset from native filename: {export_path}")


def _infer_dataset_from_header(columns: tuple[str, ...]) -> str:
    from quant_research_platform.sources.csmar.field_mappings import (
        missing_raw_column_names,
    )

    matches = [
        dataset
        for dataset in RAW_TABLE_CONTRACTS
        if not missing_raw_column_names(columns, dataset)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer CSMAR dataset from raw header: {matches}")
    return matches[0]


def read_export_header(path: str | Path) -> tuple[str, ...]:
    """Read the header row from a supported CSV raw export."""

    export_path = Path(path)
    suffix = export_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(export_path, nrows=0)
    else:
        raise ValueError(f"Unsupported CSMAR export extension: {suffix}")
    return tuple(str(column) for column in frame.columns)


def expected_metadata_sidecar_path(path: str | Path) -> Path:
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
        if export_path.suffix.lower() != ".csv":
            dataset = infer_dataset_from_native_export(export_path)
            columns = read_export_header(export_path)
        else:
            columns = read_export_header(export_path)
            try:
                dataset = infer_dataset_from_native_export(export_path)
            except ValueError:
                dataset = _infer_dataset_from_header(columns)
    else:
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
        has_metadata=expected_metadata_sidecar_path(export_path).exists(),
    )
