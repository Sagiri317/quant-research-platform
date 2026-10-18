"""Trust-boundary reader for the designated A-share fee reference CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import (
    _DigestingReader,
    open_filesystem_file,
    require_unaliased_path,
)

A_SHARE_FEE_CSV_NAME = "chatgpt_compiled_a_share_fee_schedule_2014_2025.csv"
A_SHARE_FEE_SIDECAR_NAME = (
    "chatgpt_compiled_a_share_fee_schedule_2014_2025.metadata.json"
)
FEE_CSV_COLUMNS = (
    "规则类别",
    "费用代码",
    "费用中文名",
    "生效日",
    "截止日",
    "市场",
    "方向",
    "计费基数",
    "十进制费率",
    "费率(‰)",
    "费率(bps)",
    "进入正式账本",
    "包含于",
    "权威机构",
    "官方/主要文件",
    "主要来源URL",
    "交叉验证URL",
    "证据等级",
    "说明",
)
_FEE_CODES = frozenset(
    {
        "BROKER_COMMISSION",
        "STAMP_DUTY",
        "TRANSFER_FEE",
        "EXCHANGE_HANDLING_FEE",
        "REGULATORY_FEE",
    }
)
_INCLUDED_CODES = frozenset({"BROKER_COMMISSION", "STAMP_DUTY", "TRANSFER_FEE"})
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class FeeSourceRow:
    """Canonical immutable row produced by the external reference reader."""

    fee_code: str
    effective_date: date
    end_date: date
    market: str
    side: str
    basis: str
    rate: float
    included_in_ledger: bool
    included_in: str | None = None


@dataclass(frozen=True, slots=True)
class AShareFeeSource:
    rows: tuple[FeeSourceRow, ...]
    csv_sha256: str
    source_identity: str


def load_a_share_fee_source(
    *,
    reference_root: Path,
    fee_csv: Path | None = None,
    fee_sidecar: Path | None = None,
) -> AShareFeeSource:
    """Read an explicit CSV/sidecar pair, or the existing reference-root default."""

    if (fee_csv is None) != (fee_sidecar is None):
        raise ValueError("a_share_fee_source_pair_required")

    root = Path(reference_root) / "a_share_fee_schedule"
    csv_path = root / A_SHARE_FEE_CSV_NAME if fee_csv is None else Path(fee_csv)
    sidecar_path = (
        root / A_SHARE_FEE_SIDECAR_NAME if fee_sidecar is None else Path(fee_sidecar)
    )
    require_unaliased_path(csv_path, error="a_share_fee_source_unavailable")
    require_unaliased_path(sidecar_path, error="a_share_fee_source_unavailable")
    metadata, sidecar_sha256 = _read_sidecar(sidecar_path)
    rows, data_row_count, csv_sha256, csv_bytes = _read_rows(csv_path)
    _validate_sidecar(metadata, csv_sha256, csv_bytes=csv_bytes)
    if metadata.get("row_count") != data_row_count:
        raise ValueError("a_share_fee_source_row_count_mismatch")
    digest = hashlib.sha256()
    for name, member_hash in sorted(
        (
            (csv_path.name, csv_sha256),
            (sidecar_path.name, sidecar_sha256),
        )
    ):
        for value in (name, member_hash):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    source_identity = digest.hexdigest()
    return AShareFeeSource(
        rows=rows,
        csv_sha256=csv_sha256,
        source_identity=source_identity,
    )


def _read_sidecar(path: Path) -> tuple[dict[str, object], str]:
    try:
        with closing_iterator(open_filesystem_file(path)) as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("a_share_fee_sidecar_invalid") from exc
    if type(value) is not dict:
        raise ValueError("a_share_fee_sidecar_invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_sidecar(
    metadata: dict[str, object],
    csv_sha256: str,
    *,
    csv_bytes: int,
) -> None:
    if (
        metadata.get("file_sha256") != csv_sha256
        or type(metadata.get("file_bytes")) is not int
        or metadata.get("file_bytes") != csv_bytes
        or type(metadata.get("row_count")) is not int
        or metadata.get("row_count") < 0
        or metadata.get("fields_raw") != list(FEE_CSV_COLUMNS)
    ):
        raise ValueError("a_share_fee_source_metadata_invalid")


def _read_rows(path: Path) -> tuple[tuple[FeeSourceRow, ...], int, str, int]:
    rows: list[FeeSourceRow] = []
    seen: set[tuple[object, ...]] = set()
    data_row_count = 0
    try:
        with closing_iterator(open_filesystem_file(path)) as binary:
            hashed = _DigestingReader(binary)
            with closing_iterator(
                io.TextIOWrapper(hashed, encoding="utf-8-sig", newline="")
            ) as text:
                csv.field_size_limit(sys.maxsize)
                reader = csv.DictReader(text, strict=True)
                if tuple(reader.fieldnames or ()) != FEE_CSV_COLUMNS:
                    raise ValueError("a_share_fee_source_schema_invalid")
                for row in reader:
                    data_row_count += 1
                    if set(row) != set(FEE_CSV_COLUMNS) or None in row:
                        raise ValueError("a_share_fee_source_schema_invalid")
                    values = {}
                    for key in FEE_CSV_COLUMNS:
                        value = row[key]
                        if not isinstance(value, str):
                            raise ValueError("a_share_fee_source_schema_invalid")
                        values[key] = value
                    fee_code = values["费用代码"]
                    if fee_code not in _FEE_CODES:
                        raise ValueError("a_share_fee_source_fee_code_invalid")
                    market = _canonical_market(values["市场"])
                    basis = _canonical_basis(values["计费基数"])
                    sides = _canonical_sides(values["方向"])
                    start = _parse_date(values["生效日"])
                    end = _parse_date(values["截止日"])
                    rate = _parse_rate(values["十进制费率"])
                    per_mille = _parse_rate(values["费率(‰)"])
                    basis_points = _parse_rate(values["费率(bps)"])
                    if not math.isclose(
                        per_mille, rate * 1_000.0, abs_tol=1e-12, rel_tol=0.0
                    ):
                        raise ValueError("a_share_fee_source_rate_unit_invalid")
                    if not math.isclose(
                        basis_points,
                        rate * 10_000.0,
                        abs_tol=1e-12,
                        rel_tol=0.0,
                    ):
                        raise ValueError("a_share_fee_source_rate_unit_invalid")
                    included_text = values["进入正式账本"]
                    if included_text not in {"是", "否"}:
                        raise ValueError("a_share_fee_source_inclusion_invalid")
                    included = included_text == "是"
                    included_in = values["包含于"] or None
                    if included and included_in is not None:
                        raise ValueError("a_share_fee_source_inclusion_invalid")
                    if not included and included_in not in _INCLUDED_CODES:
                        raise ValueError("a_share_fee_source_inclusion_invalid")
                    for side in sides:
                        key = (fee_code, start, end, market, side, basis)
                        if key in seen:
                            raise ValueError("a_share_fee_source_duplicate")
                        seen.add(key)
                        rows.append(
                            FeeSourceRow(
                                fee_code=fee_code,
                                effective_date=start,
                                end_date=end,
                                market=market,
                                side=side,
                                basis=basis,
                                rate=rate,
                                included_in_ledger=included,
                                included_in=included_in,
                            )
                        )
                    row = values = value = None
    except csv.Error as exc:
        raise ValueError("a_share_fee_source_schema_invalid") from exc
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc
    if not rows:
        raise ValueError("a_share_fee_source_empty")
    return tuple(rows), data_row_count, hashed.hexdigest(), hashed.bytes_read


def _parse_date(value: str) -> date:
    if _DATE_RE.fullmatch(value) is None:
        raise ValueError("a_share_fee_source_date_invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("a_share_fee_source_date_invalid") from exc


def _parse_rate(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("a_share_fee_source_rate_invalid") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("a_share_fee_source_rate_invalid")
    return result


def _canonical_market(value: str) -> str:
    values = {"沪深": "a_share", "沪市": "shanghai", "深市": "shenzhen"}
    try:
        return values[value]
    except KeyError as exc:
        raise ValueError("a_share_fee_source_market_invalid") from exc


def _canonical_sides(value: str) -> tuple[str, ...]:
    if value == "买入+卖出":
        return ("buy", "sell")
    if value == "买入":
        return ("buy",)
    if value == "卖出":
        return ("sell",)
    raise ValueError("a_share_fee_source_side_invalid")


def _canonical_basis(value: str) -> str:
    values = {"成交金额": "amount", "成交面值": "face_value"}
    try:
        return values[value]
    except KeyError as exc:
        raise ValueError("a_share_fee_source_basis_invalid") from exc
