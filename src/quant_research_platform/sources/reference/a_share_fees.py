"""Trust-boundary reader for the designated A-share fee reference CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    open_filesystem_file,
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
_MAX_SIDECAR_BYTES = 128 * 1024
_MAX_ROWS = 128
_MAX_FIELD_CHARS = 16 * 1024
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
    reference_root: Path | None = None,
    csv_path: Path | None = None,
    sidecar_path: Path | None = None,
) -> AShareFeeSource:
    """Read the exact reference member or explicit test paths."""

    csv_path, sidecar_path = _resolve_paths(
        reference_root=reference_root,
        csv_path=csv_path,
        sidecar_path=sidecar_path,
    )
    csv_sha256, csv_identity = _sha256_file(csv_path)
    sidecar_identity = _file_identity(sidecar_path)
    metadata = _read_sidecar(sidecar_path, sidecar_identity)
    csv_bytes = csv_identity.size
    _validate_sidecar(metadata, csv_sha256, csv_bytes=csv_bytes)
    rows, data_row_count = _read_rows(csv_path, csv_identity)
    if metadata.get("row_count") != data_row_count:
        raise ValueError("a_share_fee_source_row_count_mismatch")
    source_identity = content_hash_for_files(
        (
            (A_SHARE_FEE_CSV_NAME, csv_path, csv_identity),
            (A_SHARE_FEE_SIDECAR_NAME, sidecar_path, sidecar_identity),
        ),
        error="a_share_fee_source_identity_invalid",
    )
    return AShareFeeSource(
        rows=rows,
        csv_sha256=csv_sha256,
        source_identity=source_identity,
    )


def _resolve_paths(
    *,
    reference_root: Path | None,
    csv_path: Path | None,
    sidecar_path: Path | None,
) -> tuple[Path, Path]:
    if reference_root is not None and csv_path is not None:
        raise ValueError("a_share_fee_source_path_ambiguous")
    if csv_path is None:
        if reference_root is None:
            raise ValueError("a_share_fee_source_path_required")
        root = Path(reference_root)
        csv_path = root / "a_share_fee_schedule" / A_SHARE_FEE_CSV_NAME
    else:
        csv_path = Path(csv_path)
    if sidecar_path is None:
        sidecar_path = csv_path.with_suffix(".metadata.json")
    else:
        sidecar_path = Path(sidecar_path)
    return csv_path, sidecar_path


def _file_identity(path: Path):
    try:
        return dataset_file_identity(path)
    except ValueError as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc


def _sha256_file(path: Path):
    digest = hashlib.sha256()
    expected = _file_identity(path)
    try:
        with closing_iterator(open_filesystem_file(path)) as handle:
            if dataset_file_identity_from_stat(os.fstat(handle.fileno())) != expected:
                raise ValueError("a_share_fee_source_identity_changed")
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            final = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc
    if final != expected or _file_identity(path) != expected:
        raise ValueError("a_share_fee_source_identity_changed")
    return digest.hexdigest(), expected


def _read_sidecar(path: Path, expected) -> dict[str, object]:
    try:
        if expected.size > _MAX_SIDECAR_BYTES:
            raise ValueError("a_share_fee_sidecar_too_large")
        with closing_iterator(open_filesystem_file(path)) as handle:
            if dataset_file_identity_from_stat(os.fstat(handle.fileno())) != expected:
                raise ValueError("a_share_fee_source_identity_changed")
            raw = handle.read(_MAX_SIDECAR_BYTES + 1)
            final = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc
    if len(raw) > _MAX_SIDECAR_BYTES:
        raise ValueError("a_share_fee_sidecar_too_large")
    if final != expected or _file_identity(path) != expected:
        raise ValueError("a_share_fee_source_identity_changed")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("a_share_fee_sidecar_invalid") from exc
    if type(value) is not dict:
        raise ValueError("a_share_fee_sidecar_invalid")
    return value


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


def _read_rows(path: Path, expected) -> tuple[tuple[FeeSourceRow, ...], int]:
    rows: list[FeeSourceRow] = []
    seen: set[tuple[object, ...]] = set()
    data_row_count = 0
    try:
        with closing_iterator(open_filesystem_file(path)) as binary:
            if dataset_file_identity_from_stat(os.fstat(binary.fileno())) != expected:
                raise ValueError("a_share_fee_source_identity_changed")
            handle = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FEE_CSV_COLUMNS:
                raise ValueError("a_share_fee_source_schema_invalid")
            for row_number, row in enumerate(reader, start=2):
                data_row_count += 1
                if row_number > _MAX_ROWS + 1:
                    raise ValueError("a_share_fee_source_row_count_invalid")
                if set(row) != set(FEE_CSV_COLUMNS) or None in row:
                    raise ValueError("a_share_fee_source_schema_invalid")
                values = {}
                for key in FEE_CSV_COLUMNS:
                    value = row[key]
                    if not isinstance(value, str) or len(value) > _MAX_FIELD_CHARS:
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
            handle.detach()
            final = dataset_file_identity_from_stat(os.fstat(binary.fileno()))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("a_share_fee_source_unavailable") from exc
    if final != expected or _file_identity(path) != expected:
        raise ValueError("a_share_fee_source_identity_changed")
    if not rows:
        raise ValueError("a_share_fee_source_empty")
    return tuple(rows), data_row_count


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
