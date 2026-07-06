"""Local Binance raw JSONL envelope readers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

DATA_ID_BY_RAW_KIND = {
    "klines": "market_bars_1m",
    "agg_trades": "aggregate_trades_tick",
    "depth": "depth_levels_tick",
}


def dataset_files_by_id(raw_root: Path) -> dict[str, tuple[Path, ...]]:
    root = Path(raw_root)
    return {
        data_id: tuple(sorted(_raw_kind_files(root, raw_kind)))
        for raw_kind, data_id in DATA_ID_BY_RAW_KIND.items()
    }


def iter_raw_envelopes(path: Path) -> Iterator[dict[str, Any]]:
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if raw_path.stat().st_size == 0:
        raise ValueError("empty_binance_raw_file")
    if raw_path.suffix == ".jsonl":
        yielded = False
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yielded = True
                yield _validate_envelope(
                    _loads_object(line, raw_path, line_number=line_number)
                )
        if not yielded:
            raise ValueError("empty_binance_raw_file")
        return
    raise ValueError("unsupported_binance_raw_file_extension")


def _raw_kind_files(root: Path, raw_kind: str) -> tuple[Path, ...]:
    if raw_kind == "klines":
        base = root / "spot" / "klines"
        patterns = ("*" + "/1m/*.jsonl",)
    elif raw_kind == "agg_trades":
        base = root / "spot" / "agg_trades"
        patterns = ("*" + "/*.jsonl",)
    elif raw_kind == "depth":
        base = root / "spot" / "depth"
        patterns = ("*" + "/*.jsonl",)
    else:
        raise ValueError(f"unsupported_binance_raw_kind: {raw_kind}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return tuple(path for path in files if path.is_file())


def _loads_object(text: str, path: Path, *, line_number: int) -> Any:
    return _loads_json(text, path, line_number=line_number)


def _loads_json(text: str, path: Path, *, line_number: int | None = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        location = f":{line_number}" if line_number is not None else ""
        raise ValueError(f"malformed_binance_raw_row: {path.name}{location}") from exc


def _validate_envelope(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError("binance_raw_envelope_required")
    required = {
        "source",
        "market_type",
        "data_kind",
        "symbol",
        "captured_at",
        "payload",
    }
    if not required.issubset(envelope):
        raise ValueError("binance_raw_envelope_required")
    if envelope["source"] != "binance":
        raise ValueError("invalid_binance_raw_envelope_source")
    if envelope["market_type"] != "spot":
        raise ValueError("unsupported_binance_market_type")
    data_kind = envelope["data_kind"]
    if data_kind not in DATA_ID_BY_RAW_KIND:
        raise ValueError("unsupported_binance_data_kind")
    if not isinstance(envelope["symbol"], str) or not envelope["symbol"]:
        raise ValueError("invalid_binance_raw_envelope_symbol")
    _parse_captured_at(envelope["captured_at"])
    payload = envelope["payload"]
    if data_kind == "klines":
        if envelope.get("interval") != "1m":
            raise ValueError("unsupported_binance_kline_interval")
        if not isinstance(payload, list) or len(payload) < 11:
            raise ValueError("invalid_binance_kline_payload")
    elif data_kind == "agg_trades":
        if not isinstance(payload, dict):
            raise ValueError("invalid_binance_agg_trade_payload")
    elif data_kind == "depth" and not isinstance(payload, dict):
        raise ValueError("invalid_binance_depth_payload")
    return envelope


def _parse_captured_at(value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not _has_utc_offset(value):
        raise ValueError("invalid_binance_raw_envelope_captured_at")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("invalid_binance_raw_envelope_captured_at")
    return timestamp


def _has_utc_offset(value: str) -> bool:
    if value.endswith("Z"):
        return True
    if len(value) < 6:
        return False
    offset = value[-6:]
    return (
        offset[0] in {"+", "-"}
        and offset[1:3].isdigit()
        and offset[3] == ":"
        and offset[4:6].isdigit()
    )


__all__ = ["DATA_ID_BY_RAW_KIND", "dataset_files_by_id", "iter_raw_envelopes"]
