"""Local Binance raw JSON/JSONL export readers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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


def iter_raw_rows(path: Path) -> Iterator[dict[str, Any]]:
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
                yield _loads_object(line, raw_path, line_number=line_number)
        if not yielded:
            raise ValueError("empty_binance_raw_file")
        return
    if raw_path.suffix == ".json":
        payload = _loads_json(raw_path.read_text(encoding="utf-8"), raw_path)
        rows = payload if isinstance(payload, list) else [payload]
        if not rows:
            raise ValueError("empty_binance_raw_file")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("malformed_binance_raw_row")
            yield row
        return
    raise ValueError("unsupported_binance_raw_file_extension")


def _raw_kind_files(root: Path, raw_kind: str) -> tuple[Path, ...]:
    if raw_kind == "klines":
        base = root / "spot" / "klines"
        patterns = ("*" + "/1m/*.jsonl", "*" + "/1m/*.json")
    elif raw_kind == "agg_trades":
        base = root / "spot" / "agg_trades"
        patterns = ("*" + "/*.jsonl", "*" + "/*.json")
    elif raw_kind == "depth":
        base = root / "spot" / "depth"
        patterns = ("*" + "/*.jsonl", "*" + "/*.json")
    else:
        raise ValueError(f"unsupported_binance_raw_kind: {raw_kind}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return tuple(path for path in files if path.is_file())


def _loads_object(text: str, path: Path, *, line_number: int) -> dict[str, Any]:
    payload = _loads_json(text, path, line_number=line_number)
    if not isinstance(payload, dict):
        raise ValueError("malformed_binance_raw_row")
    return payload


def _loads_json(text: str, path: Path, *, line_number: int | None = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        location = f":{line_number}" if line_number is not None else ""
        raise ValueError(f"malformed_binance_raw_row: {path.name}{location}") from exc


__all__ = ["DATA_ID_BY_RAW_KIND", "dataset_files_by_id", "iter_raw_rows"]
