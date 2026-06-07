"""Fail-if-exists atomic write helpers."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import pandas as pd

from quant_research_platform.storage.errors import OutputAlreadyExistsError

T = TypeVar("T")


def reject_if_exists(path: Path) -> None:
    if Path(path).exists():
        raise OutputAlreadyExistsError()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_sibling(final_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp_path.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        reject_if_exists(final_path)
        temp_path.rename(final_path)
    except Exception:
        _remove_path(temp_path)
        raise
    return final_path


def atomic_write_json(path: Path, payload: object) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_sibling(final_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(temp_path, **kwargs)
        reject_if_exists(final_path)
        temp_path.rename(final_path)
    except Exception:
        _remove_path(temp_path)
        raise
    return final_path


def atomic_write_parquet(frame: pd.DataFrame, path: Path, **kwargs: object) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_sibling(final_path)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(temp_path, **kwargs)
        reject_if_exists(final_path)
        temp_path.rename(final_path)
    except Exception:
        _remove_path(temp_path)
        raise
    return final_path


def atomic_write_directory(root: Path, writer: Callable[[Path], T]) -> T:
    final_root = Path(root)
    reject_if_exists(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = _temp_sibling(final_root)
    try:
        temp_root.mkdir()
        result = writer(temp_root)
        reject_if_exists(final_root)
        temp_root.rename(final_root)
    except Exception:
        _remove_path(temp_root)
        raise
    return result


def _temp_sibling(path: Path) -> Path:
    return path.with_name(f".tmp-{uuid4().hex[:12]}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


__all__ = [
    "atomic_write_csv",
    "atomic_write_directory",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
    "reject_if_exists",
]
