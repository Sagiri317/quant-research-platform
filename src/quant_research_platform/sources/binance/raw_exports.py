"""Local Binance raw JSONL envelope readers."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.json_codec import (
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.runtime.resources import OwnedValue
from quant_research_platform.sources.binance.validation import (
    depth_side_levels,
    first_present,
    strict_boolean,
    strict_data_kind,
    strict_iso_timestamp,
    strict_market_type,
    strict_millisecond_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    strict_symbol,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _file_identity_bytes,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    require_unaliased_path,
)

DATA_ID_BY_RAW_KIND = {
    "klines": "market_bars_1m",
    "agg_trades": "aggregate_trades_tick",
    "depth": "depth_levels_tick",
}
_SCAN_BUFFER_BYTES = 64 * 1024
_RAW_IDENTITY_ERROR = "binance_raw_export_identity_changed"
_RawInputFile = tuple[Path, DatasetFileIdentity]


@contextmanager
def _raw_path_session(root: Path) -> Iterator[Callable[[], None]]:
    raw_root = Path(root)

    def read_root_identity() -> os.stat_result | None:
        require_unaliased_path(raw_root, error=_RAW_IDENTITY_ERROR)
        try:
            metadata = raw_root.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ValueError(_RAW_IDENTITY_ERROR) from error
        _reject_raw_reparse(metadata)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(_RAW_IDENTITY_ERROR)
        return metadata

    root_identity = read_root_identity()

    def verify_root() -> None:
        current = read_root_identity()
        if root_identity is None:
            if current is not None:
                raise ValueError(_RAW_IDENTITY_ERROR)
            return
        if current is None:
            raise ValueError(_RAW_IDENTITY_ERROR)
        try:
            unchanged = os.path.samestat(root_identity, current)
        except OSError as error:
            raise ValueError(_RAW_IDENTITY_ERROR) from error
        if not unchanged:
            raise ValueError(_RAW_IDENTITY_ERROR)

    try:
        yield verify_root
    finally:
        primary = sys.exception()
        try:
            verify_root()
        except BaseException as verify_error:
            if primary is not None:
                primary.add_note(
                    "Binance raw root verification also failed: "
                    f"{type(verify_error).__name__}"
                )
            else:
                raise


def _iter_raw_envelope_batches(
    path: Path,
    *,
    batch_size: int,
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterator[object]:
    raw_path = Path(path)
    expected_identity = _capture_raw_file_identity(
        raw_path,
        expected_identity=expected_identity,
    )
    if expected_identity.size == 0:
        raise ValueError("empty_binance_raw_file")
    if raw_path.suffix != ".jsonl":
        raise ValueError("unsupported_binance_raw_file_extension")
    producer_bound = 256 * 1024
    line_number = 0
    yielded = False

    try:
        handle = raw_path.open("rb")
    except OSError as open_error:
        try:
            current_identity = _capture_raw_file_identity(
                raw_path,
                missing_is_file_not_found=False,
            )
        except ValueError:
            raise ValueError(_RAW_IDENTITY_ERROR) from open_error
        if current_identity != expected_identity:
            raise ValueError(_RAW_IDENTITY_ERROR) from open_error
        raise
    try:
        _verify_raw_file_handle(
            handle,
            expected_identity=expected_identity,
        )
        while True:
            lease = yield producer_bound
            _verify_raw_file_handle(
                handle,
                expected_identity=expected_identity,
            )
            scan_bytes = min(_SCAN_BUFFER_BYTES, max(1, producer_bound))
            if scan_bytes + 64 * 1024 > lease.reserved_bytes:
                raise RuntimeError("binance_record_envelope_bound_invalid")
            buffer = bytearray(scan_bytes)
            batch: list[dict[str, Any]] = []
            batch_working_bytes = 0
            end_of_file = False
            records_in_batch = 0
            while records_in_batch < batch_size:
                record_length = _next_record_length(handle, buffer)
                if record_length == 0:
                    end_of_file = True
                    break
                record_bound = json_record_working_set_bytes(record_length)
                required_bound = batch_working_bytes + record_bound + 64 * 1024
                if required_bound > lease.reserved_bytes:
                    if batch:
                        break
                    lease.grow(required_bound - lease.reserved_bytes)
                raw = handle.read(record_length)
                line_number += 1
                if raw.isspace():
                    del raw
                    continue
                records_in_batch += 1
                batch_working_bytes += record_bound
                yielded = True
                decoded: dict[str, Any] | None = None
                validated: dict[str, Any] | None = None
                try:
                    decoded = _loads_json(
                        raw,
                        raw_path,
                        line_number=line_number,
                    )
                    validated = _validate_envelope(decoded)
                    batch.append(validated)
                except BaseException as error:
                    clear_exception_frames(error)
                    raw = None
                    decoded = None
                    validated = None
                    raise
                raw = None
                decoded = None
                validated = None
            if batch:
                actual_bound = batch_working_bytes + 64 * 1024
                del buffer
                if actual_bound > lease.reserved_bytes:
                    raise RuntimeError("binance_record_envelope_bound_invalid")
                _verify_raw_file_handle(
                    handle,
                    expected_identity=expected_identity,
                )
                yield OwnedValue(tuple(batch), lease)
                del batch
            else:
                del buffer
            if end_of_file:
                break
        if not yielded:
            raise ValueError("empty_binance_raw_file")
    finally:
        primary = sys.exception()
        first_failure: BaseException | None = None
        try:
            _verify_raw_file_handle(
                handle,
                expected_identity=expected_identity,
            )
        except BaseException as verify_error:
            if primary is not None:
                primary.add_note(
                    "Binance raw file verification also failed: "
                    f"{type(verify_error).__name__}"
                )
            else:
                first_failure = verify_error
        try:
            handle.close()
        except BaseException as close_error:
            if primary is not None:
                primary.add_note(
                    f"Binance raw file close also failed: {type(close_error).__name__}"
                )
            elif first_failure is None:
                first_failure = close_error
            else:
                first_failure.add_note(
                    f"Binance raw file close also failed: {type(close_error).__name__}"
                )
        if first_failure is not None:
            raise first_failure


def _capture_raw_file_identity(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
    missing_is_file_not_found: bool = True,
) -> DatasetFileIdentity:
    raw_path = Path(path)
    try:
        raw_path.lstat()
    except FileNotFoundError as error:
        if missing_is_file_not_found and expected_identity is None:
            raise FileNotFoundError(raw_path) from error
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    except OSError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    try:
        identity = dataset_file_identity(raw_path)
    except ValueError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(_RAW_IDENTITY_ERROR)
    return identity


def _verify_raw_file_handle(
    handle: Any,
    *,
    expected_identity: DatasetFileIdentity,
) -> None:
    try:
        handle_identity = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except (OSError, ValueError) as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    if handle_identity != expected_identity:
        raise ValueError(_RAW_IDENTITY_ERROR)


def _next_record_length(handle: Any, buffer: bytearray) -> int:
    start = handle.tell()
    length = 0
    while True:
        count = handle.readinto(buffer)
        if count == 0:
            handle.seek(start)
            return length
        newline = buffer.find(b"\n", 0, count)
        if newline >= 0:
            length += newline + 1
            handle.seek(start)
            return length
        length += count


def _raw_kind_location(
    root: Path,
    raw_kind: str,
) -> tuple[Path, str | None]:
    if raw_kind == "klines":
        return root / "spot" / "klines", "1m"
    elif raw_kind == "agg_trades":
        return root / "spot" / "agg_trades", None
    elif raw_kind == "depth":
        return root / "spot" / "depth", None
    else:
        raise ValueError(f"unsupported_binance_raw_kind: {raw_kind}")


def _iter_raw_kind_files(
    root: Path,
    raw_kind: str,
) -> Iterator[Path]:
    base, interval = _raw_kind_location(root, raw_kind)
    for symbol_dir, symbol_metadata in _iter_raw_directory_entries(base):
        if not stat.S_ISDIR(symbol_metadata.st_mode):
            continue
        data_dir = symbol_dir if interval is None else symbol_dir / interval
        for path, metadata in _iter_raw_directory_entries(data_dir):
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if path.suffix != ".jsonl":
                continue
            yield path


def _iter_raw_directory_entries(
    path: Path,
) -> Iterator[tuple[Path, os.stat_result]]:
    require_unaliased_path(path, error=_RAW_IDENTITY_ERROR)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    _reject_raw_reparse(metadata)
    if not stat.S_ISDIR(metadata.st_mode):
        return
    try:
        children = path.iterdir()
    except OSError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    try:
        for child in children:
            require_unaliased_path(child, error=_RAW_IDENTITY_ERROR)
            try:
                metadata = child.lstat()
            except OSError as error:
                raise ValueError(_RAW_IDENTITY_ERROR) from error
            _reject_raw_reparse(metadata)
            yield child, metadata
    except OSError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error


def _index_raw_kind_files(
    root: Path,
    *,
    raw_kind: str,
    index: ExactPrimaryKeyIndex,
    namespace: str,
) -> int:
    """Index one requested raw kind without retaining its directory listing."""

    discovered = 0

    def payloads() -> Iterator[tuple[bytes, bytes]]:
        nonlocal discovered
        for path in _iter_raw_kind_files(Path(root), raw_kind):
            identity = _capture_raw_file_identity(
                path,
                missing_is_file_not_found=False,
            )
            relative = path.relative_to(root).as_posix().encode("utf-8")
            discovered += 1
            yield relative, relative + b"\0" + _file_identity_bytes(identity)

    inserted = index.add_identity_payloads(
        namespace=namespace,
        payloads=payloads(),
    )
    if inserted != discovered:
        raise ValueError(_RAW_IDENTITY_ERROR)
    return discovered


def _iter_indexed_raw_kind_files(
    root: Path,
    *,
    index: ExactPrimaryKeyIndex,
    namespace: str,
) -> Iterator[_RawInputFile]:
    for payload in index.identity_payload_values(namespace=namespace):
        relative, device, inode, size, modified_ns = payload.split(b"\0")
        yield (
            Path(root) / relative.decode("utf-8"),
            DatasetFileIdentity(
                device=int(device),
                inode=int(inode),
                size=int(size),
                modified_ns=int(modified_ns),
            ),
        )


def _verify_indexed_raw_kind_files(
    root: Path,
    *,
    raw_kind: str,
    index: ExactPrimaryKeyIndex,
    namespace: str,
    expected_count: int,
) -> None:
    current_count = 0
    for path in _iter_raw_kind_files(Path(root), raw_kind):
        identity = _capture_raw_file_identity(
            path,
            missing_is_file_not_found=False,
        )
        relative = path.relative_to(root).as_posix().encode("utf-8")
        expected = index.identity_payload_value(
            namespace=namespace,
            payload=relative,
        )
        actual = relative + b"\0" + _file_identity_bytes(identity)
        if expected != actual:
            raise ValueError(_RAW_IDENTITY_ERROR)
        current_count += 1
    if current_count != expected_count:
        raise ValueError(_RAW_IDENTITY_ERROR)


def _reject_raw_reparse(metadata: os.stat_result) -> None:
    reparse_flag = getattr(metadata, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if reparse_flag:
        raise ValueError(_RAW_IDENTITY_ERROR)


def _loads_json(raw: bytes, path: Path, *, line_number: int | None = None) -> Any:
    try:
        return parse_qrp_json_object(raw)
    except ValueError as exc:
        if str(exc) == "qrp_json_top_level_object_required":
            raise ValueError("binance_raw_envelope_required") from exc
        location = f":{line_number}" if line_number is not None else ""
        raise ValueError(f"malformed_binance_raw_row: {path.name}{location}") from exc


def _validate_envelope(
    envelope: Any,
) -> dict[str, Any]:
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
    strict_market_type(envelope["market_type"])
    data_kind = strict_data_kind(envelope["data_kind"])
    strict_symbol(envelope["symbol"], error="invalid_binance_raw_envelope_symbol")
    strict_iso_timestamp(
        envelope["captured_at"],
        error="invalid_binance_raw_envelope_captured_at",
    )
    payload = envelope["payload"]
    if data_kind == "klines":
        if envelope.get("interval") != "1m":
            raise ValueError("unsupported_binance_kline_interval")
        if not isinstance(payload, list) or len(payload) < 11:
            raise ValueError("invalid_binance_kline_payload")
        _validate_kline_payload(payload)
    elif data_kind == "agg_trades":
        if "interval" in envelope:
            raise ValueError("unexpected_binance_interval")
        if not isinstance(payload, dict):
            raise ValueError("invalid_binance_agg_trade_payload")
        _validate_agg_trade_payload(payload)
    elif data_kind == "depth":
        if "interval" in envelope:
            raise ValueError("unexpected_binance_interval")
        if not isinstance(payload, dict):
            raise ValueError("invalid_binance_depth_payload")
        bids = depth_side_levels(
            payload.get("bids", ()),
            side="bid",
            error="invalid_binance_depth_bid",
        )
        asks = depth_side_levels(
            payload.get("asks", ()),
            side="ask",
            error="invalid_binance_depth_ask",
        )
        if bids and asks and bids[0][0] >= asks[0][0]:
            raise ValueError("crossed_binance_depth_book")
    return envelope


def _validate_kline_payload(payload: list[object]) -> None:
    open_time = strict_millisecond_timestamp(
        payload[0],
        error="invalid_binance_kline_open_time",
    )
    close_time = strict_millisecond_timestamp(
        payload[6],
        error="invalid_binance_kline_close_time",
    )
    if open_time != open_time.floor("min"):
        raise ValueError("invalid_binance_kline_open_time_alignment")
    if close_time < open_time:
        raise ValueError("invalid_binance_kline_time_order")
    if close_time != open_time + pd.Timedelta(milliseconds=59_999):
        raise ValueError("invalid_binance_kline_close_time")
    open_price = strict_positive_float(payload[1], error="invalid_binance_kline_ohlc")
    high = strict_positive_float(payload[2], error="invalid_binance_kline_ohlc")
    low = strict_positive_float(payload[3], error="invalid_binance_kline_ohlc")
    close = strict_positive_float(payload[4], error="invalid_binance_kline_ohlc")
    if high < max(open_price, close) or low > min(open_price, close):
        raise ValueError("invalid_binance_kline_ohlc")
    volume = strict_non_negative_float(payload[5], error="invalid_binance_kline_volume")
    quote_volume = strict_non_negative_float(
        payload[7],
        error="invalid_binance_kline_volume",
    )
    taker_buy_volume = strict_non_negative_float(
        payload[9],
        error="invalid_binance_kline_volume",
    )
    taker_buy_quote_volume = strict_non_negative_float(
        payload[10],
        error="invalid_binance_kline_volume",
    )
    if taker_buy_volume > volume or taker_buy_quote_volume > quote_volume:
        raise ValueError("invalid_binance_kline_volume")
    strict_non_negative_integer(payload[8], error="invalid_binance_kline_trade_count")


def _validate_agg_trade_payload(payload: dict[str, Any]) -> None:
    strict_millisecond_timestamp(
        payload.get("timestamp", payload.get("T")),
        error="invalid_binance_agg_trade_timestamp",
    )
    strict_positive_float(
        first_present(payload, "price", "p"),
        error="invalid_binance_agg_trade_price",
    )
    strict_positive_float(
        first_present(payload, "quantity", "q"),
        error="invalid_binance_agg_trade_quantity",
    )
    first = strict_non_negative_integer(
        first_present(payload, "first_trade_id", "f"),
        error="invalid_binance_agg_trade_id_range",
    )
    last = strict_non_negative_integer(
        first_present(payload, "last_trade_id", "l"),
        error="invalid_binance_agg_trade_id_range",
    )
    aggregate = strict_non_negative_integer(
        first_present(payload, "aggregate_trade_id", "a"),
        error="invalid_binance_agg_trade_id_range",
    )
    strict_boolean(
        payload.get("is_buyer_maker", payload.get("m")),
        error="invalid_binance_agg_trade_side",
    )
    if aggregate < 0 or first < 0 or last < first:
        raise ValueError("invalid_binance_agg_trade_id_range")
