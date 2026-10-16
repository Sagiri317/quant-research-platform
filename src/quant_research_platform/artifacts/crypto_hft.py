"""Native Crypto HFT episode ledgers and economic summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
    qrp_json_object_snapshot,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.manifests import read_qrp_json_file, write_manifest
from quant_research_platform.storage.parquet import parquet_table_writer

CRYPTO_EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("stop_ns", pa.int64(), nullable=False),
        pa.field("terminal_valuation_ns", pa.int64(), nullable=False),
        pa.field("termination", pa.string(), nullable=False),
        pa.field("stop_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("ledger_root", pa.string(), nullable=False),
    ]
)
_EPISODE_STOP_REASONS = frozenset(
    {
        "planned_oos_end",
        "book_coverage_end",
        "trade_coverage_end",
        "funding_coverage_end",
        "archive_end",
    }
)


def _episode_id(start_ns, stop_ns):
    return f"{start_ns}_{stop_ns}"


def _episode_window(metadata):
    window = metadata.get("window")
    if not isinstance(window, dict) or set(window) != {"start", "end"}:
        raise ValueError("crypto_episode_window_invalid")
    try:
        start, end = (pd.Timestamp(window[key]) for key in ("start", "end"))
        if pd.isna(start) or pd.isna(end) or start > end:
            raise ValueError("crypto_episode_window_invalid")
        return start.value, end.value
    except (TypeError, OverflowError) as error:
        raise ValueError("crypto_episode_window_invalid") from error


def _validate_episode_row(row, first, last, previous_stop):
    start, stop, terminal = (
        row["start_ns"],
        row["stop_ns"],
        row["terminal_valuation_ns"],
    )
    if any(type(value) is not int for value in (start, stop, terminal)):
        raise ValueError("crypto_episode_index_invalid")
    reasons = list(row["stop_reasons"])
    episode_id = _episode_id(start, stop)
    child_root = f"episodes/{episode_id}"
    if (
        row["episode_id"] != episode_id
        or row["ledger_root"] != child_root
        or not first <= start <= terminal < stop <= last + 1
        or previous_stop is not None
        and start < previous_stop
        or not reasons
        or reasons != sorted(set(reasons))
        or not set(reasons) <= _EPISODE_STOP_REASONS
        or row["termination"] != ("PLANNED_END" if stop == last + 1 else "TRUNCATED")
        or ("planned_oos_end" in reasons) != (stop == last + 1)
    ):
        raise ValueError("crypto_episode_index_invalid")
    row["stop_reasons"] = reasons
    return stop


def _episode_rows(artifact, consume_episode=None):
    """Check chronological episode references while consuming each ledger once."""
    first, last = _episode_window(artifact.metadata)
    previous_stop = None

    def validate_schema(metadata):
        if not metadata.schema.equals(CRYPTO_EPISODE_SCHEMA, check_metadata=False):
            raise ValueError("crypto_episode_schema_invalid")

    chunks = iter_declared_parquet_member_batches(
        artifact.root / "episodes.parquet", metadata_observer=validate_schema
    )
    count = 0
    with closing_iterator(chunks):
        for frame in chunks:
            for row in frame.to_dict("records"):
                start, stop, terminal = (
                    int(row["start_ns"]),
                    int(row["stop_ns"]),
                    int(row["terminal_valuation_ns"]),
                )
                previous_stop = _validate_episode_row(row, first, last, previous_stop)
                child_root = row["ledger_root"]
                terminal_observed = None

                def consume(child, batches, row=row, start=start, stop=stop):
                    if child.capital != artifact.capital or child.metadata[
                        "window"
                    ] != {
                        "start": pd.Timestamp(start).isoformat(),
                        "end": pd.Timestamp(stop - 1).isoformat(),
                    }:
                        raise ValueError("crypto_episode_child_contract_invalid")

                    def observe():
                        nonlocal terminal_observed
                        for batch in batches:
                            for value in batch.valuations:
                                if value.phase == "end":
                                    terminal_observed = (
                                        value.timestamp_ns
                                        if terminal_observed is None
                                        else max(terminal_observed, value.timestamp_ns)
                                    )
                            yield batch

                    with closing_iterator(observe()) as observed:
                        if consume_episode is not None:
                            consume_episode(row, child, observed)
                        for _ in observed:
                            pass

                child = read_portfolio_ledger_artifact(
                    artifact.root / child_root, consume_batches=consume
                )
                if terminal_observed != terminal:
                    raise ValueError("crypto_episode_child_contract_invalid")
                count += 1
                yield row, child
    if not count:
        raise ValueError("crypto_episode_child_members_invalid")


@dataclass(frozen=True, slots=True)
class CryptoEpisodeArtifact:
    root: Path
    capital: PortfolioCapital
    metadata: dict

    def iter_episodes(self, *, consume_episode=None):
        with closing_iterator(_episode_rows(self, consume_episode)) as episodes:
            yield from episodes


def read_crypto_episode_artifact(root, *, consume_episode=None):
    root = Path(root)
    _, manifest = read_qrp_json_file(root / "manifest.json")
    if (
        manifest.get("schema_id") != "qrp-crypto-episodes"
        or manifest.get("members") != ["episodes.parquet"]
        or not isinstance(manifest.get("metadata"), dict)
    ):
        raise ValueError("crypto_episode_manifest_invalid")
    artifact = CryptoEpisodeArtifact(
        root, PortfolioCapital(**manifest["capital"]), manifest["metadata"]
    )
    _episode_window(artifact.metadata)
    if consume_episode is not None:
        with closing_iterator(
            artifact.iter_episodes(consume_episode=consume_episode)
        ) as rows:
            for _ in rows:
                pass
    return artifact


def write_crypto_episode_artifact(output_path, *, capital, metadata, produce):
    """Publish new explicit output only after every indexed child validates."""
    if not isinstance(capital, PortfolioCapital):
        raise ValueError("crypto_episode_capital_invalid")
    metadata = qrp_json_object_snapshot(metadata)
    first, last = _episode_window(metadata)

    def write(staging):
        count = 0
        previous_stop = None
        with parquet_table_writer(
            native_filesystem_path(staging / "episodes.parquet"),
            compression="zstd",
            schema=CRYPTO_EPISODE_SCHEMA,
        ) as write_index:

            def emit_episode(*, start_ns, stop_ns, termination, stop_reasons, produce):
                nonlocal count, previous_stop
                if type(start_ns) is not int or type(stop_ns) is not int:
                    raise ValueError("crypto_episode_index_invalid")
                episode_id = _episode_id(start_ns, stop_ns)
                terminal = None

                def produce_child(emit):
                    def capture(batch):
                        nonlocal terminal
                        for value in batch.valuations:
                            if value.phase == "end":
                                terminal = (
                                    value.timestamp_ns
                                    if terminal is None
                                    else max(terminal, value.timestamp_ns)
                                )
                        emit(batch)

                    return produce(capture)

                child = write_portfolio_ledger_artifact(
                    staging / "episodes" / episode_id,
                    capital=capital,
                    metadata={
                        **metadata,
                        "window": {
                            "start": pd.Timestamp(start_ns).isoformat(),
                            "end": pd.Timestamp(stop_ns - 1).isoformat(),
                        },
                    },
                    produce=produce_child,
                )
                row = {
                    "episode_id": episode_id,
                    "start_ns": start_ns,
                    "stop_ns": stop_ns,
                    "terminal_valuation_ns": terminal,
                    "termination": termination,
                    "stop_reasons": list(stop_reasons),
                    "ledger_root": child.relative_to(staging).as_posix(),
                }
                previous_stop = _validate_episode_row(row, first, last, previous_stop)
                write_index(pa.Table.from_pylist([row], schema=CRYPTO_EPISODE_SCHEMA))
                count += 1

            produce(emit_episode)
            if not count:
                raise PortfolioTargetBlocked(
                    "crypto_execution_qualified_episode_missing"
                )
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": "qrp-crypto-episodes",
                "capital": asdict(capital),
                "metadata": metadata,
                "members": ["episodes.parquet"],
            },
        )

    atomic_write_directory(Path(output_path), write)
    return Path(output_path)


def summarize_crypto_ledger(artifact, batches, *, windows_ns=None, consume_batch=None):
    """Project execution and interval diagnostics during the economic summary pass."""
    expected = (
        None
        if windows_ns is None
        else [
            (name, float(index), timestamp)
            for index, bounds in enumerate(windows_ns)
            for name, timestamp in zip(
                ("oos_window_start", "oos_window_end"), bounds, strict=True
            )
        ]
    )
    if expected == []:
        raise ValueError("crypto_ledger_windows_missing")
    execution = funding_qualification = None
    marker_index = 0
    previous = None
    intervals = []
    accumulated = dict.fromkeys(
        ("net_pnl", "fees", "funding", "turnover", "fill_events"), Fraction()
    )

    def observe(batch, economics, increment):
        nonlocal execution, funding_qualification, marker_index, previous
        for name in accumulated:
            accumulated[name] += increment[name]
        for diagnostic in batch.diagnostics:
            if diagnostic.name == "funding_qualification":
                if (
                    funding_qualification is not None
                    or type(diagnostic.text) is not str
                ):
                    raise ValueError("crypto_funding_qualification_invalid")
                funding_qualification = parse_qrp_json_object(
                    diagnostic.text.encode("utf-8")
                )
            if diagnostic.name == "execution_summary":
                if execution is not None or type(diagnostic.text) is not str:
                    raise ValueError("crypto_execution_summary_invalid")
                execution = parse_qrp_json_object(diagnostic.text.encode("utf-8"))
                if PORTFOLIO_ECONOMIC_SUMMARY_FIELDS.intersection(execution) or any(
                    type(value) not in {int, float, str, bool, type(None)}
                    for value in execution.values()
                ):
                    raise ValueError("crypto_execution_duplicate_economic_fields")
            if expected is None or diagnostic.name not in {
                "oos_window_start",
                "oos_window_end",
            }:
                continue
            if (
                marker_index >= len(expected)
                or (
                    diagnostic.name,
                    diagnostic.number,
                    economics["valuation_timestamp_ns"],
                )
                != expected[marker_index]
            ):
                raise ValueError("crypto_ledger_window_marker_invalid")
            current = tuple(
                economics[name]
                for name in ("nav", "fees", "funding", "turnover", "fill_events")
            )
            if previous is None:
                if current != (
                    artifact.capital.initial_capital,
                    0.0,
                    0.0,
                    0.0,
                    0,
                ) or any(accumulated.values()):
                    raise ValueError("crypto_ledger_before_window_economics")
            else:
                net, cost, cash, traded, count = accumulated.values()
                intervals.append(
                    {
                        "kind": "oos" if marker_index % 2 else "gap",
                        "start_ns": expected[marker_index - 1][2],
                        "end_ns": expected[marker_index][2],
                        "start_nav": previous[0],
                        "end_nav": economics["nav"],
                        "net_pnl": float(net),
                        "gross_pnl": float(net + cost - cash),
                        "fees": float(cost),
                        "funding": float(cash),
                        "turnover": float(traded),
                        "fill_events": int(count),
                    }
                )
            previous = current
            accumulated.update(dict.fromkeys(accumulated, Fraction()))
            marker_index += 1
        if consume_batch is not None:
            consume_batch(batch)

    economics = summarize_portfolio_ledger(
        artifact.capital, batches, consume_batch=observe
    )
    if execution is None:
        raise ValueError("crypto_execution_summary_missing")
    partial = execution.get("account_path_status") == "PARTIAL"
    if expected is not None and not partial:
        if marker_index != len(expected):
            raise ValueError("crypto_ledger_window_marker_missing")
        if any(accumulated.values()) or previous != tuple(
            economics[name]
            for name in ("nav", "fees", "funding", "turnover", "fill_events")
        ):
            raise ValueError("crypto_ledger_after_window_economics")
    return {
        "economics": economics,
        "execution": execution,
        "intervals": intervals,
        "funding_qualification": funding_qualification,
    }
