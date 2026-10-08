"""Crypto HFT execution evidence owner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.portfolio_ledger import (
    PORTFOLIO_LEDGER_TABLE_COLUMNS,
    _capital,
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_ECONOMIC_SUMMARY_FIELDS,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    native_filesystem_path,
)
from quant_research_platform.storage.parquet import (
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    _member_inventory,
    _visit_member_paths,
    read_semantic_artifact_record,
    write_semantic_artifact,
)

CRYPTO_EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("stop_ns", pa.int64(), nullable=False),
        pa.field("terminal_valuation_ns", pa.int64(), nullable=False),
        pa.field("termination", pa.string(), nullable=False),
        pa.field("stop_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("ledger_root", pa.string(), nullable=False),
        pa.field("ledger_identity", pa.string(), nullable=False),
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


def crypto_episode_schema_payload():
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in CRYPTO_EPISODE_SCHEMA
    ]


def _episode_identity(parent_identity, start_ns, stop_ns):
    return canonical_json_hash(
        {
            "execution_instance_identity": parent_identity,
            "start_ns": start_ns,
            "stop_ns": stop_ns,
        }
    )


def _episode_envelope(parent, start, stop):
    episode_id = _episode_identity(parent["instance_identity"], start, stop)
    return build_business_manifest_envelope(
        logical_artifact_id="episode_" + episode_id,
        schema_identity=canonical_json_hash(PORTFOLIO_LEDGER_TABLE_COLUMNS),
        protocol_or_spec_identity=parent["protocol_or_spec_identity"],
        software_identity=parent["software_identity"],
        research_window={
            **parent["research_window"],
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(stop - 1).isoformat(),
        },
        upstream_identities={"episode_parent": parent["instance_identity"]},
        semantic_metadata={
            "portfolio_capital": parent["semantic_metadata"]["portfolio_capital"]
        },
    )


def _episode_rows(
    root, envelope, members, identities, manifest_identity=None, consume_episode=None
):
    """Keep the parent's captured physical member barrier through child consumption."""
    previous_stop = None
    expected_paths = {"episodes.parquet"}
    window = envelope["research_window"]
    first, last = (
        pd.Timestamp(window["start"]).value,
        pd.Timestamp(window["end"]).value,
    )
    primary = None
    try:

        def validate_schema(metadata):
            if not metadata.schema.equals(CRYPTO_EPISODE_SCHEMA, check_metadata=False):
                raise ValueError("crypto_episode_schema_invalid")

        chunks = iter_declared_parquet_member_batches(
            root / "episodes.parquet",
            expected_identity=identities["episodes.parquet"],
            metadata_observer=validate_schema,
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
                    reasons = list(row["stop_reasons"])
                    episode_id = _episode_identity(
                        envelope["instance_identity"], start, stop
                    )
                    child_root = (
                        f"episodes/{episode_id}/instances/{row['ledger_identity']}"
                    )
                    if (
                        row["episode_id"] != episode_id
                        or row["ledger_root"] != child_root
                        or (not first <= start <= terminal < stop <= last + 1)
                        or (previous_stop is not None and start < previous_stop)
                        or (not reasons)
                        or (reasons != sorted(set(reasons)))
                        or (not set(reasons) <= _EPISODE_STOP_REASONS)
                        or (
                            row["termination"]
                            != ("PLANNED_END" if stop == last + 1 else "TRUNCATED")
                        )
                        or (("planned_oos_end" in reasons) != (stop == last + 1))
                    ):
                        raise ValueError("crypto_episode_index_invalid")
                    previous_stop = stop
                    row["stop_reasons"] = reasons
                    child_paths = {
                        child_root + "/" + member
                        for member in (
                            "manifest.json",
                            *(
                                name + ".parquet"
                                for name in PORTFOLIO_LEDGER_TABLE_COLUMNS
                            ),
                        )
                    }
                    if not child_paths <= identities.keys():
                        raise ValueError("crypto_episode_child_members_invalid")
                    expected_paths.update(child_paths)
                    terminal_observed = None

                    def consume(artifact, batches, row=row):

                        def observe():
                            nonlocal terminal_observed
                            for batch in batches:
                                for value in batch.valuations:
                                    if value.phase == "end":
                                        terminal_observed = (
                                            value.timestamp_ns
                                            if terminal_observed is None
                                            else max(
                                                terminal_observed, value.timestamp_ns
                                            )
                                        )
                                yield batch

                        with closing_iterator(observe()) as observed:
                            if consume_episode is not None:
                                consume_episode(row, artifact, observed)
                            for _ in observed:
                                pass

                    expected_child = _episode_envelope(envelope, start, stop)
                    if row["ledger_identity"] != expected_child.instance_identity:
                        raise ValueError("crypto_episode_child_contract_invalid")
                    child_identities = {
                        path.removeprefix(child_root + "/"): identities[path]
                        for path in child_paths
                    }
                    child = read_portfolio_ledger_artifact(
                        root / child_root,
                        expected_identity=expected_child.instance_identity,
                        expected_member_identities=child_identities,
                        consume_batches=consume,
                    )
                    if terminal_observed != terminal:
                        raise ValueError("crypto_episode_child_contract_invalid")
                    count += 1
                    yielded_value = (row, child)
                    try:
                        yield yielded_value
                    except GeneratorExit:
                        return
        if not count or expected_paths != {member["path"] for member in members}:
            raise ValueError("crypto_episode_child_members_invalid")
    except BaseException as body_error:
        primary = body_error
        raise
    finally:
        try:
            observed_paths = set()
            _visit_member_paths(
                root, lambda relative, path: observed_paths.add(relative)
            )
            if observed_paths != set(identities) or any(
                (
                    dataset_file_identity(root / path) != identity
                    for path, identity in identities.items()
                )
            ):
                raise ValueError("crypto_episode_member_identity_changed")
            if (
                manifest_identity is not None
                and dataset_file_identity(root / "manifest.json") != manifest_identity
            ):
                raise ValueError("crypto_episode_manifest_identity_changed")
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(
                "episode member verification also failed: " + type(error).__name__
            )


@dataclass(frozen=True, slots=True)
class CryptoEpisodeArtifact:
    root: Path
    identity: str
    capital: PortfolioCapital
    _envelope: dict
    _members: tuple
    _identities: dict
    _manifest_identity: object

    def iter_episodes(self, *, consume_episode=None):
        with closing_iterator(
            _episode_rows(
                self.root,
                self._envelope,
                self._members,
                self._identities,
                self._manifest_identity,
                consume_episode,
            )
        ) as delegated_chunks:
            for yielded_value in delegated_chunks:
                try:
                    yield yielded_value
                except GeneratorExit:
                    return


def read_crypto_episode_artifact(
    root, *, expected_identity, expected_member_identities=None, consume_episode=None
):
    root = Path(root)
    identities = {}
    envelope, members = read_semantic_artifact_record(
        root,
        expected_identity=expected_identity,
        member_identities=identities,
        expected_member_identities=expected_member_identities,
    )
    manifest_identity = identities.pop("manifest.json")
    artifact = CryptoEpisodeArtifact(
        root,
        expected_identity,
        _capital(envelope),
        envelope,
        members,
        identities,
        manifest_identity,
    )
    with closing_iterator(
        artifact.iter_episodes(consume_episode=consume_episode)
    ) as episodes:
        for _row, _ledger in episodes:
            pass
    return artifact


def write_crypto_episode_artifact(logical_root, *, envelope, produce):
    """Publish the complete episode collection only after every child validates."""

    def write(staging):
        count = 0
        with parquet_table_writer(
            native_filesystem_path(staging / "episodes.parquet"),
            compression="zstd",
            schema=CRYPTO_EPISODE_SCHEMA,
        ) as write_index:

            def emit_episode(*, start_ns, stop_ns, termination, stop_reasons, produce):
                nonlocal count
                episode_id = _episode_identity(
                    envelope.instance_identity, start_ns, stop_ns
                )
                child = _episode_envelope(envelope.payload(), start_ns, stop_ns)
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

                root = write_portfolio_ledger_artifact(
                    staging / "episodes" / episode_id,
                    envelope=child,
                    produce=produce_child,
                )
                row = {
                    "episode_id": episode_id,
                    "start_ns": start_ns,
                    "stop_ns": stop_ns,
                    "terminal_valuation_ns": terminal,
                    "termination": termination,
                    "stop_reasons": list(stop_reasons),
                    "ledger_root": root.relative_to(staging).as_posix(),
                    "ledger_identity": child.instance_identity,
                }
                write_index(pa.Table.from_pylist([row], schema=CRYPTO_EPISODE_SCHEMA))
                count += 1

            produce(emit_episode)
            if not count:
                raise PortfolioTargetBlocked(
                    "crypto_execution_qualified_episode_missing"
                )

    def validate(staging, members):
        identities = {}
        if (
            _member_inventory(
                staging,
                expected_paths={member["path"] for member in members},
                verified_identities=identities,
            )
            != members
        ):
            raise ValueError("crypto_episode_member_integrity_invalid")
        with closing_iterator(
            _episode_rows(staging, envelope.payload(), members, identities)
        ) as rows:
            for _row, _ledger in rows:
                pass

    return write_semantic_artifact(
        Path(logical_root), envelope, write, staged_validator=validate
    )


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
    execution = None
    marker_index = 0
    previous = None
    intervals = []

    def observe(batch, economics):
        nonlocal execution, marker_index, previous
        for diagnostic in batch.diagnostics:
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
                if current != (artifact.capital.initial_capital, 0.0, 0.0, 0.0, 0):
                    raise ValueError("crypto_ledger_before_window_economics")
            else:
                net, cost, cash, traded, count = (
                    value - before
                    for value, before in zip(current, previous, strict=True)
                )
                intervals.append(
                    {
                        "kind": "oos" if marker_index % 2 else "gap",
                        "start_ns": expected[marker_index - 1][2],
                        "end_ns": expected[marker_index][2],
                        "start_nav": previous[0],
                        "end_nav": economics["nav"],
                        "net_pnl": net,
                        "gross_pnl": net + cost - cash,
                        "fees": cost,
                        "funding": cash,
                        "turnover": traded,
                        "fill_events": count,
                    }
                )
            previous = current
            marker_index += 1
        if consume_batch is not None:
            consume_batch(batch)

    economics = summarize_portfolio_ledger(
        artifact.capital, batches, consume_batch=observe
    )
    if execution is None:
        raise ValueError("crypto_execution_summary_missing")
    if expected is not None:
        if marker_index != len(expected):
            raise ValueError("crypto_ledger_window_marker_missing")
        if previous != tuple(
            economics[name]
            for name in ("nav", "fees", "funding", "turnover", "fill_events")
        ):
            raise ValueError("crypto_ledger_after_window_economics")
    return {"economics": economics, "execution": execution, "intervals": intervals}
