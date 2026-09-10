"""One immutable typed economic ledger for every research portfolio."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, fields
from itertools import groupby
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.business_manifest import BusinessManifestEnvelope
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    ExecutedTrade,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
    validate_portfolio_transition,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.storage.exact_keys import (
    EXACT_KEY_SQLITE_CACHE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    _parquet_frame_reservation,
    _parquet_table_write_state,
    _parquet_writer_fixed_reservation,
    iter_dataset_part_batches,
    record_batch_decoded_working_set_bounds,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    write_semantic_artifact,
)

_TABLE_TYPES = {
    "trades": ExecutedTrade,
    "cashflows": PortfolioCashFlow,
    "valuations": PortfolioValuation,
    "diagnostics": PortfolioDiagnostic,
}
_NUMERIC = {
    "quantity",
    "notional",
    "amount",
    "value",
    "number",
    "mark_price",
    "cost_basis",
}
PORTFOLIO_LEDGER_TABLE_COLUMNS = {
    name: ("sequence", *(field.name for field in fields(kind)))
    for name, kind in _TABLE_TYPES.items()
}
_SCHEMAS = {
    name: pa.schema(
        pa.field(
            column,
            pa.int64()
            if column in {"sequence", "timestamp_ns"}
            else pa.float64()
            if column in _NUMERIC
            else pa.string(),
        )
        for column in columns
    )
    for name, columns in PORTFOLIO_LEDGER_TABLE_COLUMNS.items()
}
_BUFFER_ROWS = 256


@contextmanager
def _preserving_close(resource):
    try:
        yield resource
    except BaseException as primary:
        try:
            resource.close()
        except BaseException as cleanup:
            primary.add_note(
                f"portfolio resource close also failed: {type(cleanup).__name__}"
            )
        raise
    else:
        resource.close()


@dataclass(frozen=True, slots=True)
class PortfolioLedgerArtifact:
    root: Path
    identity: str
    capital: PortfolioCapital

    def iter_chunks(self) -> Iterator[PortfolioLedgerBatch]:
        envelope, _members = read_semantic_artifact_record(
            self.root, expected_identity=self.identity
        )
        if _capital(envelope) != self.capital:
            raise ValueError("portfolio_ledger_capital_invalid")
        yield from _iter_batches(self.root, self.capital, envelope["research_window"])


def _capital(envelope: Mapping[str, object]) -> PortfolioCapital:
    metadata = envelope["semantic_metadata"]
    if not isinstance(metadata, Mapping) or not isinstance(
        metadata.get("portfolio_capital"), Mapping
    ):
        raise ValueError("portfolio_ledger_capital_missing")
    return PortfolioCapital(**dict(metadata["portfolio_capital"]))


def write_portfolio_ledger_artifact(
    logical_root: Path,
    *,
    envelope: BusinessManifestEnvelope,
    produce: Callable[[Callable[[PortfolioLedgerBatch], None]], object],
) -> Path:
    """Stream bounded tables, validate persisted economics, then atomic publish."""
    capital = _capital(envelope.payload())

    def write(staging: Path) -> None:
        with (
            _preserving_close(ExitStack()) as resources,
            _preserving_close(ExitStack()) as stack,
        ):
            policy = current_runtime_resource_policy()
            buffer_lease = (
                resources.enter_context(policy.lease("artifacts.portfolio.buffers", 1))
                if policy is not None
                else None
            )
            continuity_lease = (
                resources.enter_context(
                    policy.lease("artifacts.portfolio.write_continuity", 1)
                )
                if policy is not None
                else None
            )
            resources.enter_context(
                _parquet_writer_fixed_reservation(writer_count=4, index_bytes=0)
            )
            writers = {
                name: stack.enter_context(
                    _preserving_close(
                        pq.ParquetWriter(
                            native_filesystem_path(staging / f"{name}.parquet"),
                            schema,
                            compression="zstd",
                            use_dictionary=False,
                            write_statistics=False,
                        )
                    )
                )
                for name, schema in _SCHEMAS.items()
            }
            buffers = {name: [] for name in _TABLE_TYPES}
            buffer_bytes = {name: 0 for name in _TABLE_TYPES}
            retained = {name: (0, 0) for name in _TABLE_TYPES}
            metadata = {name: None for name in _TABLE_TYPES}
            sequence = 0
            previous = None

            def flush(name: str) -> None:
                records = buffers[name]
                if not records:
                    return
                frame = pd.DataFrame.from_records(
                    records, columns=PORTFOLIO_LEDGER_TABLE_COLUMNS[name]
                )
                with _parquet_frame_reservation(frame):
                    table = pa.Table.from_pandas(
                        frame, schema=_SCHEMAS[name], preserve_index=False
                    )
                    persisted, state, batch_rows, retained[name] = (
                        _parquet_table_write_state(
                            table,
                            previous_metadata=metadata[name],
                            retained_memory=retained[name],
                        )
                    )
                    resources.enter_context(state)
                    metadata[name] = dict(persisted.schema.metadata or {})
                    writers[name].write_table(persisted, row_group_size=batch_rows)
                records.clear()
                buffer_bytes[name] = 0
                if buffer_lease is not None:
                    buffer_lease.shrink_to(max(1, sum(buffer_bytes.values())))

            def emit(batch: PortfolioLedgerBatch) -> None:
                nonlocal sequence, previous
                if (
                    type(batch) is not PortfolioLedgerBatch
                    or batch.sequence != sequence
                ):
                    raise ValueError("portfolio_ledger_sequence_invalid")
                continuity_bytes = sum(
                    1024 + 8 * len(row.asset_id)
                    for row in batch.valuations
                    if row.phase == "end"
                )
                if continuity_lease is not None:
                    continuity_lease.grow(continuity_bytes)
                previous = validate_portfolio_transition(batch, capital, previous)
                if continuity_lease is not None:
                    continuity_lease.shrink_to(max(1, continuity_bytes))
                for name in _TABLE_TYPES:
                    for row in getattr(batch, name):
                        bound = (
                            256
                            + 128 * len(fields(row))
                            + 4
                            * sum(
                                len(getattr(row, field.name))
                                for field in fields(row)
                                if isinstance(getattr(row, field.name), str)
                            )
                        )
                        if buffer_lease is not None:
                            buffer_lease.grow(bound)
                        buffer_bytes[name] += bound
                        buffers[name].append({"sequence": sequence, **asdict(row)})
                        if len(buffers[name]) >= _BUFFER_ROWS:
                            flush(name)
                sequence += 1

            produce(emit)
            if not sequence:
                raise ValueError("portfolio_ledger_empty")
            for name in writers:
                flush(name)
                writers[name].add_key_value_metadata(metadata[name] or {})
        with closing_iterator(
            _iter_batches(staging, capital, envelope.research_window)
        ) as batches:
            for _batch in batches:
                pass

    return write_semantic_artifact(Path(logical_root), envelope, write)


def read_portfolio_ledger_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
) -> PortfolioLedgerArtifact:
    envelope, members = read_semantic_artifact_record(
        instance_root, expected_identity=expected_identity
    )
    if {member["path"] for member in members} != {
        f"{name}.parquet" for name in _TABLE_TYPES
    }:
        raise ValueError("portfolio_ledger_member_invalid")
    artifact = PortfolioLedgerArtifact(
        Path(instance_root), expected_identity, _capital(envelope)
    )
    with closing_iterator(artifact.iter_chunks()) as batches:
        for _batch in batches:
            pass
    return artifact


def summarize_portfolio_ledger(artifact: PortfolioLedgerArtifact) -> dict[str, object]:
    fees = funding = turnover = nav = fill_quantity = 0.0
    valuation_timestamp_ns = 0
    fills = 0
    alpha_calibration = {}
    with closing_iterator(artifact.iter_chunks()) as batches:
        for batch in batches:
            calibration = {
                row.name.removeprefix("alpha_"): row.text
                if row.text is not None
                else row.number
                for row in batch.diagnostics
                if row.asset_id == PORTFOLIO_CASH_ASSET_ID
                and row.name.startswith("alpha_")
            }
            if "status" in calibration:
                alpha_calibration = calibration
            fills += len(batch.trades)
            fill_quantity += sum(abs(row.quantity) for row in batch.trades)
            turnover += sum(abs(row.notional) for row in batch.trades)
            fees -= sum(
                row.amount for row in batch.cashflows if row.kind.endswith("_cost")
            )
            funding += sum(
                row.amount for row in batch.cashflows if row.kind == "funding"
            )
            nav = sum(row.value for row in batch.valuations if row.phase == "end")
            valuation_timestamp_ns = max(
                row.timestamp_ns for row in batch.valuations if row.phase == "end"
            )
    net = nav - artifact.capital.initial_capital
    return {
        "candidate_id": artifact.capital.candidate_id,
        "initial_capital": artifact.capital.initial_capital,
        "valuation_currency": artifact.capital.valuation_currency,
        "gross_pnl": net + fees - funding,
        "net_pnl": net,
        "nav": nav,
        "capital_return": net / artifact.capital.initial_capital,
        "fees": fees,
        "funding": funding,
        "turnover": turnover,
        "fill_events": fills,
        "fill_quantity": fill_quantity,
        "valuation_timestamp_ns": valuation_timestamp_ns,
        **({"alpha_calibration": alpha_calibration} if alpha_calibration else {}),
    }


def _table_records(root: Path, name: str) -> Iterator[dict[str, object]]:
    with _opened_parquet(root / f"{name}.parquet", expected_identity=None) as (
        parquet,
        identity,
        _handle,
    ):
        if not parquet.schema_arrow.remove_metadata().equals(_SCHEMAS[name]):
            raise ValueError("portfolio_ledger_schema_invalid")
    chunks = iter_dataset_part_batches(
        root / f"{name}.parquet",
        expected_identity=identity,
        target_chunk_bytes=256 * 1024,
        source_batch_rows=_BUFFER_ROWS,
    )
    with closing_iterator(chunks):
        for chunk in chunks:
            if not chunk.schema.remove_metadata().equals(_SCHEMAS[name]):
                raise ValueError("portfolio_ledger_schema_invalid")
            row_bytes, fixed = record_batch_decoded_working_set_bounds(chunk)
            with _preserving_close(ExitStack()) as ownership:
                policy = current_runtime_resource_policy()
                if policy is not None:
                    ownership.enter_context(
                        policy.lease(
                            "artifacts.portfolio.decoded_rows",
                            4 * fixed
                            + chunk.num_rows
                            * (4 * row_bytes + 128 * chunk.num_columns),
                        )
                    )
                records = chunk.to_pylist()
                try:
                    yield from records
                except GeneratorExit:
                    return
                finally:
                    records = None


def _iter_batches(
    root: Path, capital: PortfolioCapital, research_window: Mapping[str, object]
) -> Iterator[PortfolioLedgerBatch]:
    bounds = tuple(pd.Timestamp(research_window[key]).value for key in ("start", "end"))
    with _preserving_close(ExitStack()) as stack:
        policy = current_runtime_resource_policy()
        if policy is not None:
            stack.enter_context(
                policy.lease("artifacts.portfolio.index", EXACT_KEY_SQLITE_CACHE_BYTES)
            )
            continuity_lease = stack.enter_context(
                policy.lease("artifacts.portfolio.continuity", 1)
            )
        else:
            continuity_lease = None
        key_index = stack.enter_context(
            ExactPrimaryKeyIndex(
                primary_key=("fill_id",),
                timestamp_columns=(),
            )
        )
        readers = {
            name: stack.enter_context(closing_iterator(_table_records(root, name)))
            for name in _TABLE_TYPES
        }
        groups = {
            name: iter(groupby(reader, key=lambda row: row["sequence"]))
            for name, reader in readers.items()
        }
        pending = {name: next(group, None) for name, group in groups.items()}
        sequence = 0
        previous = None
        while pending["valuations"] is not None:
            if pending["valuations"][0] != sequence:
                raise ValueError("portfolio_ledger_sequence_invalid")
            with _preserving_close(ExitStack()) as retained:
                lease = (
                    retained.enter_context(
                        policy.lease("artifacts.portfolio.transition", 1)
                    )
                    if policy is not None
                    else None
                )
                tables = {}
                for name, kind in _TABLE_TYPES.items():
                    group = pending[name]
                    if group is not None and group[0] < sequence:
                        raise ValueError("portfolio_ledger_sequence_invalid")
                    records = []
                    if group is not None and group[0] == sequence:
                        for row in group[1]:
                            if lease is not None:
                                lease.grow(
                                    256
                                    + 128 * len(row)
                                    + 4
                                    * sum(
                                        len(value)
                                        for value in row.values()
                                        if isinstance(value, str)
                                    )
                                )
                            records.append(
                                kind(
                                    **{
                                        key: value
                                        for key, value in row.items()
                                        if key != "sequence"
                                    }
                                )
                            )
                        pending[name] = next(groups[name], None)
                    tables[name] = tuple(records)
                batch = PortfolioLedgerBatch(sequence, **tables)
                if any(
                    not bounds[0] <= row.timestamp_ns <= bounds[1]
                    for row in batch.valuations
                ):
                    raise ValueError("portfolio_ledger_valuation_window_invalid")
                if continuity_lease is not None:
                    bound = max(
                        1,
                        sum(
                            1024 + 8 * len(row.asset_id)
                            for row in batch.valuations
                            if row.phase == "end"
                        ),
                    )
                    continuity_lease.grow(bound)
                previous = validate_portfolio_transition(batch, capital, previous)
                if continuity_lease is not None:
                    continuity_lease.shrink_to(bound)
                inserted = key_index.add_identity_payloads(
                    namespace="fills",
                    payloads=(
                        (row.fill_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                        for row in batch.trades
                    ),
                )
                if inserted != len(batch.trades):
                    raise ValueError("portfolio_ledger_duplicate_fill")
                key_index.add_identity_payloads(
                    namespace="fill_orders",
                    payloads=(
                        (row.fill_id.encode("utf-8"), row.order_id.encode("utf-8"))
                        for row in batch.trades
                    ),
                )
                for diagnostic in batch.diagnostics:
                    if (
                        diagnostic.fill_id is not None
                        and key_index.identity_payload_value(
                            namespace="fills",
                            payload=diagnostic.fill_id.encode("utf-8"),
                        )
                        != diagnostic.asset_id.encode("utf-8")
                    ):
                        raise ValueError("portfolio_ledger_diagnostic_fill_mismatch")
                    if diagnostic.fill_id is not None and diagnostic.name == "order_id":
                        order_id = (
                            diagnostic.text
                            if diagnostic.text is not None
                            else format(diagnostic.number, "g")
                        )
                        if key_index.identity_payload_value(
                            namespace="fill_orders",
                            payload=diagnostic.fill_id.encode("utf-8"),
                        ) != order_id.encode("utf-8"):
                            raise ValueError(
                                "portfolio_ledger_diagnostic_order_mismatch"
                            )
                inserted = key_index.add_identity_payloads(
                    namespace="cashflows",
                    payloads=(
                        (row.flow_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                        for row in batch.cashflows
                    ),
                )
                if inserted != len(batch.cashflows):
                    raise ValueError("portfolio_ledger_duplicate_cashflow")
                try:
                    yield batch
                except GeneratorExit:
                    return
                sequence += 1
        if not sequence or any(group is not None for group in pending.values()):
            raise ValueError("portfolio_ledger_sequence_invalid")
