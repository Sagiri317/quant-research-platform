"""One immutable typed economic ledger for every research portfolio."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from itertools import groupby, islice
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.business_manifest import BusinessManifestEnvelope
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    ExecutedTrade,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioValuation,
    validate_portfolio_transition,
)
from quant_research_platform.contracts.validation import (
    decode_exact_numeric,
    encode_exact_numeric,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    open_filesystem_file,
)
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.parquet import (
    iter_dataset_part_batches,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    semantic_artifact_read_session,
    write_semantic_artifact,
)

_TABLE_TYPES = {
    "trades": ExecutedTrade,
    "cashflows": PortfolioCashFlow,
    "valuations": PortfolioValuation,
    "diagnostics": PortfolioDiagnostic,
    "rights_adjustments": PortfolioRightsAdjustment,
}
_EXACT_FIELDS = {
    "quantity",
    "quantity_before",
    "quantity_after",
    "target_quantity",
    "mark_price",
}
_NUMERIC = {
    "notional",
    "amount",
    "value",
    "number",
    "cost_basis",
    "cash_amount",
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


@dataclass(frozen=True, slots=True)
class PortfolioLedgerArtifact:
    envelope: dict
    root: Path
    identity: str
    capital: PortfolioCapital
    member_identities: Mapping[str, DatasetFileIdentity]
    operational_metadata: Mapping[str, object]
    final_alpha_calibration: Mapping[str, object] | None = None

    def iter_chunks(self) -> Iterator[PortfolioLedgerBatch]:
        identities: dict[str, DatasetFileIdentity] = {}
        with semantic_artifact_read_session(
            self.root,
            expected_identity=self.identity,
            member_identities=identities,
            expected_member_identities=self.member_identities,
        ) as (envelope, _members):
            if _capital(envelope) != self.capital:
                raise ValueError("portfolio_ledger_capital_invalid")
            with closing_iterator(
                _iter_batches(
                    self.root, self.capital, envelope["research_window"], identities
                )
            ) as delegated_chunks:
                for yielded_value in delegated_chunks:
                    try:
                        yield yielded_value
                    except GeneratorExit:
                        return


def _capital(envelope: Mapping[str, object]) -> PortfolioCapital:
    metadata = envelope["semantic_metadata"]
    if not isinstance(metadata, Mapping) or not isinstance(
        metadata.get("portfolio_capital"), Mapping
    ):
        raise ValueError("portfolio_ledger_capital_missing")
    return PortfolioCapital(**dict(metadata["portfolio_capital"]))


def _final_calibration_required(envelope):
    contract = envelope["semantic_metadata"].get("target_contract")
    return (
        isinstance(contract, Mapping)
        and contract.get("protocol", {}).get("protocol_id") == "a_share_low_frequency"
        and (contract["portfolio_candidate"]["construction"] == "benchmark_enhanced")
        and (envelope["research_window"]["access"] == "development")
    )


def write_portfolio_ledger_artifact(
    logical_root: Path,
    *,
    envelope: BusinessManifestEnvelope,
    produce: Callable[[Callable[[PortfolioLedgerBatch], None]], object],
    final_calibration: Callable[[], dict[str, object]] | None = None,
) -> Path:
    """Stream bounded tables, validate persisted economics, then atomic publish."""
    capital = _capital(envelope.payload())
    calibration_required = _final_calibration_required(envelope.payload())
    operational = {}
    if calibration_required != (final_calibration is not None):
        raise ValueError("portfolio_final_calibration_contract_invalid")

    def write(staging: Path) -> None:
        with closing_iterator(ExitStack()) as stack:
            writers = {}
            for name, schema in _SCHEMAS.items():
                handle = stack.enter_context(
                    closing_iterator(
                        open_filesystem_file(staging / f"{name}.parquet", "xb")
                    )
                )
                writers[name] = stack.enter_context(
                    parquet_table_writer(handle, compression="zstd", schema=schema)
                )
            buffers = {name: [] for name in _TABLE_TYPES}
            sequence = 0
            previous = None

            def flush(name: str) -> None:
                records = buffers[name]
                if not records:
                    return
                frame = pd.DataFrame.from_records(
                    (
                        {
                            key: encode_exact_numeric(value)
                            if key in _EXACT_FIELDS
                            else value
                            for key, value in record.items()
                        }
                        for record in records
                    ),
                    columns=PORTFOLIO_LEDGER_TABLE_COLUMNS[name],
                )
                table = pa.Table.from_pandas(
                    frame, schema=_SCHEMAS[name], preserve_index=False
                )
                writers[name](table)
                records.clear()

            def emit(batch: PortfolioLedgerBatch) -> None:
                nonlocal sequence, previous
                if (
                    type(batch) is not PortfolioLedgerBatch
                    or batch.sequence != sequence
                ):
                    raise ValueError("portfolio_ledger_sequence_invalid")
                previous = validate_portfolio_transition(batch, capital, previous)
                for name in _TABLE_TYPES:
                    for row in getattr(batch, name):
                        buffers[name].append({"sequence": sequence, **asdict(row)})
                        if len(buffers[name]) >= DEFAULT_BATCH_ROWS:
                            flush(name)
                sequence += 1

            produce(emit)
            if not sequence:
                raise ValueError("portfolio_ledger_empty")
            for name in writers:
                flush(name)
        with closing_iterator(
            _iter_batches(staging, capital, envelope.research_window)
        ) as batches:
            for _ in batches:
                pass
        if calibration_required:
            payload = dict(final_calibration())
            operational["final_alpha_observations"] = payload.pop("observations")
            write_internal_artifact_manifest(
                staging / "final_alpha_calibration.json", payload
            )

    return write_semantic_artifact(
        Path(logical_root), envelope, write, operational_metadata=operational
    )


def read_portfolio_ledger_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
    consume_batches: Callable[
        [PortfolioLedgerArtifact, Iterator[PortfolioLedgerBatch]], object
    ]
    | None = None,
    expected_member_identities: Mapping[str, DatasetFileIdentity] | None = None,
) -> PortfolioLedgerArtifact:
    """Validate once, optionally consuming batches during that same full pass."""
    identities: dict[str, DatasetFileIdentity] = {}
    operational = {}
    with semantic_artifact_read_session(
        instance_root,
        expected_identity=expected_identity,
        member_identities=identities,
        expected_member_identities=expected_member_identities,
        operational_metadata=operational,
    ) as (envelope, members):
        calibration_required = _final_calibration_required(envelope)
        required = {f"{name}.parquet" for name in _TABLE_TYPES}
        if calibration_required:
            required.add("final_alpha_calibration.json")
        if {member["path"] for member in members} != required:
            raise ValueError("portfolio_ledger_member_invalid")
        calibration = (
            read_qrp_json_file(
                Path(instance_root) / "final_alpha_calibration.json",
                expected_identity=identities["final_alpha_calibration.json"],
            )[1]
            if calibration_required
            else None
        )
        if calibration_required and (
            type(calibration) is not dict
            or set(calibration)
            != {
                "schema_id",
                "selection_group",
                "cutoff",
                "upstream_identities",
                "calibration",
            }
            or calibration["schema_id"] != "qrp-final-alpha-calibration"
            or (
                calibration["selection_group"]
                != envelope["semantic_metadata"]["target_contract"][
                    "portfolio_candidate"
                ]["selection_group"]
            )
            or (
                pd.Timestamp(calibration["cutoff"])
                != pd.Timestamp(envelope["research_window"]["end"])
            )
            or (calibration["upstream_identities"] != envelope["upstream_identities"])
        ):
            raise ValueError("portfolio_final_calibration_binding_invalid")
        artifact = PortfolioLedgerArtifact(
            envelope,
            Path(instance_root),
            expected_identity,
            _capital(envelope),
            identities,
            operational,
            calibration,
        )
        with closing_iterator(
            _iter_batches(
                artifact.root, artifact.capital, envelope["research_window"], identities
            )
        ) as batches:
            if consume_batches is not None:
                consume_batches(artifact, islice(batches, None))
            for _ in batches:
                pass
        return artifact


def summarize_portfolio_ledger(
    capital: PortfolioCapital,
    batches: Iterator[PortfolioLedgerBatch],
    *,
    consume_batch: Callable[
        [PortfolioLedgerBatch, Mapping[str, object], Mapping[str, object]], None
    ]
    | None = None,
) -> dict[str, object]:
    economics = _LedgerEconomics(capital)
    result = {
        "candidate_id": capital.candidate_id,
        "initial_capital": capital.initial_capital,
        "valuation_currency": capital.valuation_currency,
        **economics.summary(),
    }
    with closing_iterator(batches):
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
                result["alpha_calibration"] = calibration
            increment = economics.consume(batch)
            result.update(economics.summary())
            if consume_batch is not None:
                consume_batch(batch, result, increment)
    return result


def _table_records(
    root: Path, name: str, expected_identity: DatasetFileIdentity | None
) -> Iterator[dict[str, object]]:
    def validate_schema(metadata):
        if not metadata.schema.equals(_SCHEMAS[name]):
            raise ValueError("portfolio_ledger_schema_invalid")

    chunks = iter_dataset_part_batches(
        root / f"{name}.parquet",
        expected_identity=expected_identity,
        metadata_observer=validate_schema,
    )
    with closing_iterator(chunks):
        for chunk in chunks:
            if not chunk.schema.remove_metadata().equals(_SCHEMAS[name]):
                raise ValueError("portfolio_ledger_schema_invalid")
            records = chunk.to_pylist()
            for yielded_value in records:
                for column in _EXACT_FIELDS.intersection(yielded_value):
                    yielded_value[column] = decode_exact_numeric(yielded_value[column])
                try:
                    yield yielded_value
                except GeneratorExit:
                    return


def _iter_batches(
    root: Path,
    capital: PortfolioCapital,
    research_window: Mapping[str, object],
    identities: Mapping[str, DatasetFileIdentity] | None = None,
) -> Iterator[PortfolioLedgerBatch]:
    bounds = tuple(pd.Timestamp(research_window[key]).value for key in ("start", "end"))
    with closing_iterator(ExitStack()) as stack:
        key_index = ExactPrimaryKeyIndex(primary_key=("fill_id",))
        readers = {
            name: stack.enter_context(
                closing_iterator(
                    _table_records(
                        root,
                        name,
                        None if identities is None else identities[f"{name}.parquet"],
                    )
                )
            )
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
            tables = {}
            for name, kind in _TABLE_TYPES.items():
                group = pending[name]
                if group is not None and group[0] < sequence:
                    raise ValueError("portfolio_ledger_sequence_invalid")
                records = []
                if group is not None and group[0] == sequence:
                    for row in group[1]:
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
            previous = validate_portfolio_transition(batch, capital, previous)
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
                if diagnostic.fill_id is not None and key_index.identity_payload_value(
                    namespace="fills", payload=diagnostic.fill_id.encode("utf-8")
                ) != diagnostic.asset_id.encode("utf-8"):
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
                        raise ValueError("portfolio_ledger_diagnostic_order_mismatch")
            inserted = key_index.add_identity_payloads(
                namespace="rights_adjustments",
                payloads=(
                    (row.event_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                    for row in batch.rights_adjustments
                ),
            )
            if inserted != len(batch.rights_adjustments):
                raise ValueError("portfolio_ledger_duplicate_rights_adjustment")
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
