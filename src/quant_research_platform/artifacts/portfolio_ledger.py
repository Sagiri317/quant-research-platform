"""One immutable typed economic ledger for every research portfolio."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from itertools import groupby
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.compute.portfolio_economics import _LedgerEconomics
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
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
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    decode_exact_numeric,
    encode_exact_numeric,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    open_filesystem_file,
)
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
    write_manifest,
)
from quant_research_platform.storage.parquet import (
    iter_dataset_part_batches,
    parquet_table_writer,
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
    "locked_quantity",
    "locked_quantity_before",
    "locked_quantity_after",
    "entitlement_quantity",
}
_NUMERIC = {
    "notional",
    "amount",
    "value",
    "number",
    "cost_basis",
    "cash_amount",
    "receivable",
    "receivable_before",
    "receivable_after",
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
            if column in {"sequence", "timestamp_ns", "entitlement_record_at_ns"}
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
    metadata: dict
    root: Path
    capital: PortfolioCapital
    _final_alpha_calibration_member: str | None = None

    @property
    def final_alpha_calibration(self) -> Mapping[str, object] | None:
        """Load optional final calibration only when explicitly consumed."""
        if self._final_alpha_calibration_member is None:
            return None
        return read_qrp_json_file(self.root / self._final_alpha_calibration_member)[1]

    def iter_chunks(self) -> Iterator[PortfolioLedgerBatch]:
        yield from _iter_batches(self.root, self.capital, self.metadata["window"])


def write_portfolio_ledger_artifact(
    logical_root: Path,
    *,
    capital: PortfolioCapital,
    metadata: Mapping[str, object],
    produce: Callable[[Callable[[PortfolioLedgerBatch], None]], object],
    final_calibration: Callable[[], dict[str, object]] | None = None,
) -> Path:
    """Validate canonical economic batches once, then atomically publish tables."""
    if not isinstance(capital, PortfolioCapital):
        raise ValueError("portfolio_ledger_capital_invalid")
    metadata = dict(metadata)
    if "window" not in metadata:
        raise ValueError("portfolio_ledger_window_missing")

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
            validation = _LedgerValidation(capital, metadata["window"])

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
                nonlocal sequence
                validation.observe(batch)
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
        calibration = (
            dict(final_calibration()) if final_calibration is not None else None
        )
        if calibration is not None:
            write_manifest(staging / "final_alpha_calibration.json", calibration)
        write_manifest(
            staging / "manifest.json",
            {
                "schema_id": "qrp-portfolio-ledger",
                "capital": asdict(capital),
                "metadata": metadata,
                "final_alpha_calibration": "final_alpha_calibration.json"
                if calibration is not None
                else None,
                "members": [f"{name}.parquet" for name in _TABLE_TYPES],
            },
        )

    atomic_write_directory(Path(logical_root), write)
    return Path(logical_root)


def read_portfolio_ledger_artifact(
    instance_root: Path,
    *,
    consume_batches: Callable[
        [PortfolioLedgerArtifact, Iterator[PortfolioLedgerBatch]], object
    ]
    | None = None,
) -> PortfolioLedgerArtifact:
    """Read native metadata; consume requested economics once without pre-scans."""
    root = Path(instance_root)
    _, manifest = read_qrp_json_file(root / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_id") != "qrp-portfolio-ledger"
        or manifest.get("members") != [f"{name}.parquet" for name in _TABLE_TYPES]
        or not isinstance(manifest.get("metadata"), dict)
        or "window" not in manifest["metadata"]
        or manifest.get("final_alpha_calibration")
        not in {None, "final_alpha_calibration.json"}
    ):
        raise ValueError("portfolio_ledger_manifest_invalid")
    artifact = PortfolioLedgerArtifact(
        manifest["metadata"],
        root,
        PortfolioCapital(**manifest["capital"]),
        manifest.get("final_alpha_calibration"),
    )
    if consume_batches is not None:
        with closing_iterator(artifact.iter_chunks()) as batches:
            consume_batches(artifact, batches)
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
    account_stop_reason = None
    result = {
        "candidate_id": capital.candidate_id,
        "initial_capital": capital.initial_capital,
        "valuation_currency": capital.valuation_currency,
        **economics.summary(),
    }
    with closing_iterator(batches):
        for batch in batches:
            for row in batch.diagnostics:
                if row.name == "execution_summary" and row.text is not None:
                    execution = parse_qrp_json_object(row.text.encode("utf-8"))
                    if execution.get("account_path_status") == "PARTIAL":
                        account_stop_reason = execution["account_stop_reason"]
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
    if account_stop_reason is not None:
        result["account_path_status"] = "PARTIAL"
        result["observed_totals_scope"] = "known_prefix"
        for name in ("gross_pnl", "net_pnl", "nav", "capital_return", "funding"):
            result[name] = None
            result[f"{name}_reason"] = account_stop_reason
    return result


def _table_records(root: Path, name: str) -> Iterator[dict[str, object]]:
    def validate_schema(metadata):
        if not metadata.schema.equals(_SCHEMAS[name]):
            raise ValueError("portfolio_ledger_schema_invalid")

    chunks = iter_dataset_part_batches(
        root / f"{name}.parquet", metadata_observer=validate_schema
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


class _LedgerValidation:
    """Validate the same canonical stream at publication and consumption."""

    def __init__(self, capital, research_window):
        self.capital = capital
        start, end, closed = time_window_timestamp_interval(
            TimeWindow(research_window["start"], research_window["end"])
        )
        self.bounds = (
            pd.Timestamp(start).value,
            pd.Timestamp(end).value - (0 if closed else 1),
        )
        self.key_index = ExactPrimaryKeyIndex(primary_key=("fill_id",))
        self.sequence = 0
        self.previous = None

    def observe(self, batch):
        if type(batch) is not PortfolioLedgerBatch or batch.sequence != self.sequence:
            raise ValueError("portfolio_ledger_sequence_invalid")
        if any(
            not self.bounds[0] <= row.timestamp_ns <= self.bounds[1]
            for row in batch.valuations
        ):
            raise ValueError("portfolio_ledger_valuation_window_invalid")
        self.previous = validate_portfolio_transition(
            batch, self.capital, self.previous
        )
        inserted = self.key_index.add_identity_payloads(
            namespace="fills",
            payloads=(
                (row.fill_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                for row in batch.trades
            ),
        )
        if inserted != len(batch.trades):
            raise ValueError("portfolio_ledger_duplicate_fill")
        for diagnostic in batch.diagnostics:
            if diagnostic.fill_id is not None and self.key_index.identity_payload_value(
                namespace="fills", payload=diagnostic.fill_id.encode("utf-8")
            ) != diagnostic.asset_id.encode("utf-8"):
                raise ValueError("portfolio_ledger_diagnostic_fill_mismatch")
        for event in batch.rights_adjustments:
            if event.kind == "recovery":
                continue
            identity = event.event_id.rsplit(":", 1)[0].encode("utf-8")
            payload = "|".join(
                (
                    event.asset_id,
                    event.lifecycle_id,
                    event.evidence_identity,
                    encode_exact_numeric(Fraction(event.entitlement_quantity)),
                    str(event.entitlement_record_at_ns),
                )
            ).encode("utf-8")
            if event.kind == "registration":
                if (
                    self.key_index.add_identity_payloads(
                        namespace="entitlement_registration",
                        payloads=((identity, payload),),
                    )
                    != 1
                ):
                    raise ValueError("portfolio_registration_duplicate")
            elif (
                self.key_index.identity_payload_value(
                    namespace="entitlement_registration", payload=identity
                )
                != payload
            ):
                raise ValueError("portfolio_registration_entitlement_invalid")
        inserted = self.key_index.add_identity_payloads(
            namespace="rights_adjustments",
            payloads=(
                (row.event_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                for row in batch.rights_adjustments
            ),
        )
        if inserted != len(batch.rights_adjustments):
            raise ValueError("portfolio_ledger_duplicate_rights_adjustment")
        inserted = self.key_index.add_identity_payloads(
            namespace="cashflows",
            payloads=(
                (row.flow_id.encode("utf-8"), row.asset_id.encode("utf-8"))
                for row in batch.cashflows
            ),
        )
        if inserted != len(batch.cashflows):
            raise ValueError("portfolio_ledger_duplicate_cashflow")
        self.sequence += 1


def _iter_batches(
    root: Path,
    capital: PortfolioCapital,
    research_window: Mapping[str, object],
) -> Iterator[PortfolioLedgerBatch]:
    validation = _LedgerValidation(capital, research_window)
    with closing_iterator(ExitStack()) as stack:
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
            validation.observe(batch)
            try:
                yield batch
            except GeneratorExit:
                return
            sequence += 1
        if not sequence or any(group is not None for group in pending.values()):
            raise ValueError("portfolio_ledger_sequence_invalid")
