"""Single-file portfolio ledger artifact."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.artifacts.outputs.parquet import (
    _artifact_frame_reservation,
    _artifact_table_write_state,
    _artifact_writer_fixed_reservation,
)
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_BLOCK_REASONS,
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_DECISION_REASONS,
    PORTFOLIO_LEDGER_COLUMNS,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    PORTFOLIO_LEDGER_KEY,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    verify_semantic_artifact_member,
    write_semantic_artifact,
)

_LEDGER_MEMBER = "ledger.parquet"
_DECISION_KEY_COLUMN = PORTFOLIO_LEDGER_KEY[0]
_ASSET_KEY_COLUMN = PORTFOLIO_LEDGER_KEY[1]
_TIMESTAMP_COLUMNS = ("decision_timestamp", "execution_date")
_NOTIONAL_COLUMNS = (
    "pre_trade_notional",
    "target_notional",
    "post_trade_notional",
    "end_notional",
)
_SIGNED_COLUMNS = ("executed_delta_notional",)
_OPTIONAL_NUMERIC_COLUMNS = (
    "adv20",
    "alpha_score",
    "expected_alpha",
    "benchmark_weight",
    "asset_specific_variance",
    "alpha_slope",
    "alpha_slope_se",
    "alpha_slope_t",
    "alpha_slope_pvalue",
    "alpha_slope_ci_low",
    "alpha_slope_ci_high",
    "alpha_weighted_r2",
    "alpha_coverage",
    "alpha_valid_dates",
    "alpha_scheduled_dates",
    "predicted_tracking_error",
    "portfolio_factor_variance",
    "portfolio_specific_variance",
    "expected_transaction_cost",
)
_NUMERIC_COLUMNS = (
    *_NOTIONAL_COLUMNS,
    *_SIGNED_COLUMNS,
    "holding_return",
    *PORTFOLIO_LEDGER_COST_COLUMNS,
    *_OPTIONAL_NUMERIC_COLUMNS,
)
_STRING_COLUMNS = ("asset_id", "block_reason", "decision_reason")
_COUNT_COLUMNS = ("alpha_valid_dates", "alpha_scheduled_dates")
_PER_ASSET_NUMERIC_COLUMNS = (
    "adv20",
    "alpha_score",
    "expected_alpha",
    "benchmark_weight",
    "asset_specific_variance",
)
_CASH_DIAGNOSTIC_COLUMNS = tuple(
    column
    for column in _OPTIONAL_NUMERIC_COLUMNS
    if column not in _PER_ASSET_NUMERIC_COLUMNS
)
_ARROW_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            pa.timestamp("ns")
            if column in _TIMESTAMP_COLUMNS
            else pa.int64()
            if column in _COUNT_COLUMNS
            else pa.string()
            if column in _STRING_COLUMNS
            else pa.float64(),
        )
        for column in PORTFOLIO_LEDGER_COLUMNS
    ]
)


@dataclass(frozen=True, slots=True)
class PortfolioLedgerArtifact:
    """Validated ledger location; rows are read one row group at a time."""

    root: Path
    identity: str
    _member: Mapping[str, object]

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        yield from _iter_validated_chunks(
            self.root,
            expected_identity=self.identity,
            member=self._member,
        )


def write_portfolio_ledger_artifact(
    logical_root: Path,
    *,
    envelope: BusinessManifestEnvelope,
    ledger_chunks: Iterable[pd.DataFrame],
) -> Path:
    """Publish one parquet ledger with one complete session per row group."""

    def write(staging: Path) -> None:
        path = staging / _LEDGER_MEMBER
        writer: pq.ParquetWriter | None = None
        writer_schema: pa.Schema | None = None
        fixed_state = None
        table_states = []
        retained_memory = (0, 0)
        previous_metadata = None
        previous_end: dict[str, float] | None = None
        previous_execution: pd.Timestamp | None = None
        primary: BaseException | None = None
        try:
            fixed_state = _artifact_writer_fixed_reservation(
                writer_count=1,
                index_bytes=0,
            )
            with closing_iterator(iter(ledger_chunks)) as chunks:
                for chunk in chunks:
                    if type(chunk) is not pd.DataFrame:
                        raise TypeError("portfolio_ledger_chunk_invalid")
                    with _artifact_frame_reservation(
                        chunk,
                        key_columns=PORTFOLIO_LEDGER_KEY,
                    ):
                        frame, _decision_timestamp, execution_date, previous_end = (
                            _validated_frame(
                                chunk,
                                previous_end=previous_end,
                                previous_execution=previous_execution,
                            )
                        )
                        table = pa.Table.from_pandas(
                            frame,
                            schema=_ARROW_SCHEMA,
                            preserve_index=False,
                        )
                        (
                            persisted,
                            table_state,
                            batch_rows,
                            retained_memory,
                        ) = _artifact_table_write_state(
                            table,
                            previous_metadata=previous_metadata,
                            retained_memory=retained_memory,
                        )
                        table_states.append(table_state)
                        if batch_rows < len(frame):
                            raise MemoryError(
                                "portfolio_session_exceeds_task_memory_budget"
                            )
                        previous_metadata = dict(persisted.schema.metadata or {})
                        if writer is None:
                            writer = pq.ParquetWriter(
                                path,
                                persisted.schema,
                                compression="zstd",
                                use_dictionary=False,
                                write_statistics=False,
                                write_page_index=False,
                                write_page_checksum=False,
                            )
                            writer_schema = persisted.schema
                        elif (
                            writer_schema is None
                            or persisted.schema.names != writer_schema.names
                            or tuple(field.type for field in persisted.schema)
                            != tuple(field.type for field in writer_schema)
                        ):
                            raise ValueError("portfolio_ledger_schema_invalid")
                        writer.write_table(
                            persisted,
                            row_group_size=batch_rows,
                        )
                        previous_execution = execution_date
            if writer is None:
                raise ValueError("portfolio_ledger_empty")
        except BaseException as exc:
            primary = exc

        cleanup_error: BaseException | None = None

        if writer is not None:
            try:
                writer.close()
            except BaseException as close_error:
                if primary is None:
                    cleanup_error = close_error
                else:
                    primary.add_note(
                        "portfolio ledger parquet close also failed: "
                        f"{type(close_error).__name__}"
                    )
        for table_state in table_states:
            try:
                table_state.close()
            except BaseException as state_error:
                if primary is None and cleanup_error is None:
                    cleanup_error = state_error
                elif primary is not None:
                    primary.add_note(
                        "portfolio ledger table state release also failed: "
                        f"{type(state_error).__name__}"
                    )
                else:
                    cleanup_error.add_note(
                        "portfolio ledger table state release also failed: "
                        f"{type(state_error).__name__}"
                    )
        if fixed_state is not None:
            try:
                fixed_state.close()
            except BaseException as state_error:
                if primary is None and cleanup_error is None:
                    cleanup_error = state_error
                elif primary is not None:
                    primary.add_note(
                        "portfolio ledger fixed state release also failed: "
                        f"{type(state_error).__name__}"
                    )
                else:
                    cleanup_error.add_note(
                        "portfolio ledger fixed state release also failed: "
                        f"{type(state_error).__name__}"
                    )
        if primary is not None:
            raise primary
        if cleanup_error is not None:
            raise cleanup_error

    return write_semantic_artifact(Path(logical_root), envelope, write)


def read_portfolio_ledger_artifact(
    instance_root: Path,
    *,
    expected_identity: str,
) -> PortfolioLedgerArtifact:
    """Read and fully validate a published ledger without concatenating it."""

    _, members = read_semantic_artifact_record(
        Path(instance_root),
        expected_identity=expected_identity,
    )
    matching = tuple(
        member for member in members if member.get("path") == _LEDGER_MEMBER
    )
    if len(members) != 1 or len(matching) != 1:
        raise ValueError("portfolio_ledger_member_invalid")
    artifact = PortfolioLedgerArtifact(
        root=Path(instance_root),
        identity=expected_identity,
        _member=MappingProxyType(dict(matching[0])),
    )
    for _frame in artifact.iter_chunks():
        pass
    return artifact


def _iter_validated_chunks(
    root: Path,
    *,
    expected_identity: str,
    member: dict[str, object],
) -> Iterator[pd.DataFrame]:
    if root.name != expected_identity:
        raise ValueError("portfolio_ledger_identity_invalid")
    _, current_members = read_semantic_artifact_record(
        root,
        expected_identity=expected_identity,
    )
    if len(current_members) != 1 or dict(current_members[0]) != dict(member):
        raise ValueError("portfolio_ledger_member_invalid")
    verify_semantic_artifact_member(
        root,
        relative_path=_LEDGER_MEMBER,
        expected=dict(member),
    )
    path = Path(root) / _LEDGER_MEMBER
    parquet = pq.ParquetFile(native_filesystem_path(path))
    previous_end: dict[str, float] | None = None
    previous_execution: pd.Timestamp | None = None
    primary: BaseException | None = None
    try:
        if parquet.schema_arrow != _ARROW_SCHEMA:
            raise ValueError("portfolio_ledger_schema_invalid")
        if parquet.metadata is None or parquet.metadata.num_row_groups <= 0:
            raise ValueError("portfolio_ledger_empty")
        for row_group in range(parquet.metadata.num_row_groups):
            frame = parquet.read_row_group(row_group).to_pandas()
            frame, _decision, execution, previous_end = _validated_frame(
                frame,
                previous_end=previous_end,
                previous_execution=previous_execution,
            )
            previous_execution = execution
            yield frame
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close = getattr(parquet, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "portfolio ledger parquet close also failed: "
                    f"{type(close_error).__name__}"
                )
        try:
            verify_semantic_artifact_member(
                root,
                relative_path=_LEDGER_MEMBER,
                expected=dict(member),
            )
        except BaseException as verify_error:
            if primary is None:
                raise
            primary.add_note(
                "portfolio ledger member verification also failed: "
                f"{type(verify_error).__name__}"
            )


def _validated_frame(
    frame: pd.DataFrame,
    *,
    previous_end: dict[str, float] | None,
    previous_execution: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, dict[str, float]]:
    if (
        type(frame) is not pd.DataFrame
        or tuple(frame.columns) != PORTFOLIO_LEDGER_COLUMNS
    ):
        raise ValueError("portfolio_ledger_schema_invalid")
    result = frame.copy(deep=True)
    for column in _TIMESTAMP_COLUMNS:
        try:
            result[column] = pd.to_datetime(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("portfolio_ledger_timestamp_invalid") from exc
        if result[column].isna().any():
            raise ValueError("portfolio_ledger_timestamp_invalid")
        result[column] = result[column].dt.normalize()
    if result.empty:
        raise ValueError("portfolio_ledger_session_empty")
    decision_values = result[_DECISION_KEY_COLUMN].drop_duplicates()
    execution_values = result["execution_date"].drop_duplicates()
    if len(decision_values) != 1 or len(execution_values) != 1:
        raise ValueError("portfolio_ledger_session_boundary_invalid")
    decision_timestamp = pd.Timestamp(decision_values.iloc[0])
    execution_date = pd.Timestamp(execution_values.iloc[0])
    if execution_date <= decision_timestamp:
        raise ValueError("portfolio_ledger_execution_timing_invalid")
    if previous_execution is not None and decision_timestamp != previous_execution:
        raise ValueError("portfolio_ledger_cross_session_timing_invalid")

    if result[_ASSET_KEY_COLUMN].duplicated().any():
        raise ValueError("portfolio_ledger_primary_key_invalid")
    if not bool(
        result["asset_id"].map(lambda value: type(value) is str and bool(value)).all()
    ):
        raise ValueError("portfolio_ledger_primary_key_invalid")
    assets = result.loc[
        result["asset_id"] != PORTFOLIO_CASH_ASSET_ID, "asset_id"
    ].tolist()
    if assets != sorted(assets) or PORTFOLIO_CASH_ASSET_ID not in set(
        result["asset_id"]
    ):
        raise ValueError("portfolio_ledger_canonical_order_invalid")
    if result["asset_id"].iloc[-1] != PORTFOLIO_CASH_ASSET_ID:
        raise ValueError("portfolio_ledger_cash_row_invalid")
    if int(result["asset_id"].eq(PORTFOLIO_CASH_ASSET_ID).sum()) != 1:
        raise ValueError("portfolio_ledger_cash_row_invalid")

    for column in _NUMERIC_COLUMNS:
        raw_values = result[column].dropna()
        if raw_values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError("portfolio_ledger_numeric_invalid")
        try:
            numeric = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("portfolio_ledger_numeric_invalid") from exc
        values = numeric.dropna().to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("portfolio_ledger_numeric_invalid")
        if column in _COUNT_COLUMNS:
            if (values < 0.0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError("portfolio_ledger_count_invalid")
            result[column] = numeric.astype("Int64")
        else:
            result[column] = numeric.astype("float64")
    for column in _STRING_COLUMNS:
        if column == "asset_id":
            continue
        values = result[column].dropna()
        if not bool(values.map(lambda value: type(value) is str and bool(value)).all()):
            raise ValueError("portfolio_ledger_string_invalid")
    if set(result["block_reason"].dropna()) - PORTFOLIO_BLOCK_REASONS:
        raise ValueError("portfolio_ledger_block_reason_invalid")
    if set(result["decision_reason"].dropna()) - PORTFOLIO_DECISION_REASONS:
        raise ValueError("portfolio_ledger_decision_reason_invalid")

    if (result.loc[:, list(_NOTIONAL_COLUMNS)] < 0.0).any().any():
        raise ValueError("portfolio_ledger_notional_invalid")
    if result["holding_return"].lt(-1.0).any():
        raise ValueError("portfolio_ledger_return_invalid")
    if result["adv20"].dropna().lt(0.0).any():
        raise ValueError("portfolio_ledger_adv_invalid")
    if (result.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)] < 0.0).any().any():
        raise ValueError("portfolio_ledger_cost_invalid")

    nav_before = float(result["pre_trade_notional"].sum())
    if not np.isfinite(nav_before) or nav_before <= 0.0:
        raise ValueError("portfolio_ledger_nav_invalid")
    tolerance = _scaled_tolerance(nav_before)
    desired = result["target_notional"] - result["pre_trade_notional"]
    executed = result["executed_delta_notional"]
    cash = result["asset_id"].eq(PORTFOLIO_CASH_ASSET_ID)
    asset_rows = ~cash
    if (
        (
            ((desired > 0.0) & (executed < 0.0))
            | ((desired < 0.0) & (executed > 0.0))
            | ((desired == 0.0) & (executed != 0.0))
            | (executed.abs() > desired.abs() + tolerance)
        )
        & asset_rows
    ).any():
        raise ValueError("portfolio_ledger_direction_invalid")
    if not np.allclose(
        result["pre_trade_notional"] + executed,
        result["post_trade_notional"],
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("portfolio_ledger_state_invalid")
    if not np.allclose(
        result["post_trade_notional"] * (1.0 + result["holding_return"]),
        result["end_notional"],
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("portfolio_ledger_return_state_invalid")

    assets_frame = result.loc[~cash]
    total_cost = float(result.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum())
    if (
        result.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)]
        .gt(0.0)
        .any(axis=1)
        .loc[executed.eq(0.0)]
        .any()
    ):
        raise ValueError("portfolio_ledger_phantom_cost_invalid")
    mismatch = (
        result["target_notional"] - result["post_trade_notional"]
    ).abs() > tolerance
    if not bool((result["block_reason"].notna() == (mismatch & asset_rows)).all()):
        raise ValueError("portfolio_ledger_block_reason_invalid")
    if result.loc[cash, "block_reason"].notna().any():
        raise ValueError("portfolio_ledger_block_reason_invalid")
    if not bool(result.loc[cash, "holding_return"].eq(0.0).all()):
        raise ValueError("portfolio_ledger_cash_return_invalid")
    if (result.loc[cash, list(PORTFOLIO_LEDGER_COST_COLUMNS)] != 0.0).any().any():
        raise ValueError("portfolio_ledger_cash_cost_invalid")
    if result.loc[cash, list(_PER_ASSET_NUMERIC_COLUMNS)].notna().any().any():
        raise ValueError("portfolio_ledger_cash_diagnostic_invalid")
    if result.loc[~cash, list(_CASH_DIAGNOSTIC_COLUMNS)].notna().any().any():
        raise ValueError("portfolio_ledger_decision_diagnostic_invalid")
    if result.loc[~cash, "decision_reason"].notna().any():
        raise ValueError("portfolio_ledger_decision_reason_invalid")
    if result.loc[cash, "decision_reason"].isna().any():
        raise ValueError("portfolio_ledger_decision_reason_invalid")
    executed_assets = asset_rows & executed.ne(0.0)
    if (executed_assets & (result["adv20"].isna() | result["adv20"].le(0.0))).any():
        raise ValueError("portfolio_ledger_adv_invalid")

    if not np.isclose(
        result["target_notional"].sum(),
        nav_before,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("portfolio_ledger_target_state_invalid")
    if not np.isclose(
        result["post_trade_notional"].sum(),
        nav_before - total_cost,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("portfolio_ledger_self_financing_invalid")
    cash_delta = float(result.loc[cash, "executed_delta_notional"].iloc[0])
    asset_delta = float(assets_frame["executed_delta_notional"].sum())
    if not np.isclose(
        cash_delta,
        -asset_delta - total_cost,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("portfolio_ledger_cash_reconciliation_invalid")

    if previous_end is not None:
        current = dict(
            zip(
                result["asset_id"].astype(str),
                result["pre_trade_notional"].to_numpy(dtype="float64"),
                strict=True,
            )
        )
        for asset in set(previous_end) | set(current):
            if not np.isclose(
                previous_end.get(asset, 0.0),
                current.get(asset, 0.0),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("portfolio_ledger_cross_session_state_invalid")
    next_end = dict(
        zip(
            result["asset_id"].astype(str),
            result["end_notional"].to_numpy(dtype="float64"),
            strict=True,
        )
    )
    return result, decision_timestamp, execution_date, next_end


def _scaled_tolerance(nav_before: float) -> float:
    return max(np.finfo("float64").eps * max(abs(nav_before), 1.0) * 512.0, 1e-12)
