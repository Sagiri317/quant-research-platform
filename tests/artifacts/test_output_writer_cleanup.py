from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._parquet_writers as parquet_writers
import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_portfolio_artifacts,
    make_output_edge_portfolio_spec,
    make_output_edge_report_spec,
    write_output_edge_portfolio_chunks,
)

_TERMINAL_PRIMARY_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
)

_WRITER_BODY_PRIMARY_TYPES = (*_TERMINAL_PRIMARY_TYPES, GeneratorExit)


class _TerminalInput:
    def __init__(
        self,
        *,
        values: tuple[object, ...] = (),
        primary: BaseException | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        self._values = iter(values)
        self.primary = primary
        self.close_failure = close_failure
        self.close_calls = 0

    def __iter__(self) -> _TerminalInput:
        return self

    def __next__(self) -> object:
        if self.primary is not None:
            raise self.primary
        return next(self._values)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


def _run_terminal_writer(
    owner: str,
    source: _TerminalInput,
    tmp_path: Path,
) -> object:
    if owner == "prediction":
        return parquet_writers.write_parquet_chunks(
            source,  # type: ignore[arg-type]
            tmp_path / "prediction.parquet",
            required=("timestamp", "value"),
            key_columns=("timestamp",),
            compression="snappy",
        )
    return portfolio_outputs._write_portfolio_row_artifact_chunks(
        source,  # type: ignore[arg-type]
        temp_root=tmp_path,
        row_artifacts=(),
        required_columns={},
        spec=object(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("owner", ("prediction", "portfolio"))
@pytest.mark.parametrize("primary_type", _TERMINAL_PRIMARY_TYPES)
def test_terminal_writer_closes_input_once_without_replacing_primary(
    owner: str,
    primary_type: type[BaseException],
    tmp_path: Path,
) -> None:
    primary = primary_type("body_primary")
    source = _TerminalInput(
        primary=primary,
        close_failure=OSError("close_secondary"),
    )

    with pytest.raises(primary_type) as raised:
        _run_terminal_writer(owner, source, tmp_path)

    assert raised.value is primary
    assert source.close_calls == 1


@pytest.mark.parametrize("owner", ("prediction", "portfolio"))
def test_terminal_writer_propagates_close_only_failure_once(
    owner: str,
    tmp_path: Path,
) -> None:
    close_failure = OSError("close_failed")
    source = _TerminalInput(
        values=(
            pd.DataFrame(
                {"timestamp": [pd.Timestamp("2024-01-01")], "value": [1.0]}
            ),
        )
        if owner == "prediction"
        else (),
        close_failure=close_failure,
    )

    with pytest.raises(OSError) as raised:
        _run_terminal_writer(owner, source, tmp_path)

    assert raised.value is close_failure
    assert source.close_calls == 1


@pytest.mark.parametrize("owner", ("prediction", "portfolio"))
def test_terminal_writer_preserves_validation_primary_over_input_close(
    owner: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary = ValueError("validation_primary")
    source = _TerminalInput(
        values=(pd.DataFrame(),) if owner == "prediction" else ({},),
        close_failure=OSError("close_secondary"),
    )
    if owner == "prediction":
        monkeypatch.setattr(
            parquet_writers,
            "_prediction_output_chunk_frame",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )
    else:
        monkeypatch.setattr(
            portfolio_outputs,
            "validate_portfolio_artifact_mapping_keys",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        )

    with pytest.raises(ValueError) as raised:
        _run_terminal_writer(owner, source, tmp_path)

    assert raised.value is primary
    assert source.close_calls == 1


def test_prediction_path_preflight_does_not_activate_terminal_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    activations = 0

    class Input:
        def __iter__(self) -> object:
            nonlocal activations
            activations += 1
            raise AssertionError("input_activated_before_path_preflight")

    preflight_failure = FileExistsError("output_exists")
    monkeypatch.setattr(
        parquet_writers,
        "reject_if_exists",
        lambda _path: (_ for _ in ()).throw(preflight_failure),
    )

    with pytest.raises(FileExistsError) as raised:
        parquet_writers.write_parquet_chunks(
            Input(),  # type: ignore[arg-type]
            tmp_path / "prediction.parquet",
            required=("timestamp", "value"),
            key_columns=("timestamp",),
            compression="snappy",
        )

    assert raised.value is preflight_failure
    assert activations == 0


def test_prediction_writer_close_is_attempted_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    close_failure = OSError("writer_close_failed")

    class KeyIndex:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> KeyIndex:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def add_table(self, _table: object) -> int:
            return 0

    class Writer:
        close_calls = 0

        def write_table(self, _table: object, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1
            raise close_failure

    writer = Writer()
    monkeypatch.setattr(parquet_writers, "ExactPrimaryKeyIndex", KeyIndex)
    monkeypatch.setattr(
        parquet_writers.pq,
        "ParquetWriter",
        lambda *_args, **_kwargs: writer,
    )
    frames = (
        pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "value": [1.0]})
        for _ in range(2)
    )

    with pytest.raises(OSError) as raised:
        parquet_writers.write_parquet_chunks(
            frames,
            tmp_path / "prediction.parquet",
            required=("timestamp", "value"),
            key_columns=("timestamp",),
            compression="snappy",
        )

    assert raised.value is close_failure
    assert writer.close_calls == 1


@pytest.mark.parametrize("failure_position", (0, 1, 2))
def test_portfolio_writers_close_once_and_attempt_all(
    failure_position: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    close_failure = OSError("writer_close_failed")

    class Writer:
        def __init__(self, position: int) -> None:
            self.position = position
            self.close_calls = 0

        def write_table(self, _table: object, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            self.close_calls += 1
            if self.position == failure_position:
                raise close_failure

    writers: list[Writer] = []

    def writer_factory(*_args: object, **_kwargs: object) -> Writer:
        writer = Writer(len(writers))
        writers.append(writer)
        return writer

    row_artifacts = tuple(portfolio_outputs.PORTFOLIO_ARTIFACTS)[:3]
    frame = pd.DataFrame({"value": [1.0]})
    monkeypatch.setattr(
        portfolio_outputs,
        "validate_portfolio_artifact_mapping_keys",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        portfolio_outputs,
        "_portfolio_output_chunk_frame",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(portfolio_outputs.pq, "ParquetWriter", writer_factory)

    with pytest.raises(OSError) as raised:
        portfolio_outputs._write_portfolio_row_artifact_chunks(
            [{name: frame for name in row_artifacts}],
            temp_root=tmp_path,
            row_artifacts=row_artifacts,
            required_columns={name: ("value",) for name in row_artifacts},
            spec=object(),  # type: ignore[arg-type]
        )

    assert raised.value is close_failure
    assert [writer.close_calls for writer in writers] == [1, 1, 1]


@pytest.mark.parametrize("owner", ("prediction", "portfolio"))
@pytest.mark.parametrize("primary_type", _WRITER_BODY_PRIMARY_TYPES)
def test_parquet_writers_close_once_without_replacing_body_primary(
    owner: str,
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary = primary_type("body_primary")
    close_failure = OSError("writer_close_secondary")
    row_artifacts = tuple(portfolio_outputs.PORTFOLIO_ARTIFACTS)[:3]
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "value": [1.0]})
    first = (
        frame
        if owner == "prediction"
        else {name: frame[["value"]] for name in row_artifacts}
    )

    class BodyPrimaryInput:
        yielded = False

        def __iter__(self) -> BodyPrimaryInput:
            return self

        def __next__(self) -> object:
            if not self.yielded:
                self.yielded = True
                return first
            raise primary

        def close(self) -> None:
            pass

    class KeyIndex:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> KeyIndex:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def add_table(self, _table: object) -> int:
            return 0

    class Writer:
        def __init__(self) -> None:
            self.close_calls = 0
            self.write_calls = 0

        def write_table(self, _table: object, **_kwargs: object) -> None:
            self.write_calls += 1

        def close(self) -> None:
            self.close_calls += 1
            raise close_failure

    writers: list[Writer] = []

    def writer_factory(*_args: object, **_kwargs: object) -> Writer:
        writer = Writer()
        writers.append(writer)
        return writer

    if owner == "prediction":
        monkeypatch.setattr(parquet_writers, "ExactPrimaryKeyIndex", KeyIndex)
        monkeypatch.setattr(parquet_writers.pq, "ParquetWriter", writer_factory)
    else:
        monkeypatch.setattr(
            portfolio_outputs,
            "validate_portfolio_artifact_mapping_keys",
            lambda value, **_kwargs: value,
        )
        monkeypatch.setattr(
            portfolio_outputs,
            "_portfolio_output_chunk_frame",
            lambda value, **_kwargs: value,
        )
        monkeypatch.setattr(portfolio_outputs.pq, "ParquetWriter", writer_factory)

    with pytest.raises(primary_type) as raised:
        if owner == "prediction":
            parquet_writers.write_parquet_chunks(
                BodyPrimaryInput(),  # type: ignore[arg-type]
                tmp_path / "prediction.parquet",
                required=("timestamp", "value"),
                key_columns=("timestamp",),
                compression="snappy",
            )
        else:
            portfolio_outputs._write_portfolio_row_artifact_chunks(
                BodyPrimaryInput(),  # type: ignore[arg-type]
                temp_root=tmp_path,
                row_artifacts=row_artifacts,
                required_columns={name: ("value",) for name in row_artifacts},
                spec=object(),  # type: ignore[arg-type]
            )

    assert raised.value is primary
    assert len(writers) == (1 if owner == "prediction" else 3)
    assert all(writer.write_calls == 1 for writer in writers)
    assert all(writer.close_calls == 1 for writer in writers)


def test_portfolio_outputs_clean_temp_bundle_directory_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    layout = StorageLayout(project_root=tmp_path)
    report_spec = make_output_edge_report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    run_parent = tmp_path / "outputs"

    def fail_write(frame, path, **kwargs):  # noqa: ANN001
        del frame, kwargs
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("portfolio parquet failed")

    monkeypatch.setattr(portfolio_outputs, "write_parquet", fail_write)

    with pytest.raises(RuntimeError, match="portfolio parquet failed"):
        write_output_edge_portfolio_chunks(
            make_output_edge_portfolio_artifacts(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    assert not (run_parent / report_spec.output_id).exists()
    assert not list(run_parent.glob(".tmp-*"))


def test_report_parquet_writer_cleans_temp_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1]})

    def fail_to_parquet(_table, sink, **kwargs):  # noqa: ANN001
        del kwargs
        sink.write(b"temporary")
        raise RuntimeError("write failed")

    monkeypatch.setattr(parquet_writers.pq, "write_table", fail_to_parquet)

    with pytest.raises(RuntimeError, match="write failed"):
        parquet_writers.write_parquet(
            frame,
            tmp_path / "report.parquet",
        )
    assert not list(tmp_path.glob(".f-*"))


def test_report_parquet_writer_uses_short_temp_for_long_final_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / ("p" * 230 + ".parquet")
    publications: list[tuple[Path, Path]] = []

    def publish(
        source: Path,
        target: Path,
        *,
        source_identity: object,
    ) -> None:
        publications.append((source, target))
        scratch_owner.remove_owned_scratch(source, source_identity)

    monkeypatch.setattr(parquet_writers, "publish_no_replace", publish)

    parquet_writers.write_parquet(pd.DataFrame({"value": [1]}), final_path)

    assert len(final_path.name) == 238
    assert len(publications) == 1
    assert publications[0][1] == final_path
    assert len(publications[0][0].name) < 40
    assert not list(tmp_path.glob(".f-*"))


def test_report_parquet_writer_cleans_temp_on_control_flow_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def cancel_to_parquet(_table, sink, **kwargs):  # noqa: ANN001
        del kwargs
        sink.write(b"temporary")
        raise KeyboardInterrupt

    monkeypatch.setattr(parquet_writers.pq, "write_table", cancel_to_parquet)

    with pytest.raises(KeyboardInterrupt):
        parquet_writers.write_parquet(
            pd.DataFrame({"value": [1]}),
            tmp_path / "report.parquet",
        )

    assert not list(tmp_path.glob(".f-*"))


def test_report_parquet_writer_does_not_overwrite_unclaimed_temp_collision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "report.parquet"
    live_temp = final_path.with_name(scratch_owner.owned_scratch_name(".f-", "c" * 12))
    live_temp.write_text("live\n", encoding="utf-8")
    monkeypatch.setattr(
        parquet_writers,
        "uuid4",
        lambda: type("FixedUuid", (), {"hex": "c" * 32})(),
    )

    with pytest.raises(FileExistsError):
        parquet_writers.write_parquet(pd.DataFrame({"value": [1]}), final_path)

    assert live_temp.read_text(encoding="utf-8") == "live\n"
    assert not final_path.exists()


def test_artifact_parquet_cleanup_preserves_active_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parquet_writers.pq,
        "write_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(GeneratorExit()),
    )
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unlink_failed")),
    )

    with pytest.raises(GeneratorExit):
        parquet_writers.write_parquet(
            pd.DataFrame({"value": [1]}),
            tmp_path / "x.parquet",
        )


@pytest.mark.parametrize("writer_name", ("write_parquet", "write_parquet_chunks"))
def test_parquet_cleanup_uses_shared_owned_scratch_removal(
    writer_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output.parquet"
    temp_path = final_path.with_name(scratch_owner.owned_scratch_name(".f-", "c" * 12))
    events: list[str] = []
    original_remove = scratch_owner.remove_owned_scratch

    def remove(path: Path, identity: object) -> None:
        if path == temp_path:
            events.append("remove")
            raise OSError("remove_secondary")
        original_remove(path, identity)

    monkeypatch.setattr(
        parquet_writers,
        "uuid4",
        lambda: type("Uuid", (), {"hex": "c" * 32})(),
    )
    monkeypatch.setattr(scratch_owner, "remove_owned_scratch", remove)
    if writer_name == "write_parquet":
        monkeypatch.setattr(
            parquet_writers.pq,
            "write_table",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("body_primary")
            ),
        )

        def operation() -> object:
            return parquet_writers.write_parquet(
                pd.DataFrame({"value": [1]}),
                final_path,
            )

    else:

        def frames():  # type: ignore[no-untyped-def]
            yield pd.DataFrame(
                {"timestamp": [pd.Timestamp("2024-01-01")], "value": [1.0]}
            )
            raise RuntimeError("body_primary")

        def operation() -> object:
            return parquet_writers.write_parquet_chunks(
                frames(),
                final_path,
                required=("timestamp", "value"),
                key_columns=("timestamp",),
                compression="snappy",
            )

    with pytest.raises(RuntimeError, match="body_primary"):
        operation()

    assert events == ["remove"]
