from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    select_best_validation_trial,
    validate_trial_metric_columns,
)


def test_model_selection_uses_validation_only_and_explicit_ties() -> None:
    validate_trial_metric_columns(
        selection_metric="validation_rankic",
        validation_metric_columns=("validation_rankic",),
        test_metric_columns=("test_rankic",),
    )
    trials = pd.DataFrame(
        {
            "trial_id": ["trial_b", "trial_a"],
            "validation_rankic": [0.7, 0.7],
        }
    )
    assert (
        select_best_validation_trial(
            trials,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("trial_a", "trial_b"),
        )
        == "trial_a"
    )
    with pytest.raises(ValueError, match="selection_metric_must_be_validation_metric"):
        validate_trial_metric_columns(
            selection_metric="test_rankic",
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=("test_rankic",),
        )


def test_model_selection_metrics_are_cross_sectional_and_chunk_parity_stable() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "asset_id": ["a", "b", "c"] * 2,
            "label": [1.0, 2.0, 3.0, 101.0, 102.0, 103.0],
            "prediction": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    partitioned = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:2], frame.iloc[2:5], frame.iloc[5:]),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    assert partitioned == whole
    assert whole["validation_rows"] == 6
    assert whole["validation_ic"] == pytest.approx(0.0)
    assert whole["validation_rankic"] == pytest.approx(0.0)


def test_model_selection_rejects_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 2),
            "asset_id": ["a", "b"],
            "label": [1.0, 2.0],
            "prediction": [np.nan, 1.0],
        }
    )
    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="current",
            label_column="label",
            prediction_column="prediction",
        )


@pytest.mark.parametrize("residual", [1e155, 1e-155, 2e-161, 3e-161, 0.0, 2.0])
def test_model_selection_mse_scales_before_squaring(residual: float) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "asset_id": "asset",
            "label": 0.0,
            "prediction": [residual, *([0.0] * 99)],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,), model_id="model", label_column="label", prediction_column="prediction"
    )
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1], frame.iloc[1:]),
        model_id="model",
        label_column="label",
        prediction_column="prediction",
    )
    assert whole == split
    assert whole["validation_mse"] == pytest.approx(
        float(Fraction.from_float(residual) ** 2 / len(frame)), rel=1e-12, abs=0.0
    )
    assert whole["validation_ic"] is None
    assert whole["validation_rankic"] is None


def test_model_selection_mse_rejects_unrepresentable_final_value() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "asset_id": "a",
            "label": 0.0,
            "prediction": 1e155,
        }
    )
    with pytest.raises(ValueError, match="model_selection_mse_unrepresentable"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="model",
            label_column="label",
            prediction_column="prediction",
        )


@pytest.mark.parametrize("field", ["validation_ic", "validation_rankic"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_persisted_trial_correlations_obey_exact_bounds(field, sign):
    import math

    from quant_research_platform.contracts.models import (
        validate_completed_model_trial_metrics,
    )

    metrics = {
        "validation_rows": 2,
        "validation_ic": 1.0,
        "validation_ic_reason": None,
        "validation_rankic": 1.0,
        "validation_rankic_reason": None,
        "validation_mse": 0.0,
        "n_train": 2,
    }
    metrics[field] = sign
    validate_completed_model_trial_metrics("model", metrics)
    metrics[field] = sign * math.nextafter(1.0, math.inf)
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        validate_completed_model_trial_metrics("model", metrics)


@pytest.mark.parametrize("memory_scratch", [False, True])
@pytest.mark.parametrize(
    "available,cap",
    [(256 * 1024**2, 64 * 1024**2), (256 * 1024**2, 512 * 1024), (512 * 1024, None)],
)
def test_trial_metric_scratch_and_native_workspace_share_capacity(
    tmp_path, monkeypatch, memory_scratch, available, cap
):
    import tempfile

    from quant_research_platform.research.models import trial_selection
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import _scratch_ownership
    from tests.helpers.runtime_resources import capture_test_policy

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(
        _scratch_ownership, "_memory_filesystem", lambda _path: memory_scratch
    )
    row_count = 4096
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(np.arange(row_count) // 8, unit="D"),
            "asset_id": [f"a{index % 8}" for index in range(row_count)],
            "label": np.arange(row_count) % 8,
            "prediction": (3 * np.arange(row_count)) % 8,
        }
    )
    original_summary = trial_selection._cross_sectional_metric_summary
    query_calls = []

    def observed_summary(database):
        query_calls.append(True)
        assert [
            row[1] for row in database.execute("PRAGMA table_info(validation_rows)")
        ] == ["row_id", "timestamp_ns", "label", "prediction"]
        assert database.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert database.execute("PRAGMA threads").fetchone()[0] == 0
        database_bytes = sum(
            path.stat().st_size for path in scratch.rglob("*") if path.is_file()
        )
        assert policy.used_memory_bytes > 512 * 1024
        if memory_scratch:
            assert (
                sum(
                    lease.reserved_bytes
                    for lease in policy._leases
                    if lease.owner == "storage.scratch.tmpfs"
                )
                >= database_bytes
            )
        return original_summary(database)

    monkeypatch.setattr(
        trial_selection, "_cross_sectional_metric_summary", observed_summary
    )
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=available, explicit_cap_bytes=cap)
    ) as policy:
        chunks = (frame.iloc[start : start + 128] for start in range(0, row_count, 128))
        if cap == 64 * 1024**2:
            result = evaluate_model_trial_metrics_from_chunks(
                chunks,
                model_id="model",
                label_column="label",
                prediction_column="prediction",
            )
            assert result["validation_rows"] == row_count
            assert result["validation_mse"] == pytest.approx(6.0)
            assert result["validation_ic"] == pytest.approx(3 / 7)
            assert result["validation_rankic"] == pytest.approx(3 / 7)
            assert result["validation_ic_reason"] is None
            assert result["validation_rankic_reason"] is None
            assert query_calls == [True]
        else:
            with pytest.raises(
                ValueError,
                match="^(storage_scratch|model_selection_working_set)_exceeds_memory_capacity$",
            ):
                evaluate_model_trial_metrics_from_chunks(
                    chunks,
                    model_id="model",
                    label_column="label",
                    prediction_column="prediction",
                )
            assert query_calls == []
        assert policy.used_memory_bytes == 0
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure_owner", ["database", "directory"])
@pytest.mark.parametrize("primary_failure", [False, True])
def test_trial_metric_cleanup_keeps_live_reservations_and_primary(
    tmp_path, monkeypatch, failure_owner, primary_failure
):
    import sqlite3
    import tempfile

    from quant_research_platform.research.models import trial_selection
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import _scratch_ownership
    from tests.helpers.runtime_resources import capture_test_policy

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(_scratch_ownership, "_memory_filesystem", lambda _path: True)
    original_connect = sqlite3.connect
    original_directory = tempfile.TemporaryDirectory
    connections, directories, reservations = [], [], []
    fail_cleanup = True

    class Connection(sqlite3.Connection):
        def close(self):
            if fail_cleanup and failure_owner == "database":
                raise OSError("database_close_failed")
            super().close()

    def connect(*args, **kwargs):
        connection = original_connect(*args, factory=Connection, **kwargs)
        connections.append(connection)
        return connection

    class Directory(original_directory):
        def cleanup(self):
            if fail_cleanup and failure_owner == "directory":
                raise OSError("directory_cleanup_failed")
            super().cleanup()

    def directory(*args, **kwargs):
        result = Directory(*args, **kwargs)
        directories.append(result)
        return result

    class Reservation(_scratch_ownership.ScratchReservation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            reservations.append(self)

    monkeypatch.setattr(trial_selection.sqlite3, "connect", connect)
    monkeypatch.setattr(trial_selection.tempfile, "TemporaryDirectory", directory)
    monkeypatch.setattr(trial_selection, "ScratchReservation", Reservation)

    def fail_summary(_database):
        raise ValueError("metric_primary_failure")

    if primary_failure:
        monkeypatch.setattr(
            trial_selection, "_cross_sectional_metric_summary", fail_summary
        )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 2),
            "asset_id": ["a", "b"],
            "label": [1.0, 2.0],
            "prediction": [1.0, 2.0],
        }
    )
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        metric_lease = None
        try:
            with pytest.raises(
                ValueError if primary_failure else OSError,
                match="metric_primary_failure"
                if primary_failure
                else f"{failure_owner}_.*failed",
            ):
                evaluate_model_trial_metrics_from_chunks(
                    (frame,),
                    model_id="model",
                    label_column="label",
                    prediction_column="prediction",
                )
            assert reservations[0].reserved_bytes > 0
            assert policy.used_memory_bytes >= reservations[0].reserved_bytes
            metric_lease = next(
                (
                    lease
                    for lease in policy._leases
                    if lease.owner == "research.model.selection.metric"
                ),
                None,
            )
            if failure_owner == "database":
                assert metric_lease is not None and metric_lease.active
                assert metric_lease.reserved_bytes > 0
            else:
                assert metric_lease is None
        finally:
            # Complete the real cleanup before releasing the retained test owners.
            fail_cleanup = False
            for connection in connections:
                connection.close()
            for item in directories:
                item.cleanup()
            for reservation in reservations:
                reservation.resize(0)
                reservation.close()
            if metric_lease is not None:
                metric_lease.close()
        assert policy.used_memory_bytes == 0
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("cap", [48 * 1024**2, 128 * 1024**2])
def test_forced_memory_sqlite_accounts_unspilled_sorters_and_trees(
    tmp_path, monkeypatch, cap
):
    import sqlite3
    import tempfile

    from quant_research_platform.research.models import trial_selection
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import _scratch_ownership
    from tests.helpers.runtime_resources import capture_test_policy

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(_scratch_ownership, "_memory_filesystem", lambda _path: False)
    original_connect = sqlite3.connect

    class MemoryConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            # Exercise SQLite's real memory sorter/B-tree path. The installed
            # binary is unchanged; only its reported compile fact is injected.
            if sql == "PRAGMA temp_store=FILE":
                sql = "PRAGMA temp_store=MEMORY"
            if sql == "PRAGMA compile_options":
                return iter(
                    ("TEMP_STORE=3" if value.startswith("TEMP_STORE=") else value,)
                    for (value,) in super().execute(sql, *args, **kwargs)
                )
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        return original_connect(*args, factory=MemoryConnection, **kwargs)

    monkeypatch.setattr(trial_selection.sqlite3, "connect", connect)
    summary = trial_selection._cross_sectional_metric_summary
    query_calls = []

    def observed_summary(database):
        assert database.execute("PRAGMA temp_store").fetchone()[0] == 2
        query_calls.append(True)
        return summary(database)

    monkeypatch.setattr(
        trial_selection, "_cross_sectional_metric_summary", observed_summary
    )
    count = 16384
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(np.arange(count) // 8, unit="D"),
            "asset_id": [f"a{index % 8}" for index in range(count)],
            "label": np.arange(count) % 8,
            "prediction": (3 * np.arange(count)) % 8,
        }
    )
    with runtime_resource_policy_scope(
        capture_test_policy(
            available_memory_bytes=512 * 1024**2, explicit_cap_bytes=cap
        )
    ) as policy:
        chunks = (frame.iloc[start : start + 256] for start in range(0, count, 256))
        if cap == 48 * 1024**2:
            with pytest.raises(
                ValueError,
                match="^model_selection_working_set_exceeds_memory_capacity$",
            ):
                evaluate_model_trial_metrics_from_chunks(
                    chunks,
                    model_id="model",
                    label_column="label",
                    prediction_column="prediction",
                )
            assert query_calls == []
        else:
            result = evaluate_model_trial_metrics_from_chunks(
                chunks,
                model_id="model",
                label_column="label",
                prediction_column="prediction",
            )
            assert result["validation_rows"] == count
            assert result["validation_mse"] == pytest.approx(6.0)
            assert result["validation_ic"] == pytest.approx(3 / 7)
            assert result["validation_rankic"] == pytest.approx(3 / 7)
            assert query_calls == [True]
        assert policy.used_memory_bytes == 0
    assert list(scratch.iterdir()) == []


def test_metric_consumer_drops_owned_input_before_next_production():
    import gc
    import weakref

    from quant_research_platform.runtime.resources import (
        OwnedValue,
        pandas_frame_payload_bytes,
        runtime_resource_policy_scope,
    )

    references = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with runtime_resource_policy_scope() as policy:

            def chunks():
                for day in range(3):
                    assert all(reference() is None for reference in references)
                    frame = pd.DataFrame(
                        {
                            "timestamp": pd.Timestamp("2023-01-01")
                            + pd.Timedelta(days=day),
                            "asset_id": ["a", "b", "c"],
                            "label": [1.0, 2.0, 3.0],
                            "prediction": [1.0, 2.0, 3.0],
                        }
                    )
                    references.append(weakref.ref(frame))
                    item = OwnedValue(
                        frame,
                        policy.lease(
                            "test.metric.input", pandas_frame_payload_bytes(frame)
                        ),
                    )
                    frame = None
                    yield item
                    item = None

            actual = evaluate_model_trial_metrics_from_chunks(
                chunks(),
                model_id="fixture",
                label_column="label",
                prediction_column="prediction",
            )
            assert actual["validation_mse"] == 0
            assert all(reference() is None for reference in references)
            assert policy.used_memory_bytes == 0
    finally:
        if was_enabled:
            gc.enable()
