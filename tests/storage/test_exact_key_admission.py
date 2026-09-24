"""Physical admission boundaries for bulk exact-key comparison."""

from pathlib import Path

import pyarrow as pa
import pytest

import quant_research_platform.storage.exact_keys as key_tracking
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


@pytest.mark.parametrize("count", (1_000, 5_000, 20_000))
def test_bulk_comparison_admission_tracks_physical_scale(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    unique = count - 100
    table = pa.table(
        {
            "asset_id": [f"asset-{position % unique:08d}" for position in range(count)],
            "value": [position % unique for position in range(count)],
        }
    )
    requests: list[int] = []

    def gate(_path: Path, *, reserve_bytes: int) -> int:
        requests.append(reserve_bytes)
        if reserve_bytes > 64 * 1024**2:
            raise RuntimeError("insufficient_free_disk")
        return 64 * 1024**2

    with ExactPrimaryKeyIndex(primary_key=("asset_id",)) as index:
        with monkeypatch.context() as patch:
            patch.setattr(key_tracking, "require_free_disk_headroom", gate)
            keep, duplicates = index.classify_comparison_table(
                table,
                namespace="physical-scale",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
        assert list(keep) == list(range(unique))
        assert duplicates == 100
        assert requests and max(requests) < 64 * 1024**2
        assert index.database_path.stat().st_size < 12 * 1024**2
        assert index._scratch_reservation is not None
        assert (
            index._scratch_reservation.reserved_bytes
            >= index.database_path.stat().st_size
        )


def test_bulk_comparison_disk_boundary_fails_before_exact_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "asset_id": [f"asset-{position:08d}" for position in range(20_000)],
            "value": list(range(20_000)),
        }
    )
    requests: list[int] = []

    def passing_gate(_path: Path, *, reserve_bytes: int) -> int:
        requests.append(reserve_bytes)
        return 64 * 1024**2

    with ExactPrimaryKeyIndex(primary_key=("asset_id",)) as index:
        with monkeypatch.context() as patch:
            patch.setattr(key_tracking, "require_free_disk_headroom", passing_gate)
            index.classify_comparison_table(
                table,
                namespace="boundary",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
    boundary = max(requests)
    assert 0 < boundary < 64 * 1024**2

    def boundary_gate(_path: Path, *, reserve_bytes: int) -> int:
        if reserve_bytes > boundary:
            raise RuntimeError("insufficient_free_disk")
        return boundary

    with ExactPrimaryKeyIndex(primary_key=("asset_id",)) as index:
        with monkeypatch.context() as patch:
            patch.setattr(key_tracking, "require_free_disk_headroom", boundary_gate)
            keep, duplicates = index.classify_comparison_table(
                table,
                namespace="boundary",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
        assert len(keep) == 20_000 and duplicates == 0

    with ExactPrimaryKeyIndex(primary_key=("asset_id",)) as index:
        assert index._connection is not None
        connection = index._connection

        def failing_gate(_path: Path, *, reserve_bytes: int) -> int:
            if reserve_bytes >= boundary:
                assert connection.execute(
                    "SELECT COUNT(*) FROM exact_keys"
                ).fetchone() == (0,)
                raise RuntimeError("insufficient_free_disk")
            return boundary - 1

        with monkeypatch.context() as patch:
            patch.setattr(key_tracking, "require_free_disk_headroom", failing_gate)
            with pytest.raises(RuntimeError, match="insufficient_free_disk"):
                index.classify_comparison_table(
                    table,
                    namespace="boundary",
                    value_columns=("value",),
                    payload_target_bytes=1024**2,
                )
        assert connection.execute("SELECT COUNT(*) FROM exact_keys").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone() == (0,)


def test_bulk_comparison_conflict_rolls_back_then_accepts_valid_rows() -> None:
    valid = pa.table({"asset_id": ["a", "b"], "value": [1, 2]})
    conflicting = pa.table({"asset_id": ["c", "c"], "value": [3, 4]})
    with ExactPrimaryKeyIndex(primary_key=("asset_id",)) as index:
        index.classify_comparison_table(
            valid,
            namespace="conflict",
            value_columns=("value",),
            payload_target_bytes=1024,
        )
        with pytest.raises(ValueError, match="conflicting_exact_primary_key_payload"):
            index.classify_comparison_table(
                conflicting,
                namespace="conflict",
                value_columns=("value",),
                payload_target_bytes=1024,
            )
        assert index._connection is not None
        assert index._connection.execute(
            "SELECT COUNT(*) FROM exact_keys"
        ).fetchone() == (2,)
        assert index._connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone() == (0,)
