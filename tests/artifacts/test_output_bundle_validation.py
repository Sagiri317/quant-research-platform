"""Actual native members are checked once; no filesystem mutation snapshots."""

import json

import pandas as pd
import pyarrow as pa
import pytest

from quant_research_platform.artifacts.outputs.bundle_validation import (
    validate_output_bundle,
)
from quant_research_platform.artifacts.outputs.prediction import (
    iter_predictions,
    read_prediction_output_bundle,
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.contracts.time import TimeWindow
from tests.artifacts.test_outputs_core import native_bundle
from tests.helpers.output_frames import make_price_volume_prediction_output_frame


def test_bundle_returns_actual_members_and_private_paths(tmp_path):
    root, _, metadata = native_bundle(tmp_path)
    manifest, members = validate_output_bundle(root)
    assert members == (root / "prediction.parquet",)
    assert manifest["metadata"] == metadata


@pytest.mark.parametrize("mutation", ["extra", "missing", "symlink"])
def test_bundle_exact_members_and_no_alias(tmp_path, mutation):
    root, _, _ = native_bundle(tmp_path)
    if mutation == "extra":
        (root / "extra").write_text("extra")
    elif mutation == "missing":
        (root / "prediction.parquet").unlink()
    else:
        (root / "prediction.parquet").rename(tmp_path / "actual.parquet")
        (root / "prediction.parquet").symlink_to(tmp_path / "actual.parquet")
    with pytest.raises(ValueError):
        validate_output_bundle(root)


def test_nanosecond_window_and_independent_lookups(tmp_path):
    frame = make_price_volume_prediction_output_frame()
    frame = pd.concat([frame] * 3, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(
        [
            "2014-01-03T00:00:00.000000001",
            "2014-01-03T00:00:00.000000002",
            "2014-01-03T00:00:00.000000003",
        ]
    )
    metadata = frame.attrs["prediction_metadata"]
    metadata.update(
        prediction_rows=3, effective_prediction_end=frame.timestamp.max().isoformat()
    )
    root = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    window = TimeWindow(
        "2014-01-03T00:00:00.000000002", "2014-01-03T00:00:00.000000002"
    )
    selected = read_predictions(root, time_window=window)[0]
    assert len(selected) == 1 and selected.timestamp.iloc[0] == frame.timestamp.iloc[1]
    for _ in range(2):
        _, lookup = read_prediction_output_bundle(root, time_window=window)
        values, found = lookup.values_for_frame(selected, mark_matched=True)
        assert list(found) == [1] and values[0] == 0.25


def test_actual_member_corruption_and_end_count_rejected(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"]["prediction_rows"] = 2
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        list(iter_predictions(root))
    (root / "prediction.parquet").write_bytes(b"bad parquet")
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        read_predictions(root)


def test_projected_iterator_closes_source_on_rejection(tmp_path, monkeypatch):
    root, frame, _ = native_bundle(tmp_path)

    import quant_research_platform.storage.parquet as parquet

    class Source:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            bad = frame.copy()
            bad["prediction_score"] = float("inf")
            return pa.RecordBatch.from_pandas(bad, preserve_index=False)

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(parquet, "iter_dataset_part_batches", lambda *a, **k: source)
    with pytest.raises(ValueError):
        list(iter_predictions(root))
    assert source.closed


def test_lookup_matches_are_exact_and_each_consumer_is_independent(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    _, first = read_prediction_output_bundle(root)
    _, second = read_prediction_output_bundle(root)
    assert first.coverage_counts() == second.coverage_counts() == (1, 0)
    first.values_for_frame(frame, mark_matched=True)
    assert first.coverage_counts() == (0, 0)
    assert second.coverage_counts() == (1, 0)
