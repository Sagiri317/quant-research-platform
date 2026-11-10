"""Actual native members are checked once; no filesystem mutation snapshots."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts.outputs.manifest import (
    validate_output_bundle,
)
from quant_research_platform.artifacts.outputs.prediction import (
    iter_predictions,
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.contracts.time import TimeWindow
from tests.helpers.output_frames import (
    make_price_volume_prediction_output_frame,
    native_bundle,
)


def test_bundle_returns_actual_members_and_private_paths(tmp_path):
    root, _, metadata = native_bundle(tmp_path)
    manifest, members = validate_output_bundle(root)
    assert members == (root / "prediction.parquet",)
    assert manifest["metadata"] == metadata


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


def test_prediction_member_decode_failure_propagates(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    (root / "prediction.parquet").write_bytes(b"bad parquet")
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        read_predictions(root)


def test_projected_iterator_closes_source_on_early_stop(tmp_path, monkeypatch):
    root, frame, _ = native_bundle(tmp_path)

    import quant_research_platform.storage.parquet as parquet

    class Source:
        closed = False
        emitted = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.emitted:
                raise StopIteration
            self.emitted = True
            return pa.RecordBatch.from_pandas(frame, preserve_index=False)

        def close(self):
            self.closed = True

    source = Source()
    monkeypatch.setattr(parquet, "iter_dataset_part_batches", lambda *a, **k: source)
    chunks = iter_predictions(root)
    next(chunks)
    chunks.close()
    assert source.closed


def test_unrelated_files_do_not_invalidate_declared_members(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    (root / "notes.txt").write_text("Research notes")
    pd.testing.assert_frame_equal(
        read_predictions(root)[0], frame, check_dtype=False, check_exact=True
    )


def test_read_parses_one_manifest_and_projects_only_consumed_scores(
    tmp_path, monkeypatch
):
    root, _, _ = native_bundle(tmp_path)
    import quant_research_platform.artifacts.outputs.manifest as manifests
    import quant_research_platform.storage.parquet as parquet

    calls = []
    original = manifests.read_qrp_json_file

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(manifests, "read_qrp_json_file", read)
    physical = []
    iterate = parquet.iter_dataset_part_batches

    def projected(path, **kwargs):
        physical.append(kwargs["columns"])
        return iterate(path, **kwargs)

    monkeypatch.setattr(parquet, "iter_dataset_part_batches", projected)
    frame, _ = read_predictions(root, columns=("prediction_score",))
    assert list(frame) == ["prediction_score"]
    assert len(calls) == 1
    assert physical == [("timestamp", "asset_id", "prediction_score")]


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("future_maturity", "model_training_label_after_fit_cutoff"),
        ("future_selection", "model_selection_after_fit_cutoff"),
        ("outside_training", "model_training_window_outside_declared_fit"),
        ("fitting_state", "prediction_training_facts_invalid"),
        ("forbidden_input_columns", "prediction_training_facts_invalid"),
    ],
)
def test_prediction_publication_rejects_invalid_training_facts(
    tmp_path, mutation, error
):
    root, _, _ = native_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    metadata = manifest["metadata"]
    if mutation == "future_maturity":
        metadata["model_metadata"]["label_available_max"] = "2014-01-03"
    elif mutation == "future_selection":
        metadata["model_facts"]["selection_cutoff"] = "2014-01-03"
    elif mutation == "outside_training":
        metadata["model_metadata"]["train_start"] = "2014-01-01"
    elif mutation == "fitting_state":
        metadata["model_metadata"]["scaler"] = None
    else:
        metadata["model_metadata"]["input_columns"] = ["momentum_20d"]
    with pytest.raises(ValueError, match=error):
        write_prediction_outputs(
            output_path=tmp_path / "invalid",
            output_chunks=[make_price_volume_prediction_output_frame()],
            metadata=metadata,
        )


def test_prediction_writer_statistics_prune_actual_decoding(tmp_path, monkeypatch):
    frame = pd.concat(
        [make_price_volume_prediction_output_frame()] * 4, ignore_index=True
    )
    frame["timestamp"] = pd.to_datetime(
        [
            "2014-01-03T00:00:00",
            "2014-02-03T00:00:00",
            "2014-02-03T23:59:59.999999999",
            "2014-03-03T00:00:00",
        ],
        format="ISO8601",
    )
    frame["prediction_score"] = [0.25, 0.5, 0.75, 1.0]
    metadata = frame.attrs["prediction_metadata"]
    metadata.update(
        prediction_window={"start": "2014-01-03", "end": "2014-03-03"},
        prediction_rows=len(frame),
        effective_prediction_end=frame.timestamp.max().isoformat(),
    )
    root = write_prediction_outputs(
        output_path=tmp_path / "prediction",
        output_chunks=[frame.iloc[:1], frame.iloc[1:3], frame.iloc[3:]],
        metadata=metadata,
    )
    with pq.ParquetFile(root / "prediction.parquet") as parquet:
        assert parquet.metadata.num_row_groups == 3
        for index in range(3):
            group = parquet.metadata.row_group(index)
            assert group.column(0).statistics.has_min_max
            assert all(group.column(i).statistics is None for i in range(1, 4))
    decoded = []
    original = pq.ParquetFile.iter_batches

    def observe(self, *args, **kwargs):
        decoded.extend(kwargs["row_groups"])
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observe)
    for window, expected in (
        (TimeWindow("2014-02-03", "2014-02-03"), frame.iloc[1:3]),
        (
            TimeWindow(
                "2014-02-04T07:59:59.999999999+08:00",
                "2014-02-04T07:59:59.999999999+08:00",
            ),
            frame.iloc[2:3],
        ),
    ):
        decoded.clear()
        selected, _ = read_predictions(root, time_window=window)
        pd.testing.assert_frame_equal(
            selected, expected.reset_index(drop=True), check_exact=True
        )
        assert decoded == [1]
    decoded.clear()
    pd.testing.assert_frame_equal(read_predictions(root)[0], frame, check_exact=True)
    assert decoded == [0, 1, 2]
