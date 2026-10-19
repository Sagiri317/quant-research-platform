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
from tests.artifacts.test_outputs_core import native_bundle
from tests.helpers.output_frames import make_price_volume_prediction_output_frame


def test_bundle_returns_actual_members_and_private_paths(tmp_path):
    root, _, metadata = native_bundle(tmp_path)
    manifest, members = validate_output_bundle(root)
    assert members == (root / "prediction.parquet",)
    assert manifest["metadata"] == metadata


@pytest.mark.parametrize("mutation", ["missing", "symlink"])
def test_bundle_exact_members_and_no_alias(tmp_path, mutation):
    root, _, _ = native_bundle(tmp_path)
    if mutation == "missing":
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


def test_unrelated_files_do_not_invalidate_declared_members(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    (root / "notes.txt").write_text("Research notes")
    pd.testing.assert_frame_equal(read_predictions(root)[0], frame, check_dtype=False)


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


def test_projection_does_not_decode_or_validate_an_unused_score(tmp_path):
    root, frame, _ = native_bundle(tmp_path)
    frame["prediction_rank_score"] = float("inf")
    frame.to_parquet(root / "prediction.parquet", index=False)
    projected, _ = read_predictions(root, columns=("prediction_score",))
    assert projected.prediction_score.tolist() == [0.25]
    with pytest.raises(ValueError, match="invalid_prediction_values"):
        read_predictions(root)


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("future_maturity", "model_training_label_after_fit_cutoff"),
        ("future_selection", "model_selection_after_fit_cutoff"),
        ("outside_training", "model_training_window_outside_declared_fit"),
        ("fitting_state", "prediction_training_facts_invalid"),
        ("duplicate_recipe", "prediction_training_facts_invalid"),
    ],
)
def test_compact_prediction_facts_preserve_science_without_fitting_state(
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
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        read_predictions(root)


@pytest.mark.parametrize("rank", [-0.5001, 0.5001])
def test_consumed_prediction_rank_rejects_values_outside_centered_rank_domain(
    tmp_path, rank
):
    root, frame, _ = native_bundle(tmp_path)
    frame["prediction_rank_score"] = rank
    frame.to_parquet(root / "prediction.parquet", index=False)
    with pytest.raises(ValueError, match="invalid_prediction_rank_values"):
        read_predictions(root, columns=("prediction_rank_score",))
    projected, _ = read_predictions(root, columns=("prediction_score",))
    assert projected.prediction_score.tolist() == [0.25]


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
        pd.testing.assert_frame_equal(selected, expected.reset_index(drop=True))
        assert decoded == [1]
    decoded.clear()
    pd.testing.assert_frame_equal(read_predictions(root)[0], frame)
    assert decoded == [0, 1, 2]


@pytest.mark.parametrize("damage", ("missing", "extra", "end", "no_statistics"))
def test_filtered_predictions_check_footer_without_scanning_other_periods(
    tmp_path, monkeypatch, damage
):
    from contextlib import contextmanager

    import pyarrow.parquet as pq

    from quant_research_platform.storage import parquet as storage_parquet

    frame = pd.concat(
        [make_price_volume_prediction_output_frame()] * 3, ignore_index=True
    )
    frame["timestamp"] = pd.to_datetime(["2014-01-03", "2014-02-03", "2014-03-03"])
    metadata = frame.attrs["prediction_metadata"]
    metadata.update(
        prediction_window={"start": "2014-01-03", "end": "2014-03-03"},
        prediction_rows=3,
        effective_prediction_end=frame.timestamp.max().isoformat(),
    )
    root = write_prediction_outputs(
        output_path=tmp_path / "prediction",
        output_chunks=[frame.iloc[[i]] for i in range(3)],
        metadata=metadata,
    )
    changed = frame.copy()
    if damage == "missing":
        changed = changed.iloc[:2]
    elif damage == "extra":
        changed = pd.concat(
            [changed, changed.iloc[[-1]].assign(timestamp=pd.Timestamp("2014-03-02"))],
            ignore_index=True,
        )
    elif damage == "end":
        changed.loc[2, "timestamp"] = pd.Timestamp("2014-03-02")
    changed.to_parquet(
        root / "prediction.parquet",
        index=False,
        row_group_size=1,
        write_statistics=False if damage == "no_statistics" else ["timestamp"],
    )
    opens = []
    opened = storage_parquet._opened_parquet

    @contextmanager
    def observed_open(path):
        opens.append(path)
        with opened(path) as source:
            yield source

    monkeypatch.setattr(storage_parquet, "_opened_parquet", observed_open)
    decoded = []
    original = pq.ParquetFile.iter_batches

    def observed(self, *args, **kwargs):
        decoded.append((kwargs["row_groups"], kwargs["columns"]))
        yield from original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observed)
    window = TimeWindow("2014-02-03", "2014-02-03")
    if damage in {"no_statistics", "end"}:
        result, _ = read_predictions(
            root, columns=("prediction_score",), time_window=window
        )
        assert (
            result.prediction_score.tolist()
            == frame.iloc[[1]].prediction_score.tolist()
        )
        assert all(
            columns == ["timestamp", "asset_id", "prediction_score"]
            for _, columns in decoded
        )
    else:
        with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
            read_predictions(root, columns=("prediction_score",), time_window=window)
        assert decoded == []
    assert len(opens) == 1
    if damage == "end":
        assert [groups for groups, _columns in decoded] == [[1]]
        with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
            read_predictions(root)
