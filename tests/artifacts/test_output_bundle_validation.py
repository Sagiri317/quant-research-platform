"""Actual native members are checked once; no filesystem mutation snapshots."""

import json

import pandas as pd
import pyarrow as pa
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


@pytest.mark.parametrize("rank", [-1e100, 1e100, -0.5001, 0.5001])
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
