"""Native prediction metadata, atomic publication and streaming safety."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.manifest import (
    output_manifest_payload,
    validate_output_manifest,
)
from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from tests.helpers.output_frames import make_price_volume_prediction_output_frame


def native_bundle(tmp_path):
    frame = make_price_volume_prediction_output_frame()
    metadata = deepcopy(frame.attrs["prediction_metadata"])
    path = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    return path, frame, metadata


def test_manifest_actual_model_and_paths(tmp_path):
    root, frame, metadata = native_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    assert validate_output_manifest(manifest) is manifest
    assert manifest["metadata"] == metadata
    assert "run_id" not in manifest
    pd.testing.assert_frame_equal(
        read_predictions(root)[0],
        frame,
        check_flags=False,
        check_dtype=False,
        check_exact=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_window", {}),
        ("prediction_window", {"start": "2014-01-03"}),
        (
            "prediction_window",
            {"start": "2014-01-03", "end": "2014-01-03", "extra": True},
        ),
        ("effective_prediction_end", "NaT"),
        ("sample_role", ""),
        ("prediction_rows", True),
        ("scope", "LOCAL_ONLY"),
    ],
)
def test_metadata_rejected_before_publish_and_after_read(tmp_path, field, value):
    root, frame, metadata = native_bundle(tmp_path)
    metadata[field] = value
    with pytest.raises((ValueError, TypeError)):
        write_prediction_outputs(
            output_path=tmp_path / "bad", output_chunks=[frame], metadata=metadata
        )
    assert not (tmp_path / "bad").exists()
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"] = metadata
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, TypeError)):
        read_predictions(root)


@pytest.mark.parametrize("key", ["Authorization", "auth", "private_key"])
def test_manifest_rejects_credentials(tmp_path, key):
    _, frame, metadata = native_bundle(tmp_path)
    metadata["model_facts"]["model"]["hyperparameters"][key] = "secret-value"
    metadata["model_facts"]["candidate_parameters"][key] = "secret-value"
    with pytest.raises(ValueError):
        output_manifest_payload(
            definition_id="prediction_outputs",
            metadata=metadata,
            files=["prediction.parquet"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "before",
        "after",
        "nan",
        "duplicate",
        "duplicate_chunks",
        "score_text",
        "rank_text",
    ],
)
def test_writer_validates_actual_rows_and_global_keys(tmp_path, mutation):
    root, frame, metadata = native_bundle(tmp_path)
    if mutation == "before":
        frame["timestamp"] = pd.Timestamp("2014-01-02")
    elif mutation == "after":
        frame["timestamp"] = pd.Timestamp("2014-01-04")
    elif mutation == "nan":
        frame["prediction_score"] = float("nan")
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame], ignore_index=True)
    elif mutation in {"score_text", "rank_text"}:
        column = (
            "prediction_score" if mutation == "score_text" else "prediction_rank_score"
        )
        frame[column] = frame[column].astype(str)
    chunks = [frame, frame] if mutation == "duplicate_chunks" else [frame]
    with pytest.raises(ValueError):
        write_prediction_outputs(
            output_path=tmp_path / "bad", output_chunks=chunks, metadata=metadata
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["string", "date", "ns", "us"])
def test_timestamp_normalization_and_readback(tmp_path, kind):
    frame = make_price_volume_prediction_output_frame()
    expected = frame.copy(deep=True)
    expected["timestamp"] = expected.timestamp.dt.as_unit("ns")
    metadata = frame.attrs["prediction_metadata"]
    if kind == "string":
        frame["timestamp"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    elif kind == "date":
        frame["timestamp"] = frame.timestamp.dt.date
    else:
        frame["timestamp"] = frame.timestamp.dt.as_unit(kind)
    before = frame.copy(deep=True)
    root = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    actual, _ = read_predictions(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)


def test_writer_collision_preserves_bytes_and_closes_unconsumed_source(tmp_path):
    root, frame, metadata = native_bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}

    class Source:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            pytest.fail("collision consumed source")

        def close(self):
            self.closed = True

    source = Source()
    with pytest.raises(OutputAlreadyExistsError):
        write_prediction_outputs(
            output_path=root, output_chunks=source, metadata=metadata
        )
    assert source.closed
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
