"""Native prediction metadata, atomic publication and streaming safety."""

import json

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from tests.helpers.output_frames import (
    make_price_volume_prediction_output_frame,
    native_bundle,
)


def test_manifest_actual_model_and_paths(tmp_path):
    root, frame, metadata = native_bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    assert not {"model_id", "family_id", "label_unit"} & set(manifest["metadata"])
    assert manifest["metadata"] == metadata
    assert "run_id" not in manifest
    actual, restored = read_predictions(root)
    assert restored == metadata
    pd.testing.assert_frame_equal(
        actual,
        frame,
        check_flags=False,
        check_dtype=False,
        check_exact=True,
    )


@pytest.mark.parametrize("key", ["Authorization", "auth", "private_key"])
def test_manifest_writer_rejects_credentials(tmp_path, key):
    frame = make_price_volume_prediction_output_frame()
    metadata = frame.attrs["prediction_metadata"]
    metadata["model_facts"]["model"]["model_algorithm"] = "lasso_regression"
    metadata["model_facts"]["model"]["hyperparameters"][key] = "secret-value"
    metadata["model_facts"]["candidate_parameters"][key] = "secret-value"
    with pytest.raises(
        ValueError, match="unsafe_internal_artifact_metadata: credential"
    ):
        write_prediction_outputs(
            output_path=tmp_path / "invalid", output_chunks=[frame], metadata=metadata
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("kind", ["ns", "us"])
def test_timestamp_normalization_and_readback(tmp_path, kind):
    frame = make_price_volume_prediction_output_frame()
    expected = frame.copy(deep=True)
    expected["timestamp"] = expected.timestamp.dt.as_unit("ns")
    metadata = frame.attrs["prediction_metadata"]
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


@pytest.mark.parametrize(
    "field,value", [("input_spec", {"unused": True}), ("scope", "notes-only")]
)
def test_prediction_ignores_unconsumed_extra_metadata(tmp_path, field, value):
    root, frame, _ = native_bundle(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "input_spec" not in manifest["metadata"]
    manifest["metadata"][field] = value
    manifest_path.write_text(json.dumps(manifest))
    actual, metadata = read_predictions(root)
    assert metadata[field] == value
    pd.testing.assert_frame_equal(actual, frame, check_dtype=False, check_exact=True)
