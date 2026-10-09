from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from quant_research_platform.artifacts.outputs.manifest import (
    output_manifest_payload,
    prediction_output_metadata,
    validate_output_manifest_business_envelope,
)
from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_output_metadata_shape,
)
from tests.helpers.output_frames import (
    make_prediction_report_spec,
    make_price_volume_prediction_output_frame,
    make_price_volume_prediction_spec,
)


def _prediction_manifest_payload(*, ridge: bool = False) -> dict[str, object]:
    frame = make_price_volume_prediction_output_frame()
    if ridge:
        from sklearn import __version__

        frame.attrs["model_metadata"].update(
            model_algorithm="ridge_regression",
            dependency_status={
                "status": "available",
                "package": "sklearn",
                "version": __version__,
            },
        )
    return output_manifest_payload(
        output_spec=make_prediction_report_spec(),
        metadata=prediction_output_metadata(
            spec=make_price_volume_prediction_spec(),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata=frame.attrs["prediction_metadata"],
        ),
        files=("prediction.parquet",),
    )


def test_output_manifest_builder_validates_current_business_envelope() -> None:
    payload = _prediction_manifest_payload()
    assert validate_output_manifest_business_envelope(payload) is payload


def test_output_manifest_preserves_actual_external_dependency_version() -> None:
    from sklearn import __version__

    payload = _prediction_manifest_payload(ridge=True)
    model = payload["metadata"]["model_metadata"]
    assert validate_output_manifest_business_envelope(payload) is payload
    assert model["dependency_status"]["version"] == __version__


@pytest.mark.parametrize(
    "algorithm,dependency",
    (
        (
            "linear_regression",
            {"status": "available", "package": "internal", "version": "1"},
        ),
        ("ridge_regression", {"status": "available", "package": "sklearn"}),
        (
            "ridge_regression",
            {"status": "available", "package": "sklearn", "version": ""},
        ),
        ("linear_regression", {"status": "available", "package": "sklearn"}),
    ),
)
def test_output_manifest_rejects_dependency_contract_drift(algorithm, dependency):
    payload = _prediction_manifest_payload()
    model = payload["metadata"]["model_metadata"]
    model["model_algorithm"] = algorithm
    model["dependency_status"] = dependency
    with pytest.raises(ValueError, match="^output_manifest_metadata_invalid$"):
        validate_output_manifest_business_envelope(payload)


def test_output_manifest_preserves_typed_identifier_values() -> None:
    output_spec = replace(
        make_prediction_report_spec(),
        output_id="provider_summary",
        run_id="run_600000_sh",
    )
    base_payload = _prediction_manifest_payload()
    payload = output_manifest_payload(
        output_spec=output_spec,
        metadata=prediction_output_metadata(
            spec=make_price_volume_prediction_spec(),
            model_metadata=base_payload["metadata"]["model_metadata"],
            prediction_metadata=base_payload["metadata"]["prediction_metadata"],
        ),
        files=("prediction.parquet",),
    )
    assert payload["output_id"] == "provider_summary"
    assert payload["run_id"] == "run_600000_sh"


@pytest.mark.parametrize(
    "mutation",
    (
        "absolute_path",
        "provider_marker",
        "credential_marker",
        "traceback_marker",
        "row_sample",
    ),
)
def test_output_manifest_rejects_unsafe_free_form_metadata_before_publish(
    mutation: str,
) -> None:
    payload = _prediction_manifest_payload(ridge=True)
    model_metadata = payload["metadata"]["model_metadata"]
    if mutation == "absolute_path":
        model_metadata["model_algorithm"] = "ridge_regression"
        model_metadata["dependency_status"] = {
            "status": "available",
            "package": "sklearn",
            "version": "C:" + chr(92) + "private" + chr(92) + "environment",
        }
    else:
        marker = {
            "provider_marker": "csmar",
            "credential_marker": "token=abcdefghijk",
            "traceback_marker": "Traceback (most recent call last): secret",
            "row_sample": "row_samples: market_id=a asset_id=b timestamp=t",
        }[mutation]
        model_metadata["hyperparameters"][mutation] = marker
        model_metadata["resolved_hyperparameters"][mutation] = marker
    assert (
        validate_prediction_output_metadata_shape(payload["metadata"])
        is payload["metadata"]
    )
    if mutation in {"provider_marker", "row_sample"}:
        validate_output_manifest_business_envelope(payload)
    else:
        with pytest.raises(ValueError, match="^output_manifest_metadata_invalid$"):
            validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize("key", ("Authorization", "auth", "private_key"))
def test_output_manifest_rejects_structured_credentials_before_publish(
    key: str,
) -> None:
    payload = _prediction_manifest_payload(ridge=True)
    model_metadata = payload["metadata"]["model_metadata"]
    model_metadata["hyperparameters"][key] = "secret-value"
    model_metadata["resolved_hyperparameters"][key] = "secret-value"
    assert (
        validate_prediction_output_metadata_shape(payload["metadata"])
        is payload["metadata"]
    )
    with pytest.raises(ValueError, match="^output_manifest_metadata_invalid$"):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("equal_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition", "output_manifest_definition_id_invalid"),
        ("nested_schema_id", "output_manifest_metadata_invalid"),
        ("nested_mapping_schema_id", "output_manifest_metadata_invalid"),
        ("nested_file", "output_file_path_invalid"),
        ("casefold_duplicate", "output_files_invalid"),
    ),
)
def test_output_manifest_rejects_noncurrent_identity(mutation: str, error: str) -> None:
    payload = _prediction_manifest_payload()
    if mutation == "equal_ids":
        payload["output_id"] = "prediction_outputs"
    elif mutation == "unknown_definition":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "nested_mapping_schema_id":
        payload["metadata"] = {"nested": MappingProxyType({"schema_id": "forbidden"})}
    elif mutation == "nested_file":
        payload["files"] = ["nested/prediction.parquet"]
    else:
        payload["files"] = ["prediction.parquet", "PREDICTION.PARQUET"]
    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)


@pytest.mark.parametrize(
    "mutation", ("prediction_mode", "substituted_file", "missing_file", "extra_file")
)
def test_output_manifest_binds_prediction_metadata_and_exact_files(
    mutation: str,
) -> None:
    payload = _prediction_manifest_payload()
    if mutation == "prediction_mode":
        payload["metadata"]["prediction_mode"] = "unknown"
        error = "output_manifest_metadata_invalid"
    elif mutation == "substituted_file":
        payload["files"] = ["opaque.parquet"]
        error = "output_manifest_definition_files_mismatch"
    elif mutation == "missing_file":
        payload["files"] = []
        error = "output_files_invalid"
    else:
        payload["files"].append("extra.parquet")
        error = "output_manifest_definition_files_mismatch"
    with pytest.raises(ValueError, match=error):
        validate_output_manifest_business_envelope(payload)


def _native_prediction_bundle(tmp_path, *, crypto=False):
    import pandas as pd

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.models import (
        CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    )
    from quant_research_platform.storage.layout import StorageLayout

    frame = make_price_volume_prediction_output_frame()
    frame = frame.loc[frame.index.repeat(2)].reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(["2014-01-03", "2014-01-04"])
    frame["asset_id"] = ["asset-a", "asset-b"]
    frame["prediction_score"] = [0.25, 0.5]
    spec = replace(make_price_volume_prediction_spec(), test_end="2014-01-04")
    model = dict(frame.attrs["model_metadata"])
    model.update(test_end=spec.test_end, oos_end=spec.test_end)
    prediction = dict(frame.attrs["prediction_metadata"])
    prediction.update(
        prediction_rows=2,
        requested_test_end=spec.test_end,
        effective_test_end=spec.test_end,
    )
    output_spec = make_prediction_report_spec()
    if crypto:
        spec = replace(spec, train_end=None, prediction_mode="crypto_oos")
        output_spec = replace(output_spec, market_id="crypto")
        prediction.update(train_end=None, prediction_mode="crypto_oos")
        model = {
            "model_artifact_id": "crypto_models",
            "model_artifact_identity": "a" * 64,
            "model_artifact_schema": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
            "model_id": spec.model_id,
            "asset_ids": ["asset-a", "asset-b"],
            "score_to_label": None,
            "fold_models": [
                {
                    "fold_id": "fold_one",
                    "prediction_window": {
                        "start": spec.test_start,
                        "end": spec.test_end,
                    },
                    "fit_windows": [
                        {"start": "2013-12-30", "end": "2014-01-01"},
                        {"start": "2014-01-02", "end": "2014-01-02"},
                    ],
                    "fit_cutoff": "2014-01-02",
                    "training_asset_ids": [],
                    "evaluation_asset_ids": ["asset-a", "asset-b"],
                }
            ],
        }
    root = write_prediction_outputs(
        spec=spec,
        output_spec=output_spec,
        layout=StorageLayout(tmp_path),
        output_chunks=(frame,),
        model_metadata=model,
        prediction_metadata_factory=lambda: prediction,
    )
    return root, spec, output_spec, frame, model, prediction


@pytest.mark.parametrize("failure", ("none", "reject", "close", "both"))
@pytest.mark.parametrize("validation_only", (False, True))
def test_prediction_reader_closes_native_batch_iterator_before_outer_file(
    tmp_path, monkeypatch, failure, validation_only
):
    import sys
    from contextlib import contextmanager

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs import bundle_validation
    from quant_research_platform.contracts.time import TimeWindow

    root, spec, _output, _frame, _model, _prediction = _native_prediction_bundle(
        tmp_path
    )
    path = root / "prediction.parquet"
    if failure in {"reject", "both"}:
        table = pq.read_table(path)
        field = table.schema.field("prediction_score")
        table = table.set_column(
            table.schema.get_field_index("prediction_score"),
            field,
            pa.array([float("nan"), 0.5], type=field.type),
        )
        pq.write_table(table, path)
    original = bundle_validation._opened_parquet
    secondary = RuntimeError("synthetic_batch_close")
    resources = []
    events = []

    class Parquet:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def iter_batches(self, **_kwargs):
            native = self.native.iter_batches(batch_size=1, use_threads=False)
            resources.append(native)

            class Batches:
                def __iter__(self):
                    return self

                def __next__(self):
                    return next(native)

                def close(self):
                    events.append(("batches_close", sys.exception()))
                    native.close()
                    if failure in {"close", "both"}:
                        raise secondary

            return Batches()

    @contextmanager
    def observed_open(*args, **kwargs):
        with original(*args, **kwargs) as (parquet, identity, handle):
            resources.append(handle)
            try:
                yield Parquet(parquet), identity, handle
            finally:
                events.append(("outer_exit", None))

    monkeypatch.setattr(bundle_validation, "_opened_parquet", observed_open)
    reader = (
        bundle_validation._validate_prediction_output_bundle
        if validation_only
        else bundle_validation.read_prediction_output_bundle
    )
    if failure == "none":
        _manifest, lookup = reader(
            root,
            expected_output_id=spec.prediction_id,
            time_window=TimeWindow("2014-01-04", "2014-01-04"),
        )
        if validation_only:
            assert lookup is None
        else:
            assert lookup.row_count == 1
    else:
        with pytest.raises(BaseException) as caught:
            reader(
                root,
                expected_output_id=spec.prediction_id,
                time_window=TimeWindow("2014-01-04", "2014-01-04"),
            )
        if failure in {"reject", "both"}:
            assert isinstance(caught.value, ValueError)
            assert str(caught.value) == "invalid_exact_float_value"
            assert events[0][1] is caught.value
            if failure == "both":
                assert caught.value.__notes__ == [
                    "iterator close also failed: RuntimeError"
                ]
        else:
            assert caught.value is secondary
    assert [kind for kind, _primary in events] == ["batches_close", "outer_exit"]
    assert resources[0].closed
    assert resources[1].gi_frame is None


@pytest.mark.parametrize(
    "window",
    (
        {},
        {"start": "2014-01-03"},
        {"end": "2014-01-04"},
        {"start": "2014-01-03", "end": "2014-01-04", "extra": True},
        [],
        None,
        1,
        True,
        "2014-01-03",
    ),
    ids=(
        "empty",
        "missing_end",
        "missing_start",
        "extra",
        "array",
        "null",
        "int",
        "bool",
        "string",
    ),
)
def test_crypto_prediction_window_shape_rejects_at_native_write_and_read(
    tmp_path, window
):
    import json
    from copy import deepcopy

    from quant_research_platform.artifacts.outputs import (
        read_prediction_output_bundle,
        write_prediction_outputs,
    )
    from quant_research_platform.artifacts.outputs.prediction_validation import (
        validate_model_metadata_shape,
    )
    from quant_research_platform.storage.layout import StorageLayout

    root, spec, output, frame, model, prediction = _native_prediction_bundle(
        tmp_path, crypto=True
    )
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    assert validate_model_metadata_shape(model) is model
    _metadata, lookup = read_prediction_output_bundle(
        root, expected_output_id=spec.prediction_id
    )
    assert lookup.row_count == 2
    bad = deepcopy(model)
    bad["fold_models"][0]["prediction_window"] = window
    with pytest.raises(ValueError, match="^invalid_model_metadata$"):
        validate_model_metadata_shape(bad)
    with pytest.raises(ValueError, match="^invalid_model_metadata$"):
        write_prediction_outputs(
            spec=spec,
            output_spec=output,
            layout=StorageLayout(tmp_path),
            output_chunks=(frame,),
            model_metadata=bad,
            prediction_metadata_factory=lambda: prediction,
        )
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    manifest = root / "manifest.json"
    mutated = json.loads(before["manifest.json"])
    mutated["metadata"]["model_metadata"] = bad
    manifest.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(
        ValueError, match="^output_manifest_metadata_invalid$"
    ) as caught:
        read_prediction_output_bundle(root, expected_output_id=spec.prediction_id)
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "invalid_model_metadata"
    manifest.write_bytes(before["manifest.json"])
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    _metadata, lookup = read_prediction_output_bundle(
        root, expected_output_id=spec.prediction_id
    )
    assert lookup.row_count == 2


@pytest.mark.parametrize(
    "case", ("endpoints", "before", "after", "open_before", "empty")
)
def test_prediction_writer_enforces_stream_window_before_publication(tmp_path, case):
    import pandas as pd

    from quant_research_platform.artifacts.outputs import (
        validate_output_bundle,
        write_prediction_outputs,
    )
    from quant_research_platform.storage.errors import OutputAlreadyExistsError
    from quant_research_platform.storage.layout import StorageLayout

    root, spec, output_spec, frame, model, prediction = _native_prediction_bundle(
        tmp_path / "good"
    )
    original = {path.name: path.read_bytes() for path in root.iterdir()}
    if case in {"before", "open_before"}:
        frame.loc[0, "timestamp"] = pd.Timestamp("2014-01-01")
    elif case == "after":
        frame.loc[0, "timestamp"] = pd.Timestamp("2014-01-05")
    elif case == "empty":
        frame = frame.iloc[:0]
    if case == "open_before":
        spec = replace(spec, test_end=None, is_final_oos_window=True)
        model.update(test_end=None)
        prediction.update(requested_test_end=None, is_final_oos_window=True)
    kwargs = dict(
        spec=spec,
        output_spec=output_spec,
        output_chunks=(frame.iloc[:1], frame.iloc[1:]),
        model_metadata=model,
        prediction_metadata_factory=lambda: prediction,
    )
    if case == "endpoints":
        published = write_prediction_outputs(
            layout=StorageLayout(tmp_path / "new"), **kwargs
        )
        validate_output_bundle(published)
    else:
        with pytest.raises(
            ValueError,
            match="empty_prediction_window"
            if case == "empty"
            else "prediction_metadata_inconsistent",
        ):
            write_prediction_outputs(layout=StorageLayout(tmp_path / "new"), **kwargs)
        assert not list((tmp_path / "new").rglob("prediction.parquet"))
    with pytest.raises(OutputAlreadyExistsError):
        write_prediction_outputs(layout=StorageLayout(tmp_path / "good"), **kwargs)
    assert {path.name: path.read_bytes() for path in root.iterdir()} == original
