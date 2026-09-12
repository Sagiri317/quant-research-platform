"""Production-fold boundary checks using the workflow's published dataset."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.business_manifest import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.models import fitting, production
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.experiments import read_trial_ledger_reference


def check_crypto_fold_boundaries(*, layout, dataset_root, profile, saved, monkeypatch):
    dataset_id = saved.envelope["semantic_metadata"]["target_contract"]["protocol"][
        "scenario_id"
    ]
    protocol = deepcopy(
        saved.envelope["semantic_metadata"]["target_contract"]["protocol"]
    )
    # A genuine undeclared gap containing two observed grid points.
    protocol["research_plan"]["folds"][1]["validation"]["start"] = (
        "1970-01-01T00:00:04.300000"
    )
    ids = tuple(spec.model_id for spec in profile.models.list())
    windows = []
    mutated = False
    with runtime_resource_policy_scope() as policy:
        frame = read_dataset(
            dataset_root, mode=DatasetReadMode.MATERIALIZED, resource_policy=policy
        )

        def chunks(_layout, *, time_window, columns, **kwargs):
            windows.append(time_window)
            selected = frame.loc[
                frame.timestamp.between(
                    pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
                ),
                list(columns),
            ].copy()
            if mutated:
                for label in {spec.label_column for spec in profile.models.list()}:
                    if label in selected:
                        late = selected[f"{label}_available_at"] > pd.Timestamp(
                            "1970-01-01T00:00:06"
                        )
                        selected.loc[late, label] = 100.0
            yield selected

        def results(run_id):
            return production.build_crypto_fold_models(
                layout=layout.with_target_roots({dataset_id: dataset_root}),
                dataset_catalog=profile.research_datasets,
                model_catalog=profile.models,
                dataset_id=dataset_id,
                model_ids=ids,
                tie_break_order=ids,
                time_window=TimeWindow(
                    "1970-01-01T00:00:06.500000", "1970-01-01T00:00:08.500000"
                ),
                resource_policy=policy,
                protocol_manifest=protocol,
                run_id=run_id,
                research_protocol_identity=canonical_json_hash(protocol),
                model_target_id=f"{dataset_id}_models",
                backend_versions={"fixture": "1"},
            )

        def fit(run_id):
            observed = {}
            with closing_iterator(results(run_id)) as source:
                for owned in source:
                    member = owned.value

                    def count(
                        parts, cutoff, spec=member.spec, assets=member.training_assets
                    ):
                        mask = pd.Series(False, index=frame.index)
                        for part in parts:
                            mask |= frame.timestamp.between(
                                pd.Timestamp(part.start), pd.Timestamp(part.end)
                            )
                        if assets:
                            mask &= frame.asset_id.isin(assets)
                        mask &= frame.eligible
                        mask &= frame[f"{spec.label_column}_available_at"].le(
                            pd.Timestamp(cutoff)
                        )
                        return len(
                            frame.loc[mask].dropna(
                                subset=[
                                    *spec.input_columns,
                                    spec.label_column,
                                ]
                            )
                        )

                    _, records = read_trial_ledger_reference(
                        layout, member.trial_reference
                    )
                    assert records[0].validation_metrics["n_train"] == count(
                        (member.fold.train,), member.fold.train.end
                    )
                    assert records[0].validation_metrics["validation_rows"] == count(
                        (member.fold.validation,), member.fold.validation.end
                    )
                    assert member.result.metadata[
                        "rows_after_input_label_dropna"
                    ] == count(
                        (member.fold.train, member.fold.validation),
                        member.fold.validation.end,
                    )
                    probe = frame.dropna(subset=list(member.spec.input_columns)).iloc[
                        :6
                    ]
                    predict = fitting.prediction_callable(
                        member.spec.input_columns,
                        member.result.prediction_model,
                    )
                    observed[member.spec.model_id] = (
                        predict(probe).copy(),
                        member.result.metadata["rows_after_input_label_dropna"],
                    )
                    predict = member = probe = None
                    owned.release(owner="test.crypto.fold")
            return observed

        with monkeypatch.context() as patch:
            patch.setattr(production, "iter_research_dataset_chunks", chunks)
            before = fit("run_crypto_boundaries_before")
            # Mutate only excluded times in the original borrowed source frame.
            excluded = (
                (frame.timestamp < pd.Timestamp("1970-01-01T00:00:01"))
                | frame.timestamp.between(
                    pd.Timestamp("1970-01-01T00:00:04.100000"),
                    pd.Timestamp("1970-01-01T00:00:04.200000"),
                )
                | (frame.timestamp > pd.Timestamp("1970-01-01T00:00:06"))
            )
            labels = list({spec.label_column for spec in profile.models.list()})
            frame.loc[excluded, labels] = 100.0
            mutated = True
            after = fit("run_crypto_boundaries_after")
            assert len(before) == len(after) == 9
            for model_id in ids:
                np.testing.assert_array_equal(before[model_id][0], after[model_id][0])
                assert before[model_id][1] == after[model_id][1]
            assert {(pd.Timestamp(w.start), pd.Timestamp(w.end)) for w in windows} == {
                (
                    pd.Timestamp("1970-01-01T00:00:01"),
                    pd.Timestamp("1970-01-01T00:00:04"),
                ),
                (
                    pd.Timestamp("1970-01-01T00:00:04.300000"),
                    pd.Timestamp("1970-01-01T00:00:06"),
                ),
            }
            assert policy.used_memory_bytes == 0

            primary = KeyboardInterrupt("after refit")
            finish = production.TrialRecorder.finish
            release = OwnedValue.release

            def fail_finish(self, *args, **kwargs):
                if kwargs.get("error") is None:
                    raise primary
                return finish(self, *args, **kwargs)

            def fail_release(self, *, owner):
                release(self, owner=owner)
                if owner == "research.model.crypto":
                    raise OSError("release failed after closing")

            patch.setattr(production.TrialRecorder, "finish", fail_finish)
            patch.setattr(OwnedValue, "release", fail_release)
            with pytest.raises(KeyboardInterrupt) as caught:
                with closing_iterator(
                    results("run_crypto_boundaries_failure")
                ) as source:
                    next(source)
            assert caught.value is primary
            assert any("release also failed" in note for note in primary.__notes__)
            assert policy.used_memory_bytes == 0
        frame = None


def check_crypto_prediction_staging(*, layout, prediction_root, saved):
    from quant_research_platform.artifacts.outputs import write_prediction_outputs
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
        write_semantic_artifact,
    )

    old = read_semantic_artifact_record(prediction_root)
    payload = old[0]
    entry = next(item for item in saved.models if "leave_btc_out" in item["model_id"])
    spec = PredictionSpec(
        prediction_id=f"{entry['model_id']}_prediction",
        model_id=entry["model_id"],
        research_dataset_id=payload["semantic_metadata"]["target_contract"]["protocol"][
            "scenario_id"
        ],
        train_end=None,
        test_start="1970-01-01T00:00:04",
        test_end="1970-01-01T00:00:08.500000",
        prediction_mode="crypto_oos",
        training_asset_ids=tuple(entry["training_asset_ids"]),
        evaluation_asset_ids=tuple(entry["evaluation_asset_ids"]),
    )
    logical_root = prediction_root.parent.parent
    prior = {p.name for p in prediction_root.parent.iterdir()}
    with runtime_resource_policy_scope() as policy:
        for index, (stamp, asset) in enumerate(
            (("05.500000", "btc_usdt"), ("04.500000", "eth_usdt"))
        ):
            envelope = build_business_manifest_envelope(
                **{
                    key: (
                        dict(payload[key], start=f"1970-01-01T00:00:04.00{index + 1}")
                        if key == "research_window"
                        else payload[key]
                    )
                    for key in (
                        "logical_artifact_id",
                        "schema_identity",
                        "protocol_or_spec_identity",
                        "software_identity",
                        "research_window",
                        "upstream_identities",
                        "semantic_metadata",
                    )
                }
            )
            frame = pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp(f"1970-01-01T00:00:{stamp}")],
                    "asset_id": [asset],
                    "prediction_score": [0.1],
                }
            )

            def write(staging, frame=frame):
                write_prediction_outputs(
                    spec=spec,
                    output_spec=OutputSpec(
                        spec.prediction_id,
                        "prediction_outputs",
                        f"run_{spec.prediction_id}",
                        "crypto",
                        "prediction",
                        OutputScope.LOCAL_ONLY,
                    ),
                    layout=StorageLayout(staging),
                    output_chunks=(frame,),
                    model_metadata=saved.prediction_lineage(entry["model_id"]),
                    prediction_metadata_factory=lambda: pytest.fail(
                        "invalid row must reject before metadata publication"
                    ),
                    resource_policy=policy,
                )

            with pytest.raises(
                ValueError, match="crypto_prediction_fold_scope_misaligned"
            ):
                write_semantic_artifact(logical_root, envelope, write)
            assert {p.name for p in prediction_root.parent.iterdir()} == prior
            assert read_semantic_artifact_record(prediction_root) == old
            assert policy.used_memory_bytes == 0


def check_crypto_model_staging(*, layout, profile, saved):
    import hashlib
    import json
    import pickle
    import shutil
    from pathlib import Path
    from types import SimpleNamespace

    from sklearn.dummy import DummyRegressor

    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.research.models import persisted
    from quant_research_platform.research.models.persisted import (
        _read_model_member,
        write_crypto_model_artifact,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        protocol as crypto_protocol,
    )
    from quant_research_platform.runtime import resources
    from quant_research_platform.storage import parquet as parquet_storage
    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_operational_metadata,
        read_semantic_artifact_record,
    )

    original = read_semantic_artifact_record(saved.instance_root)
    references = read_semantic_artifact_operational_metadata(saved.instance_root)[
        "model_trial_ledgers"
    ]
    contract = saved.envelope["semantic_metadata"]["target_contract"]["protocol"]
    protocol = crypto_protocol(
        contract["scenario_id"], research_plan=contract["research_plan"]
    )
    logical_root = saved.instance_root.parent.parent
    previous_instances = {p.name for p in saved.instance_root.parent.iterdir()}
    with runtime_resource_policy_scope() as policy:
        with policy.lease("test.crypto.model_corruption", 8 * 1024**2):
            entry = saved.models[0]
            columns = entry["model_metadata"]["input_columns"]
            probe = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
            wrong = DummyRegressor(strategy="constant", constant=0.125).fit(
                probe, np.zeros(2)
            )
            copied = layout.project_root / "corrupt-model" / saved.instance_root.name
            shutil.copytree(
                native_filesystem_path(saved.instance_root),
                native_filesystem_path(copied),
            )
            relative = f"models/{entry['fold_id']}/{entry['model_id']}.pkl"
            raw = pickle.dumps(wrong, protocol=pickle.HIGHEST_PROTOCOL)
            Path(native_filesystem_path(copied / relative)).write_bytes(raw)
            manifest_path = Path(native_filesystem_path(copied / "manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            member = next(row for row in manifest["members"] if row["path"] == relative)
            member.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            changed = persisted.read_crypto_model_artifact(
                copied,
                expected_identity=saved.envelope["instance_identity"],
                trial_layout=layout,
            )
            with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
                with persisted.read_crypto_model_member(
                    changed, changed.models[0], resource_policy=policy
                ):
                    pytest.fail("wrong predictor reached the consumer")
            del probe, wrong, raw, manifest, changed, member
        assert policy.used_memory_bytes == 0
        assert read_semantic_artifact_record(saved.instance_root) == original
        validation_entry = next(
            entry for entry in saved.models if entry["validation_prediction_path"]
        )
        validation_fold = next(
            fold
            for fold in protocol.research_plan.folds
            if fold.fold_id == validation_entry["fold_id"]
        )
        validation_spec = profile.models.get(validation_entry["model_id"])
        validation = pd.concat(
            persisted._crypto_validation_chunks(
                saved.instance_root, saved.member_records, validation_entry
            ),
            ignore_index=True,
        ).rename(columns={"label": validation_spec.label_column})
        for failure in ("disk", "primary"):
            primary = KeyboardInterrupt("validation_training_primary")
            opened = []
            closed = []

            def observed_results(*, on_validation, primary=primary):
                on_validation(validation_fold, validation_spec, validation.iloc[:1])
                raise primary

            writer_type = parquet_storage.pq.ParquetWriter

            class ClosingWriter:
                def __init__(
                    self, *args, opened=opened, writer_type=writer_type, **kwargs
                ):
                    opened.append(args[0])
                    self.writer = writer_type(*args, **kwargs)

                def write_table(self, *args, **kwargs):
                    return self.writer.write_table(*args, **kwargs)

                def add_key_value_metadata(self, *args, **kwargs):
                    return self.writer.add_key_value_metadata(*args, **kwargs)

                def close(self, closed=closed):
                    self.writer.close()
                    closed.append(True)
                    raise RuntimeError("validation_close_secondary")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(parquet_storage.pq, "ParquetWriter", ClosingWriter)
                if failure == "disk":
                    disk_admission = parquet_storage.require_free_disk_headroom

                    def reject_validation_disk(
                        path, disk_admission=disk_admission, **kwargs
                    ):
                        if Path(path).name == "validation":
                            raise RuntimeError("validation_disk_admission")
                        return disk_admission(path, **kwargs)

                    patch.setattr(
                        parquet_storage,
                        "require_free_disk_headroom",
                        reject_validation_disk,
                    )
                with pytest.raises(
                    RuntimeError if failure == "disk" else KeyboardInterrupt,
                    match=(
                        "validation_disk_admission"
                        if failure == "disk"
                        else "validation_training_primary"
                    ),
                ) as raised:
                    write_crypto_model_artifact(
                        logical_root,
                        envelope=build_business_manifest_envelope(
                            **{
                                key: (
                                    dict(
                                        saved.envelope[key],
                                        start="1970-01-01T00:00:04.000005",
                                    )
                                    if key == "research_window"
                                    else saved.envelope[key]
                                )
                                for key in (
                                    "logical_artifact_id",
                                    "schema_identity",
                                    "protocol_or_spec_identity",
                                    "software_identity",
                                    "research_window",
                                    "upstream_identities",
                                    "semantic_metadata",
                                )
                            }
                        ),
                        model_results=observed_results,
                        trial_layout=layout,
                    )
                if failure == "disk":
                    assert not opened and not closed
                else:
                    assert raised.value is primary
                    assert len(opened) == len(closed) == 1
                    assert any("close" in note for note in primary.__notes__)
            assert policy.used_memory_bytes == 0
            assert read_semantic_artifact_record(saved.instance_root) == original
            assert {
                p.name for p in saved.instance_root.parent.iterdir()
            } == previous_instances
            assert not tuple(logical_root.glob(".*staging*"))
        validation = None
        for index, failure in enumerate(("seed", "missing_fold", "disk", "primary")):
            envelope = build_business_manifest_envelope(
                **{
                    key: (
                        dict(
                            saved.envelope[key],
                            start=f"1970-01-01T00:00:04.00{index + 1}",
                        )
                        if key == "research_window"
                        else saved.envelope[key]
                    )
                    for key in (
                        "logical_artifact_id",
                        "schema_identity",
                        "protocol_or_spec_identity",
                        "software_identity",
                        "research_window",
                        "upstream_identities",
                        "semantic_metadata",
                    )
                }
            )

            def members(failure=failure):
                for number, entry in enumerate(saved.models):
                    if failure == "missing_fold" and entry["fold_id"] == "fold_2":
                        break
                    with _read_model_member(
                        saved.instance_root,
                        saved.member_records,
                        f"models/{entry['fold_id']}/{entry['model_id']}.pkl",
                        resource_policy=policy,
                        workspace_factor=4,
                    ) as owned:
                        payload = owned.value
                        metadata = deepcopy(entry["model_metadata"])
                        if failure == "seed" and number == 0:
                            metadata["random_seed"] += 1
                        member = production.CryptoFoldModelResult(
                            next(
                                f
                                for f in protocol.research_plan.folds
                                if f.fold_id == entry["fold_id"]
                            ),
                            profile.models.get(entry["model_id"]),
                            fitting.ModelFitResult(metadata, payload),
                            tuple(entry["training_asset_ids"]),
                            tuple(entry["evaluation_asset_ids"]),
                            references[number],
                        )
                        owned.value = member
                        member = payload = None
                        try:
                            yield owned
                        finally:
                            assert owned.value is None
                            assert not owned.lease.active

            error = (
                "crypto_model_parameters_misaligned"
                if failure == "seed"
                else "crypto_model_fold_coverage_invalid"
            )
            primary = KeyboardInterrupt("model_pickle_primary")
            with pytest.MonkeyPatch.context() as patch:
                if failure == "disk":
                    patch.setattr(
                        resources.shutil,
                        "disk_usage",
                        lambda _path: SimpleNamespace(free=1),
                    )

                    def reject_dump(*_args, **_kwargs):
                        raise AssertionError("pickle_started_before_disk_admission")

                    patch.setattr(persisted.pickle, "dump", reject_dump)
                if failure == "primary":
                    close = resources.MemoryLease.close

                    def close_then_fail(lease, close=close):
                        close(lease)
                        if lease.owner == "research.model.crypto.persist":
                            raise RuntimeError("secondary_model_release")

                    def fail_dump(*_args, primary=primary, **_kwargs):
                        raise primary

                    patch.setattr(resources.MemoryLease, "close", close_then_fail)
                    patch.setattr(persisted.pickle, "dump", fail_dump)
                expected = (
                    RuntimeError
                    if failure == "disk"
                    else KeyboardInterrupt
                    if failure == "primary"
                    else ValueError
                )
                message = (
                    "insufficient_free_disk"
                    if failure == "disk"
                    else "model_pickle_primary"
                    if failure == "primary"
                    else error
                )
                with pytest.raises(expected, match=message) as raised:
                    write_crypto_model_artifact(
                        logical_root,
                        envelope=envelope,
                        model_results=lambda **_kwargs: members(),
                        trial_layout=layout,
                    )
                if failure == "primary":
                    assert raised.value is primary
                    assert any(
                        "lease close also failed" in note for note in primary.__notes__
                    )
            assert {
                p.name for p in saved.instance_root.parent.iterdir()
            } == previous_instances
            assert read_semantic_artifact_record(saved.instance_root) == original
            assert not tuple(logical_root.glob(".*staging*"))
            assert policy.used_memory_bytes == 0


def check_crypto_wide_fold_artifact(
    *, layout, dataset_root, profile, saved, monkeypatch
):
    """Persist 22 synthetic folds through real model fitting and trial ledgers."""
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.research.models import persisted
    from quant_research_platform.research.protocols.crypto_hft import (
        protocol as crypto_protocol,
    )

    original = saved.envelope["semantic_metadata"]["target_contract"]["protocol"]
    plan = deepcopy(original["research_plan"])
    template = deepcopy(plan["folds"][0])
    plan["folds"] = []
    for index in range(22):
        fold = deepcopy(template)
        fold["fold_id"] = f"wide_{index}"
        for name in ("train", "validation", "development_test"):
            fold[name] = {
                key: (
                    pd.Timestamp(value) + pd.Timedelta(seconds=100 * index)
                ).isoformat()
                for key, value in fold[name].items()
            }
        plan["folds"].append(fold)
    last = pd.Timestamp(plan["folds"][-1]["development_test"]["end"])
    plan["final_holdout"] = {
        "start": (last + pd.Timedelta(seconds=10)).isoformat(),
        "end": (last + pd.Timedelta(seconds=11)).isoformat(),
    }
    protocol = crypto_protocol(original["scenario_id"], research_plan=plan).manifest()
    fields = {
        key: saved.envelope[key]
        for key in (
            "logical_artifact_id",
            "schema_identity",
            "software_identity",
            "upstream_identities",
        )
    }
    envelope = build_business_manifest_envelope(
        **fields,
        protocol_or_spec_identity=canonical_json_hash(protocol),
        research_window={
            "start": plan["folds"][0]["development_test"]["start"],
            "end": plan["folds"][-1]["development_test"]["end"],
            "access": "development",
        },
        semantic_metadata={"target_contract": {"protocol": protocol}},
    )
    from quant_research_platform.research.models.supervised import preflight_model_specs

    preflight_model_specs(profile.models.list())
    ids = tuple(spec.model_id for spec in profile.models.list())
    with runtime_resource_policy_scope() as policy:
        frame = read_dataset(
            dataset_root, mode=DatasetReadMode.MATERIALIZED, resource_policy=policy
        )
        frame = frame.loc[
            frame.timestamp.between(
                pd.Timestamp(template["train"]["start"]),
                pd.Timestamp(template["validation"]["end"]),
            )
        ].copy()

        def chunks(_layout, *, time_window, columns, **kwargs):
            index = int(
                (
                    pd.Timestamp(time_window.start)
                    - pd.Timestamp(template["train"]["start"])
                ).total_seconds()
                // 100
            )
            shifted = frame.copy()
            for column in shifted:
                if pd.api.types.is_datetime64_any_dtype(shifted[column]):
                    shifted[column] += pd.Timedelta(seconds=100 * index)
            yield shifted.loc[
                shifted.timestamp.between(
                    pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
                ),
                list(columns),
            ].copy()

        def results(*, on_validation):
            return production.build_crypto_fold_models(
                layout=layout.with_target_roots(
                    {original["scenario_id"]: dataset_root}
                ),
                dataset_catalog=profile.research_datasets,
                model_catalog=profile.models,
                dataset_id=original["scenario_id"],
                model_ids=ids,
                tie_break_order=ids,
                time_window=TimeWindow(
                    envelope.research_window["start"], envelope.research_window["end"]
                ),
                resource_policy=policy,
                protocol_manifest=protocol,
                run_id="run_crypto_wide",
                research_protocol_identity=canonical_json_hash(protocol),
                model_target_id=envelope.logical_artifact_id,
                backend_versions={"fixture": "1"},
                on_validation=on_validation,
            )

        with monkeypatch.context() as patch:
            patch.setattr(production, "iter_research_dataset_chunks", chunks)
            root = persisted.write_crypto_model_artifact(
                layout.project_root / "wide-models",
                envelope=envelope,
                model_results=results,
                trial_layout=layout,
            )
        loaded = persisted.read_crypto_model_artifact(
            root, expected_identity=envelope.instance_identity, trial_layout=layout
        )
        assert len(loaded.models) == 22 * 9
        assert len(loaded.member_records) == 265
        for entry in loaded.models:
            with persisted.read_crypto_model_member(
                loaded, entry, resource_policy=policy
            ) as model:
                assert model is not None
        assert policy.used_memory_bytes == 0
