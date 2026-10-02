"""Production-fold boundary checks using the workflow's published dataset."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.business_manifest import canonical_json_hash
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.models import fitting, production
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.experiments import read_trial_ledger_reference
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_iterdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
)


def check_crypto_fold_boundaries(*, layout, dataset_root, profile, saved, monkeypatch):
    dataset_id = saved.envelope["semantic_metadata"]["target_contract"]["protocol"][
        "scenario_id"
    ]
    protocol = deepcopy(
        saved.envelope["semantic_metadata"]["target_contract"]["protocol"]
    )
    # A real gap between training and validation contains excluded grid points.
    protocol["research_plan"]["folds"][1]["validation"]["start"] = (
        "1970-01-01T00:00:06.700000"
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
                            "1970-01-01T00:00:08.500000"
                        )
                        selected.loc[late, label] = 100.0
            yield selected

        def results(run_id):
            requested = TimeWindow(
                "1970-01-01T00:00:06.500000", "1970-01-01T00:00:08.500000"
            )
            folds, _holdout = production._crypto_model_chronology(protocol, requested)
            with open_dataset_session(
                dataset_root,
                columns=None,
                time_window=TimeWindow(
                    min(fold.train.start for fold in folds),
                    max(fold.validation.end for fold in folds),
                ),
                resource_policy=policy,
            ) as dataset_session:
                yield from production.build_crypto_fold_models(
                    dataset_session=dataset_session,
                    layout=layout.with_target_roots({dataset_id: dataset_root}),
                    dataset_catalog=profile.research_datasets,
                    model_catalog=profile.models,
                    dataset_id=dataset_id,
                    model_ids=ids,
                    tie_break_order=ids,
                    time_window=requested,
                    resource_policy=policy,
                    protocol_manifest=protocol,
                    run_id=run_id,
                    research_protocol_identity=canonical_json_hash(protocol),
                    model_target_id=f"{dataset_id}_models",
                    backend_versions={"fixture": "1"},
                    on_validation=lambda *args: None,
                    on_selection=lambda *args: None,
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

                    with read_trial_ledger_reference(
                        layout, member.selection_trial_reference
                    ) as (_, records):
                        assert records[0].metrics["n_train"] == count(
                            (member.fold.train,), member.fold.train.end
                        )
                        assert records[0].metrics["validation_rows"] == count(
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
            patch.setattr(production, "_model_input_chunks", chunks)
            before = fit("run_crypto_boundaries_before")
            # Mutate only excluded times in the original borrowed source frame.
            excluded = (
                (frame.timestamp < pd.Timestamp("1970-01-01T00:00:01"))
                | frame.timestamp.between(
                    pd.Timestamp("1970-01-01T00:00:04.100000"),
                    pd.Timestamp("1970-01-01T00:00:06.600000"),
                )
                | (frame.timestamp > pd.Timestamp("1970-01-01T00:00:08.500000"))
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
                    pd.Timestamp("1970-01-01T00:00:06.700000"),
                    pd.Timestamp("1970-01-01T00:00:08.500000"),
                ),
            }
            assert policy.used_memory_bytes == 0

            def no_legal_validation(*args, **kwargs):
                for selected in chunks(*args, **kwargs):
                    if pd.Timestamp(kwargs["time_window"].start) == pd.Timestamp(
                        "1970-01-01T00:00:06.700000"
                    ):
                        selected.loc[:, labels] = np.nan
                    yield selected

            with monkeypatch.context() as empty_patch:
                empty_patch.setattr(
                    production, "_model_input_chunks", no_legal_validation
                )
                empty_patch.setattr(
                    production.fitting,
                    "_fit_model_result_owned_from_frames",
                    lambda *args, **kwargs: pytest.fail(
                        "empty validation reached fitting"
                    ),
                )
                empty_patch.setattr(
                    production.TrialRecorder,
                    "begin",
                    lambda *args, **kwargs: pytest.fail(
                        "empty validation wrote a trial"
                    ),
                )
                with pytest.raises(
                    PortfolioTargetBlocked,
                    match="^crypto_model_validation_legal_rows_missing$",
                ):
                    with closing_iterator(
                        results("run_crypto_empty_validation")
                    ) as source:
                        next(source)
            assert policy.used_memory_bytes == 0

            primary = KeyboardInterrupt("after refit")
            finish = production.TrialRecorder.finish
            release = OwnedValue.release

            def fail_finish(self, *args, **kwargs):
                if self.plan.stage == "refit" and kwargs.get("error") is None:
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
    entry = next(
        item for item in saved.final_models if "leave_btc_out" in item["model_id"]
    )
    spec = PredictionSpec(
        prediction_id=f"{entry['model_id']}_prediction",
        model_id=entry["model_id"],
        research_dataset_id=payload["semantic_metadata"]["target_contract"]["protocol"][
            "scenario_id"
        ],
        train_end=None,
        test_start="1970-01-01T00:00:08.600000",
        test_end="1970-01-01T00:00:09",
        prediction_mode="crypto_oos",
        training_asset_ids=tuple(entry["training_asset_ids"]),
        evaluation_asset_ids=tuple(entry["evaluation_asset_ids"]),
    )
    logical_root = prediction_root.parent.parent
    prior = {p.name for p in filesystem_iterdir(prediction_root.parent)}
    with runtime_resource_policy_scope() as policy:
        for index, (stamp, asset) in enumerate(
            (("08.500000", "btc_usdt"), ("08.700000", "eth_usdt"))
        ):
            envelope = build_business_manifest_envelope(
                **{
                    key: (
                        dict(
                            payload[key], start=f"1970-01-01T00:00:08.60000{index + 1}"
                        )
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
                    model_metadata=saved.prediction_lineage(
                        entry["model_id"],
                        TimeWindow(spec.test_start, spec.test_end),
                        access_window="holdout",
                    ),
                    prediction_metadata_factory=lambda: pytest.fail(
                        "invalid row must reject before metadata publication"
                    ),
                    resource_policy=policy,
                )

            with pytest.raises(
                ValueError, match="crypto_prediction_fold_scope_misaligned"
            ):
                write_semantic_artifact(logical_root, envelope, write)
            assert {p.name for p in filesystem_iterdir(prediction_root.parent)} == prior
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
    protocol = profile.protocol(contract["scenario_id"])
    logical_root = saved.instance_root.parent.parent
    previous_instances = {
        p.name for p in filesystem_iterdir(saved.instance_root.parent)
    }
    with runtime_resource_policy_scope() as policy:
        from quant_research_platform.research.protocols.crypto_hft import (
            HFT_FACTOR_COLUMNS,
        )

        for entry in saved.final_models:
            spec = profile.models.get(entry["model_id"])
            if spec.model_algorithm != "ridge_regression":
                continue
            assert tuple(entry["model_metadata"]["input_columns"]) == spec.input_columns
            with persisted.read_crypto_model_member(
                saved, entry, resource_policy=policy
            ) as model:
                assert model.scaler.columns == spec.input_columns
                probe = pd.DataFrame(
                    {
                        column: np.arange(3, dtype=float) + 1
                        for column in HFT_FACTOR_COLUMNS
                    }
                )
                predict = fitting.prediction_callable(spec.input_columns, model)
                before = predict(probe)
                for column in set(HFT_FACTOR_COLUMNS) - set(spec.input_columns):
                    probe[column] *= 1000
                np.testing.assert_array_equal(before, predict(probe))
        assert policy.used_memory_bytes == 0
        with policy.lease("test.crypto.model_corruption", 8 * 1024**2):
            from dataclasses import replace
            from uuid import uuid4

            from quant_research_platform.contracts.experiments import (
                trial_ledger_semantic_identity,
            )
            from quant_research_platform.storage.experiments import (
                append_trial_record,
                read_model_trial_ledger_reference,
            )

            with read_model_trial_ledger_reference(layout, references[0]) as (
                plan,
                terminals,
            ):
                for kind, changes in (
                    ("lineage", {"input_lineage": ("e" * 64,)}),
                    ("family", {"candidate_family_id": "different_family"}),
                ):
                    changed_plan = replace(
                        plan, experiment_id=f"experiment_{uuid4().hex}", **changes
                    )
                    terminal = replace(
                        terminals[0],
                        event_id=f"event_{uuid4().hex}",
                        run_id=f"run_{uuid4().hex}",
                        experiment_id=changed_plan.experiment_id,
                        protocol_hash=changed_plan.protocol_hash,
                        candidate_family_id=changed_plan.candidate_family_id,
                        artifact_lineage=changed_plan.input_lineage,
                    )
                    append_trial_record(
                        layout.trial_ledger_root(
                            terminal.run_id, changed_plan.experiment_id
                        ),
                        changed_plan,
                        terminal,
                    )
                    reference = {
                        "run_id": terminal.run_id,
                        "experiment_id": changed_plan.experiment_id,
                        "protocol_hash": changed_plan.protocol_hash,
                        "semantic_identity": trial_ledger_semantic_identity(
                            changed_plan, (terminal,)
                        ),
                    }
                    # The replacement trial is valid by itself, but belongs elsewhere.
                    with read_model_trial_ledger_reference(layout, reference):
                        pass
                    copied = (
                        layout.project_root / f"wrong-{kind}" / saved.instance_root.name
                    )
                    shutil.copytree(
                        native_filesystem_path(saved.instance_root),
                        native_filesystem_path(copied),
                    )
                    manifest_path = copied / "manifest.json"
                    manifest = json.loads(
                        filesystem_read_text(manifest_path, encoding="utf-8")
                    )
                    manifest["operational_metadata"]["model_trial_ledgers"][0] = (
                        reference
                    )
                    relative = f"models/{saved.models[0]['fold_id']}/metadata.json"
                    metadata_path = copied / relative
                    metadata = json.loads(
                        filesystem_read_text(metadata_path, encoding="utf-8")
                    )
                    metadata["models"][0]["trial_semantic_identity"] = reference[
                        "semantic_identity"
                    ]
                    filesystem_write_text(
                        metadata_path, json.dumps(metadata), encoding="utf-8"
                    )
                    raw = filesystem_read_bytes(metadata_path)
                    member = next(
                        row for row in manifest["members"] if row["path"] == relative
                    )
                    member.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                    filesystem_write_text(
                        manifest_path, json.dumps(manifest), encoding="utf-8"
                    )
                    with pytest.raises(
                        ValueError, match="crypto_model_trial_binding_invalid"
                    ):
                        persisted.read_crypto_model_artifact(
                            copied,
                            expected_identity=saved.envelope["instance_identity"],
                            trial_layout=layout,
                        )
                    assert (
                        read_semantic_artifact_record(saved.instance_root) == original
                    )
            entry = saved.final_models[0]
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
            relative = f"models/final/{entry['model_id']}.pkl"
            raw = pickle.dumps(wrong, protocol=pickle.HIGHEST_PROTOCOL)
            filesystem_write_bytes(copied / relative, raw)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
            member = next(row for row in manifest["members"] if row["path"] == relative)
            member.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
            changed = persisted.read_crypto_model_artifact(
                copied,
                expected_identity=saved.envelope["instance_identity"],
                trial_layout=layout,
            )
            with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
                with persisted.read_crypto_model_member(
                    changed, changed.final_models[0], resource_policy=policy
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
            saved.validation_chunks(
                fold_id=validation_entry["fold_id"],
                model_id=validation_entry["model_id"],
            ),
            ignore_index=True,
        ).rename(columns={"label": validation_spec.label_column})
        for failure in ("disk", "primary"):
            primary = KeyboardInterrupt("validation_training_primary")
            opened = []
            closed = []

            def observed_results(*, on_validation, on_selection, primary=primary):
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
                        verify_source=lambda: pytest.fail(
                            "invalid_model_reached_source_verification"
                        ),
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
                p.name for p in filesystem_iterdir(saved.instance_root.parent)
            } == previous_instances
            assert not tuple(filesystem_glob(logical_root, ".*staging*"))
        validation = None
        for index, failure in enumerate(
            (
                "seed",
                "parameters",
                "missing_fold",
                "training_assets",
                "evaluation_assets",
                "refit_window",
                "trial_reference",
                "trial_binding",
                "disk",
                "primary",
            )
        ):
            envelope = build_business_manifest_envelope(
                **{
                    key: (
                        dict(
                            saved.envelope[key],
                            start=f"1970-01-01T00:00:04.{index + 1:06d}",
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

            def members(*, on_validation, on_selection, failure=failure):
                by_identity = {ref["semantic_identity"]: ref for ref in references}
                folds = {fold.fold_id: fold for fold in protocol.research_plan.folds}
                for number, entry in enumerate(saved.models):
                    if failure == "missing_fold" and entry["fold_id"] == "fold_2":
                        continue
                    fold = folds[entry["fold_id"]]
                    spec = profile.models.get(entry["model_id"])
                    with closing_iterator(
                        saved.validation_chunks(
                            fold_id=entry["fold_id"], model_id=entry["model_id"]
                        )
                    ) as chunks:
                        for frame in chunks:
                            on_validation(fold, spec, frame)
                    reference = by_identity[
                        saved.models[1 - number]["trial_semantic_identity"]
                        if failure == "trial_binding" and number < 2
                        else entry["trial_semantic_identity"]
                    ]
                    if failure == "trial_reference" and number == 1:
                        reference = by_identity[
                            saved.models[0]["trial_semantic_identity"]
                        ]
                    on_selection(
                        fold,
                        spec,
                        deepcopy(entry["model_metadata"]),
                        ("outside_scope",)
                        if failure == "training_assets" and number == 0
                        else tuple(entry["training_asset_ids"]),
                        ("outside_scope",)
                        if failure == "evaluation_assets" and number == 0
                        else tuple(entry["evaluation_asset_ids"]),
                        reference,
                    )
                for number, entry in enumerate(saved.final_models):
                    with _read_model_member(
                        saved.instance_root,
                        saved.member_records,
                        f"models/final/{entry['model_id']}.pkl",
                        resource_policy=policy,
                        workspace_factor=4,
                    ) as owned:
                        payload = owned.value
                        metadata = deepcopy(entry["model_metadata"])
                        if failure == "seed" and number == 0:
                            metadata["random_seed"] += 1
                        if failure == "parameters" and number == 0:
                            metadata["max_train_rows"] = (
                                1
                                if metadata["max_train_rows"] is None
                                else metadata["max_train_rows"] + 1
                            )
                        if failure == "refit_window" and number == 0:
                            metadata["train_end"] = (
                                protocol.research_plan.test.end.isoformat()
                            )
                        selection = next(
                            item
                            for item in saved.models
                            if item["fold_id"] == entry["fold_id"]
                            and item["model_id"] == entry["model_id"]
                        )
                        member = production.CryptoFoldModelResult(
                            folds[entry["fold_id"]],
                            profile.models.get(entry["model_id"]),
                            fitting.ModelFitResult(metadata, payload),
                            tuple(entry["training_asset_ids"]),
                            tuple(entry["evaluation_asset_ids"]),
                            by_identity[selection["trial_semantic_identity"]],
                            by_identity[entry["refit_trial_semantic_identity"]],
                            entry["fit_coverage_id"],
                        )
                        owned.value = member
                        member = payload = None
                        try:
                            yield owned
                        finally:
                            assert owned.value is None
                            assert not owned.lease.active

            message = {
                "seed": "crypto_model_refit_binding_invalid",
                "parameters": "crypto_model_parameters_misaligned",
                "missing_fold": "crypto_model_fold_coverage_invalid",
                "training_assets": "crypto_model_fold_binding_invalid",
                "evaluation_assets": "crypto_model_fold_binding_invalid",
                "refit_window": "crypto_model_fit_window_invalid",
                "trial_reference": "crypto_model_trial_references_invalid",
                "trial_binding": "crypto_model_trial_binding_invalid",
                "disk": "insufficient_free_disk",
                "primary": "model_pickle_primary",
            }[failure]
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
                with pytest.raises(expected, match=message) as raised:
                    write_crypto_model_artifact(
                        logical_root,
                        envelope=envelope,
                        model_results=members,
                        trial_layout=layout,
                        verify_source=lambda: pytest.fail(
                            "invalid_model_reached_source_verification"
                        ),
                    )
                if failure == "primary":
                    assert raised.value is primary
                    assert any(
                        "lease close also failed" in note for note in primary.__notes__
                    )
            assert {
                p.name for p in filesystem_iterdir(saved.instance_root.parent)
            } == previous_instances
            assert read_semantic_artifact_record(saved.instance_root) == original
            assert not tuple(filesystem_glob(logical_root, ".*staging*"))
            assert policy.used_memory_bytes == 0


def check_crypto_wide_fold_artifact(
    *, layout, dataset_root, profile, saved, monkeypatch
):
    """Persist 22 synthetic folds through real model fitting and trial ledgers."""
    from quant_research_platform.profiles.research import bind_research_configuration
    from quant_research_platform.research.models import persisted
    from quant_research_platform.workflows.research import (
        _target_envelope_for_spec,
        production_software_identity,
    )

    original = saved.envelope["semantic_metadata"]["target_contract"]["protocol"]
    full_protocol = profile.protocol(original["scenario_id"])
    plan = deepcopy(full_protocol.manifest()["research_plan"])
    template = deepcopy(plan["folds"][0])
    plan["folds"] = []
    for index in range(22):
        fold = deepcopy(template)
        fold["fold_id"] = f"wide_{index}"
        for name in ("train", "validation"):
            fold[name] = {
                key: (
                    pd.Timestamp(value) + pd.Timedelta(seconds=100 * index)
                ).isoformat()
                for key, value in fold[name].items()
            }
        plan["folds"].append(fold)
    last = pd.Timestamp(plan["folds"][-1]["validation"]["end"])
    plan["test"] = {
        "start": (last + pd.Timedelta(seconds=10)).isoformat(),
        "end": (last + pd.Timedelta(seconds=11)).isoformat(),
    }
    configuration = deepcopy(full_protocol.configuration())
    configuration["research_plan"] = plan
    configuration["windows"] = {
        "development": {
            "start": plan["folds"][0]["validation"]["start"],
            "end": plan["folds"][-1]["validation"]["end"],
        },
        "holdout": dict(plan["test"]),
    }
    profile = bind_research_configuration(profile, configuration)
    bound = profile.protocol(original["scenario_id"])
    target = next(
        item
        for item in bound.target_specs("research run")
        if item.target_id == saved.envelope["logical_artifact_id"]
    )
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=bound,
        target_spec=target,
        data_spec=None,
        research_window={
            "start": plan["folds"][0]["validation"]["start"],
            "end": plan["folds"][-1]["validation"]["end"],
            "access": "development",
        },
        upstream_identities=saved.envelope["upstream_identities"],
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    protocol = envelope.payload()["semantic_metadata"]["target_contract"]["protocol"]
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

        def results(*, on_validation, on_selection):
            return production.build_crypto_fold_models(
                dataset_session=dataset_session,
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
                on_selection=on_selection,
            )

        folds, _holdout = production._crypto_model_chronology(
            protocol,
            TimeWindow(
                envelope.research_window["start"], envelope.research_window["end"]
            ),
        )
        with (
            monkeypatch.context() as patch,
            open_dataset_session(
                dataset_root,
                columns=None,
                time_window=TimeWindow(
                    min(fold.train.start for fold in folds),
                    max(fold.validation.end for fold in folds),
                ),
                resource_policy=policy,
            ) as dataset_session,
        ):
            patch.setattr(production, "_model_input_chunks", chunks)
            root = persisted.write_crypto_model_artifact(
                layout.project_root / "wide-models",
                envelope=envelope,
                model_results=results,
                trial_layout=layout,
                verify_source=dataset_session.verify,
            )
        loaded = persisted.read_crypto_model_artifact(
            root, expected_identity=envelope.instance_identity, trial_layout=layout
        )
        assert len(loaded.models) == 22 * 9
        assert len(loaded.final_models) == 9
        assert len(loaded.member_records) == 1 + 22 * (1 + 9) + 9
        for entry in loaded.final_models:
            with persisted.read_crypto_model_member(
                loaded, entry, resource_policy=policy
            ) as model:
                assert model is not None
        assert policy.used_memory_bytes == 0


def check_crypto_single_refit_lifecycle(tmp_path, monkeypatch):
    """Exercise one real fitted artifact and two terminal refit failures."""
    import hashlib

    def file_hashes(root):
        return {
            (parent / name).relative_to(root): hashlib.sha256(
                filesystem_read_bytes(parent / name)
            ).hexdigest()
            for parent, _directories, files in filesystem_walk(root)
            for name in files
        }

    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.models import persisted
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows.research import (
        _target_envelope_for_spec,
        production_software_identity,
    )
    from tests.helpers.binance_hft import hft_research_plan
    from tests.helpers.research_configuration import research_configuration

    plan = hft_research_plan()
    plan["folds"] = plan["folds"][:1]
    window = TimeWindow(**plan["folds"][0]["validation"])
    configuration = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=(window.start.isoformat(), window.end.isoformat()),
    )
    model = next(
        item
        for item in configuration["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    model.update(
        input_factor_columns=["imbalance_1"],
        input_feature_columns=[],
        training_asset_ids=["btc_usdt"],
        evaluation_asset_ids=["btc_usdt"],
    )
    configuration.update(
        models=[model], portfolios=[], comparison_model_id=model["model_id"]
    )
    configuration["evaluation"].update(
        factors=["imbalance_1"],
        labels=[model["label_column"]],
        primary_label=model["label_column"],
    )
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), configuration
    )
    bound = profile.protocol(configuration["scenario_id"])
    target_id = configuration["scenario_id"] + "_models"
    target = next(
        item
        for item in bound.target_specs("research run")
        if item.target_id == target_id
    )
    dataset_id = configuration["scenario_id"]
    dataset_identity = "a" * 64
    layout = StorageLayout(tmp_path)
    dataset_root = layout.instance_root(
        layout.research_dataset_root("crypto", dataset_id), dataset_identity
    )
    layout = layout.with_target_roots({dataset_id: dataset_root})
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=bound,
        target_spec=target,
        data_spec=None,
        research_window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "access": "development",
        },
        upstream_identities={dataset_id: dataset_identity},
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    protocol = envelope.payload()["semantic_metadata"]["target_contract"]["protocol"]
    timestamps = pd.date_range("1970-01-01", periods=56, freq="100ms")
    x = np.sin(np.arange(56) / 3)
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": timestamps,
            "imbalance_1": x,
            model["label_column"]: 0.001 * x + 0.0001 * np.cos(np.arange(56)),
            model["label_column"] + "_available_at": timestamps
            + pd.Timedelta(milliseconds=500),
            "eligible": True,
            "book_segment_id": "book",
            "trade_observation_id": "trade",
            "segment_id": "segment",
        }
    )

    from contextlib import contextmanager
    from dataclasses import replace

    from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
    from quant_research_platform.storage.datasets import write_dataset

    dataset = profile.research_datasets.get(dataset_id)
    dataset = replace(
        dataset,
        context_columns=tuple(
            column for column in dataset.context_columns if column in frame
        ),
        column_roles={column: dataset.column_roles[column] for column in frame},
    )
    profile = replace(profile, research_datasets=ResearchDatasetCatalog((dataset,)))
    write_dataset(dataset_root, frame, spec=dataset)
    opened_sessions = []
    folds, _holdout = production._crypto_model_chronology(protocol, window)
    read_window = TimeWindow(
        min(fold.train.start for fold in folds),
        max(fold.validation.end for fold in folds),
    )

    @contextmanager
    def tracked_session():
        with open_dataset_session(
            dataset_root, columns=None, time_window=read_window, resource_policy=policy
        ) as session:
            opened_sessions.append(session)
            yield session

    original_begin = production.TrialRecorder.begin_refit
    original_fit = fitting.ModelMatrixScaler.fit
    captured = []
    failure = None
    mutate_source = None
    observed_attempts = []
    mutation_points = []

    def begin_refit(recorder, *args, **kwargs):
        refit, attempt = original_begin(recorder, *args, **kwargs)
        captured.append((recorder, recorder.reference(), refit))
        return refit, attempt

    def change_source():
        nonlocal mutate_source
        mutation_points.append(mutate_source)
        mutate_source = None
        session = opened_sessions[-1]
        assert session._lease.active and session.inventory._lease.active
        part = session.read_paths[0]
        payload = filesystem_read_bytes(part)
        filesystem_write_bytes(part, payload[:-1] + bytes((payload[-1] ^ 1,)))

    def fit_scaler(cls, matrix, **kwargs):
        session = opened_sessions[-1]
        assert session._lease.active and session.inventory._lease.active
        assert policy.used_memory_bytes >= (
            session._lease.reserved_bytes + session.inventory._lease.reserved_bytes
        )
        if mutate_source == "selection" or (mutate_source == "refit" and captured):
            change_source()
        if captured:
            selection, reference, refit = captured[-1]
            with read_trial_ledger_reference(layout, reference) as (
                selected_plan,
                selected_records,
            ):
                with read_trial_ledger_reference(layout, refit.reference()) as (
                    refit_plan,
                    refit_records,
                ):
                    assert (
                        selected_plan.stage == "selection"
                        and selected_records[0].status == "completed"
                    )
                    assert (
                        refit_plan.stage == "refit"
                        and refit_records[0].status == "attempted"
                    )
                    assert refit_plan.selection_ledger_semantic_identities == (
                        reference["semantic_identity"],
                    )
                    assert (
                        refit_records[0].common_coverage_id
                        != selected_records[0].common_coverage_id
                    )
            observed_attempts.append(len(matrix))
            if failure is not None:
                raise failure
        return original_fit(matrix, **kwargs)

    original_write_member = persisted._write_model_member
    original_validate = persisted._validate_crypto_model_payload

    def write_member(*args, **kwargs):
        result = original_write_member(*args, **kwargs)
        if mutate_source == "member_write":
            change_source()
        return result

    def validate_payload(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        if mutate_source == "staged_validation":
            change_source()
        return result

    monkeypatch.setattr(persisted, "_write_model_member", write_member)
    monkeypatch.setattr(persisted, "_validate_crypto_model_payload", validate_payload)
    monkeypatch.setattr(production.TrialRecorder, "begin_refit", begin_refit)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", classmethod(fit_scaler))
    logical_root = tmp_path / "models"
    with runtime_resource_policy_scope() as policy:

        def results(
            run_id,
            *,
            dataset_session,
            model_window=window,
            on_validation=lambda *args: None,
            on_selection=lambda *args: None,
        ):
            return production.build_crypto_fold_models(
                layout=layout,
                dataset_session=dataset_session,
                dataset_catalog=profile.research_datasets,
                model_catalog=profile.models,
                dataset_id=dataset_id,
                model_ids=(model["model_id"],),
                tie_break_order=(model["model_id"],),
                time_window=model_window,
                resource_policy=policy,
                protocol_manifest=protocol,
                run_id=run_id,
                research_protocol_identity=canonical_json_hash(protocol),
                model_target_id=target_id,
                backend_versions={"fixture": "1"},
                on_validation=on_validation,
                on_selection=on_selection,
            )

        def publish(target_envelope, run_id):
            with tracked_session() as dataset_session:
                return persisted.write_crypto_model_artifact(
                    logical_root,
                    envelope=target_envelope,
                    model_results=lambda **kwargs: results(
                        run_id, dataset_session=dataset_session, **kwargs
                    ),
                    trial_layout=layout,
                    verify_source=dataset_session.verify,
                )

        assert (
            list(
                results(
                    "run_crypto_no_fold",
                    dataset_session=None,
                    model_window=TimeWindow(
                        "1969-12-31T00:00:00", "1969-12-31T00:00:01"
                    ),
                )
            )
            == []
        )
        assert opened_sessions == []
        from quant_research_platform.workflows import research

        empty_window = TimeWindow("1969-12-31T00:00:00", "1969-12-31T00:00:01")
        empty_research_window = {
            "start": empty_window.start.isoformat(),
            "end": empty_window.end.isoformat(),
            "access": "development",
        }
        software = production_software_identity()
        empty_envelope = _target_envelope_for_spec(
            profile=profile,
            protocol=bound,
            target_spec=target,
            data_spec=None,
            research_window=empty_research_window,
            upstream_identities={dataset_id: dataset_identity},
            software_identity=software,
            raw_content_identity=None,
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                research,
                "open_dataset_session",
                lambda *_args, **_kwargs: pytest.fail("empty_fold_opened_source"),
            )
            empty_result = research._build_semantic_target(
                profile=profile,
                protocol=bound,
                layout=layout,
                spec=target,
                planned={
                    "requested_identity": empty_envelope.instance_identity,
                    "upstream_identities": {dataset_id: dataset_identity},
                },
                target_layout=layout,
                policy=policy,
                time_window=empty_window,
                research_window=empty_research_window,
                software_identity=software,
                raw_root=tmp_path / "raw",
                run_id="run_crypto_empty_artifact",
            )
        assert empty_result.instance_identity == empty_envelope.instance_identity
        empty_root = layout.instance_root(
            layout.semantic_artifact_root("crypto", target_id),
            empty_envelope.instance_identity,
        )
        assert (
            persisted.read_crypto_model_artifact(
                empty_root,
                expected_identity=empty_envelope.instance_identity,
                trial_layout=layout,
            ).models
            == ()
        )
        assert policy.used_memory_bytes == 0

        root = publish(envelope, "run_crypto_refit_success")
        saved = persisted.read_crypto_model_artifact(
            root, expected_identity=envelope.instance_identity, trial_layout=layout
        )
        assert len(saved.models) == len(saved.final_models) == 1
        assert len(captured) == 1
        selected_entry = saved.models[0]
        entry = saved.final_models[0]
        selection, selection_ref, refit = captured[0]
        with read_trial_ledger_reference(layout, selection_ref) as (
            selection_plan,
            selected,
        ):
            with read_trial_ledger_reference(layout, refit.reference()) as (
                refit_plan,
                refitted,
            ):
                assert (
                    selected_entry["trial_semantic_identity"]
                    == selection_ref["semantic_identity"]
                )
                assert (
                    entry["refit_trial_semantic_identity"]
                    == refit.reference()["semantic_identity"]
                )
                assert entry["fit_coverage_id"] == refitted[0].common_coverage_id
                assert refit_plan.selection_ledger_semantic_identities == (
                    selection_ref["semantic_identity"],
                )
                assert refitted[0].status == "completed"
                assert refitted[0].metrics["n_train"] == observed_attempts[0]
                assert refitted[0].metrics["n_train"] > selected[0].metrics["n_train"]
        with persisted.read_crypto_model_member(
            saved, entry, resource_policy=policy
        ) as predictor:
            predictions = fitting.prediction_callable(("imbalance_1",), predictor)(
                frame.iloc[:3]
            )
            assert np.isfinite(predictions).all()
        assert policy.used_memory_bytes == 0
        original = file_hashes(root)
        assert len(opened_sessions) == 1
        captured.clear()
        with (
            tracked_session() as dataset_session,
            closing_iterator(
                results("run_crypto_early_close", dataset_session=dataset_session)
            ) as source,
        ):
            owned = next(source)
            owned.release(owner="test.crypto.early_close")
        assert len(opened_sessions) == 2
        assert policy.used_memory_bytes == 0
        for index, error in enumerate(
            (RuntimeError("refit_failure"), KeyboardInterrupt("refit_cancel"))
        ):
            captured.clear()
            failure = error
            payload = envelope.payload()
            other = build_business_manifest_envelope(
                **{
                    key: (str(index + 1) * 64 if key == "software_identity" else value)
                    for key, value in payload.items()
                    if key not in {"instance_identity", "status"}
                }
            )
            with pytest.raises(type(error)) as raised:
                publish(other, "run_crypto_refit_failure_" + str(index))
            assert raised.value is error and len(captured) == 1
            selection, selection_ref, refit = captured[0]
            assert selection.reference() == selection_ref
            with (
                read_trial_ledger_reference(layout, selection_ref) as (_, selected),
                read_trial_ledger_reference(layout, refit.reference()) as (_, refitted),
            ):
                assert selected[0].status == "completed"
                assert refitted[0].status == (
                    "failed" if isinstance(error, Exception) else "abandoned"
                )
            assert not filesystem_exists(
                logical_root / "instances" / other.instance_identity
            )
            assert original == file_hashes(root)
            assert policy.used_memory_bytes == 0
        assert len(observed_attempts) == 4
        part = next(
            parent / name
            for parent, _directories, files in filesystem_walk(dataset_root)
            for name in files
            if name.endswith(".parquet")
        )
        original_part = filesystem_read_bytes(part)
        payload = envelope.payload()
        for index, phase in enumerate(
            ("selection", "refit", "member_write", "staged_validation"), start=3
        ):
            captured.clear()
            failure = None
            mutate_source = phase
            changed_source_envelope = build_business_manifest_envelope(
                **{
                    key: (str(index) * 64 if key == "software_identity" else value)
                    for key, value in payload.items()
                    if key not in {"instance_identity", "status"}
                }
            )
            try:
                with pytest.raises(ValueError, match="dataset_file_identity_changed"):
                    publish(
                        changed_source_envelope, "run_crypto_source_changed_" + phase
                    )
                assert mutation_points[-1] == phase
                assert len(filesystem_read_bytes(part)) == len(original_part)
            finally:
                filesystem_write_bytes(part, original_part)
            if phase != "selection":
                assert len(captured) == 1
                selection, selection_ref, refit = captured[0]
                with (
                    read_trial_ledger_reference(layout, selection_ref) as (_, selected),
                    read_trial_ledger_reference(layout, refit.reference()) as (
                        _,
                        refitted,
                    ),
                ):
                    assert selected[0].status == refitted[0].status == "completed"
            assert original == file_hashes(root)
            assert not filesystem_exists(
                logical_root / "instances" / changed_source_envelope.instance_identity
            )
            assert policy.used_memory_bytes == 0
        assert mutation_points == [
            "selection",
            "refit",
            "member_write",
            "staged_validation",
        ]
        assert len(opened_sessions) == 8
        assert all(not session._lease.active for session in opened_sessions)

        metadata_failure = RuntimeError("crypto_metadata_boundary_primary")
        copied_metadata_ids = []
        original_manifest_write = persisted.write_internal_artifact_manifest

        def fail_metadata_write(path, payload, **kwargs):
            if path.name == "metadata.json":
                copied_metadata_ids.extend(
                    id(entry["model_metadata"]) for entry in payload["models"]
                )
                assert any(
                    lease.owner == "research.model.artifact.metadata"
                    and lease.reserved_bytes > 0
                    for lease in policy._leases
                )
                raise metadata_failure
            return original_manifest_write(path, payload, **kwargs)

        payload = envelope.payload()
        metadata_envelope = build_business_manifest_envelope(
            **{
                key: ("f" * 64 if key == "software_identity" else value)
                for key, value in payload.items()
                if key not in {"instance_identity", "status"}
            }
        )
        captured.clear()
        with monkeypatch.context() as patch:
            patch.setattr(
                persisted, "write_internal_artifact_manifest", fail_metadata_write
            )
            with pytest.raises(RuntimeError) as raised:
                publish(metadata_envelope, "run_crypto_metadata_failure")
        assert raised.value is metadata_failure
        assert copied_metadata_ids
        current = metadata_failure.__traceback__
        while current is not None:
            local_values = current.tb_frame.f_locals
            if current.tb_frame.f_code.co_name == "fail_metadata_write":
                assert not local_values
            assert all(
                id(value) not in copied_metadata_ids for value in local_values.values()
            )
            current = current.tb_next
        assert policy.used_memory_bytes == 0
        assert original == file_hashes(root)
        assert not filesystem_exists(
            logical_root / "instances" / metadata_envelope.instance_identity
        )
        assert len(opened_sessions) == 9
        assert all(not session._lease.active for session in opened_sessions)
