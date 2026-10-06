"""Independent fold-maturity counts and publication failure checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.evaluations import runner
from quant_research_platform.research.evaluations.engine import EvaluationResult
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    read_semantic_artifact_record,
)
from quant_research_platform.workflows import research
from tests.helpers.research_configuration import research_configuration


def check_crypto_evaluation_readback(
    *, layout, profile, targets, frame, plan, prediction_root
):

    def read_parquet(path):
        return pd.read_parquet(native_filesystem_path(path))

    oos = pd.Series(False, index=frame.index)
    cutoff = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    window = plan["test"]
    oos = frame.timestamp.between(
        pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    )
    cutoff.loc[oos] = pd.Timestamp(window["end"])
    for spec in profile.evaluations.list():
        identity = targets[spec.evaluation_id]["instance_identity"]
        root = layout.instance_root(
            layout.semantic_artifact_root("crypto", spec.evaluation_id), identity
        )
        read_semantic_artifact(root, expected_identity=identity)
        bundle = root / "outputs" / spec.evaluation_id
        manifest, _ = validate_output_bundle(
            bundle,
            expected_output_id=spec.evaluation_id,
            expected_run_id=f"run_{spec.evaluation_id}",
        )
        metrics = read_parquet(bundle / "metrics.parquet")
        work = frame.copy()
        assets = ()
        if spec.signals[0].source_kind == "prediction":
            scopes = profile.protocol(
                profile.research_datasets.list()[0].data_id
            ).manifest()["model_scopes"]
            asset_sets = []
            for signal in spec.signals:
                model_id = signal.signal_id
                predictions = read_parquet(
                    prediction_root
                    / "outputs"
                    / f"{model_id}_prediction"
                    / "prediction.parquet"
                )
                work = work.merge(
                    predictions[["timestamp", "asset_id", "prediction_score"]].rename(
                        columns={"prediction_score": signal.source_column}
                    ),
                    on=["timestamp", "asset_id"],
                    how="left",
                    validate="one_to_one",
                )
                asset_sets.append(set(scopes[model_id]["evaluation_asset_ids"]))
            assets = tuple(set.intersection(*asset_sets))
        scope = oos & work.eligible
        if spec.signals[0].source_kind == "prediction":
            scope &= work.asset_id.isin(assets)
            for signal in spec.signals:
                scope &= np.isfinite(work[signal.source_column])
        for signal in spec.signals:
            for label, availability in spec.label_availability_columns.items():
                mask = (
                    scope
                    & work[availability].le(cutoff)
                    & work[label].notna()
                    & work[signal.source_column].notna()
                )
                pair = metrics.loc[
                    metrics.signal.eq(signal.signal_id) & metrics.label.eq(label)
                ]
                assert int(pair.n_obs.sum()) == int(mask.sum()), (
                    spec.evaluation_id,
                    label,
                )
                if (
                    spec.evaluation_axis == "single_asset_time_series"
                    and mask.sum() > 1
                ):
                    values = work.loc[mask, [signal.source_column, label]].to_numpy(
                        dtype=float
                    )
                    if np.std(values[:, 0]) > 0 and np.std(values[:, 1]) > 0:
                        direction = (
                            -1.0
                            if signal.source_kind == "factor"
                            and signal.source_column in {"spread", "relative_spread"}
                            else 1.0
                        )
                        assert float(pair.ic.iloc[0]) == pytest.approx(
                            direction * float(np.corrcoef(values.T)[0, 1]), abs=1e-10
                        )
        diagnostics = manifest["metadata"]["evaluation_diagnostics"]
        if diagnostics["timestamp_count"]:
            assert diagnostics["min_timestamp"] >= plan["test"]["start"]
        else:
            assert diagnostics["min_timestamp"] is None
            assert diagnostics["max_timestamp"] is None


def check_crypto_evaluation_old_good(
    *, layout, profile, targets, raw, plan, development_run_id
):
    from quant_research_platform.profiles import research as profiles

    bind = profiles.bind_research_configuration
    target = next(
        spec.evaluation_id
        for spec in profile.evaluations.list()
        if spec.signals[0].source_kind == "factor"
    )
    logical = layout.semantic_artifact_root("crypto", target)
    root = layout.instance_root(logical, targets[target]["instance_identity"])
    original = read_semantic_artifact_record(root)
    primary = KeyboardInterrupt("evaluation_interrupted_after_output")

    def bind_with_interruption(*args, **kwargs):
        rebound = bind(*args, **kwargs)
        binding = rebound.producer_bindings[target]

        def interrupted(**kwargs):
            result = binding.producer(**kwargs)

            def chunks():
                with result:
                    for chunk in result.output_chunks:
                        yield chunk
                        raise primary

            source = chunks()

            def close():
                source.close()
                raise RuntimeError("evaluation_cleanup_failed")

            return EvaluationResult(source, result.summaries, close)

        return replace(
            rebound,
            producer_bindings={
                **rebound.producer_bindings,
                target: replace(binding, producer=interrupted),
            },
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(profiles, "bind_research_configuration", bind_with_interruption)
        with pytest.raises(KeyboardInterrupt) as caught:
            research.research_run(
                profile=profile,
                layout=layout,
                raw_root=raw,
                rebuild=True,
                holdout=True,
                development_run_id=development_run_id,
                research_plan=research_configuration(
                    "crypto",
                    "baseline",
                    small_models=True,
                    development=(
                        plan["folds"][0]["validation"]["start"],
                        plan["folds"][-1]["validation"]["end"],
                    ),
                    plan=plan,
                ),
            )
    assert caught.value is primary
    assert any("RuntimeError" in note for note in primary.__notes__)
    assert read_semantic_artifact_record(root) == original
    read_semantic_artifact(root, expected_identity=targets[target]["instance_identity"])


def check_crypto_evaluation_input_boundaries(
    *, layout, profile, targets, dataset_root, prediction_root, plan, monkeypatch
):
    dataset_id = profile.research_datasets.list()[0].data_id
    target_layout = layout.with_target_roots(
        {dataset_id: dataset_root, f"{dataset_id}_predictions": prediction_root}
    )
    protocol = profile.protocol(
        profile.research_datasets.list()[0].data_id, access_window="holdout"
    ).manifest()
    window = TimeWindow(**plan["test"])
    original_reader = runner.iter_dataset_chunks
    mode = "late_labels"

    def changed_reader(*args, **kwargs):
        for frame in original_reader(*args, **kwargs):
            changed = frame.copy()
            if mode == "late_labels":
                for availability in changed.columns:
                    if availability.endswith("_available_at"):
                        label = availability.removesuffix("_available_at")
                        if label in changed:
                            changed.loc[
                                changed[availability].gt(pd.Timestamp(window.end)),
                                label,
                            ] = 999.0
            elif mode == "extra_prediction":
                changed = changed.iloc[1:].copy()
            else:
                changed.loc[changed.index[0], "timestamp"] += pd.Timedelta(
                    microseconds=1
                )
            yield changed

    with monkeypatch.context() as patch:
        patch.setattr(runner, "iter_dataset_chunks", changed_reader)

        def compute(spec):
            with profile.producer_bindings[spec.evaluation_id].producer(
                layout=target_layout,
                dataset_catalog=profile.research_datasets,
                evaluation_catalog=profile.evaluations,
                evaluation_id=spec.evaluation_id,
                time_window=window,
                protocol_manifest=protocol,
            ) as result:
                return pd.concat(
                    [chunk[0].copy() for chunk in result.output_chunks],
                    ignore_index=True,
                )

        for spec in profile.evaluations.list():
            actual = compute(spec)
            root = layout.instance_root(
                layout.semantic_artifact_root("crypto", spec.evaluation_id),
                targets[spec.evaluation_id]["instance_identity"],
            )
            path = root / "outputs" / spec.evaluation_id / "metrics.parquet"
            expected = pd.read_parquet(native_filesystem_path(path))
            pd.testing.assert_frame_equal(actual, expected)
        spec = profile.evaluations.get("crypto_hft_linear_regression_oos_evaluation")
        for fault in ("extra_prediction", "missing_prediction"):
            mode = fault
            with pytest.raises(
                ValueError, match="misaligned_persisted_prediction_keys"
            ):
                compute(spec)
