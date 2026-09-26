from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import read_prediction_output_bundle
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research import statistics_production
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
    canonical_feature_transforms,
    model_input_validity_columns,
)
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.portfolios.production import _prediction_alpha
from quant_research_platform.research.protocols import a_share
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research import test_quarterly_model_artifacts as t


@pytest.mark.parametrize(
    "algorithm,input_column",
    [
        ("equal_weight_rank_ensemble", "x"),
        ("equal_weight_rank_ensemble", "feature_x"),
        ("linear_regression", "feature_x"),
        ("linear_regression", "feature_missing_x"),
        ("linear_regression", "x"),
    ],
)
@pytest.mark.parametrize("missing_required", [False, True])
def test_transform_prediction_publication_and_consumers_share_exact_validity(
    tmp_path, monkeypatch, algorithm, input_column, missing_required
):
    count = 106
    raw = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{index:03d}" for index in range(count)],
            "timestamp": pd.Timestamp("2019-02-01"),
            "eligible": True,
            "industry_level1_code": "industry",
            "log_float_market_cap": 20 + np.arange(count) * 0.01,
            "x": np.sin(np.arange(count)),
            "float_market_cap": 100.0,
            "forward_return_20d": np.arange(count) * 0.001,
            "forward_return_20d_available_at": pd.Timestamp("2019-03-01"),
        }
    )
    raw.loc[count - 2, "x"] = np.nan
    raw.loc[count - 1, "log_float_market_cap"] = np.nan
    frame = (
        apply_feature_transforms(raw, canonical_feature_transforms(("x",)))
        .iloc[-4:]
        .copy()
    )
    assert frame.feature_x.notna().sum() == 3
    expected = (
        frame.asset_id.tolist()
        if algorithm == "equal_weight_rank_ensemble" and input_column == "x"
        else frame.loc[frame.x.notna(), "asset_id"].tolist()
        if input_column == "x"
        else frame.asset_id.iloc[:3].tolist()
    )
    spec = t.ModelSpec(
        model_id="fixture_model",
        model_algorithm=algorithm,
        input_factor_columns=(input_column,) if input_column == "x" else (),
        input_feature_columns=() if input_column == "x" else (input_column,),
        label_column="label",
    )
    protocol = t._fixture_protocol((spec,))
    dataset = make_demo_research_dataset_spec(
        data_id="research_demo",
        factor_columns=("x",),
        feature_columns=("feature_x", "feature_missing_x", "feature_observed_x"),
        feature_transforms=canonical_feature_transforms(("x",)),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    frame = frame.loc[:, list(dataset.required_columns)]
    layout = t.StorageLayout(tmp_path)
    predictor = (
        RankEnsemblePredictor(columns=(input_column,), directions=(1.0,))
        if algorithm == "equal_weight_rank_ensemble"
        else t._column_predictor(input_column, 1.0)
    )
    window = TimeWindow("2019-01-01", "2019-03-31")
    write_dataset(
        layout.research_dataset_root("a_share", dataset.data_id),
        frame,
        spec=dataset,
    )
    with t.runtime_resource_policy_scope() as policy:
        quarterly = t._quarterly_result(
            1,
            trial_layout=layout,
            model_spec=spec,
            protocol=protocol,
            validation_specs=(spec,),
            predictor=predictor,
        )
        if algorithm == "equal_weight_rank_ensemble":
            quarterly.model_result.metadata["rank_parameters"] = {
                "columns": [input_column],
                "directions": [1.0],
            }
        owned = t.OwnedValue(quarterly, policy.lease("test.model", 8192))
        model_root = t.write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=layout,
            envelope=t._envelope(specs=(spec,)),
            quarterly_results=t.fixture_quarterly_results([owned]),
        )
        artifact = t.read_quarterly_model_artifact(model_root, trial_layout=layout)
        prediction_spec = t.PredictionSpec(
            prediction_id="prediction",
            model_id="model-artifact",
            research_dataset_id=dataset.data_id,
            train_end=None,
            test_start="2019-01-01",
            test_end="2019-03-31",
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        if missing_required:
            t._mock_prediction_dataset(
                monkeypatch, frame.loc[frame.asset_id.ne(expected[-1])]
            )
        result = t.runner.build_prediction_result_from_persisted_model(
            layout=layout,
            dataset_catalog=t.ResearchDatasetCatalog((dataset,)),
            prediction_spec=prediction_spec,
            persisted_model=artifact,
            resource_policy=policy,
        )
        bundle = t.write_prediction_result(layout, result)
        with read_prediction_output_bundle(
            bundle,
            expected_output_id="prediction",
            expected_run_id="test_run",
            family_selection="champion",
        ) as (_manifest, lookup):
            assert lookup.row_count == len(expected) - int(missing_required)
            validity = model_input_validity_columns(spec.input_columns, algorithm)
            if missing_required:
                with pytest.raises(
                    ValueError, match="^missing_required_prediction_keys$"
                ):
                    _prediction_alpha(frame, lookup, validity)
            else:
                alpha = _prediction_alpha(frame, lookup, validity)
                assert alpha.dropna().index.tolist() == expected
                assert alpha.index.tolist() == frame.asset_id.tolist()
                assert lookup.coverage_counts() == (0, 0)

        statistics_protocol = deepcopy(a_share.protocol("pv_all").manifest())
        statistics_protocol["statistics"].update(
            factor_fdr_families={"fixture": ["x"]},
            fama_macbeth_core_factors=["x"],
        )
        statistics_protocol["research_configuration"] = {
            "evaluation": {"factors": ["x"], "primary_label": "forward_return_20d"},
            "comparison_model_id": spec.model_id,
        }
        with read_prediction_output_bundle(
            bundle,
            expected_output_id="prediction",
            expected_run_id="test_run",
        ) as (_manifest, lookup):

            def statistics():
                return statistics_production.build_a_share_statistics_tables(
                    layout=layout,
                    dataset_catalog=t.ResearchDatasetCatalog((dataset,)),
                    dataset_id=dataset.data_id,
                    prediction_id="prediction",
                    persisted_model=artifact,
                    protocol_manifest=statistics_protocol,
                    time_window=window,
                    prediction_lookup=lookup,
                    resource_policy=policy,
                )

            if missing_required:
                with pytest.raises(
                    ValueError, match="^missing_required_prediction_keys$"
                ):
                    statistics()
            else:
                tables = statistics()
                try:
                    assert lookup.coverage_counts() == (0, 0)
                    assert tables.value["model_losses"].observation_count.tolist() == [
                        len(expected)
                    ]
                    envelope = t.build_business_manifest_envelope(
                        logical_artifact_id="statistics",
                        schema_identity="a" * 64,
                        protocol_or_spec_identity="b" * 64,
                        software_identity="c" * 64,
                        research_window={
                            "start": "2019-01-01",
                            "end": "2019-03-31",
                            "access": "development",
                        },
                        upstream_identities={},
                        semantic_metadata={},
                    )
                    root = write_statistics_artifact(
                        tmp_path / "statistics", envelope=envelope, tables=tables.value
                    )
                finally:
                    tables.release(owner="research.statistics.output")
                restored = read_statistics_artifact(
                    root, expected_identity=envelope.instance_identity
                )
                with restored.table("model_losses.parquet") as saved:
                    assert saved.observation_count.tolist() == [len(expected)]
        assert policy.used_memory_bytes == 0
