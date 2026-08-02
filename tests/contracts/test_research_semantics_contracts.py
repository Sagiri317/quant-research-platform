from __future__ import annotations

from collections import UserDict
from dataclasses import replace
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    normalize_data_frequency,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.signals import SignalSpec
from tests.helpers import frequency_ids


class _DictSubclass(dict[str, str]):
    pass


class _ColumnRoleDictSubclass(dict[str, ColumnRole]):
    pass


class _StringSubclass(str):
    pass


def test_data_frequency_vocabulary_is_current_and_exact() -> None:
    assert tuple(frequency.value for frequency in DataFrequency) == (
        frequency_ids.DAILY_FREQUENCY_SLUG,
        "1m",
        "tick",
    )
    with pytest.raises(ValueError, match="^invalid_data_frequency:"):
        normalize_data_frequency("unknown")


def test_research_dataset_splits_factor_and_feature_columns_without_alias() -> None:
    spec = ResearchDatasetSpec(
        data_id="a_share_research",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "mom_20d": ColumnRole.FACTOR,
            "mom_20d_zscore": ColumnRole.FEATURE,
            "forward_return_20d": ColumnRole.LABEL,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id=MarketId("a_share"),
        input_data_ids=("factor_panel", "feature_panel"),
        base_input_data_id="factor_panel",
        factor_columns=("mom_20d",),
        feature_columns=("mom_20d_zscore",),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="mom_20d_zscore",
                source_columns=("mom_20d",),
                output_columns=("mom_20d_zscore",),
                transform="cross_sectional_zscore",
            ),
        ),
        label_columns=("forward_return_20d",),
    )

    assert spec.factor_columns == ("mom_20d",)
    assert spec.feature_columns == ("mom_20d_zscore",)

    with pytest.raises(ValueError, match="feature_columns_without_transform"):
        ResearchDatasetSpec(
            data_id="bad_missing_transform",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                "mom_20d_zscore": ColumnRole.FEATURE,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id=MarketId("a_share"),
            input_data_ids=("factor_panel",),
            base_input_data_id="factor_panel",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d_zscore",),
            label_columns=("forward_return_20d",),
        )

    with pytest.raises(ValueError, match="research_factor_feature_overlap"):
        ResearchDatasetSpec(
            data_id="bad_research",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "mom_20d": ColumnRole.FACTOR,
                "forward_return_20d": ColumnRole.LABEL,
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id=MarketId("a_share"),
            input_data_ids=("factor_panel",),
            base_input_data_id="factor_panel",
            factor_columns=("mom_20d",),
            feature_columns=("mom_20d",),
            label_columns=("forward_return_20d",),
        )


@pytest.mark.parametrize(
    ("field_name", "missing_column"),
    [
        ("factor_columns", "missing_factor"),
        ("feature_columns", "missing_feature"),
        ("label_columns", "missing_label"),
        ("filter_columns", "missing_filter"),
        ("context_columns", "missing_context"),
    ],
)
def test_research_dataset_spec_requires_semantic_columns_in_required_columns(
    field_name: str,
    missing_column: str,
) -> None:
    spec = _complete_research_dataset_spec()
    changes: dict[str, object] = {field_name: (missing_column,)}
    if field_name == "feature_columns":
        changes["feature_transforms"] = (
            FeatureTransformSpec(
                transform_id="missing_feature_from_factor_1",
                source_columns=("factor_1",),
                output_columns=(missing_column,),
                transform="cross_sectional_zscore",
            ),
        )

    with pytest.raises(
        ValueError,
        match=f"research_semantic_column_not_required: {field_name}",
    ):
        replace(spec, **changes)


@pytest.mark.parametrize(
    ("field_name", "column", "invalid_role"),
    [
        ("factor_columns", "factor_1", ColumnRole.MASK),
        ("feature_columns", "feature_1", ColumnRole.CONTEXT),
        ("label_columns", "label_1", ColumnRole.CONTEXT),
        ("filter_columns", "eligible", ColumnRole.CONTEXT),
        ("context_columns", "context_1", ColumnRole.MASK),
    ],
)
def test_research_dataset_spec_requires_semantic_column_roles(
    field_name: str,
    column: str,
    invalid_role: ColumnRole,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(
        ValueError,
        match=f"research_semantic_column_role_invalid: {field_name}",
    ):
        replace(
            spec,
            column_roles={**spec.column_roles, column: invalid_role},
        )


def test_research_dataset_spec_allows_mask_or_universe_filter_roles() -> None:
    spec = _complete_research_dataset_spec()

    universe_filter = replace(
        spec,
        column_roles={**spec.column_roles, "eligible": ColumnRole.UNIVERSE},
    )

    assert universe_filter.column_roles["eligible"] is ColumnRole.UNIVERSE


def test_research_dataset_spec_freezes_semantic_mapping_snapshots() -> None:
    original = _complete_research_dataset_spec()
    source_roles = dict(original.column_roles)
    source_key_coverage = {"factors": "sparse"}

    spec = replace(
        original,
        column_roles=source_roles,
        key_coverage=source_key_coverage,
    )
    source_roles["factor_1"] = ColumnRole.MASK
    source_key_coverage["factors"] = "strict"
    source_key_coverage["annotations"] = "sparse"

    assert type(spec.column_roles) is MappingProxyType
    assert type(spec.key_coverage) is MappingProxyType
    assert spec.column_roles["factor_1"] is ColumnRole.FACTOR
    assert spec.key_coverage == {
        "factors": "sparse",
        "labels": "strict",
        "annotations": "strict",
    }
    with pytest.raises(TypeError):
        spec.column_roles["factor_1"] = ColumnRole.MASK  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.key_coverage["annotations"] = "sparse"  # type: ignore[index]

    replacement = replace(spec)

    assert replacement == spec
    assert replacement is not spec
    assert replacement.column_roles is not spec.column_roles
    assert replacement.key_coverage is not spec.key_coverage
    assert replacement.key_coverage["annotations"] == "strict"


@pytest.mark.parametrize(
    "wrong_container",
    (
        [("market_id", ColumnRole.KEY)],
        (("market_id", ColumnRole.KEY),),
        "market_id",
        None,
        _ColumnRoleDictSubclass({"market_id": ColumnRole.KEY}),
        UserDict({"market_id": ColumnRole.KEY}),
    ),
    ids=("list_pairs", "tuple_pairs", "string", "none", "dict_subclass", "mapping"),
)
def test_research_dataset_spec_requires_exact_column_roles_container(
    wrong_container: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_column_roles$"):
        replace(spec, column_roles=wrong_container)


@pytest.mark.parametrize(
    "wrong_key",
    (1, "", _StringSubclass("factor_1_alias")),
    ids=("integer", "empty", "str_subclass"),
)
def test_research_dataset_spec_requires_exact_nonempty_column_role_keys(
    wrong_key: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_column_roles$"):
        replace(
            spec,
            column_roles={**spec.column_roles, wrong_key: ColumnRole.FACTOR},
        )


@pytest.mark.parametrize(
    "wrong_role",
    ("factor", _StringSubclass("factor"), 1),
    ids=("string", "str_subclass", "integer"),
)
def test_research_dataset_spec_requires_exact_column_role_values(
    wrong_role: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_column_roles$"):
        replace(
            spec,
            column_roles={**spec.column_roles, "factor_1": wrong_role},
        )


@pytest.mark.parametrize(
    "wrong_kind",
    ([], {}, _StringSubclass("strict")),
    ids=("list", "dict", "str_subclass"),
)
def test_research_dataset_spec_rejects_invalid_key_coverage_policy_with_owner_error(
    wrong_kind: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_key_coverage"):
        replace(
            spec,
            key_coverage={**spec.key_coverage, "factors": wrong_kind},
        )


@pytest.mark.parametrize(
    "wrong_container",
    ([], (), None, _DictSubclass()),
    ids=("list", "tuple", "none", "dict_subclass"),
)
def test_research_dataset_spec_requires_exact_key_coverage_dict(
    wrong_container: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_key_coverage$"):
        replace(spec, key_coverage=wrong_container)


@pytest.mark.parametrize("wrong_key", (1, _StringSubclass("factors")))
def test_research_dataset_spec_requires_exact_string_key_coverage_keys(
    wrong_key: object,
) -> None:
    spec = _complete_research_dataset_spec()

    with pytest.raises(ValueError, match="^invalid_research_key_coverage$"):
        replace(spec, key_coverage={wrong_key: "strict"})


def test_feature_transform_spec_rejects_raw_factor_pass_through() -> None:
    spec = FeatureTransformSpec(
        transform_id="mom_20d_zscore",
        source_columns=("mom_20d",),
        output_columns=("mom_20d_zscore",),
        transform="cross_sectional_zscore",
    )

    assert spec.source_columns == ("mom_20d",)
    assert spec.output_columns == ("mom_20d_zscore",)

    with pytest.raises(ValueError, match="feature_transform_raw_factor_passthrough"):
        FeatureTransformSpec(
            transform_id="bad_passthrough",
            source_columns=("mom_20d",),
            output_columns=("mom_20d",),
            transform="cross_sectional_zscore",
        )


def test_model_spec_uses_algorithm_and_explicit_factor_feature_inputs() -> None:
    spec = ModelSpec(
        model_id="linear_mom",
        model_algorithm="linear_regression",
        input_factor_columns=("mom_20d",),
        input_feature_columns=("mom_20d_zscore",),
        label_column="forward_return_20d",
    )

    assert spec.model_algorithm == "linear_regression"
    assert spec.input_columns == ("mom_20d", "mom_20d_zscore")

    with pytest.raises(ValueError, match="model_input_columns_required"):
        ModelSpec(
            model_id="empty_model",
            model_algorithm="linear_regression",
            input_factor_columns=(),
            input_feature_columns=(),
            label_column="forward_return_20d",
        )


def test_evaluation_and_portfolio_specs_are_signal_owned() -> None:
    signal = SignalSpec(
        signal_id="linear_prediction_score",
        source_column="prediction_score",
        source_kind="prediction",
        direction="higher_is_better",
    )
    evaluation = EvaluationSpec(
        run_id="signal_eval",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(signal,),
        label_columns=("forward_return_20d",),
    )
    portfolio = PortfolioSpec(
        portfolio_id="signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(signal,),
        return_source_column="forward_return_20d",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
    )

    assert evaluation.signals == (signal,)
    assert portfolio.signals == (signal,)


def test_signal_contracts_reject_filter_and_label_collisions() -> None:
    with pytest.raises(ValueError, match="filter_column_in_input_columns"):
        SignalSpec(
            signal_id="eligible",
            source_column="prediction_score",
            source_kind="prediction",
        )

    with pytest.raises(ValueError, match="evaluation_signal_label_overlap"):
        EvaluationSpec(
            run_id="bad_eval_signal_column",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="prediction_score",
                    source_column="forward_return_20d",
                    source_kind="prediction",
                ),
            ),
            label_columns=("forward_return_20d",),
        )

    with pytest.raises(ValueError, match="evaluation_signal_label_overlap"):
        EvaluationSpec(
            run_id="bad_eval_signal_id",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="forward_return_20d",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            label_columns=("forward_return_20d",),
        )


def test_signal_contracts_reject_cross_signal_id_column_collisions() -> None:
    own_name_signal = SignalSpec(
        signal_id="prediction_score",
        source_column="prediction_score",
        source_kind="prediction",
    )
    EvaluationSpec(
        run_id="self_named_eval",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(own_name_signal,),
        label_columns=("forward_return_20d",),
    )
    PortfolioSpec(
        portfolio_id="self_named_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="a_share_research",
        signals=(own_name_signal,),
        return_source_column="forward_return_20d",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
    )

    cross_collision = (
        SignalSpec(
            signal_id="signal_a",
            source_column="raw_a",
            source_kind="factor",
        ),
        SignalSpec(
            signal_id="raw_a",
            source_column="raw_b",
            source_kind="factor",
        ),
    )
    with pytest.raises(ValueError, match="evaluation_signal_id_column_overlap"):
        EvaluationSpec(
            run_id="bad_eval_cross_signal",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=cross_collision,
            label_columns=("forward_return_20d",),
        )
    with pytest.raises(ValueError, match="portfolio_signal_id_column_overlap"):
        PortfolioSpec(
            portfolio_id="bad_portfolio_cross_signal",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=cross_collision,
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )


def test_portfolio_signal_contract_rejects_duplicate_and_reserved_names() -> None:
    with pytest.raises(
        ValueError,
        match="portfolio_signal_columns_must_not_contain_duplicates",
    ):
        PortfolioSpec(
            portfolio_id="bad_duplicate_signal_columns",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="signal_a",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
                SignalSpec(
                    signal_id="signal_b",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )

    with pytest.raises(ValueError, match="portfolio_signal_return_overlap"):
        PortfolioSpec(
            portfolio_id="bad_label_overlap",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="forward_return_20d",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
        )

    with pytest.raises(TypeError):
        PortfolioSpec(
            portfolio_id="bad_weight_overlap",
            market_id=MarketId("a_share"),
            research_dataset_id="a_share_research",
            signals=(
                SignalSpec(
                    signal_id="prediction_score",
                    source_column="prediction_score",
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return_20d",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
            weight_column="prediction_score",
        )


def test_compute_panel_spec_requires_signal_columns() -> None:
    spec = ComputePanelSpec(signal_columns=("prediction_score",), label_column="label")

    assert spec.signal_columns == ("prediction_score",)

    with pytest.raises(ValidationError, match="signal_columns"):
        ComputePanelSpec(signal_columns=(), label_column="label")


def _complete_research_dataset_spec() -> ResearchDatasetSpec:
    return ResearchDatasetSpec(
        data_id="complete_research",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "factor_1": ColumnRole.FACTOR,
            "feature_1": ColumnRole.FEATURE,
            "label_1": ColumnRole.LABEL,
            "eligible": ColumnRole.MASK,
            "context_1": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id=MarketId("a_share"),
        input_data_ids=("factors", "labels", "annotations"),
        base_input_data_id="annotations",
        factor_columns=("factor_1",),
        feature_columns=("feature_1",),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="feature_1_from_factor_1",
                source_columns=("factor_1",),
                output_columns=("feature_1",),
                transform="cross_sectional_zscore",
            ),
        ),
        label_columns=("label_1",),
        filter_columns=("eligible",),
        context_columns=("context_1",),
    )
