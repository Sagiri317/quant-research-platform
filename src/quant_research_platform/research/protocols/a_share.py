"""A-share low-frequency research protocol."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
)
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.research.factors.expectations.definitions import (
    ALL_EXPECTATIONS_FACTORS,
    expectations_factor_definitions,
)
from quant_research_platform.research.factors.fundamental.definitions import (
    FUNDAMENTAL_FACTORS,
    fundamental_factor_definitions,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PATH_FACTORS,
    price_volume_factor_definitions,
)
from quant_research_platform.research.filtering import A_SHARE_ELIGIBILITY_FACTS
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)

PROTOCOL_ID = "a_share_low_frequency"
MARKET_MODEL_FACTORS = ("beta_252d", "residual_volatility_60d")
RESEARCH_DATASET_IDS = {
    "pv_all": "pv_all",
    "pv_expanded": "pv_expanded",
    "pv_nf_expectations": "pv_nf_expectations",
    "fusion_nf_core": "fusion_nf_core",
    "fusion_nf_expectations": "fusion_nf_expectations",
}
SCENARIO_IDS = tuple(RESEARCH_DATASET_IDS)
LABEL_HORIZONS = (1, 5, 10, 20, 60, 120)
PRIMARY_LABEL_COLUMN = forward_return_label_column(20)
MODEL_TARGET_COLUMN = "model_target"


EXECUTION_ASSUMPTIONS = ExecutionAssumptions(
    initial_nav=1_000_000_000.0,
    slippage_bps=5.0,
    impact_bps=25.0,
    adv_participation=0.05,
    adv_sensitivities=(0.01, 0.02, 0.05, 0.10),
)
RISK_STYLE_COLUMNS = (
    "log_float_market_cap",
    "beta_252d",
    "momentum_252_20",
    "residual_volatility_60d",
    "amihud_60d",
    "earnings_yield",
)


def _inference_blocker(manifest):
    """Exclude independent robustness runs only from formal joint inference."""
    if manifest.get("selection_is_exploratory", False):
        return "exploratory_selection_excluded"
    if manifest.get("expectations", {}).get("role") == "robustness":
        return "expectations_robustness_excluded"
    if manifest.get("feature_recipe", {}).get("role") == "robustness":
        return "feature_recipe_robustness_excluded"
    if manifest.get("training_weighting", {}).get("role") == "robustness":
        return "training_weighting_robustness_excluded"
    return None


def dataset_factor_columns(dataset_id: str) -> tuple[str, ...]:
    """Return the declared factor set for a static dataset definition."""
    if dataset_id not in RESEARCH_DATASET_IDS:
        raise ValueError("unknown_a_share_dataset")
    columns = (*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS, *MARKET_MODEL_FACTORS)
    if dataset_id in {"fusion_nf_core", "fusion_nf_expectations"}:
        columns += FUNDAMENTAL_FACTORS
    if dataset_id in {"pv_nf_expectations", "fusion_nf_expectations"}:
        columns += ALL_EXPECTATIONS_FACTORS
    return columns


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        (
            *price_volume_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id=market_ids.PRICE_VOLUME_DATA_ID,
                factor_columns=(*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS),
            ),
            *(
                FactorDefinition(
                    factor_id=column,
                    market_id=market_ids.MARKET_ID,
                    factor_set=FactorSet.STANDARD,
                    input_columns=tuple(
                        dict.fromkeys(
                            (
                                "period_return",
                                "float_market_cap",
                                *A_SHARE_ELIGIBILITY_FACTS,
                            )
                        )
                    ),
                    output_column=column,
                    family="market_model",
                    grain=DataGrain.ASSET_TIMESTAMP,
                    required_input_data_id="market_model_factors",
                    lookback_window=312,
                    required_windows=(252, 60),
                    window=312,
                    description=(
                        "Prior-cap weighted eligible market; causal 252-session OLS "
                        "and 60-session population residual volatility."
                    ),
                    timing_semantics="same_session_close_available_at_next_session",
                    null_policy="preserve_nulls_after_min_periods",
                )
                for column in MARKET_MODEL_FACTORS
            ),
            *fundamental_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="fundamental_pit",
            ),
            *expectations_factor_definitions(
                market_id=market_ids.MARKET_ID,
                required_input_data_id="analyst_forecast",
            ),
        )
    )


def label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}d",
            market_id=market_ids.MARKET_ID,
            family="forward_return",
            input_columns=("adj_open", "can_enter"),
            output_column=forward_return_label_column(horizon),
            label_timing="signal_close_to_next_open_to_horizon_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_market_session_adjusted_open",
            exit_rule="entry_plus_horizon_market_sessions_open",
        )
        for horizon in LABEL_HORIZONS
    )


def label_catalog() -> LabelCatalog:
    return LabelCatalog(label_definitions())


def research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    specs: list[ResearchDatasetSpec] = []
    for scenario_id in SCENARIO_IDS:
        factor_columns = dataset_factor_columns(scenario_id)
        feature_columns = tuple(
            feature
            for factor in factor_columns
            for feature in (
                f"feature_{factor}",
                f"feature_missing_{factor}",
                f"feature_observed_{factor}",
            )
        )
        labels = (
            *forward_label_columns(),
            *forward_label_available_columns(),
            MODEL_TARGET_COLUMN,
        )
        context = (
            "model_target_available_at",
            "is_a_share_scope",
            "is_trading",
            "is_st",
            "is_pt",
            "listing_age_trading_days",
            "has_252_trading_sessions",
            "close",
            "adj_close",
            "volume",
            "amount",
            "turnover",
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
            "float_market_cap",
            "total_market_cap",
        )
        if scenario_id in {
            "pv_nf_expectations",
            "fusion_nf_core",
            "fusion_nf_expectations",
        }:
            context += ("is_nonfinancial_industry",)
        if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            context += ("has_legal_fundamental_pit",)
        all_columns = (
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            *factor_columns,
            *feature_columns,
            *labels,
            *context,
        )
        roles = {
            column: _dataset_role(
                column,
                factor_columns,
                feature_columns,
                labels,
                ("eligible",),
            )
            for column in all_columns
        }
        transforms = (FeatureTransformSpec("a_share_canonical", tuple(factor_columns)),)
        input_data_ids = [
            market_ids.MARKET_PANEL_ID,
            market_ids.PRICE_VOLUME_DATA_ID,
            "forward_returns",
        ]
        if set(factor_columns).intersection(MARKET_MODEL_FACTORS):
            input_data_ids.append("market_model_factors")
        if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            input_data_ids.insert(3, "fundamental_factors")
        if scenario_id in {"pv_nf_expectations", "fusion_nf_expectations"}:
            input_data_ids.insert(-1, "expectations_factors")
        specs.append(
            ResearchDatasetSpec(
                data_id=RESEARCH_DATASET_IDS[scenario_id],
                market_id=market_ids.MARKET_ID,
                input_data_ids=tuple(input_data_ids),
                base_input_data_id=market_ids.MARKET_PANEL_ID,
                factor_columns=factor_columns,
                feature_columns=feature_columns,
                feature_transforms=transforms,
                label_columns=labels,
                label_availability_columns={
                    **{
                        forward_return_label_column(
                            h
                        ): forward_return_available_at_column(h)
                        for h in LABEL_HORIZONS
                    },
                    "model_target": "model_target_available_at",
                },
                filter_columns=("eligible",),
                eligibility_recipe=(
                    "a_share_fundamental"
                    if scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}
                    else "a_share_nonfinancial"
                    if scenario_id == "pv_nf_expectations"
                    else "a_share_all"
                ),
                context_columns=context,
                frequency=DataFrequency.DAILY,
                grain="research_row",
                column_roles=roles,
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
                key_coverage={data_id: "strict" for data_id in input_data_ids},
            )
        )
    return tuple(specs)


def research_dataset_catalog():
    from quant_research_platform.catalog.datasets import ResearchDatasetCatalog

    return ResearchDatasetCatalog(research_dataset_specs())


def _dataset_role(
    column: str,
    factors: tuple[str, ...],
    features: tuple[str, ...],
    labels: tuple[str, ...],
    filters: tuple[str, ...],
) -> ColumnRole:
    if column in factors:
        return ColumnRole.FACTOR
    if column in features:
        return ColumnRole.FEATURE
    if column in labels:
        return ColumnRole.LABEL
    if column in filters:
        return ColumnRole.UNIVERSE
    if column == "timestamp":
        return ColumnRole.TIMESTAMP
    if column in {"market_id", "asset_id"}:
        return ColumnRole.KEY
    return ColumnRole.CONTEXT


def forward_label_columns() -> tuple[str, ...]:
    return tuple(forward_return_label_column(horizon) for horizon in LABEL_HORIZONS)


def forward_label_available_columns() -> tuple[str, ...]:
    return tuple(
        forward_return_available_at_column(horizon) for horizon in LABEL_HORIZONS
    )


def validate_model_selection(models):
    """Validate A-share model family identities and sample eligibility."""
    from quant_research_platform.contracts.models import (
        ModelSpec,
        _validate_a_share_model_family_ids,
    )

    models = tuple(models)
    if not models or any(not isinstance(model, ModelSpec) for model in models):
        raise ValueError("a_share_model_selection_required")
    _validate_a_share_model_family_ids(model.model_id for model in models)
    if any(model.sample_filter_mode != "eligible" for model in models):
        raise ValueError("a_share_model_sample_filter_mode_not_supported")
