"""A-share low-frequency research protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta

import pandas as pd

from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.models import _validate_a_share_model_family_ids
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import TargetSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    a_share_analysis_windows,
    is_datetime_window,
)
from quant_research_platform.markets.implementations.a_share import ids as market_ids
from quant_research_platform.research.factors.expectations.calculations import (
    FORECAST_FRESHNESS_DAYS,
)
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
from quant_research_platform.research.features.transforms import (
    feature_transforms,
)
from quant_research_platform.research.filtering import A_SHARE_ELIGIBILITY_FACTS
from quant_research_platform.research.labels.forward_returns import (
    forward_return_available_at_column,
    forward_return_label_column,
)
from quant_research_platform.research.portfolios.accounting import AlphaCalibration
from quant_research_platform.research.portfolios.optimizer import OptimizerConfig
from quant_research_platform.research.portfolios.risk import FactorRiskConfig
from quant_research_platform.research.randomness import ROOT_SEED

PROTOCOL_ID = "a_share_low_frequency"
MARKET_MODEL_FACTORS = ("beta_252d", "residual_volatility_60d")
BENCHMARK_TOTAL_RETURN_INDICES = {"csi500": "H00905", "csi1000": "H00852"}
RESEARCH_DATASET_IDS = {
    "pv_all": "pv_all",
    "pv_expanded": "pv_expanded",
    "pv_nf_expectations": "pv_nf_expectations",
    "fusion_nf_core": "fusion_nf_core",
    "fusion_nf_expectations": "fusion_nf_expectations",
}
SCENARIO_IDS = tuple(RESEARCH_DATASET_IDS)
ACCESS_WINDOWS = ("development", "holdout")
DATA_START = date(2011, 1, 1)
DATA_END = date(2025, 12, 31)
FORMAL_DATASET_START = date(2014, 1, 1)
FORMAL_DATASET_END = date(2025, 12, 31)
DEVELOPMENT_START = date(2019, 1, 1)
DEVELOPMENT_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
LABEL_HORIZONS = (1, 5, 10, 20, 60, 120)
PRIMARY_LABEL_COLUMN = forward_return_label_column(20)
MODEL_TARGET_COLUMN = "model_target"


def quarter_id_for_fold(fold: ExperimentFold) -> str:
    """Return the A-share persisted quarter identity for one protocol fold."""

    if not isinstance(fold, ExperimentFold):
        raise TypeError("a_share_experiment_fold_required")
    year, separator, quarter = fold.fold_id.partition("-q")
    if len(year) != 4 or separator != "-q" or quarter not in {"1", "2", "3", "4"}:
        raise ValueError("a_share_experiment_fold_id_invalid")
    return f"{year}Q{quarter}"


EXECUTION_ASSUMPTIONS = ExecutionAssumptions(
    initial_nav=1_000_000_000.0,
    slippage_bps=5.0,
    impact_bps=25.0,
    adv_participation=0.05,
    adv_sensitivities=(0.01, 0.02, 0.05, 0.10),
)
STATISTICS_BLOCK = 20
STATISTICS_REPETITIONS = 10_000
HAC_LAG = 19
RISK_STYLE_COLUMNS = (
    "log_float_market_cap",
    "beta_252d",
    "momentum_252_20",
    "residual_volatility_60d",
    "amihud_60d",
    "earnings_yield",
)


@dataclass(frozen=True)
class AShareProtocol:
    scenario_id: str
    access_window: str = "development"
    root_seed: int = ROOT_SEED
    frozen_development: dict[str, object] | None = None
    frozen_alpha: tuple[tuple[tuple[str, str, str], AlphaCalibration], ...] = ()
    configuration_json: str | None = None
    risk_config: FactorRiskConfig = FactorRiskConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS:
            raise ValueError("unknown_a_share_scenario")
        if self.access_window not in ACCESS_WINDOWS:
            raise ValueError("unknown_a_share_access_window")
        if self.root_seed != ROOT_SEED:
            raise ValueError("a_share_root_seed_must_be_frozen")
        if self.configuration_json is not None:
            models = self.configuration()["models"]
            _validate_a_share_model_family_ids(item["model_id"] for item in models)
            if any(
                item.get("sample_filter_mode", "eligible") != "eligible"
                for item in models
            ):
                raise ValueError("a_share_model_sample_filter_mode_not_supported")
            half_lives = {
                item.get("training_weighting", {}).get("half_life") for item in models
            }
            if 756 in half_lives and len(half_lives) > 1:
                raise ValueError("a_share_training_weighting_research_roles_mixed")
            if half_lives.difference({None, 756}):
                raise ValueError("a_share_training_weighting_half_life_not_supported")

    @property
    def time_window(self) -> TimeWindow:
        return TimeWindow(
            DEVELOPMENT_START if self.access_window == "development" else HOLDOUT_START,
            DEVELOPMENT_END if self.access_window == "development" else HOLDOUT_END,
        )

    @property
    def data_window(self) -> TimeWindow:
        return TimeWindow(DATA_START, DATA_END)

    @property
    def formal_dataset_window(self) -> TimeWindow:
        return TimeWindow(FORMAL_DATASET_START, FORMAL_DATASET_END)

    def configuration(self):
        import json

        return (
            None
            if self.configuration_json is None
            else json.loads(self.configuration_json)
        )

    @property
    def forecast_freshness_days(self):
        configuration = self.configuration()
        return (
            FORECAST_FRESHNESS_DAYS
            if configuration is None
            else configuration["evaluation"].get(
                "forecast_freshness_days", FORECAST_FRESHNESS_DAYS
            )
        )

    @property
    def protocol_id(self) -> str:
        return PROTOCOL_ID

    @property
    def factor_columns(self) -> tuple[str, ...]:
        if self.configuration_json is not None:
            return tuple(self.configuration()["evaluation"]["factors"])
        columns = (
            *PRICE_VOLUME_LIBRARY,
            *PRICE_VOLUME_PATH_FACTORS,
            *MARKET_MODEL_FACTORS,
        )
        if self.scenario_id in {"fusion_nf_core", "fusion_nf_expectations"}:
            columns += FUNDAMENTAL_FACTORS
        if self.scenario_id in {"pv_nf_expectations", "fusion_nf_expectations"}:
            columns += ALL_EXPECTATIONS_FACTORS
        return columns

    @property
    def factor_fdr_families(self) -> dict[str, list[str]]:
        definitions = {item.output_column: item for item in factor_catalog().list()}
        families = {}
        for column in self.factor_columns:
            families.setdefault(definitions[column].family, []).append(column)
        return families

    def manifest(self) -> dict[str, object]:
        return {
            **(
                {"research_configuration": self.configuration()}
                if self.configuration_json is not None
                else {}
            ),
            **(
                {"frozen_development": self.frozen_development}
                if self.frozen_development is not None
                else {}
            ),
            "protocol_id": self.protocol_id,
            "scenario_id": self.scenario_id,
            "access_window": self.access_window,
            "data_window": {
                "start": DATA_START.isoformat(),
                "end": DATA_END.isoformat(),
            },
            "formal_dataset_window": {
                "start": FORMAL_DATASET_START.isoformat(),
                "end": FORMAL_DATASET_END.isoformat(),
            },
            "development_window": {
                "start": DEVELOPMENT_START.isoformat(),
                "end": DEVELOPMENT_END.isoformat(),
            },
            "holdout_window": {
                "start": HOLDOUT_START.isoformat(),
                "end": HOLDOUT_END.isoformat(),
            },
            "model_chronology": {
                "scoring_lookback_months": 12,
                "warmup_window": {
                    "start": (
                        pd.Timestamp(DEVELOPMENT_START) - pd.DateOffset(months=12)
                    )
                    .date()
                    .isoformat(),
                    "end": (pd.Timestamp(DEVELOPMENT_START) - pd.Timedelta(days=1))
                    .date()
                    .isoformat(),
                },
            },
            "selection_groups": {
                "global": {
                    "scoring": "all_formal_validation",
                    "development_curve": "retrospective",
                },
                "quarterly": {
                    "scoring": "past_12_months",
                    "development_curve": "causal",
                },
            },
            "root_seed": self.root_seed,
            "benchmark_total_return_indices": {
                benchmark: BENCHMARK_TOTAL_RETURN_INDICES[benchmark]
                for benchmark in dict.fromkeys(
                    candidate.benchmark_id
                    for candidate in self.portfolio_candidates()
                    if candidate.benchmark_id is not None
                )
            },
            "execution_assumptions": {
                "initial_nav": EXECUTION_ASSUMPTIONS.initial_nav,
                "valuation_currency": "CNY",
                "slippage_bps": EXECUTION_ASSUMPTIONS.slippage_bps,
                "impact_bps": EXECUTION_ASSUMPTIONS.impact_bps,
                "adv_participation": EXECUTION_ASSUMPTIONS.adv_participation,
                "adv_sensitivities": list(EXECUTION_ASSUMPTIONS.adv_sensitivities),
            },
            "expectations": {
                "freshness_days": self.forecast_freshness_days,
                "role": "robustness"
                if self.forecast_freshness_days == 180
                else "primary",
            },
            "feature_recipe": {
                "recipe_id": self.configuration().get(
                    "feature_recipe", "a_share_canonical"
                )
                if self.configuration_json is not None
                else "a_share_canonical",
                "role": "robustness"
                if self.configuration_json is not None
                and self.configuration().get("feature_recipe", "a_share_canonical")
                == "a_share_quantile_1_99"
                else "primary",
            },
            "training_weighting": {
                "role": "robustness"
                if self.configuration_json is not None
                and any(
                    model.get("training_weighting", {}).get("half_life") == 756
                    for model in self.configuration()["models"]
                )
                else "primary",
            },
            "risk": asdict(self.risk_config),
            "optimizer": asdict(self.optimizer_config),
            "statistics": {
                "inference_clock": "ordered_defined_observations",
                "inference_population": "dates_with_defined_statistic",
                "input_date_domain": "dataset_dates_in_analysis_window",
                "hac_lag": HAC_LAG,
                "bootstrap_block": STATISTICS_BLOCK,
                "bootstrap_repetitions": STATISTICS_REPETITIONS,
                "fdr_primary": "BH_5pct",
                "fdr_robustness": "BY_5pct",
                "cscv_blocks": 10,
                "group_comparison": {
                    "direction": "quarterly_minus_global",
                    "metric": "net_return",
                    "primary_portfolio_id": (
                        self.configuration()["primary_portfolio_id"]
                        if self.configuration_json is not None
                        else None
                    ),
                    "portfolio_ids": [
                        item["candidate_id"]
                        for item in self.configuration()["portfolios"]
                    ]
                    if self.configuration_json is not None
                    else [],
                    "fdr_method": "BY",
                    "alpha": 0.05,
                    "confidence": 0.95,
                    "seed": self.root_seed,
                },
                "factor_fdr_families": self.factor_fdr_families,
                "fama_macbeth_core_factors": [
                    item.output_column
                    for item in factor_catalog().list()
                    if item.factor_set is FactorSet.CORE
                    and item.output_column in self.factor_columns
                ],
            },
            "attribution_groups": {
                canonical_json_hash(list(styles)): {
                    "style_columns": list(styles),
                    "candidate_ids": [
                        candidate.candidate_id
                        for candidate in self.portfolio_candidates()
                        if candidate.risk_style_columns == styles
                    ],
                }
                for styles in dict.fromkeys(
                    candidate.risk_style_columns
                    for candidate in self.portfolio_candidates()
                )
            },
            "portfolio_candidates": [
                candidate.candidate_id for candidate in self.portfolio_candidates()
            ],
        }

    def target_manifest(self, target_id, producer_kind):
        """Bind consumed facts; upstream identities bind target inputs."""
        from quant_research_platform.catalog.models import (
            _effective_model_configurations,
        )

        full = self.manifest()
        if producer_kind == "REPORT":
            return full
        base = {
            key: full[key] for key in ("protocol_id", "scenario_id", "access_window")
        }
        if (
            self.configuration_json is not None
            and "stage_selection" in self.configuration()
        ):
            base["stage_selection"] = {
                "sources": self.configuration()["stage_selection"]["sources"]
            }
        if producer_kind == "MODEL":
            base.update(
                {
                    key: full[key]
                    for key in (
                        "formal_dataset_window",
                        "development_window",
                        "holdout_window",
                        "model_chronology",
                        "selection_groups",
                        "root_seed",
                    )
                }
            )
            base["research_configuration"] = {
                "models": _effective_model_configurations(
                    self.configuration()["models"]
                )
            }
        elif producer_kind == "PREDICTION":
            base["selection_group"] = self.selection_group_for_target(target_id)
        elif producer_kind == "EVALUATION":
            base.update(
                {
                    key: full[key]
                    for key in (
                        "formal_dataset_window",
                        "development_window",
                        "holdout_window",
                    )
                }
            )
        elif producer_kind == "STATISTIC" and target_id in (
            *self.statistics_target_ids,
            self.factor_statistics_target_id,
        ):
            base.update(
                {
                    key: full[key]
                    for key in (
                        "formal_dataset_window",
                        "development_window",
                        "holdout_window",
                        "statistics",
                        "root_seed",
                    )
                }
            )
            base["research_configuration"] = {
                key: self.configuration()[key]
                for key in (
                    ("evaluation",)
                    if target_id == self.factor_statistics_target_id
                    else ("evaluation", "comparison_model_id")
                )
            }
            if target_id != self.factor_statistics_target_id:
                base["selection_group"] = self.selection_group_for_target(target_id)
        else:
            base.update(
                {
                    key: full[key]
                    for key in (
                        "formal_dataset_window",
                        "development_window",
                        "holdout_window",
                        "execution_assumptions",
                        "risk",
                        "optimizer",
                        "benchmark_total_return_indices",
                    )
                }
            )
            if (
                producer_kind == "STATISTIC"
                and target_id == f"{self.scenario_id}_portfolio_statistics"
            ):
                base["portfolio_candidates"] = full["portfolio_candidates"]
                base["attribution_groups"] = full["attribution_groups"]
                base["statistics"] = full["statistics"]
            if producer_kind == "PORTFOLIO" and self.frozen_development is not None:
                candidate = next(
                    item
                    for item in self.portfolio_candidates()
                    if target_id == f"{self.scenario_id}_{item.candidate_id}_ledger"
                )
                if candidate.construction == "benchmark_enhanced":
                    base["frozen_alpha_source_identity"] = next(
                        source["reference"]["instance_identity"]
                        for source in self.frozen_development["alpha_sources"][
                            candidate.selection_group
                        ]
                        if source["reference"]["target_id"] == target_id
                    )
        if producer_kind == "STATISTIC":
            base["training_weighting"] = full["training_weighting"]
            base["feature_recipe"] = full["feature_recipe"]
            configuration = self.configuration()
            consumed = set(self.factor_columns)
            if configuration is not None:
                for model in configuration["models"]:
                    consumed.update(model["input_factor_columns"])
                    consumed.update(
                        column.removeprefix(prefix)
                        for column in model["input_feature_columns"]
                        for prefix in (
                            "feature_",
                            "feature_missing_",
                            "feature_observed_",
                        )
                        if column.startswith(prefix)
                    )
            if consumed.intersection(ALL_EXPECTATIONS_FACTORS):
                base["expectations"] = full["expectations"]
        return base

    def target_window(self, action, data_spec, window, target_id=None):
        if action != "research run":
            return (
                source_history_window(
                    data_spec, window, freshness_days=self.forecast_freshness_days
                )
                if isinstance(data_spec, InterimDataSpec)
                else window
            )
        if isinstance(data_spec, (InterimDataSpec, ProcessedDataSpec)):
            return TimeWindow(
                DATA_START,
                DEVELOPMENT_END if self.access_window == "development" else DATA_END,
            )
        if isinstance(data_spec, ResearchDatasetSpec):
            return TimeWindow(
                FORMAL_DATASET_START
                if self.access_window == "development"
                else HOLDOUT_START,
                DEVELOPMENT_END if self.access_window == "development" else HOLDOUT_END,
            )
        if self.access_window == "holdout":
            return window
        if target_id in {
            f"{self.scenario_id}_evaluation",
            *self.statistics_target_ids,
            self.factor_statistics_target_id,
        }:
            return a_share_analysis_windows(self.manifest(), window)[0]
        if target_id == self.model_selection_target_id:
            return TimeWindow(
                self.manifest()["model_chronology"]["warmup_window"]["start"],
                pd.Timestamp(window.end).date(),
            )
        if target_id in self.prediction_target_ids:
            return TimeWindow(
                max(
                    pd.Timestamp(self.manifest()["development_window"]["start"]),
                    pd.Timestamp(window.start) - pd.DateOffset(months=12),
                ).date(),
                pd.Timestamp(window.end).date(),
            )
        return window

    def frozen_target_ids(self):
        return (self.model_selection_target_id,)

    def freeze_development(self, record, references, frozen, artifacts):
        from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
        from quant_research_platform.research.portfolios.accounting import (
            frozen_alpha_calibration,
        )

        persisted = artifacts[self.model_selection_target_id]
        expected = {
            (group, family)
            for group in ("global", "quarterly")
            for family in self.model_ids_for_target(self.model_selection_target_id)
        }
        if {
            (entry["selection_group"], entry["family_id"])
            for entry in persisted.final_models
        } != expected:
            raise ValueError("holdout_development_final_models_incomplete")
        calibrations = {}
        sources = {}
        blocker = None
        for group in ("global", "quarterly"):
            sources[group] = []
            for candidate in self.portfolio_candidates():
                if (
                    candidate.selection_group != group
                    or candidate.construction != "benchmark_enhanced"
                ):
                    continue
                source = references[
                    f"{self.scenario_id}_{candidate.candidate_id}_ledger"
                ]
                try:
                    ledger = artifacts[source.target_id]
                    current, observed_at = frozen_alpha_calibration(
                        ledger.final_alpha_calibration,
                        cutoff=pd.Timestamp(record.research_window["end"]),
                        selection_group=group,
                    )
                except PortfolioTargetBlocked as exc:
                    blocker = str(exc)
                    break
                key = (
                    group,
                    candidate.prediction_selector,
                    candidate.rebalance_schedule,
                )
                if key in calibrations and calibrations[key] != (current, observed_at):
                    raise ValueError("holdout_alpha_sources_disagree")
                calibrations[key] = (current, observed_at)
                sources[group].append(
                    {
                        "selection_group": group,
                        "reference": source.payload(),
                        "observed_at": observed_at,
                    }
                )
            if blocker is not None:
                break
        frozen["alpha_sources"] = sources
        return replace(
            self,
            access_window="holdout",
            frozen_development=frozen,
            frozen_alpha=tuple((key, value[0]) for key, value in calibrations.items()),
        ), blocker

    def target_specs(self, action: str) -> tuple[TargetSpec, ...]:
        if action != "research run":
            return ()
        dataset = self.scenario_id
        evaluation = f"{dataset}_evaluation"
        model_selection = self.model_selection_target_id
        portfolio_candidates = self.portfolio_candidates()
        portfolios = tuple(
            TargetSpec(
                f"{dataset}_{candidate.candidate_id}_ledger",
                dependencies=(
                    candidate.prediction_target_id,
                    candidate.research_dataset_id,
                    candidate.execution_mask_target_id,
                    candidate.trading_calendar_target_id,
                    candidate.fee_schedule_target_id,
                    candidate.market_panel_target_id,
                    "market_bars_1d",
                    "stock_master",
                    *candidate.risk_context_target_ids,
                    *(
                        (candidate.benchmark_weights_target_id,)
                        if candidate.benchmark_weights_target_id
                        else ()
                    ),
                ),
            )
            for candidate in portfolio_candidates
        )
        return (
            TargetSpec(
                evaluation,
                dependencies=(dataset,),
            ),
            TargetSpec(self.factor_statistics_target_id, dependencies=(dataset,)),
            TargetSpec(
                model_selection,
                dependencies=() if self.access_window == "holdout" else (dataset,),
            ),
            *(
                TargetSpec(
                    predictions,
                    dependencies=(dataset, model_selection),
                )
                for predictions in self.prediction_target_ids
            ),
            *(
                TargetSpec(
                    statistics,
                    dependencies=(
                        predictions,
                        model_selection,
                        dataset,
                        *(
                            (self.statistics_target_ids[0],)
                            if statistics == self.statistics_target_ids[1]
                            else ()
                        ),
                    ),
                )
                for predictions, statistics in zip(
                    self.prediction_target_ids, self.statistics_target_ids, strict=True
                )
            ),
            *portfolios,
            TargetSpec(
                f"{dataset}_portfolio_statistics",
                dependencies=tuple(
                    dict.fromkeys(
                        (
                            *(item.target_id for item in portfolios),
                            dataset,
                            "market_panel_1d",
                            "trading_calendar",
                            "market_bars_1d",
                            "stock_master",
                            *(
                                (market_ids.PRICE_VOLUME_DATA_ID,)
                                if not any(
                                    candidate.benchmark_id
                                    for candidate in portfolio_candidates
                                )
                                else ("benchmark_weights", "index_market")
                            ),
                            *(
                                target_id
                                for candidate in portfolio_candidates
                                for target_id in candidate.risk_context_target_ids
                            ),
                        )
                    )
                ),
            ),
        )

    @property
    def model_selection_target_id(self) -> str:
        return f"{self.scenario_id}_model_selection_refit"

    @property
    def prediction_target_ids(self):
        return tuple(
            f"{self.scenario_id}_{group}_persisted_predictions"
            for group in ("global", "quarterly")
        )

    @property
    def factor_statistics_target_id(self):
        return f"{self.scenario_id}_factor_statistics"

    @property
    def statistics_target_ids(self):
        return tuple(
            f"{self.scenario_id}_{group}_statistics"
            for group in ("global", "quarterly")
        )

    def selection_group_for_target(self, target_id):
        for group, prediction, statistics in zip(
            ("global", "quarterly"),
            self.prediction_target_ids,
            self.statistics_target_ids,
            strict=True,
        ):
            if target_id in (prediction, statistics):
                return group
        raise ValueError("a_share_group_target_invalid")

    def model_ids_for_target(self, target_id: str) -> tuple[str, ...]:
        if target_id == self.model_selection_target_id:
            if self.configuration_json is not None:
                return tuple(
                    model["model_id"] for model in self.configuration()["models"]
                )
            model_ids = ()
        else:
            raise ValueError("unknown_a_share_model_target")
        return model_ids

    @property
    def execution_assumptions(self) -> ExecutionAssumptions:
        return EXECUTION_ASSUMPTIONS

    def portfolio_candidates(self) -> tuple[PortfolioCandidateSpec, ...]:
        if self.configuration_json is None:
            return ()
        result = []
        for item in self.configuration()["portfolios"]:
            if type(item) is not dict or set(item) != {
                "candidate_id",
                "construction",
                "prediction_selector",
                "top_quantile",
                "rebalance_schedule",
                "benchmark_id",
            }:
                raise ValueError("research_portfolio_configuration_invalid")
            enhanced = item["construction"] == "benchmark_enhanced"
            if enhanced and item["benchmark_id"] not in BENCHMARK_TOTAL_RETURN_INDICES:
                raise ValueError("a_share_benchmark_not_supported")
            for group, prediction_target in zip(
                ("global", "quarterly"), self.prediction_target_ids, strict=True
            ):
                result.append(
                    PortfolioCandidateSpec(
                        **{**item, "candidate_id": f"{group}_{item['candidate_id']}"},
                        selection_group=group,
                        prediction_target_id=prediction_target,
                        benchmark_weights_target_id="benchmark_weights"
                        if enhanced
                        else None,
                        risk_style_columns=RISK_STYLE_COLUMNS
                        if enhanced
                        else RISK_STYLE_COLUMNS[:-1],
                        risk_context_target_ids=(
                            market_ids.PRICE_VOLUME_DATA_ID,
                            "market_model_factors",
                            *(("fundamental_factors",) if enhanced else ()),
                        ),
                        research_dataset_id=self.scenario_id,
                        market_panel_target_id=market_ids.MARKET_PANEL_ID,
                        execution_mask_target_id="execution_mask",
                        trading_calendar_target_id="trading_calendar",
                        fee_schedule_target_id="a_share_fee_schedule",
                    )
                )
        return tuple(result)

    def portfolio_candidate_for_target(
        self,
        target_id: str,
    ) -> PortfolioCandidateSpec:
        for candidate in self.portfolio_candidates():
            if target_id == f"{self.scenario_id}_{candidate.candidate_id}_ledger":
                return candidate
        raise ValueError("unknown_a_share_portfolio_target")

    def terminal_targets(self, action: str) -> tuple[str, ...]:
        if action == "dataset build":
            return (self.scenario_id,)
        if action == "research run":
            if self.configuration_json is not None:
                if self.configuration()["stage"] == "dataset":
                    return (self.scenario_id,)
                if self.configuration()["stage"] == "evaluation":
                    return (
                        f"{self.scenario_id}_evaluation",
                        self.factor_statistics_target_id,
                    )
                return (
                    f"{self.scenario_id}_evaluation",
                    self.factor_statistics_target_id,
                    *self.statistics_target_ids,
                    *(
                        f"{self.scenario_id}_{candidate.candidate_id}_ledger"
                        for candidate in self.portfolio_candidates()
                    ),
                    *(
                        (f"{self.scenario_id}_portfolio_statistics",)
                        if self.portfolio_candidates()
                        else ()
                    ),
                )
            return (
                f"{self.scenario_id}_evaluation",
                self.factor_statistics_target_id,
                *self.statistics_target_ids,
                f"{self.scenario_id}_portfolio_statistics",
                *(
                    f"{self.scenario_id}_{candidate.candidate_id}_ledger"
                    for candidate in self.portfolio_candidates()
                ),
            )
        if action == "data materialize":
            return ()
        raise ValueError("unknown_a_share_action")


def _inference_blocker(manifest):
    """Exclude independent robustness runs only from formal joint inference."""
    if manifest["access_window"] == "development" and (
        "stage_selection" in manifest
        or "stage_selection" in manifest.get("research_configuration", {})
    ):
        return "development_stage_selection_exploratory"
    if manifest.get("expectations", {}).get("role") == "robustness":
        return "expectations_robustness_excluded"
    if manifest.get("feature_recipe", {}).get("role") == "robustness":
        return "feature_recipe_robustness_excluded"
    if manifest.get("training_weighting", {}).get("role") == "robustness":
        return "training_weighting_robustness_excluded"
    return None


def model_folds(
    sessions: pd.DatetimeIndex,
    *,
    chronology: Mapping[str, object],
    formal_dataset_window: Mapping[str, object],
    development_window: Mapping[str, object],
) -> tuple[ExperimentFold, ...]:
    if type(chronology) is not dict or set(chronology) != {
        "scoring_lookback_months",
        "warmup_window",
    }:
        raise ValueError("model_chronology_missing")
    history_start = pd.Timestamp(formal_dataset_window["start"])
    development_start = pd.Timestamp(development_window["start"])
    development_end = pd.Timestamp(development_window["end"])
    months = chronology["scoring_lookback_months"]
    if type(months) is not int or months != 12:
        raise ValueError("model_chronology_scoring_months_invalid")
    warmup_start = development_start - pd.DateOffset(months=months)
    if chronology["warmup_window"] != {
        "start": warmup_start.date().isoformat(),
        "end": (development_start - pd.Timedelta(days=1)).date().isoformat(),
    }:
        raise ValueError("model_chronology_warmup_invalid")
    axis = pd.DatetimeIndex(sessions).normalize().drop_duplicates().sort_values()
    if axis.empty or axis.min() < history_start or axis.max() > development_end:
        raise ValueError("model_chronology_sessions_outside_development")
    if axis.min().year != history_start.year:
        raise ValueError("model_chronology_formal_history_missing")
    folds = []
    for period in pd.period_range(warmup_start, development_end, freq="Q"):
        validation = axis[axis.to_period("Q") == period]
        training = axis[axis < period.start_time]
        if training.empty or validation.empty:
            raise ValueError("a_share_walk_forward_window_incomplete")
        folds.append(
            ExperimentFold(
                fold_id=f"{period.year:04d}-q{period.quarter}",
                train=TimeWindow(training[0], training[-1]),
                validation=TimeWindow(validation[0], validation[-1]),
            )
        )
    return tuple(folds)


def model_selection_windows(fold_windows, *, chronology, development_window, complete):
    """Own quarterly and final selection cutoffs over already validated folds."""
    months = chronology["scoring_lookback_months"]
    start = pd.Timestamp(development_window["start"])
    cutoff = pd.Timestamp(development_window["end"])
    decisions = [
        (
            "quarterly",
            quarter,
            pd.Timestamp(train_end),
            TimeWindow(
                pd.Timestamp(validation_start).to_period("Q").start_time
                - pd.DateOffset(months=months),
                pd.Timestamp(train_end),
            ),
        )
        for quarter, train_end, validation_start in fold_windows
        if pd.Timestamp(validation_start) >= start
    ]
    if complete:
        decisions.extend(
            (
                ("global", "final", cutoff, TimeWindow(start, cutoff)),
                (
                    "quarterly",
                    "final",
                    cutoff,
                    TimeWindow(
                        cutoff + pd.Timedelta(days=1) - pd.DateOffset(months=months),
                        cutoff,
                    ),
                ),
            )
        )
    return tuple(decisions)


def source_history_window(
    data_spec: InterimDataSpec,
    window: TimeWindow,
    *,
    freshness_days: int = FORECAST_FRESHNESS_DAYS,
) -> TimeWindow:
    """Bind actual preceding state without widening the knowledge cutoff."""
    if data_spec.source_id == "choice" or data_spec.data_id == "market_bars_1d":
        return data_history_window(window)
    if data_spec.data_id == "analyst_forecast":
        return TimeWindow(window.start - timedelta(days=freshness_days), window.end)
    return window


def processed_input_window(data_spec, window: TimeWindow) -> TimeWindow:
    """Plan one frozen input envelope for factors with preceding-session inputs."""
    from quant_research_platform.research.factors.engine import factor_lookback_bars

    if (
        isinstance(data_spec, ProcessedDataSpec)
        and data_spec.kind is ProcessedDataKind.FACTOR
    ):
        definitions = tuple(
            definition
            for definition in factor_catalog().list()
            if definition.output_column in data_spec.required_columns
        )
        if factor_lookback_bars(definitions):
            return data_history_window(window)
    return window


def data_history_window(window: TimeWindow) -> TimeWindow:
    """Keep the frozen source-history envelope and the requested knowledge cutoff."""
    start = pd.Timestamp(DATA_START) if is_datetime_window(window) else DATA_START
    return TimeWindow(min(start, window.start), window.end)


def protocol(
    scenario_id: str,
    access_window: str = "development",
) -> AShareProtocol:
    return AShareProtocol(scenario_id, access_window)


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


def evaluation_catalog() -> EvaluationCatalog:
    return EvaluationCatalog(
        EvaluationSpec(
            evaluation_id=f"{selected}_evaluation",
            market_id=market_ids.MARKET_ID,
            research_dataset_id=RESEARCH_DATASET_IDS[selected],
            signals=tuple(
                SignalSpec(
                    signal_id=column,
                    source_kind="factor",
                    source_column=column,
                )
                for column in protocol(selected).factor_columns
            ),
            label_columns=(PRIMARY_LABEL_COLUMN,),
            label_availability_columns={
                PRIMARY_LABEL_COLUMN: forward_return_available_at_column(20)
            },
            n_groups=5,
        )
        for selected in SCENARIO_IDS
    )


def research_dataset_specs() -> tuple[ResearchDatasetSpec, ...]:
    specs: list[ResearchDatasetSpec] = []
    for scenario_id in SCENARIO_IDS:
        factor_columns = protocol(scenario_id).factor_columns
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
        transforms = feature_transforms(factor_columns)
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
                filter_columns=("eligible",),
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
