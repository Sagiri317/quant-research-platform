"""Stable 0.1 platform contracts."""

from quant_research_platform.contracts.columns import ColumnRole, FieldSpec, LogicalType
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
    data_frequency_slug,
    horizon_unit_suffix,
    normalize_data_frequency,
    require_supported_data_frequency,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import (
    AssetId,
    MarketId,
    SourceAssetId,
    SourceId,
)
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    PortfolioSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.privacy import PrivacyLevel
from quant_research_platform.contracts.reports import ReportDefinition, ReportSpec
from quant_research_platform.contracts.run_id import (
    RunIdValidationError,
    validate_run_id,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.workflows import WorkflowRunPolicy

CONTRACT_VERSION = "0.1"

__all__ = [
    "CONTRACT_VERSION",
    "AssetId",
    "ColumnRole",
    "CostConfig",
    "DataFrequency",
    "DataGrain",
    "DataSpec",
    "EvaluationDiagnostics",
    "EvaluationSpec",
    "FactorDefinition",
    "FactorSet",
    "FieldSpec",
    "FeatureTransformSpec",
    "InterimDataSpec",
    "LabelDefinition",
    "LogicalType",
    "MarketId",
    "ModelSpec",
    "PortfolioSpec",
    "PredictionSpec",
    "ProcessedDataKind",
    "ProcessedDataSpec",
    "PrivacyLevel",
    "ResearchDatasetSpec",
    "ReportDefinition",
    "ReportSpec",
    "RunIdValidationError",
    "SignalSpec",
    "SourceAssetId",
    "SourceId",
    "TradabilityConfig",
    "WeightConfig",
    "WorkflowRunPolicy",
    "data_frequency_slug",
    "horizon_unit_suffix",
    "normalize_data_frequency",
    "require_supported_data_frequency",
    "validate_run_id",
]
