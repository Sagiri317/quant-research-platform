"""Stable 0.1 platform contracts."""

from quant_research_platform.contracts.columns import ColumnRole, FieldSpec, LogicalType
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import (
    AssetId,
    MarketId,
    SourceAssetId,
    SourceId,
)
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.privacy import PrivacyLevel
from quant_research_platform.contracts.reports import ReportDefinition, ReportSpec
from quant_research_platform.contracts.run_id import (
    RunIdValidationError,
    validate_run_id,
)

CONTRACT_VERSION = "0.1"

__all__ = [
    "CONTRACT_VERSION",
    "AssetId",
    "ColumnRole",
    "DataFrequency",
    "DataGrain",
    "DataSpec",
    "ExecutionMode",
    "FactorDefinition",
    "FactorSet",
    "FieldSpec",
    "InterimDataSpec",
    "LabelDefinition",
    "LogicalType",
    "MarketId",
    "ProcessedDataKind",
    "ProcessedDataSpec",
    "PrivacyLevel",
    "ReportDefinition",
    "ReportSpec",
    "RunIdValidationError",
    "SourceAssetId",
    "SourceId",
    "validate_run_id",
]
