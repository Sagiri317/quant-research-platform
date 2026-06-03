"""Stable 0.1 platform contracts."""

from quant_research_platform.contracts.columns import ColumnRole, FieldSpec, LogicalType
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import (
    AssetId,
    MarketId,
    SourceAssetId,
    SourceId,
)
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.masks import MaskDefinition
from quant_research_platform.contracts.reports import ReportDefinition, ReportSpec

CONTRACT_VERSION = "0.1"

__all__ = [
    "CONTRACT_VERSION",
    "AssetId",
    "ColumnRole",
    "DataFrequency",
    "DataGrain",
    "DataSpec",
    "FactorDefinition",
    "FactorSet",
    "FieldSpec",
    "InterimDataSpec",
    "LabelDefinition",
    "LogicalType",
    "MarketId",
    "MaskDefinition",
    "ProcessedDataSpec",
    "ReportDefinition",
    "ReportSpec",
    "SourceAssetId",
    "SourceId",
]
