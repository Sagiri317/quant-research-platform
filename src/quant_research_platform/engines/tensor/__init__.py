"""Internal tensor evaluation engine foundations."""

from quant_research_platform.engines.tensor.conversion import (
    panel_to_tensor,
    panel_to_tensor_batch,
)
from quant_research_platform.engines.tensor.cross_section import (
    analyze_grouped_returns_tensor,
    evaluate_ic_by_timestamp_tensor,
    summarize_group_spread_tensor,
)
from quant_research_platform.engines.tensor.layout import (
    PanelTensor,
    TensorBatch,
    TensorBatchLayout,
    TensorLayout,
)
from quant_research_platform.engines.tensor.masks import valid_observation_mask
from quant_research_platform.engines.tensor.portfolio import (
    build_tensor_weighted_positions,
    calculate_tensor_long_short,
    calculate_tensor_turnover_and_cost,
)

__all__ = [
    "PanelTensor",
    "TensorBatch",
    "TensorBatchLayout",
    "TensorLayout",
    "analyze_grouped_returns_tensor",
    "evaluate_ic_by_timestamp_tensor",
    "panel_to_tensor",
    "panel_to_tensor_batch",
    "build_tensor_weighted_positions",
    "calculate_tensor_long_short",
    "calculate_tensor_turnover_and_cost",
    "summarize_group_spread_tensor",
    "valid_observation_mask",
]
