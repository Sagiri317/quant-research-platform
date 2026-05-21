# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Quant Research Platform Synthetic A-share Demo
#
# This notebook demonstrates the post-migration local platform contract:
#
# synthetic A-share-like panel -> `EvaluationSpec` -> platform core evaluation.
#
# It uses only toy data, writes artifacts to a temporary directory, and does not
# read private CSMAR files, call vendor APIs, use credentials, automate trading,
# or provide investment advice.

# %% [markdown]
# ### 1. Prepare Imports

# %%
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

project_root = Path.cwd().resolve()
if project_root.name == "notebooks":
    project_root = project_root.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from quant_research_platform.core import (  # noqa: E402
    CostConfig,
    EvaluationSpec,
    WeightConfig,
    run_evaluation,
)

# %% [markdown]
# ### 2. Generate A Small Synthetic Panel

# %%
rng = np.random.default_rng(42)
dates = pd.bdate_range("2024-01-02", periods=80)
assets = [f"{index:06d}.SZ" for index in range(1, 13)]

records = []
for asset_index, asset in enumerate(assets):
    returns = rng.normal(loc=0.0005, scale=0.02, size=len(dates))
    close = 100 * np.cumprod(1 + returns)
    momentum = pd.Series(close).pct_change(20)
    reversal = -pd.Series(close).pct_change(5)
    forward_return = pd.Series(returns).shift(-1)
    for row_index, date in enumerate(dates):
        records.append(
            {
                "trade_date": date,
                "stock_code": asset,
                "asset_bucket": "large" if asset_index < 6 else "small",
                "mom_20d": momentum.iloc[row_index],
                "rev_5d": reversal.iloc[row_index],
                "forward_return_1d": forward_return.iloc[row_index],
            }
        )

panel = pd.DataFrame.from_records(records).dropna().reset_index(drop=True)
panel.head()

# %%
panel.groupby("stock_code").agg(
    rows=("trade_date", "size"),
    start=("trade_date", "min"),
    end=("trade_date", "max"),
    mean_momentum=("mom_20d", "mean"),
    mean_forward_return=("forward_return_1d", "mean"),
).head()

# %% [markdown]
# ### 3. Run The Platform Core Evaluation

# %%
with TemporaryDirectory() as tmpdir:
    spec = EvaluationSpec(
        panel=panel,
        timestamp_column="trade_date",
        asset_column="stock_code",
        feature_columns=("mom_20d", "rev_5d"),
        label_columns=("forward_return_1d",),
        metadata={
            "run_id": "notebook_synthetic_ashare_demo",
            "project": "quant-research-platform",
            "kind": "factor",
            "domain": "a_share_synthetic",
            "data_stage": "synthetic_only",
            "uses_external_data": False,
        },
        output_dir=Path(tmpdir) / "notebook_synthetic_ashare_demo",
        n_groups=4,
        weight_configs=(
            WeightConfig(
                method="equal_weight_quantile",
                direction="long_high",
                portfolio_mode="long_short",
                top_quantile=0.25,
                bottom_quantile=0.25,
            ),
        ),
        cost_configs=(CostConfig(slippage_bps=5.0),),
        segment_columns=("asset_bucket",),
        time_breakdowns=("month",),
    )
    result = run_evaluation(spec)
    summary = result.summary
    combinations = result.combinations.copy()
    ic_summary = result.metrics["ic_summary"].copy()
    performance = result.metrics["performance_summary"].copy()

summary

# %%
combinations

# %%
ic_summary[["feature", "label", "mean_ic", "mean_rankic", "icir", "rankicir"]]

# %%
performance_columns = [
    "feature",
    "label",
    "final_cumulative_net_return",
    "mean_turnover",
    "total_cost",
    "max_drawdown",
]
performance[[column for column in performance_columns if column in performance]]

# %% [markdown]
# ### Boundary
#
# This notebook is a synthetic/toy research demo only. It does not use or imply
# real A-share data, benchmark data, vendor data, external downloads, API
# connections, credentials, paper trading, live trading, production automation,
# or investment advice.
