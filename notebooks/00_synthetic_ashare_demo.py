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
# ## Quant Research Platform A 股合成数据演示
#
# 本 notebook 演示迁移后的本地平台合同：
#
# A 股风格合成 panel -> `EvaluationSpec` -> platform evaluation layer。
#
# 它只使用 toy data，把 artifacts 写入临时目录；不读取私有 CSMAR 文件，
# 不调用 vendor API，不使用 credentials，不自动交易，也不提供投资建议。

# %% [markdown]
# ### 1. 准备导入

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

from quant_research_platform import (  # noqa: E402
    CostConfig,
    EvaluationSpec,
    WeightConfig,
    run_evaluation,
)

# %% [markdown]
# ### 2. 生成小型合成 panel

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
# ### 3. 运行平台 evaluation layer

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
# ### 边界
#
# 本 notebook 只是 synthetic/toy research demo；不使用或暗示真实 A 股数据、
# benchmark data、vendor data、外部下载、API 连接、credentials、paper trading、
# live trading、production automation 或投资建议。
