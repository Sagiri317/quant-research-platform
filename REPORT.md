# a-share-alpha-lab 研究报告

## 摘要

本项目构建一条本地私有 A 股因子研究流程：用户从 CSMAR 手工导出数据，项目在本地生成 interim、research panel、factor table，并通过 `alpha-research-core` 进行第一版 IC、RankIC 和分组评估。

当前报告是流程和数据资产说明，不构成投资建议，也不代表最终因子结论。

## 数据

| 项目 | 内容 |
|---|---|
| 数据源 | CSMAR，用户手工导出 |
| 数据范围 | 2014-2025 interim / panel / factor 已生成 |
| 评估范围 | 2024-2025 long-only full-sample evaluation；core 年度 breakdown |
| universe | 第一版 A 股范围，排除 B 股和北交所 |
| 私有边界 | raw/interim/processed 和 private reports 不公开、不提交 |

## 方法

```text
CSMAR raw exports
  -> interim CSV
  -> research_panel_2014_2025.parquet
  -> price_volume_factors_2014_2025.parquet
  -> alpha-research-core evaluation
```

第一批 factor set 是 `price_volume`，包括：

- momentum
- reversal
- volatility
- amount liquidity
- turnover
- Amihud illiquidity
- amplitude

默认窗口为 `20, 60, 120`，forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。

## 产物

```text
data/processed_private/csmar/
  panel_full/research_panel_2014_2025.parquet
  price_volume_factors_full/price_volume_factors_2014_2025.parquet

reports/private_real/factor_evaluation/
  full_2024_2025/
    label_availability.csv
    evaluation_manifest.json
    factor_summary.csv
    factor_summary.md
    core_runs/
  comparisons/
```

当前规模：

| 产物 | 数量 | 规模 |
|---|---:|---:|
| panel Parquet | 1 file | 0.586 GB |
| factor Parquet | 1 file | 2.318 GB |
| factor evaluation reports | 1 long-only main root + comparison summaries | 约 0.584 GB，不含已清理的历史 long-short robustness |

2024-2025 evaluation 使用 21 个 price-volume factors、`forward_return_20d` label、5 组分组设置。

## Evaluation 口径

正式主口径采用 `full_2024_2025`：一次性把 2024-2025 输入传给 `alpha-research-core`，用 `forward_return_20d` 做预测标签、`forward_return_1d` 做 PnL 日收益列，显式设置 `portfolio_mode=long_only`，并启用 core 的年度 breakdown。

年度 breakdown 不在年底清仓，也不在次年重新初始化组合状态。它先保留连续回测产生的组合路径，再把 `net_portfolio_return` 按自然年切分，并在每个年度内部重新统计 IC、RankIC、分组收益、PnL、Sharpe、drawdown 和 turnover。因此，2024 年底无法退出的持仓可以自然带入 2025，并按 `can_exit` 约束继续影响 2025 年初 PnL。

这个口径与“每年空仓重启”的 yearly independent evaluation 不同。后者只能作为诊断对照，不作为主报告口径。旧 long-short 大结果目录已清理，仅保留轻量迁移对比摘要；当前 long-only 结果的不变量检查显示，`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，且 0 成本下 `net_portfolio_return` 与 `gross_portfolio_return`、`long_return` 完全一致。

## 当前观察

long-only 年度结果里，`rev_20d`、`rev_60d`、`rev_120d` 仍是第一版最主要的候选信号。2024 年 `rev_120d` 的 `mean_rankic` 约 0.114，`mean_net_portfolio_return` 约 0.003086；2025 年 `rev_60d` 的 `mean_rankic` 约 0.095，`mean_net_portfolio_return` 约 0.001717。

旧成本敏感性和 924 regime 切片是在 long-short 口径下生成的，已作为历史诊断清理，不再作为 A 股可执行口径结论。下一轮稳健性需要基于 long-only 重新生成。

## 限制

- 当前只覆盖第一批 price-volume factors。
- 当前 evaluation 是 2024-2025 full-sample 第一版诊断。
- long-only 主结果已生成；成本敏感性、924 regime、行业中性化、风险暴露和正式组合约束分析仍需在 long-only 口径下继续完成。
- CSMAR 明细数据和可还原 vendor 数据的报告不公开。
