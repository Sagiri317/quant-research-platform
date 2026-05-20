# a-share-alpha-lab 研究证据卡

本卡只保留面试或复盘时最需要的证据链索引；当前进展细节看 [PROJECT_STATUS.md](PROJECT_STATUS.md)，正式研究说明看 [REPORT.md](REPORT.md)。

## 核心证据

| 字段 | 内容 |
|---|---|
| 记录名称 | `csmar_workflow_first_factor_pipeline` |
| 当前状态 | active |
| 日期 | 2026-05-20 |
| 数据 | CSMAR 手工导出；2014-2025 raw -> interim -> full panel -> full factor |
| 主要产物 | `panel_full/research_panel_2014_2025.parquet`；`price_volume_factors_full/price_volume_factors_2014_2025.parquet` |
| evaluation | `full_2024_2025` long-only full-sample run + core `metrics_by_year`；历史 long-short 结果仅作迁移对照 |
| 公开边界 | CSMAR 明细数据、private reports 和可还原 vendor 数据不公开 |

## 方法证据

- `code/csmar/`：CSMAR raw -> interim，含字段合同和列裁剪。
- `code/datasets/`：interim -> full research panel，含生命周期、流动性和可交易约束。
- `code/factors/`：full panel -> price-volume factors + forward labels。
- `code/evaluation/`：slim factor Parquet -> core `EvaluationSpec` -> `run_evaluation(spec)`。
- `code/validation/` / `code/runtime/`：质量检查、空间检查和路径合同。

## 结果证据

- 第一批 factor set 是 `price_volume`：momentum、reversal、volatility、amount liquidity、turnover、Amihud illiquidity、amplitude。
- 默认窗口为 `20, 60, 120`；forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。
- 2024-2025 long-only 第一版诊断中，反转类因子仍是主要候选。
- 2024 年 `rev_120d` 的 `mean_rankic` 约 0.114，`mean_net_portfolio_return` 约 0.003086；2025 年 `rev_60d` 的 `mean_rankic` 约 0.095，`mean_net_portfolio_return` 约 0.001717。
- long-only 不变量已检查：`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，0 成本下 `net_portfolio_return == gross_portfolio_return == long_return`。
- 历史 long-short robustness 不再作为 A 股可执行口径结论；需要按 long-only 重新生成成本和 regime 稳健性。

## 仍需验证

- 当前 evaluation 是 pipeline 和统计诊断，不是最终投资结论。
- 尚未在 long-only 口径下完成成本敏感性、924 regime、行业/市值中性化、风险暴露和正式组合约束分析。
