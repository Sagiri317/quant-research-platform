# Quant Research Platform 研究证据卡

本卡只保留面试、复盘或项目交接时最需要的证据链索引；当前迁移状态看 [PROJECT_STATUS.md](PROJECT_STATUS.md)，正式研究说明看 [REPORT.md](REPORT.md)。

## 核心证据

| 字段 | 内容 |
|---|---|
| 记录名称 | `quant_research_platform_a_share_module` |
| 当前状态 | Phase 1 completed; N1/N2/N3/N4/N7/N8 merged to `main`; 2023-2025 canonical / `rev_120d` evidence and chart pagination synchronized |
| 日期 | 2026-05-27 |
| 项目定位 | 本地多市场量化研究平台；A 股为第一版完整 market module |
| 数据 | CSMAR 手工导出；2014-2025 raw -> interim -> full panel -> full factor |
| 主要产物 | `panel_full/research_panel_2014_2025.parquet`；`price_volume_factors_full/price_volume_factors_2014_2025.parquet`；2023-2025 canonical / `rev_120d` evaluation samples；Factor Factory `rev_120d` smoke |
| evaluation | schema `0.6` evaluation-layer-backed evaluation；`manifest.display` 驱动 API、下载、CSV preview 和 evidence workbench charts |
| 公开边界 | CSMAR 明细数据、private reports 和可还原 vendor 数据不公开 |

## 方法证据

- `src/quant_research_platform/evaluations/`：`EvaluationSpec`、panel validation、`run_evaluation`。
- `src/quant_research_platform/artifacts/`：schema `0.6` writer / loader / display contract。
- `src/quant_research_platform/interfaces/api/`：本地 evidence API、CSV preview、direct download、ZIP download。
- `src/quant_research_platform/markets/implementations/a_share/`：CSMAR-backed A-share data、factor、validation、evaluation adapter。
- `src/quant_research_platform/pipelines/a_share/`：A 股 workflow runner、checkpoint 和 pipeline contracts。
- `web/`：本地 React/Vite evidence workbench。
- `docs/code-map/`：当前平台全局地图、A 股数据/pipeline 地图、evaluation/artifact/API/frontend 合同地图。

## 历史结果证据

- 第一批 factor set 是 `price_volume`：momentum、reversal、volatility、amount liquidity、turnover、Amihud illiquidity、amplitude。
- 默认窗口为 `20, 60, 120`；forward labels 为 `1, 5, 10, 20, 60, 120` 个交易日。
- 历史 2024-2025 long-only 第一版诊断中，反转类因子是主要候选。
- 历史 long-only 不变量曾检查通过：`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，0 成本下 `net_portfolio_return == gross_portfolio_return == long_return`。
- 当前 2023-2025 canonical sample 覆盖 `mom_20d`、`vol_20d`、`avg_amount_20d` 和 `rev_120d`；`rev_120d` parity shard 覆盖 dataframe/tensor 与 long-only/long-short。
- Evidence workbench 图表通过 bounded preview pagination 展示当前组合时间轴；CSV 明细仍保持 preview row limit，完整数据走下载。

## 下一阶段事项

- N1/N2/N3/N4/N7/N8 均已合入 `main`，后续扩展应复用 display preview/download helper、factor-template API、A-share service entry 和 legacy/dev upload 边界。
- 2023-2025 canonical evaluation、`rev_120d` parity shard 和 Factor Factory smoke 可作为后续展示与回归验收入口。
- long-only 成本敏感性、regime、行业/市值中性化、风险暴露和正式组合约束分析仍需单独研究。
- ML prediction pipeline 的 time split、防泄漏、model manifest 和失败实验记录仍需单独设计。
