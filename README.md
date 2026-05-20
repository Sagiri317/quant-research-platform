# a-share-alpha-lab

本项目是本地私有 CSMAR A 股因子研究项目。目标是把用户手工导出的 CSMAR 数据转换为可复现的研究数据集、因子表，并通过 `alpha-research-core` 跑第一版因子评估。

## 边界

- Codex 不登录 CSMAR，不自动下载数据，不接触账号、密码、cookie、token 或 `.env`。
- `data/raw_private/`、`data/interim_private/`、`data/processed_private/`、`reports/private_real/` 不提交、不公开。
- 代码、方法说明和脱敏后的聚合结果未来可以公开。
- 本项目不是投资建议、实盘交易、paper trading 或生产交易系统。

## 当前结构

```text
code/
  csmar/        CSMAR raw -> interim，字段合同、列裁剪、原始导出检查
  datasets/     interim -> research panel
  factors/      panel -> factor table + forward labels
  evaluation/   调用 alpha-research-core
  validation/   检查 interim / panel / factor 产物
  runtime/      空间估算和运行安全工具

data/
  raw_private/csmar/          CSMAR 原始导出，完整保留
  interim_private/csmar/      规范化并裁剪后的 interim CSV
  processed_private/csmar/    总体 panel 和 factor Parquet

reports/private_real/
  factor_evaluation/          私有真实数据因子评估结果；每个分析口径一个子目录
```

## 运行

基础验证：

```powershell
conda run -n analysis python -m pytest
conda run -n analysis python -m ruff check .
```

从 raw 重建 2014-2025 数据，并只对 2024-2025 跑因子评估：

```powershell
conda run -n analysis python code/csmar/inspect_raw_exports.py data/raw_private/csmar
conda run -n analysis python code/csmar/build_interim.py --start-date 2014-01-01 --end-date 2025-12-31
conda run -n analysis python code/csmar/inspect_interim.py --start-year 2014 --end-year 2025

conda run -n analysis python code/datasets/build_panel.py --start-year 2014 --end-year 2025
conda run -n analysis python code/datasets/inspect_panel.py --start-year 2014 --end-year 2025

conda run -n analysis python code/factors/build_factors.py --start-year 2014 --end-year 2025 --factor-set price_volume
conda run -n analysis python code/factors/inspect_factors.py --start-year 2014 --end-year 2025

conda run -n analysis python code/runtime/check_processed_space.py --target-start-year 2014 --target-end-year 2025

conda run -n analysis python code/evaluation/run_factor_evaluation.py --start-year 2024 --end-year 2025 --labels forward_return_20d --factor-set price_volume --all-factors --n-groups 5 --pnl-return-column forward_return_1d --portfolio-mode long_only
conda run -n analysis python code/evaluation/build_factor_summary.py
```

默认 evaluation 输出到：

```text
reports/private_real/factor_evaluation/full_2024_2025/
```

当前 A 股主报告口径是一次性 full run 加 core 年度 breakdown，并显式使用 `--portfolio-mode long_only`。core 不在年底清仓，只把连续回测产生的日收益按年切分，并在年度内重算 IC、RankIC、分组收益、PnL、Sharpe、drawdown 和 turnover。若要做“每年空仓重启”的诊断，应显式指定独立 `report-root` 另跑 yearly independent evaluation，不作为主报告口径。

可选 core 参数已透出到 CLI：

```powershell
conda run -n analysis python code/evaluation/run_factor_evaluation.py --start-year 2024 --end-year 2025 --labels forward_return_20d --factor-set price_volume --all-factors --n-groups 5 --pnl-return-column forward_return_1d --portfolio-mode long_only --cost-bps 2.5 --engine dataframe
```

支持的可选参数包括 `--pnl-return-column`、`--portfolio-mode`、`--cost-bps`、`--engine`、`--segment-columns`、`--risk-column`、`--neutralization-method`、`--neutralization-group-columns` 和 `--neutralization-exposure-columns`。`--portfolio-mode` 的 CLI 默认值仍跟随 core 为 `long_short`；A 股主报告需要显式传 `long_only`。日期过滤在本项目内存中完成，不生成额外 `core_inputs/` 或 subperiod Parquet。

总体 processed 输出位置：

```text
data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet
data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet
```

## 阅读顺序

1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
2. [docs/code-map/README.md](docs/code-map/README.md)
3. [docs/csmar_data_manual.md](docs/csmar_data_manual.md)
4. [RESEARCH_CARD.md](RESEARCH_CARD.md)
5. [REPORT.md](REPORT.md)
