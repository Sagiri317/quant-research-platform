# 配置入口

配置分为当前研究入口和原生操作示例。示例中的一两只证券用于说明输入契约，
不代表当前研究只覆盖这些证券，也不会自动扩展到其他证券。
路径须对应实际已准备的数据；修改路径或窗口直接修改传入的配置，不复制证券专用版本。
文件名以实际 CLI 操作为前缀；参数片段和因子处理表除外。

## 当前研究

| 配置 | 操作与范围 |
| --- | --- |
| [run-quarterly-a-share.json](run-quarterly-a-share.json) | `run-quarterly`：A 股四类预测模型，2022 年起按季度预测，训练和验证向后滚动 |
| [high-frequency/run-sessions-etf.json](high-frequency/run-sessions-etf.json) | `run-sessions`：ETF 四类预测模型的共同参数。输入路径以 518880 举例，调用者替换真实证券输入和可用起点；137 列特征的顺序保留原协议 |
| [high-frequency/run-etf-execution.json](high-frequency/run-etf-execution.json) | `run-etf-execution --stage calibrate/direct/train/ppo`：明确列出 510300、510500、159915、518880，各自独立 100 万元账户，直接策略和 PPO 共用规则，2026 年回放；历史幅度校准默认关闭 |
| [factor-analysis-policy.csv](factor-analysis-policy.csv) | 因子检验的去极值、标准化、中性化等处理政策；按分析粒度与因子列区分，不是模型输入清单 |

ETF 执行入口默认依次运行四只证券，`--asset-id` 可限定其中一只；跨证券并行由调用者安排。
同一证券的直接策略、训练和冻结回放分别保存到 `direct`、`policy`、`ppo`。
可选 `prediction_calibration` 开启后，先以 `calibrate` 准备共用日参数；用新输出目录重跑直接策略并重新训练 PPO，配置和缺失语义见[历史幅度校准](../docs/research-learning.md#etf-历史预测幅度校准)。
PPO 使用 2025 年前三季度训练、第四季度验证；训练插补及观察统计不从 2026 年估计。
金额上限、初始半仓、费用、参与率及恢复时间只在该配置维护一份。
完整规则和术语见[策略学习](../docs/research-learning.md)与[高频账户](../docs/a-share-hft.md)。

## 数据与因子示例

| 文件 | 用途 |
| --- | --- |
| `prepare-source.json`、`prepare-source-hft.json` | 分别读取结构化日频和逐笔来源；不下载或整理 raw |
| `prepare-financial.json` | 结构化财务事实的 PIT 计算接口；当前研究不启用 Choice 财务或分析师预期 |
| `prepare-on-id.json` | 隔夜与日内收益 |
| `prepare-hft-panel.json`、`prepare-hft-factors.json`、`prepare-hft-labels.json` | 共同高频面板、盘口因子、收益标签，各阶段独立 |
| `assemble-selected.json`、`assemble-hft.json` | 已有列的拼接与高频研究数据集组装 |
| `high-frequency/sample-hft-common.json` | 共同决策时钟采样 |
| `high-frequency/prepare-event-factors.json`、`prepare-hawkes-factors.json`、`fit-hawkes.json` | 逐笔事件因子、Hawkes 冻结应用与拟合 |
| `high-frequency/assemble-deeplob.json` | DeepLOB 盘口序列输入，契约不同于普通因子表 |
| `macro-liquidity/prepare-source.json`、`prepare-decision-panel.json`、`prepare-factors.json`、`prepare-monthly-beta.json`、`assemble.json` | 宏观事实、决策时点、宏观因子、月频更新所需长窗 Beta、组装 |
| `linked-markets/prepare-sector-proxies.json` | 显式行业代理映射与跨市场因子 |
| `linked-markets/index-group.json` | 参数片段：指数组映射，不能直接作为 CLI 完整配置运行 |

## 模型与评价示例

| 文件 | 用途 |
| --- | --- |
| `fit-a-share.json`、`fit-hft.json` | 单次拟合接口；当前完整输入以滚动研究配置为准 |
| `high-frequency/fit-deeplob.json`、`fit-xgboost-early-stopping.json` | 不同模型输入契约与早停训练示例 |
| `predict.json`、`explain-model.json` | 冻结模型预测与训练域背景解释 |
| `evaluate-hft.json`、`high-frequency/evaluate-etf.json` | 截面评价与单证券时序评价，统计轴不同 |
| `evaluate-stratified-a-share.json`、`statistics-factors-stratified-a-share.json` | 共同样本分层评价与已保存日表汇总 |
| `evaluate-model-periods.json` | 滚动期训练、验证和预测指标汇总 |
| `statistics-portfolio-hft.json`、`statistics-portfolio-benchmark-a-share.json` | 高频账本绩效与含基准比较的组合统计 |
| `plot-risk.json` | 已保存组合风险结果的图表 |

## 账户与研究方法示例

| 文件 | 用途 |
| --- | --- |
| `backtest-robust.json` | 低频稳健组合的接口示例，保留显式历史校准输入；不是当前 2022–2026 年任务。该任务从首日运行需要此前真实样本外预测做收益校准，不能只改日期或用后续收益补齐 |
| `backtest-hft.json`、`high-frequency/backtest-hft-passive.json`、`backtest-policy-ac.json` | 信号阈值、被动排队与 AC 执行进度，研究假设不同 |
| `research-learning/train-policy-execution.json`、`backtest-policy-execution.json` | 固定母单执行，目标由调用者给定 |
| `research-learning/train-policy-inventory.json`、`backtest-policy-etf-inventory.json` | 学习库存目标及透明库存策略的原生接口；不同于当前 PPO 执行时机任务 |
| `research-learning/train-policy-portfolio.json`、`backtest-policy-portfolio.json`、`backtest-policy-stock-portfolio.json` | 日频学习组合与股票高频全市场组合；多证券共用账户，与四只 ETF 独立账户不同 |
| `research-learning/search-factors.json`、`apply-formula.json` | 公式搜索与冻结公式应用 |
| `research-learning/calibrate-uncertainty.json`、`apply-uncertainty.json`、`evaluate-uncertainty.json` | 不确定性校准、冻结应用及评价 |
| `research-learning/fit-cost-model.json`、`backtest-cost-model.json` | 可微成本感知模型及冻结回放，属于模型研究接口 |
| `research-learning/agent.json`、`agent-report-review.json` | 通用研究 Agent 默认设置与只读报告复核；复核样例不再充当聊天默认配置 |

以上独立示例保留其科学差异。历史报告中的参数属于当时实验事实，不随当前配置改写。
环境与打包配置仍分别由 `environment.yml`、`pyproject.toml` 和 `Dockerfile.research` 负责；
配置整理不改变依赖，也不要求重算数值未变的模型和因子。
