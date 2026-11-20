# 配置入口

配置分为当前研究入口和原生操作示例。示例中的一两只证券用于说明输入契约，
不代表当前研究只覆盖这些证券，也不会自动扩展到其他证券。
路径须对应实际已准备的数据；修改路径或窗口直接修改传入的配置，不复制证券专用版本。
文件名以实际 CLI 操作为前缀；参数片段和因子处理表除外。
相对路径从运行命令的目录解释。从仓库根运行股票或 ETF 因子组装时，把 `input_policy.policy` 改为 `configs/factor-analysis-policy.csv`；服务器从包含 `code/main/` 的数据目录运行时可直接用示例路径。

## 当前研究

| 配置 | 操作与范围 |
| --- | --- |
| [run-quarterly-a-share.json](run-quarterly-a-share.json) | `run-quarterly`：A 股四类预测模型，2022 年起按季度预测，训练和验证向后滚动 |
| [high-frequency/run-sessions-etf.json](high-frequency/run-sessions-etf.json) | 单只 ETF 的四种预测模型。每折只用当时已成熟的历史选特征和参数。 |
| [high-frequency/backtest-policy-etf.json](high-frequency/backtest-policy-etf.json) | 单只 ETF、一个预测来源、一个账户。示例使用 100 万元、初始半仓，14:50 起恢复最初的份额。 |
| [backtest-stock-weekly.json](backtest-stock-weekly.json) | 单个股票模型的周频回测，配置单股上限和方差惩罚。 |
| [factor-analysis-policy.csv](factor-analysis-policy.csv) | 按研究粒度和因子记录处理方法与研究用途，组装配置选择使用的行及表示。 |

ETF 示例使用 518880。510300、510500 的盘口历史从 2022 年起，159915、518880 从 2023 年 9 月起。换证券时按该证券实际滚动产生的会话块填写预测路径。

ETF 底仓单独运行同一账户的 `hold` 策略，使用新的输出路径。将两个输出目录各自的 `ledger/` 交给 `analyze-inventory` 比较。股票配置的 `risk_aversion=0` 优先成本后收益，正值惩罚持有期方差。收益优先账户的独立调用见[模型与账户设置](../docs/low-frequency-models.md#股票组合和-etf-底仓)。两种账户均保留 2% 的单股上限，可以留现金。

## 数据与因子示例

| 文件 | 用途 |
| --- | --- |
| `prepare-source.json`、`prepare-source-hft.json` | 分别读取结构化日频和逐笔来源；不下载或整理 raw |
| `prepare-financial.json` | 结构化财务事实的 PIT 计算接口；当前研究不启用 Choice 财务或分析师预期 |
| [prepare-cash-distributions.json](prepare-cash-distributions.json) | 从实际现金分红和除息前价格准备股息收益，持续分红交互还需要决策面板的 `volatility_60d`。 |
| [prepare-calendar-seasonality.json](prepare-calendar-seasonality.json)、[prepare-size-volatility-interaction.json](prepare-size-volatility-interaction.json) | 用实际股票行情、交易日历及收盘市值和残差波动计算两个收盘信号。 |
| [assemble-stock-close-signals.json](assemble-stock-close-signals.json)、[prepare-stock-signal-decisions.json](prepare-stock-signal-decisions.json) | 收盘观测与下一会话开盘前的决策时点。这里的输入窗口按信号日填写，首尾分别取目标决策区间之前的一个实际交易日。 |
| [assemble-stock-model-inputs.json](assemble-stock-model-inputs.json) | 合并同一决策时点的因子，选用每项因子的一种表示，形成四个模型的共同候选输入。 |
| [prepare-quote-trade-pressure.json](high-frequency/prepare-quote-trade-pressure.json)、[assemble-etf-signals.json](high-frequency/assemble-etf-signals.json)、[sample-etf-model-inputs.json](high-frequency/sample-etf-model-inputs.json) | 单只 ETF 的报价因子、逐报价组装与 3 秒决策时钟采样。各操作均可单独调用。 |
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
