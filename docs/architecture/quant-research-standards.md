# Quant Research Standards / 量化研究规范

本文定义 `quant-research-platform` 的专业量化研究底线。它回答如何避免穿越、偏差、过拟合、错误归因和不可复现结果；工程边界和测试规则见 [engineering-standards.md](engineering-standards.md)。

## 1. 时间语义

所有研究数据和评价结果必须明确时间含义。

```text
event_time                                            # 事件发生时间，例如成交、公告、财报期末或新闻发布时间
effective_time                                        # 数据或规则生效时间，例如指数成分生效日
as_of_time                                            # 当时可知时间，决定研究是否穿越
ingest_time                                           # 系统摄入时间，用于数据延迟和审计
decision_time                                         # 策略或评价假设的决策时间
label_time                                            # label 对应的未来窗口时间
bar_start_time                                        # bar 起始时间
bar_end_time                                          # bar 结束时间
session_date                                          # 交易 session 日期
timezone                                              # 时区
frequency                                             # 频率，例如 1d、1h、1m、tick
```

财务、成分、评级、宏观、新闻、链上指标和 vendor 修订数据必须有 point-in-time 或 as-of 规则。不能用未来才知道的数据构造历史信号。

## 2. Universe / Eligibility / Tradability

股票池和可交易性必须拆成三层。

```text
universe                                              # 理论研究范围，例如全 A 股、沪深 300、某交易所全部期货合约
eligibility                                           # 样本是否可纳入研究，例如上市天数、数据完整性、行业、流动性过滤
tradability                                           # 当天是否可交易，例如停牌、涨跌停、成交额、borrow、保证金和交易限制
```

评价时要记录每层过滤结果，避免把不可交易样本纳入收益计算。

## 3. 主数据和生命周期

不同资产有不同生命周期，不能作为普通字段随意处理。

```text
corporate_actions                                     # 股票分红、拆股、配股、复权、并购和退市
contract_lifecycle                                    # 期货、期权、远期和互换的上市、到期、交割、行权和展期
symbol_mapping                                        # ticker、symbol、stock_code、secid、contract code 的历史映射
index_membership                                      # 指数成分、权重、再平衡和生效时间
delisting                                             # 退市收益、退市原因和样本退出处理
restatement                                           # 财务重述和历史数据修订
```

主数据影响所有研究、回测、实盘和报告，应独立于普通行情表管理。

## 4. 数据偏差控制

研究必须显式检查以下偏差：

```text
lookahead_bias                                        # 未来函数或未来可知数据
survivorship_bias                                     # 只保留幸存资产，遗漏退市或失效资产
selection_bias                                        # 事后选择样本、行业、市场或时间段
data_snooping                                         # 多次试验后只报告最优结果
restatement_bias                                      # 使用重述后的历史财务数据
liquidity_bias                                        # 忽略流动性、成交限制或不可交易样本
corporate_action_bias                                 # 复权、分红、拆股或退市处理不一致
```

重要研究输出必须记录使用的数据版本、字段来源、过滤规则和偏差控制状态。

## 5. 因子和预测

因子、预测和权重必须记录来源。

```text
factor_definition                                     # 因子公式、字段、窗口、变换和版本
factor_availability                                   # 因子在 decision_time 是否可知
prediction_source                                     # 预测来自哪个模型、版本、样本和特征集
weight_source                                         # 权重来自规则、优化器、模型还是人工配置
dependency_fields                                     # 输入字段和依赖数据源
sample_range                                          # 样本范围和排除规则
failure_reason                                        # 候选失败原因，例如缺字段、样本不足或无法计算
```

Factor Factory 和 ML/DL/RL 产物必须能回答：这个信号怎么来、用了哪些字段、什么时候可知、为什么失败或保留。

## 6. 统计验证

专业研究不能只看单次回测结果。

```text
out_of_sample                                         # 样本外验证
walk_forward                                          # 滚动训练和验证
purged_cv                                             # 清除重叠 label 的交叉验证
embargo                                               # 防止相邻样本泄露的时间隔离
multiple_testing                                      # 多重检验和 false discovery 控制
parameter_stability                                   # 参数稳定性和敏感性
subperiod_stability                                   # 分时期稳定性
cross_section_stability                               # 分行业、分市值、分流动性稳定性
permutation_baseline                                  # 随机或置换基线
cost_adjusted_performance                             # 扣成本表现
turnover_adjusted_performance                         # 换手调整表现
capacity_adjusted_performance                         # 容量调整表现
```

研究结论必须区分诊断、候选信号、可复现证据和可交易策略。不得把初步诊断写成真实收益承诺。

## 7. 成本、流动性和容量

成本不能长期停留在固定 bps。

```text
commission                                            # 佣金和交易费
tax                                                   # 印花税、交易税或市场税费
bid_ask_spread                                        # 买卖价差
slippage                                              # 滑点
market_impact                                         # 市场冲击
participation_rate                                    # 参与率
borrow_cost                                           # 融券或借券成本
short_availability                                    # 可借券数量和做空限制
margin                                                # 保证金和杠杆要求
funding                                               # 融资、永续 funding 或资金成本
capacity_curve                                        # 策略容量曲线
```

容量结果必须写成诊断和约束，不写成真实可交易规模承诺。

## 8. 基准和对照

每个评价应有明确基准。

```text
market_benchmark                                      # 市场基准，例如沪深 300、S&P 500、BTC index
sector_neutral_benchmark                              # 行业中性或风险中性基准
equal_weight_baseline                                 # 等权基线
random_baseline                                       # 随机或置换基线
cost_baseline                                         # 只考虑成本后的基线
cash_or_risk_free_baseline                            # 现金或无风险利率基线
```

没有基准的表现只能作为描述性诊断，不应作为 alpha 结论。

## 9. 风险、场景和 Regime

研究应能解释风险来源，而不只报告收益。

```text
exposure                                              # 行业、市值、风格、国家、货币、期限、商品品种等暴露
covariance                                            # 协方差和风险矩阵
attribution                                           # 收益和风险归因
stress_test                                           # 压力测试
scenario_analysis                                     # 情景分析，例如利率冲击、商品冲击、汇率冲击
regime                                                # 牛熊、高低波、趋势/震荡、危机期和事件窗口
tail_risk                                             # 尾部风险、极端回撤和流动性冲击
```

Regime 必须在当时可知，不能用未来收益定义。

## 10. 复现和 Lineage

每个重要 run 必须能复现或解释为什么不能复现。

```text
data_version                                          # 数据版本
code_version                                          # 代码版本或 Git commit
config_version                                        # 配置版本
factor_definition_version                             # 因子定义版本
model_version                                         # 模型版本
artifact_schema_version                               # artifact schema 版本
environment_summary                                   # Python、Conda、关键依赖和运行平台摘要
random_seed                                           # 随机种子
run_timestamp                                         # run 创建时间
lineage                                               # 输入、输出和中间产物关系
```

不可确定的训练、采样、并行计算或外部数据状态必须显式标记。

## 11. 跨资产和衍生品规范

不同资产和工具不能只套用股票日频因子评价口径。

```text
equity_fundamentals                                  # PE、PB、ROE、利润质量、成长、估值分位和财报可知时间
futures_roll                                         # 主力切换、连续合约、展期收益、交割月和保证金
options_surface                                      # IV surface、skew、term structure、Greeks、exercise style 和 dividend assumption
rates_curve                                          # 收益率曲线、久期、凸性、DV01、roll-down 和 carry
credit_spread                                        # 信用利差、评级迁移、违约、回收率和流动性折价
fx_carry                                             # 利差、远期点、交割日、NDF、节假日和跨货币换算
commodity_curve                                      # 期限结构、库存、仓单、现货升贴水、季节性和交割成本
digital_asset_microstructure                         # funding、open interest、perp basis、链上指标、交易所差异和 24/7 session
```

规则：

- 股票估值和质量因子必须记录财报 `as_of_time`，不能用重述后或未来披露数据。
- 期货研究必须显式说明连续合约构造、展期规则、到期处理和交易合约选择。
- 期权研究必须区分欧式、美式、百慕大式，记录定价假设、希腊字母、波动率曲面和行权规则。
- 商品研究必须处理品种、交割、仓储、库存、季节性和现货/期货基差。
- 数字资产研究必须处理 24/7 session、交易所差异、永续 funding、强平、链上数据延迟和盘口深度。

## 12. 回测和仿真规范

向量化评价、组合回测和事件驱动仿真必须分层。

```text
factor_evaluation                                    # 给定因子或预测后评价 IC、RankIC、分组收益和稳定性
portfolio_evaluation                                 # 给定权重或组合后评价收益、换手、成本、暴露和风险
vectorized_backtest                                  # 快速向量化策略诊断，适合日频和低交互策略
event_driven_simulation                              # 事件驱动仿真，处理订单、成交、撮合、延迟、部分成交和市场状态
market_replay                                        # 按历史时间顺序回放 quote、trade、bar、news、fundamental updates
paper_exchange                                       # 模拟交易所或模拟券商，承载订单生命周期和成交回报
```

规则：

- factor evaluation 不是完整策略回测；portfolio evaluation 也不自动证明可交易。
- 高频、盘口、期权和多资产交易必须优先设计 event-driven simulation。
- 回测必须记录 fill model、slippage model、fee model、latency model、market impact model 和 rejected order 规则。
- 停牌、涨跌停、无成交、borrow 不可得、保证金不足和交易时段外订单必须显式处理。
- 回测结果必须能区分信号贡献、组合构建贡献、执行假设贡献和成本假设贡献。

## 13. 模型风险和策略生命周期

ML/DL/RL 研究必须经过模型风险治理，不得直接从训练结果进入实盘。

```text
idea                                                  # 研究想法，未验证
candidate                                             # 候选信号或模型，通过基本数据和泄露检查
validated                                             # 通过样本外、稳定性、成本、容量和风险诊断
paper                                                 # 纸面或模拟环境验证，记录 live-like 运行表现
limited_live                                          # 小规模实盘或受限环境，受风控和审批约束
production                                            # 生产策略，具备监控、回滚、对账、审计和负责人
retired                                               # 退役策略或模型，保留版本和退役原因
```

模型治理要求：

- 模型必须有 owner、purpose、feature set、label、training window、validation window、数据版本和代码版本。
- OOS、walk-forward、purged CV、embargo、drift、turnover、capacity 和 stress test 必须记录。
- 模型上线需要审批、回滚策略、监控指标和失效条件。
- RL policy 必须先在明确的 environment、reward、action space、transaction cost 和 risk constraint 下验证。
- LLM 或 agent 输出只能作为研究辅助或候选生成，不能绕过模型、风控、审批和审计边界。

## 14. 实盘前底线

任何 paper/live trading 前必须先满足：

```text
order_lifecycle                                       # 订单生命周期清晰，包括创建、提交、成交、撤单、拒单和过期
fill_model                                            # 成交模型和滑点模型有明确假设
position_accounting                                   # 持仓、现金、费用、分红、融资和 PnL 会计清楚
reconciliation                                        # broker statement、成交、持仓、现金和系统记录可对账
pre_trade_controls                                    # 交易前风控，包括限额、权限、可交易性和 kill switch
audit_trail                                           # 谁触发交易、参数是什么、结果是什么都可审计
secrets_boundary                                      # broker key、vendor key 和 token 不进入普通日志、artifact 或聊天
```

研究 pipeline 不得自然滑向真实下单。真实交易必须经过独立设计、审批、权限、风控和审计。
