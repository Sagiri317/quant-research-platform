# Future Capabilities / 后续能力池

本文是 `quant-research-platform` 的最终能力地图。它用于把后续研发、目录结构和规范文档对齐到“几年后功能完整的专业量化平台”，不是当前已实现功能清单，也不是部署、模拟盘、实盘或外部 API 审批记录。

当前可执行接口以 [README.md](../../README.md)、[ARCHITECTURE.md](../../ARCHITECTURE.md)、[EVALUATION_INTERFACE.md](../../EVALUATION_INTERFACE.md) 和 [docs/code-map/](../code-map/) 为准；目标目录和阶段边界以 [target-directory-structure.md](target-directory-structure.md) 为准；量化研究底线见 [quant-research-standards.md](quant-research-standards.md)，工程治理底线见 [engineering-standards.md](engineering-standards.md)。

## 1. 当前边界

当前平台定位：

```text
approved local data
-> market module
-> factor / label / prediction / risk inputs
-> EvaluationSpec
-> evaluation system
-> schema 0.5 artifacts
-> API / evidence workbench / downloads
```

当前已实现或已保留的核心能力：

- IC / RankIC。
- grouped return。
- long-only / long-short。
- transaction cost 基础口径。
- slippage / spread 字段级成本拆分。
- 因子中性化：group demean 和 cross-sectional OLS。
- DataFrame / Parquet / CSV input adapter。
- A 股 CSMAR-backed market module。
- schema `0.5` artifacts、`manifest.display`、API、下载和 React evidence workbench。
- 多市场接口骨架；crypto 当前只能显示为 planned / disabled，不保留空实现包。

硬边界：

- 不处理账号、密码、token、cookie、`.env` 或凭据。
- 不连接券商、交易所、vendor API、外部行情 API 或真实交易系统。
- 不自动下载 CSMAR，不自动登录网站。
- 不把研究诊断写成投资建议、真实收益承诺或可交易策略声明。
- 网站部署、认证、数据库、模拟盘和实盘必须单独设计并审批。

## 2. 能力分层

后续能力按三层管理，避免把长期平台愿景误读成当前实施范围。

```text
Research Platform Capabilities                       # 当前和近期：本地研究、评价、artifact、API、workbench
Web Platform Capabilities                            # 网站化前置：用户、租户、存储、任务、权限、审计、部署
Professional Quant Platform Capabilities             # 长期专业能力：回测、模型治理、交易、风控、对账、商业化
```

所有能力必须先判断归属：

- 研究能力优先落在 `research/`、`evaluations/`、`markets/`、`datasets/`、`artifacts/`、`pipelines/`。
- 网站能力优先落在 `apps/`、`storage/`、`identity/`、`tenancy/`、`access_control/`、`jobs/`、`workers/`、`audit/`、`observability/`。
- 专业量化平台能力优先落在 `strategies/`、`simulations/`、`feature_store/`、`model_registry/`、`trading/`、`execution/`、`controls/`、`reconciliation/`。

## 3. 数据和主数据能力

长期必须把“数据读取”和“市场事实”分开。

```text
datasets                                              # 数据集访问、版本、分区、缓存、schema 和读取器
reference_data                                        # instrument master、symbol mapping、公司行动、退市、合约链和指数成分
feeds                                                 # 历史回放、实时行情、盘口、新闻、基本面更新和链上数据流
feature_store                                         # point-in-time features、训练/推理一致性、offline/online 特征复用
data_quality                                          # 缺失、重复、延迟、异常、修订、vendor 差异和可用性检查
```

关键能力：

- 股票：复权、分红、拆股、配股、并购、退市、指数成分和行业分类。
- 期货：合约规格、上市、到期、交割、主力/连续合约、展期和保证金。
- 期权：期权链、行权价、到期日、欧式/美式/百慕大式、希腊字母和波动率曲面。
- 债券和利率：收益率曲线、久期、凸性、票息、赎回、信用评级和利差。
- 商品：期限结构、库存、仓单、现货升贴水、季节性、交割成本和品种规则。
- 外汇：spot、forward points、NDF、交割日、节假日、交叉汇率和 carry。
- 数字资产：spot、perp、funding、open interest、链上指标、DeFi 分类和交易所差异。

主数据能力不应写成普通行情表 helper。它影响研究、回测、交易、报告和对账，是独立平台域。

## 4. 市场和工具建模能力

`markets/` 负责描述“研究对象是什么”和“在哪里、以什么工具形态交易”，不是因子工厂。

```text
markets/common                                        # 时间、ID、货币、单位、bar、quote、trade、session 等公共对象
markets/asset_classes                                 # 股票、指数、利率、债券、信用、外汇、商品、数字资产、基金、波动率资产
markets/instruments                                   # cash、futures、options、forwards、swaps、perpetuals、convertibles
markets/venues                                        # 交易所、柜台、交易时段、撮合规则、涨跌停、停牌、保证金和结算规则
markets/data_sources                                  # vendor 字段到平台标准合同的映射
markets/implementations                               # 把 asset class、instrument、venue、data source 组合成可运行市场模块
```

原则：

- PE、PB、ROE、财务质量等字段是股票资产类别和数据源映射的一部分；作为因子使用时由 `research/factors/` 或 `research/factor_factory/` 生成定义。
- 5 日 return、20 日均线、成交量变化、波动率和高频盘口特征属于通用 factor primitive；具体字段可用性由 market implementation 提供。
- 期权、期货和其他衍生品不要按“市场”硬编码；它们是 instrument 轴，可以挂在股票、指数、利率、商品、外汇、数字资产等底层资产上。
- 同一个底层资产可以同时有 cash、futures、options、swaps、perpetuals 等工具形态。

## 5. 研究生产能力

`research/` 负责创造信号、预测、组合、风险输入和实验记录。它可以消费市场事实和数据集，也可以调用 evaluations 做批量验证，但不直接下单。

```text
research/factors                                      # 通用因子 primitives，例如量价、高频、microstructure、fundamental transforms
research/factor_factory                               # 候选生成、参数网格、批量评价、失败原因和候选汇总
research/models                                       # ML/DL 模型、label、feature set、walk-forward、prediction 产物
research/reinforcement_learning                       # RL 环境、state、action、reward、policy 和离线验证
research/portfolios                                   # 多因子组合、优化器、约束、risk parity、目标权重生成
research/risk                                         # 暴露、协方差、归因、tail risk、stress test 和 scenario analysis
research/regimes                                      # 牛熊、高低波、趋势/震荡、事件窗口、状态识别和切片诊断
research/experiments                                  # 参数搜索、实验配置、结果索引和实验对比
research/validation                                   # 泄露检查、样本切分、稳定性、多重检验和过拟合诊断
```

Factor Factory 第一阶段范围：

- 使用已批准的本地 processed panel。
- 候选表达式来自显式 registry 或受控参数网格。
- 不使用 Python `eval`，不执行任意用户代码。
- 每个候选保存公式、输入字段、窗口、样本范围、依赖字段和失败原因。
- 调用 evaluations 批量评价，输出 compact candidate summary 和 failed candidate log。
- 所有用户可见输出进入 `manifest.display`。

ML/DL/RL 的定位：

- 它们不是“更高级的 evaluation”，而是用特征、因子和 label 拟合预测、策略状态或组合决策。
- 模型输出必须成为 `prediction_*`、权重、风险输入或策略意图，再交给 evaluations 或 simulations 验证。
- 模型 manifest 必须记录 label、features、split、embargo、参数、训练日期、数据版本、代码版本、随机种子和失败候选。

## 6. 评价和回测能力

`evaluations/` 和 `simulations/` 必须分清。

```text
evaluations                                           # 给定 factor、prediction、weight 或 portfolio 后做统计评价和组合评价
simulations                                           # 事件驱动回测、撮合、成交、延迟、滑点、费用、market replay 和 paper exchange
benchmarks                                            # 指数、benchmark portfolio、peer group、随机基线和成本基线
portfolio_accounting                                  # 现金、持仓、成交、费用、分红、融资、PnL 和对账前会计
```

评价能力：

- IC / RankIC、grouped return、long-only、long-short、turnover、cost-aware PnL。
- prediction evaluation、portfolio evaluation、risk exposure diagnostics。
- 分行业、分市值、分流动性、分市场、分 regime、分时期稳定性。
- 多重检验、置换基线、walk-forward、purged CV、embargo。

回测能力：

- 向量化评价只适合快速诊断，不等价于可交易回测。
- 事件驱动回测必须显式建模订单、成交、部分成交、撤单、拒单、涨跌停、停牌、交易时段、撮合延迟和费用。
- 高频和 tick 需要 `IntradayEvaluationSpec`、`BacktestSpec` 或 `SimulationSpec`，不硬塞进当前日频 `EvaluationSpec`。

## 7. 成本、流动性、容量和风险能力

```text
research/risk                                         # 风险模型、协方差、归因、tail risk 和 scenario
research/portfolios                                   # 组合约束、优化、容量约束和目标权重
controls                                              # 长期交易风控：pre-trade、intraday、post-trade、kill switch 和限额
```

关键能力：

- commission、tax、spread、slippage、market impact、participation rate。
- borrow cost、short availability、margin、funding、capacity curve。
- 行业、市值、风格、国家、货币、期限、商品品种、波动率和流动性暴露。
- covariance、attribution、stress test、scenario analysis、tail risk。

容量结果必须写成诊断和约束，不写成真实可交易规模承诺。

## 8. 网站平台能力

网站化不是把当前本地 API 暴露出去，而是新增一层平台基础设施。

```text
apps/web                                             # 可部署前端应用
apps/api                                             # 可部署 HTTP API 应用
apps/workers                                         # 后台任务执行入口
apps/schedulers                                      # 定时任务和依赖触发入口
storage                                              # 数据库连接、migration、对象存储、缓存、搜索索引和事务边界
identity                                             # 用户、组织、登录、session、API key、角色和登录安全
tenancy                                              # workspace、project、资源隔离、配额和数据边界
access_control                                       # RBAC/ABAC、policy decision 和 policy enforcement
audit                                                # 访问、下载、运行、审批、交易和权限变化记录
jobs                                                 # pipeline、model、backtest、export 的状态、资源、日志和结果
workers                                              # CPU/GPU worker、队列消费、心跳、并发控制和资源隔离
schedulers                                           # 定时任务、依赖触发、数据到达触发和周期更新
observability                                        # 日志、指标、告警、health checks、数据延迟和任务状态
notifications                                        # 任务完成、失败、告警、审批、邮件、站内信和 webhook
deployments                                          # Docker、环境变量模板、服务拓扑、worker/scheduler 和运维说明
administration                                       # 用户、租户、任务、权限、数据源和异常处置后台
```

网站平台的核心规则：

- domain object 不因为要入库就归到 `storage/`；`storage/` 只负责持久化基础设施。
- 所有用户可见下载和预览仍必须通过 `manifest.display`。
- 多用户、多租户、权限、审计和数据隔离必须在任何对外服务前完成。
- `.env`、vendor key、broker key、OAuth secret 不进入 artifact、日志、聊天或普通报告。

## 9. 模型平台和治理能力

```text
feature_store                                         # 特征定义、point-in-time、训练/推理一致性和复用
model_registry                                        # 模型版本、lineage、stage、审批、部署状态和回滚
experiment_tracking                                   # 参数、metrics、数据版本、代码版本和 artifact 对比
approvals                                            # 报告发布、模型上线、数据接入、权限提升和实盘启用审批
compliance                                           # 数据许可、免责声明、发布边界、保留期限和投资建议边界
publishing                                           # 草稿、review、审批、公开链接、撤回和版本记录
knowledge                                            # 数据字典、市场规则、研究笔记、帮助文档和 RAG 资料
assistants                                           # 自然语言研究、报告解释、代码生成、agent runs 和工具审计
```

模型上线前必须有：

- 数据版本、代码版本、feature set、label 定义和训练窗口。
- OOS、walk-forward、稳定性、偏差检查和失败候选记录。
- 模型审批、回滚策略、漂移监控和运行审计。
- 明确说明模型输出是预测、权重、风险输入还是交易意图。

## 10. 用户产品和付费能力

长期网站如果要吸引用户付费，不能只暴露底层研究对象，还需要把量化能力包装成可理解、可持续使用的产品工作流。

```text
screening                                            # 选股和资产筛选：规则、公式、排名、分组、命中结果和筛选历史
fundamental_analysis                                 # 基本面分析：公司画像、财务质量、估值、成长、现金流和同行对比
ranking                                              # 评分体系：价值、质量、成长、动量、风险和综合 alpha score
watchlists                                           # 关注列表：用户自选、分组、跟踪状态、组合候选和手工标注
alerts                                               # 提醒：筛选命中、估值分位、财报发布、技术突破、风险阈值和组合变动
portfolio_analytics                                  # 持仓分析：暴露、归因、相关性、回撤、再平衡和情景压力测试
research_cards                                       # 人读卡片：公司卡片、因子卡片、策略卡片、组合卡片和可分享摘要
```

产品化原则：

- screening 消费 `markets`、`datasets`、`research/factors`、`feature_store` 和 `reference_data` 的结果，不直接绕过数据许可、point-in-time 和权限边界。
- PE、PB、ROE、现金流质量、估值分位等基本面字段属于 market / reference data / data source 映射；作为筛选条件或评分时进入 screening / ranking。
- 5 日 return、20 日均线、成交量变化、盘口特征等量价和高频条件属于 factor primitive；作为用户筛选条件时进入 screening。
- alerts 只能触发通知或待审核工作流，不能直接触发真实交易。
- research_cards 和报告必须区分事实、诊断、候选结论和投资建议边界。
- 付费产品能力必须经过 identity、tenancy、access_control、audit、compliance 和 billing 边界，不能只是本地 workbench 页面。

## 11. 交易、执行和对账能力

交易能力是长期平台域，不能由研究 pipeline 自然滑过去。

```text
strategies                                           # 可运行策略：universe、alpha、portfolio construction、risk policy、execution intent
trading                                              # 订单、账户、持仓、broker/exchange connector、paper/live 接口
execution                                            # 订单路由、拆单、成交质量、限速和执行策略
controls                                             # pre-trade、intraday、post-trade、kill switch 和限额
reconciliation                                       # broker statement、成交回报、持仓、现金、费用和系统记录核对
secrets                                              # broker key、vendor key、OAuth secret 和 webhook secret 管理
```

任何 paper/live trading 前必须完成：

- 订单生命周期、成交模型、费用模型、滑点模型和延迟模型。
- 持仓、现金、融资、费用、分红和 PnL 会计。
- pre-trade controls、权限、限额、kill switch 和审计。
- broker/exchange 回报与系统记录对账。
- 独立审批，不由研究 run 自动触发真实连接。

## 12. 商业化和运营能力

```text
billing                                              # 套餐、用量、额度、账单、支付和资源消耗
administration                                       # 平台运营后台、用户支持、任务干预和异常处理
notifications                                        # 邮件、站内信、webhook 和审批提醒
observability                                        # SLA、服务状态、任务延迟、数据延迟和错误追踪
compliance                                           # 许可、发布、免责声明、投资建议边界和审计保留
```

这些能力只有在平台准备对外使用时才落地。当前本地研究阶段只保留目录和边界认知，不实现。

## 13. 能力优先级

本节使用 `C0-C8` 表示能力池优先级，避免和 [architecture-refactor-design.md](architecture-refactor-design.md) 中当前重构实施阶段 `P0-P5` 混淆。

### C0: 合同和模块边界

- 拆清 `EvaluationRunPayload` 或等价中间结果合同。
- 把 evaluation 和 artifact persistence 解耦。
- 拆分 API route 边界。
- 完成 A 股 pipeline runner 设计。
- 把最终 repo-level `apps/` 与 Python package 边界写清楚。

### C1: Factor Factory

- 候选表达式 registry。
- 小参数网格。
- 批量评价。
- candidate summary。
- failed candidate log。
- workbench 浏览入口。

### C2: A 股 Pipeline Runner

- 配置驱动的 build/evaluate 编排。
- checkpoint 状态。
- 失败恢复和不覆盖规则。
- 小数据 smoke。

### C3: 风险、regime、成本和容量扩展

- exposure columns。
- risk / regime diagnostics。
- cost grid。
- liquidity / capacity diagnostics。
- benchmark 和 baseline 扩展。

### C4: ML Prediction

- label pipeline。
- walk-forward。
- model manifest。
- prediction Parquet。
- evaluations-backed prediction evaluation。

### C5: 多市场和多工具建模

- 美股、港股、A 股共同 equity contract。
- 指数、期货、期权、商品、外汇、利率、信用、数字资产的 asset / instrument 分轴建模。
- instrument master、symbol mapping、contract lifecycle 和 calendar 标准化。

### C6: 多频率、tick 和专业回测

- intraday / tick contract。
- order book、trade、quote、bar 的统一输入合同。
- event-driven backtest、market replay、fill model、latency 和 execution cost。

### C7: 网站平台基础

- apps/web、apps/api、apps/workers、apps/schedulers。
- storage、identity、tenancy、access_control、audit、jobs、workers、observability。
- 部署、认证、数据隔离和发布边界。

### C8: 实盘、治理和商业化

- paper/live trading。
- execution、controls、reconciliation、portfolio accounting。
- model registry、feature store、screening、watchlists、alerts、approvals、compliance、billing、assistants。

部署和实盘必须单独设计；它们会改变当前 local-only、安全、私有数据、API 暴露和认证边界。
