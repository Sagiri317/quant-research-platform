# Target Directory Structure / 目标目录结构

本文记录 `quant-research-platform` 的目标目录结构。它是架构蓝图，不表示所有目录都要立即创建；当前重构实施细节见 [architecture-refactor-design.md](architecture-refactor-design.md)。

收敛原则：

- 当前阶段只落地研究平台重构所需目录，对应 [architecture-refactor-design.md](architecture-refactor-design.md) 的实施范围。
- 最终网站形态采用 repo-level `apps/` 承载可部署应用，Python package 只承载可复用平台领域代码。
- 网站化前再落地用户、租户、存储、任务、权限、审计等平台域。
- 实盘、专业回测、模型治理和商业化能力作为长期平台域预留。
- 目录表示领域或对象集合；文件表示具体对象、合同、服务或算法。
- 函数名使用动词短语；类名使用名词；常量使用 `UPPER_SNAKE_CASE`。

## 1. 分层目标

```text
Phase 1: Research Platform Refactor                 # 当前架构重构，只服务本地研究、评价、artifact、API 和 workbench
Phase 2: Web Platform Foundation                    # 网站化前必须补齐用户、租户、存储、任务、权限、审计和部署基础
Future: Professional Quant Platform                 # 几年期专业平台能力，包括事件回测、实盘、模型治理、特征库和商业化
```

最终 repo-level 形态：

```text
apps/                                                # 可部署应用入口集合，不承载核心业务规则
  web/                                               # 网站前端应用，消费 API 和 artifact/display 合同
  api/                                               # HTTP API 应用，装配 routes、middleware、auth、依赖和服务入口
  workers/                                           # 后台任务执行入口，消费 jobs 队列并调用平台领域服务
  schedulers/                                        # 定时任务、数据到达触发和周期性 pipeline 入口

src/quant_research_platform/                         # 可复用平台领域代码，承载研究、评价、市场、数据、产物、流水线和平台域
```

本轮已批准的 Phase 1 目标是：本地 FastAPI 适配迁到 `src/quant_research_platform/interfaces/api/`，CLI 迁到 `src/quant_research_platform/interfaces/cli/`，通用 reader 迁到 `src/quant_research_platform/datasets/`，A 股市场实现迁到 `src/quant_research_platform/markets/implementations/a_share/`，A 股 workflow runner 迁到 `src/quant_research_platform/pipelines/a_share/`，并且不创建 repo-level `apps/api`。`interfaces/` 在 Python package 内只保留 SDK、notebook entrypoints、API 合同适配和本地装配薄层，不作为最终可部署应用目录，也不承载完整前端应用；真正可部署的 HTTP 应用入口属于 Phase 2 的 `apps/api`。

## 2. Phase 1: Research Platform Refactor

当前阶段的目标是把现有 A 股研究平台重构成清晰的本地研究系统，不引入真实认证、数据库、部署、外部数据 API、券商连接或实盘能力。

```text
web/                                                 # 当前本地 React evidence workbench；Phase 2 前迁移或重组为 apps/web

src/quant_research_platform/                         # 平台 Python 主包，当前阶段承载本地研究、评价、市场建模、产物和接口适配
  evaluations/                                       # 评价系统：给定因子、预测、权重或组合后，计算 IC、RankIC、收益、换手、成本和风险评价
  kernels/                                           # 数值计算内核：DataFrame、tensor、未来 GPU/分布式实现，供 evaluations 调用
  research/                                          # 研究系统：因子 primitives、Factor Factory、模型、组合、风险、regime、实验和验证
  markets/                                           # 市场建模：资产类别、合约工具、交易场所、数据源、字段映射、日历和市场规则
  datasets/                                          # 数据集访问：本地 CSV/Parquet、vendor dump、数据版本、读取器、缓存和数据索引
  artifacts/                                         # 产物合同：schema、manifest.display、writer、loader、报告、下载包和展示数据
  pipelines/                                         # 本地流水线：数据构建、因子生成、模型训练、评价、artifact 写出、checkpoint 和重试
  runtime/                                           # 本地运行时：路径、run roots、缓存、profile、环境探测和执行上下文
  interfaces/                                        # 本地交互入口适配：API、CLI、SDK 和 notebook entrypoints；最终可部署应用迁到 repo-level apps/
```

### 2.1 Evaluations / 评价系统

```text
evaluations/                                         # 统一评价系统，接收已给定的因子、预测、权重或组合并输出评价结果
  evaluation_spec.py                                 # 评价输入合同，例如 EvaluationSpec、WeightConfig 和未来 BacktestSpec
  evaluation_runner.py                               # 评价主入口，例如 run_evaluation 和 evaluate_to_payload
  evaluation_payload.py                              # 模块间传递的内存结果包，例如 EvaluationRunPayload
  evaluation_result.py                               # 对外返回对象，例如 EvaluationSetResult
  evaluation_metrics.py                              # 通用评价指标，例如 IC、RankIC、收益、波动、回撤、Sharpe 和成本指标
  portfolio_evaluation.py                            # 给定权重后的组合收益、持仓、换手和成本评价
  panel_validation.py                                # panel 索引、字段、缺失值、频率和样本完整性校验
  neutralization.py                                  # 中性化逻辑，例如 group demean、OLS 和未来风险暴露中性化
  period_breakdowns.py                               # 按年份、月份、regime 或事件窗口切分评价结果
  evaluation_report.py                               # 评价报告文本和解释层，不负责 artifact 写盘
```

### 2.2 Kernels / 数值计算内核

```text
kernels/                                             # 底层数值计算内核，供 evaluations 调用，不承载研究语义
  kernel_protocol.py                                 # 计算内核协议，定义评价层调用后端实现的统一接口
  kernel_registry.py                                 # 计算内核注册表，管理 dataframe、tensor、GPU 和分布式实现
  dataframe/                                         # 表格计算内核，适合 pandas/polars 风格的中小规模计算
  tensor/                                            # 张量计算内核，适合 numpy/torch 风格的大规模矩阵化计算
  gpu/                                               # GPU 计算内核预留，未来承载 torch、cupy 等加速实现
  distributed/                                       # 分布式计算内核预留，未来承载 ray、dask、spark 等实现
```

### 2.3 Research / 研究系统

```text
research/                                            # 研究生产层，负责创造信号、模型、组合、风险诊断和实验记录
  factors/                                           # 通用因子 primitives，例如量价、高频、microstructure 和 transforms
  factor_factory/                                    # 因子工厂，负责候选生成、参数网格、批量评价、失败记录和候选汇总
  models/                                            # ML/DL 模型研究，负责特征、label、训练、walk-forward 和 prediction 产物
  reinforcement_learning/                            # 强化学习研究，长期预留，隔离 environment、state、action、reward 和 policy
  portfolios/                                        # 组合研究，负责优化器、约束、多因子组合、risk parity 和目标权重生成
  risk/                                              # 风险研究，负责暴露、协方差、归因、tail risk 和 stress test
  regimes/                                           # 市场状态研究，负责牛熊、高低波、事件窗口、状态识别和切片诊断
  experiments/                                       # 实验管理，负责参数搜索、实验配置、结果索引和实验对比
  validation/                                        # 研究验证，负责泄露检查、样本切分、稳定性和过拟合诊断
```

### 2.4 Markets / 市场建模系统

```text
markets/                                             # 市场与资产事实层，负责资产类别、合约、交易场所、数据源和具体市场实现
  common/                                            # 跨市场公共对象，例如时间戳、资产 ID、bar、quote、trade、货币、单位和标识符
  asset_classes/                                     # 资产类别公共逻辑，描述底层风险是什么
    equities/                                        # 股票公共逻辑，例如估值、质量、成长、行业分类、公司行为和市值字段
    equity_indices/                                  # 股票指数公共逻辑，例如成分、权重、再平衡和指数衍生品底层合同
    rates/                                           # 利率公共逻辑，例如收益率曲线、久期、DV01、SOFR 和利率衍生品输入
    fixed_income/                                    # 现金债公共逻辑，例如国债、信用债、市政债、MBS、ABS、repo 和债券条款
    credit/                                          # 信用公共逻辑，例如信用利差、评级、违约、CDS、信用指数和信用事件
    fx/                                              # 外汇公共逻辑，例如货币对、spot、forward points、NDF、carry 和 cross rates
    commodities/                                     # 商品公共逻辑，例如期限结构、库存、仓单、现货升贴水、季节性和交割成本
    digital_assets/                                  # 数字资产公共逻辑，例如 spot、perp、funding、open interest、链上数据和 DeFi 分类
    funds_etps/                                      # 基金和交易所产品，例如 ETF、ETN、NAV、折溢价和申赎机制
    volatility/                                      # 波动率资产，例如 VIX、variance swap、vol surface 和波动率期货/期权
  instruments/                                       # 金融工具和合约形态，描述风险如何被交易或结算
    cash/                                            # 现货工具，例如股票现货、债券现券、现货外汇、现货商品和 crypto spot
    futures/                                         # 期货工具，例如合约规格、到期日、交割月、连续合约、展期和保证金
    options/                                         # 期权工具，例如 call/put、strike、expiry、美式/欧式/百慕大和 IV surface
    forwards/                                        # 远期工具，例如 FX forward、商品远期、NDF、交割和现金结算规则
    swaps/                                           # 互换工具，例如 IRS、CDS、TRS、commodity swap、cashflow legs 和 reset
    perpetuals/                                      # 永续合约，例如 crypto perp、funding、mark price、index price 和强平相关字段
    convertibles/                                    # 可转债和可交换债，例如转股、赎回、回售、债底和转股溢价
    structured_products/                             # 结构化产品，例如雪球、autocall、barrier、参与率和路径依赖 payoff
  venues/                                            # 交易场所规则，描述在哪里交易以及交易所/柜台规则
  data_sources/                                      # 市场数据源字段映射，描述 vendor 字段如何映射到标准合同
  implementations/                                   # 可运行市场实现，把 asset_class、instrument、venue 和 data_source 组合成具体模块
```

### 2.5 Datasets / 数据集访问层

```text
datasets/                                            # 平台数据集访问层，负责读取、索引、缓存和版本管理，不定义市场口径
  dataset_spec.py                                    # 数据集输入合同，记录路径、格式、schema、分区和数据版本
  dataset_registry.py                                # 数据集注册表，管理本地数据、vendor dump 和生成数据集
  dataset_reader.py                                  # 数据集读取入口，根据格式和 spec 分派到具体 reader
  parquet_reader.py                                  # Parquet 读取器，处理列裁剪、分区和 schema 检查
  csv_reader.py                                      # CSV 读取器，处理编码、列类型和 Windows Excel 兼容性
  local_store.py                                     # 本地数据根目录、版本索引和路径解析
  cache_store.py                                     # 可重建缓存数据的读取、写入和失效策略
```

### 2.6 Artifacts / 产物合同层

```text
artifacts/                                           # 研究和评价产物合同层，是 API、下载和前端展示的来源
  artifact_schema.py                                 # artifact schema 版本和顶层持久化合同
  display_manifest.py                                # manifest.display 构造，作为前端、图表、表格和下载的唯一展示合同
  manifest_validation.py                             # manifest、display、table、chart、download 和 path safety 校验
  artifact_writer.py                                 # JSON、CSV、Markdown、报告和 manifest 写出逻辑
  artifact_loader.py                                 # artifact run 读取逻辑，拒绝 unsupported schema 和退休兼容路径
  artifact_inventory.py                              # run 发现、索引、摘要和可用性检查
  artifact_comparison.py                             # 多 run 对比结果、对比表和对比下载
  artifact_package.py                                # ZIP 下载包、download preset 和文件选择规则
```

### 2.7 Pipelines / 本地流水线

```text
pipelines/                                           # 工作流管道层，负责把数据、因子、模型、评价和 artifact 串成可恢复流程
  pipeline_spec.py                                   # pipeline 输入合同，记录步骤、参数、输入、输出和运行模式
  pipeline_runner.py                                 # pipeline 执行入口，负责步骤调度、状态流转和错误处理
  pipeline_checkpoint.py                             # checkpoint 对象，记录已完成步骤、产物路径、warning 和 blocking issue
  pipeline_state.py                                  # pipeline 状态对象，记录 pending、running、failed、completed 等状态
  pipeline_dag.py                                    # pipeline DAG 结构，描述步骤依赖和可执行顺序
  retry_policy.py                                    # 重试策略，定义可重试错误、最大次数和退避规则
```

### 2.8 Runtime / 运行时

```text
runtime/                                             # 本地运行时层，负责路径、环境、缓存、profile 和执行上下文
  runtime_config.py                                  # 运行时配置对象，记录 run roots、cache roots、profile 和环境选项
  path_resolver.py                                   # 路径解析器，避免在代码和文档中硬编码机器绝对路径
  run_roots.py                                       # run root 配置和 source_alias 解析
  cache_paths.py                                     # 缓存路径和可重建产物路径管理
  environment_probe.py                               # 环境探测，例如 Python、Conda、Node、可用依赖和平台信息
  execution_context.py                               # 执行上下文，记录当前市场、数据阶段、profile、run id 和安全边界
```

### 2.9 Interfaces / 交互接口

```text
interfaces/                                          # 用户和外部系统入口适配集合，承载 API 合同、CLI、SDK 和 notebook entrypoints
  api/                                               # 本地 HTTP 服务接口，承载 FastAPI routes、services、evidence、downloads 和 pipeline endpoints；Phase 2 可部署应用入口另建 apps/api
  cli/                                               # 命令行接口，承载研究、评价、pipeline、artifact 检查和服务启动命令
  sdk/                                               # Python SDK 预留，提供 notebook、外部脚本和未来插件可调用的稳定接口
```

## 3. Phase 2: Web Platform Foundation

Phase 2 在决定对外提供网站服务前落地。它解决多人使用、长任务、持久化、权限和部署问题。当前阶段不实现这些能力，只保留边界。

```text
apps/                                                # 可部署应用入口：web、api、workers、schedulers
  web/                                               # 用户网站前端，消费公开 API、权限状态和 artifact/display 合同
  api/                                               # 后端 HTTP 服务，装配鉴权、租户、审计、routes、services 和错误码
  workers/                                           # 后台任务进程，执行 pipeline、model、backtest、export 和报告生成
  schedulers/                                        # 调度进程，负责定时任务、依赖触发、数据到达触发和周期更新
storage/                                             # 存储基础设施：数据库连接、migration、对象存储、缓存、搜索索引和事务边界
identity/                                            # 用户身份：用户、组织、登录、session、API key、角色和登录安全
tenancy/                                             # 多租户：workspace、project、资源隔离、配额和数据边界
access_control/                                      # 访问控制：资源级权限、RBAC/ABAC、policy decision 和 policy enforcement
audit/                                               # 审计：访问、下载、运行、审批、交易和权限变化记录
jobs/                                                # 执行任务：某次 pipeline、model、backtest、export 的状态、资源、日志和结果
workers/                                             # 后台执行器：CPU/GPU worker、队列消费、心跳和并发控制
schedulers/                                          # 调度器：定时任务、依赖触发、数据到达触发和周期更新
observability/                                       # 运行观测：日志、指标、告警、health checks、数据延迟和任务状态
notifications/                                       # 通知：任务完成、失败、告警、审批、邮件、站内信和 webhook
deployments/                                         # 部署资产：Docker、环境变量模板、服务拓扑、worker/scheduler 和运维说明
administration/                                      # 管理后台领域：用户、租户、任务、权限、数据源和异常处置
```

领域对象归属规则：

```text
identity/user.py                                     # User 属于身份领域，不属于 storage
tenancy/workspace.py                                 # Workspace 属于多租户领域，不属于 storage
artifacts/artifact_record.py                         # ArtifactRecord 属于产物领域，不属于 storage
pipelines/pipeline_job.py                            # PipelineJob 属于流水线领域，不属于 storage
storage/database_session.py                          # 数据库连接、事务和 session 属于 storage
storage/migrations/                                  # 数据库迁移脚本属于 storage
```

## 4. Future: Professional Quant Platform

这些是几年期专业量化平台能力。只有当对应业务对象真实出现时再创建目录，避免空目录膨胀。

```text
strategies/                                          # 可运行策略：universe、alpha、portfolio construction、risk policy 和 execution intent
simulations/                                         # 事件仿真：回测、paper exchange、fill、slippage、fee、latency 和 market replay
feeds/                                               # 数据流：historical replay、live feeds、order book、news 和 fundamental updates
reference_data/                                      # 主数据：instrument master、symbol mapping、合约链、公司行为和退市数据
feature_store/                                       # 特征库：point-in-time features、offline/online 一致性和训练/推理特征复用
model_registry/                                      # 模型注册：模型版本、lineage、stage、审批和部署状态
experiment_tracking/                                 # 实验追踪：参数、metrics、数据版本、代码版本和 artifact 对比
pricing/                                             # 定价系统：期权、债券、曲线、波动率曲面和衍生品估值
portfolio_accounting/                                # 组合会计：现金、持仓、成交、费用、分红、融资、PnL 和对账
benchmarks/                                          # 基准：指数、benchmark portfolio、peer group 和评价基准
screening/                                           # 选股和资产筛选：规则、公式、排名、命中结果、筛选历史和权限边界
fundamental_analysis/                                # 基本面分析：公司画像、财务质量、估值、成长、现金流和同行对比
ranking/                                             # 评分体系：价值、质量、成长、动量、风险和综合 alpha score
watchlists/                                          # 关注列表：用户自选、分组、标注、候选池和跟踪状态
alerts/                                              # 提醒系统：筛选命中、估值分位、财报发布、技术突破、风险阈值和通知历史
portfolio_analytics/                                 # 用户持仓分析：暴露、归因、相关性、回撤、再平衡和情景压力测试
research_cards/                                      # 人读研究卡片：公司卡片、因子卡片、策略卡片、组合卡片和可分享摘要
reports/                                             # 人读报告：研究报告、事实卡、PDF/HTML 和审批版报告
publishing/                                          # 发布流程：草稿、review、审批、公开链接、撤回和版本记录
knowledge/                                           # 知识库：数据字典、市场规则、研究笔记、帮助文档和 RAG 资料
sandboxes/                                           # 沙箱：用户代码、notebook、策略草稿和 AI 生成代码的隔离执行环境
secrets/                                             # 密钥管理：broker key、vendor key、OAuth secret 和 webhook secret
compliance/                                          # 合规：数据许可、免责声明、发布边界、保留期限和投资建议边界
approvals/                                           # 审批流：报告发布、模型上线、数据接入、权限提升和实盘启用
trading/                                             # 交易领域：订单、账户、持仓、broker/exchange connector 和 paper/live 接口
execution/                                           # 执行算法：订单路由、拆单、成交质量、限速和执行策略
controls/                                            # 风控控制：pre-trade、intraday、post-trade、kill switch 和限额
reconciliation/                                      # 对账：broker statement、成交回报、持仓、现金、费用和系统记录核对
billing/                                             # 计费：套餐、用量、额度、账单、支付和资源消耗
assistants/                                          # AI 助手：自然语言研究、报告解释、代码生成、agent runs 和工具审计
```

## 5. 文件、类、函数命名约定

```text
package_directory/                                   # 包目录使用清晰领域名词，尽量表达对象集合或稳定领域
object_name.py                                       # 文件使用 snake_case 名词，表达具体对象、合同、服务或算法
ObjectName                                           # 类使用 PascalCase 名词，表达稳定概念、合同、结果、配置或错误
function_name()                                      # 函数使用 snake_case 动词短语，表达动作和副作用边界
variable_name                                        # 变量使用 snake_case 数据名词，表达数据含义而不是临时形状
CONSTANT_NAME                                        # 常量使用 UPPER_SNAKE_CASE，表达 schema 版本、默认值和固定集合
```

推荐后缀：

```text
Spec                                                  # 调用方输入配置，例如 EvaluationSpec、PipelineSpec
Config                                                # 运行时配置或内部配置，例如 RuntimeConfig
Schema                                                # 持久化或序列化结构，例如 ArtifactSchema
Manifest                                              # 描述一组产物或模型的元数据，例如 DisplayManifest、ModelManifest
Payload                                               # 模块间传递的内存数据包，例如 EvaluationPayload
Result                                                # 计算完成后的返回对象，例如 EvaluationResult
Record                                                # 表格、日志或数据库记录中的一行对象
Snapshot                                              # 某一时点状态，例如 PipelineSnapshot
Checkpoint                                            # 可恢复节点，例如 PipelineCheckpoint
Repository                                            # run、artifact、model 等对象的存取索引
Service                                               # API 层业务服务，必须带具体领域前缀
Error                                                 # 异常类型
```

避免使用以下泛名：

```text
utils.py                                             # 过宽泛，无法表达职责边界
helpers.py                                           # 过宽泛，容易变成杂物文件
manager.py                                           # 过宽泛，无法区分管理对象类型
processor.py                                         # 过宽泛，无法表达输入输出合同
handler.py                                           # 过宽泛，只能在框架回调上下文中有限使用
service.py                                           # 单独使用过宽泛，应写成 download_service.py 等具体名称
common.py                                            # 文件名过宽泛，跨目录公共对象应放入具体模块名
```
