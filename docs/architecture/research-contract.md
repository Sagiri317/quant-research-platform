# 研究合同

本文件描述 QRP 研究层长期合同。

## 研究对象

QRP 支持本地量化研究流程，包括 data、factor、feature、label、model、prediction、signal、evaluation、portfolio 和 report。

## 命名边界

- factor 是研究信号来源。
- feature 是模型输入或因子变换后的建模字段。
- label 是监督学习目标。
- prediction 是模型输出。
- signal 是评估或组合使用的排序、权重或交易意图输入。
- evaluation 衡量 signal 或 prediction 的研究表现。
- evaluation axis 必须符合研究横截面：多资产同一时点使用
  `cross_sectional`；单资产 HFT 跨时点使用 `single_asset_time_series`，不得用
  零有效观测的横截面结果冒充通过。当前 single-asset axis 用有界 online
  moments 计算跨时间 Pearson；未实现 exact bounded rank/group evaluation 时将
  RankIC 与 group spread 明确标为 not applicable，不输出伪精确值。
- portfolio 处理组合构建和收益路径。

## 当前合同唯一

同一语义只能有一个当前入口。并行入口需要用户批准，并由对应 contract tests 覆盖 owner、生命周期和收敛条件。

## 层边界

research 是通用研究算法层，可以依赖 contracts、storage 和 runtime resources。research 不得依赖 `markets/implementations/` 或 `sources/`。A 股与 Crypto 的 factor/label binding、profile 和 scenario 分别由对应 `markets/implementations/` 领域实现持有；只有真实共享的研究算法进入 research。

compute 只提供数值计算核心，不持有 market session domain 工具，也不写 report/storage 产物。

reports 是纯 report-content owner，可由 workflows 或负责持久化的 artifacts 调用；
artifacts 只验证并写入内容，不反向定义 report 语义。允许 `artifacts -> reports`，
仍禁止 `reports -> artifacts`，避免内容层依赖物理产物层。

可移植的模型算法 ID、default/smoke 参数、dependency package、input kind 与 parallel classification facts 由 contracts 持有；fitting、prediction、feature importance、rank ensemble 与 validation behavior 由 research owners 持有。单个算法字段使用 `model_algorithm`，算法 ID 集合使用 `model_algorithm_ids`。

公共 package surface 由 current contract 管理；内部 helper 只在真实 caller 需要时保留，不以名称、可见性或 re-export 形状作为 gate。

model selection 的 metric、direction 和无警告行为必须由合同与 owner-layer tests 保护，runner 不维护平行事实源。

未接入当前 production path 的 deferred 实现、脚手架和测试不保留；仍有产品价值时只记录 open issue。

## 数据安全

研究证据不能泄漏行级 local-only 数据。可交接结果只能包含统计摘要、schema、counts、hash、status 和 sanitized 逻辑路径。

## 业务扩展边界

新增模型、因子、执行模式、artifact type 或 report contract 必须有明确 owner layer、输入输出合同、测试和验证计划。未经批准，不新增交易执行或 external connector 能力。
