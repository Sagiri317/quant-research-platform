# Research Contract

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
- portfolio 处理组合构建和收益路径。

## 当前合同唯一

同一语义只能有一个当前入口。兼容期需要用户批准，并由对应 contract tests 覆盖。

## 层边界

research 是通用研究算法层，可以依赖 contracts、storage、runtime resources 和 `markets.session_windows`。research 不得依赖 `markets/implementations/` 或 `sources/`。A 股 factor/label binding、profile 和 scenario 由 `markets/implementations/a_share/` 持有。

compute 只提供数值计算核心，不持有 market session domain 工具，也不写 report/storage 产物。模型算法事实源位于 `research/models/supervised.py`，单个算法字段使用 `model_algorithm`，算法 ID 集合使用 `model_algorithm_ids`。

## 数据安全

研究 evidence 不能泄漏 row-level local-only data。可交接结果只能包含统计摘要、schema、counts、hash、status 和 sanitized logical path。

## 业务扩展边界

新增模型、因子、执行模式、artifact type 或 report contract 必须有明确 owner layer、输入输出合同、测试和 verification plan。未经批准，不新增 live trading 或 external connector 能力。
