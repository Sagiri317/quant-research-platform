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

同一语义只能有一个当前入口。旧字段和旧命名不应通过 fallback 或 alias 保留。

## 数据安全

研究 evidence 不能泄漏 private row-level data。公开结果只能包含统计摘要、schema、counts、hash、status 和 public-safe path。

## 业务扩展边界

新增模型、因子、执行模式、artifact type 或 report contract 必须有明确 owner layer、输入输出合同、测试和 verification plan。未经批准，不新增 live trading 或 external connector 能力。
