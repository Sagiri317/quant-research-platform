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

同一语义只能有一个当前入口。并行入口需要用户批准，并由对应 contract tests 覆盖 owner、生命周期和收敛条件。

## 层边界

research 是通用研究算法层，可以依赖 contracts、storage、runtime resources 和 `markets.session_windows`。research 不得依赖 `markets/implementations/` 或 `sources/`。A 股 factor/label binding、profile 和 scenario 由 `markets/implementations/a_share/` 持有。

compute 只提供数值计算核心，不持有 market session domain 工具，也不写 report/storage 产物。

`research/factors/price_volume/` 是量价因子的当前 owner。`columns.py` 只负责列名、列前缀、列集合和 feature exclusion；`presets.py` 负责 preset 名称与 factor 列集合；`requirements.py` 负责 factor 输入列需求；`windows.py` 负责 lookback 和 valid bar 读取窗口；`definitions.py` 负责 factor definition 语义；`calculations.py` 负责因子计算；`rolling.py` 和 `safety.py` 负责滚动数值与安全计算；`selection.py` 负责特征选择。package `__init__.py` 只导出当前 public names，不承载实现。

`research/models/algorithm_facts.py` 是模型算法事实源。`estimators.py` 负责 estimator factories，`fitting.py` 负责 fit result、训练 frame 收集、线性拟合与 estimator fitting，`prediction.py` 负责预测输出 frame，`feature_importance.py` 负责特征重要性 payload，`rank_ensemble.py` 负责 rank ensemble predictor，`dependencies.py` 负责可选依赖状态，`validation_split.py` 负责训练 buffer 与 purge controls。`research/models/supervised.py` 保留当前 public boundary 并委派到这些 owner modules。单个算法字段使用 `model_algorithm`，算法 ID 集合使用 `model_algorithm_ids`。

跨文件生产调用的 owner-local internal API 使用非下划线名称，可以留在 owner module 的 `__all__`，但不得从 package `__init__.py` 重导出为 public surface。同文件私有 helper 保留下划线且不进入 `__all__`。

## 数据安全

研究 evidence 不能泄漏 row-level local-only data。可交接结果只能包含统计摘要、schema、counts、hash、status 和 sanitized logical path。

## 业务扩展边界

新增模型、因子、执行模式、artifact type 或 report contract 必须有明确 owner layer、输入输出合同、测试和 verification plan。未经批准，不新增交易执行或 external connector 能力。
