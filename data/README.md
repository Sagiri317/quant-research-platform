# 数据目录

本目录不用于提交数据文件。

- 私有数据不进入 git。
- 生成的行级数据不进入 git。
- vendor marker、本地绝对路径、股票代码样本不进入 git。
- `.run/` 只保存本地输出/证据/输入 artifact，不是受 Git 跟踪的工具源代码。

数据合同见 `docs/data/README.md`。

目录职责：sources 负责摄取，A 股与 Crypto 的对应实现层分别负责本市场 processed data 与市场语义，storage 负责通用 dataset IO、manifest、fingerprint、compare 和 data layout。本目录不定义运行时业务逻辑或 `run_root`。
