# Data Directory

本目录不用于提交数据文件。

- private data 不进入 git。
- generated row-level data 不进入 git。
- vendor marker、本地绝对路径、stock-code sample 不进入 git。
- `.run/` 只保存本地 output/evidence/input artifact，不是 tracked tool source。

数据合同见 `docs/data/README.md`。

目录职责：sources 负责摄取，A 股实现层负责 processed data 与市场语义，storage 负责通用 dataset IO、manifest、fingerprint、compare 和 data layout。本目录不定义运行时业务逻辑、`run_root` 或 handoff evidence。

REVIEW_SAFE evidence 的 fast synthetic gate 和 data/parquet contract gate 分别位于：

```text
tests/architecture/test_public_safe_evidence.py
tests/architecture/test_public_safe_evidence_data_contract.py
```
