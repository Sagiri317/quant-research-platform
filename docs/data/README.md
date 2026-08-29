# Data Contract

本目录描述数据目录和数据安全边界。

## 基本规则

- 不提交 private data。
- 不提交 vendor marker。
- 不提交 row-level sample。
- 不提交本地绝对路径。
- 不提交 stock-code sample。
- 不提交 raw traceback。

## 数据位置

本地数据应放在用户本地 workspace 或 `.run/` 下的运行产物目录中。tracked repo 只保存代码、contracts、schemas、docs、tests 和 REVIEW_SAFE templates。`.run/` 只是 output/evidence/input artifact layer，不是 tracked tool source。

`sources/` 负责摄取与 raw/interim 转换；A 股与 Crypto 的对应 `markets/implementations/` 分别负责本市场 processed data 和市场语义；`storage/` 只负责 dataset IO、manifest、fingerprint、compare 和 data layout。共享安全检测规则由 `policy/share_safety.py` 提供，storage 不维护平行正则。

Binance raw 合同见 `docs/data/binance.md`。Legacy local-envelope importer 只接受 canonical JSONL envelope；正式 HFT recorder-run 则只接受其 manifest 精确声明的 gzip JSONL members。两者都不接受裸 connector payload 作为独立输入。

每个 QRP dataset 保留：

```text
<dataset_root>/spec.json
<dataset_root>/_manifest/manifest.json
```

`dataset_root` 由 storage layout 按 layer/source/market/kind/data ID 唯一派生。两类 JSON 分别使用 `qrp-dataset-spec` 和 `qrp-dataset-manifest` 作为 `schema_id`。运行级索引只写入 `<run_root>/run.json`。

## REVIEW_SAFE Evidence

允许写入 REVIEW_SAFE evidence 的内容：

- schema
- counts
- hash
- command status
- artifact path structure
- summary statistics without private rows

数据合同使用 direct pytest 验证：

```text
conda run -n <environment.yml name> python -m pytest -q tests/storage tests/sources
```

## Agent 规则

agent 不得读取、复制或摘要 private row-level data，除非用户明确要求并且输出仍保持 REVIEW_SAFE。任何 private-data-adjacent 任务必须在报告中说明安全边界。
