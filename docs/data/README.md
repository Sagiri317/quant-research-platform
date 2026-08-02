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

`sources/` 负责摄取与 raw/interim 转换；`markets/implementations/a_share/` 负责 A 股 processed data 和市场语义；`storage/` 只负责 dataset IO、manifest、fingerprint、compare 和 data layout。共享安全检测规则由 `policy/share_safety.py` 提供，storage 不维护平行正则。

Binance raw input 合同见 `docs/data/binance.md`。当前 Binance raw 只接受 canonical JSONL envelope；裸 connector payload 不属于 current contract。

每个 QRP dataset 保留：

```text
data/<layer>/<dataset>/spec.json
data/<layer>/<dataset>/_manifest/manifest.json
```

两类 JSON 分别使用 `qrp-dataset-spec` 和 `qrp-dataset-manifest` 作为 `schema_id`。运行级索引只写入 `<run_root>/run.json`。

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
