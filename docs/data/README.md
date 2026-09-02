# 数据合同

本目录描述数据目录和数据安全边界。

## 基本规则

- 不提交私有数据。
- 不提交 vendor marker。
- 不提交行级样本。
- 不提交本地绝对路径。
- 不提交股票代码样本。
- 不提交 raw traceback。

## 数据位置

本地数据应放在用户本地 workspace 或 `.run/` 下的运行产物目录中。受 Git 跟踪的仓库只保存代码、contracts、schemas、docs、tests 和 REVIEW_SAFE 模板。`.run/` 只是输出/证据/输入 artifact 层，不是受 Git 跟踪的工具源代码。

`sources/` 负责摄取与 raw/interim 转换；A 股与 Crypto 的对应 `markets/implementations/` 分别负责本市场 processed data 和市场语义；`storage/` 只负责 dataset IO、manifest、fingerprint、compare 和 data layout。共享安全检测规则由 `policy/share_safety.py` 提供，storage 不维护平行正则。

所有 source 共用 `data/raw`，各 source owner 只在 `data/raw/<source_id>` 命名空间内递归发现并校验请求的 provider/table 文件。未能由现有 owner 合同证明的文件保留为 raw-only，不通过备用根进入生产链。每个保留 CSV 只有一个 canonical `.metadata.json` sidecar；sidecar 记录 hash、schema、计数和质量事实，不构成第二份 raw truth。

A 股历史费率使用独立的 standalone CSV reference source。当前 sidecar 将其指定为接受用于 QRP 研究的输入；CSV 内容及其 checksum、字节数和行数事实保持不变。

Binance raw 合同见 `docs/data/binance.md`。Canonical local-envelope 导入器只接受 canonical JSONL envelope；正式 HFT recorder-run 则只接受其 manifest 精确声明的 gzip JSONL members。两者都不接受裸 connector payload 作为独立输入。

每个 QRP dataset 保留：

```text
<dataset_root>/spec.json
<dataset_root>/_manifest/manifest.json
```

`dataset_root` 由 storage layout 按 layer/source/market/kind/data ID 唯一派生。两类 JSON 分别使用 `qrp-dataset-spec` 和 `qrp-dataset-manifest` 作为 `schema_id`。运行级索引只写入 `<run_root>/run.json`。

## REVIEW_SAFE 证据

允许写入 REVIEW_SAFE 证据的内容：

- schema
- counts
- hash
- 命令状态
- artifact 路径结构
- 不含私有行的汇总统计

数据合同直接使用 pytest 验证：

```text
conda run -n <environment.yml name> python -m pytest -q tests/storage tests/sources
```

## Agent 规则

agent 不得读取、复制或摘要私有行级数据，除非用户明确要求并且输出仍保持 REVIEW_SAFE。任何临近私有数据的任务必须在报告中说明安全边界。
