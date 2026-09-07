# 数据合同

本目录描述数据目录和数据安全边界。

## 基本规则

- 不提交私有数据。
- 不提交供应商标记。
- 不提交行级样本。
- 不提交本地绝对路径。
- 不提交股票代码样本。
- 不提交原始异常回溯。

## 数据位置

本地数据应放在用户本地工作区或 `.run/` 下的运行产物目录中。
受 Git 跟踪的仓库只保存代码、合同、结构定义、文档、测试和 REVIEW_SAFE 模板。
`.run/` 只是输出、证据和输入产物层，不是受 Git 跟踪的工具源代码。

`sources/` 负责摄取及原始/中间层转换；A 股与 Crypto 对应的
`markets/implementations/` 分别负责本市场已处理数据和市场语义；
`storage/` 只负责数据集 IO、清单、指纹、比较和数据布局。
共享安全检测规则由 `policy/share_safety.py` 提供，存储层不维护平行正则表达式。

所有来源共用 `data/raw`，各来源责任模块只在 `data/raw/<source_id>` 命名空间内
递归发现并校验请求的供应商/表文件。未能由现有责任合同证明的文件仅保留原始数据，
不通过备用根目录进入生产链。每个保留 CSV 只有一个规范的 `.metadata.json` 附属文件；
其中记录哈希、结构、计数和质量事实，不构成第二份原始数据真值。

A 股历史费率使用独立 CSV 参考来源。当前附属元数据将其指定为接受用于 QRP 研究的输入；
CSV 内容及其校验和、字节数和行数事实保持不变。

Binance 原始数据合同见 `docs/data/binance.md`。
规范本地封装导入器只接受规范 JSONL 封装；正式 HFT 录制运行只接受其清单精确声明的
gzip JSONL 成员。两者都不接受裸连接器载荷作为独立输入。

每个 QRP 数据集保留：

```text
<dataset_root>/spec.json
<dataset_root>/_manifest/manifest.json
```

`dataset_root` 由存储布局按层、来源、市场、种类与数据 ID 唯一派生。
两类 JSON 分别使用 `qrp-dataset-spec` 和 `qrp-dataset-manifest` 作为 `schema_id`。
运行级索引只写入 `<run_root>/run.json`。

## REVIEW_SAFE 证据

允许写入 REVIEW_SAFE 证据的内容：

- 结构定义
- 计数
- 哈希
- 命令状态
- 产物路径结构
- 不含私有行的汇总统计

数据合同直接使用 pytest 验证：

```text
conda run -n <environment.yml name> python -m pytest -q tests/storage tests/sources
```

## 代理规则

代理不得读取、复制或摘要私有行级数据，除非用户明确要求并且输出仍保持 REVIEW_SAFE。
任何临近私有数据的任务必须在报告中说明安全边界。
