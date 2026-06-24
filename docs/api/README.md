# API Docs

本目录描述 QRP 当前 API 边界。

## 当前原则

API 文档只描述当前公开边界，不记录历史阶段或未来承诺。新增 API、CLI、registry、store 或 facade 前必须得到用户明确批准，并同步 contracts、tests 和 docs。

## 接口边界

`src/quant_research_platform/interfaces/api/` 提供 API adapter。interfaces 不写业务算法，不绕过 contracts、privacy、runtime logging 或 public-safe evidence，也不新增未批准 public surface。

当前 public API 只有：

```text
GET  /api/0.1/catalog/factors
GET  /api/0.1/catalog/labels
GET  /api/0.1/research-datasets/{dataset_id}/requirements
```

`/api/0.1` 是当前合法 API 版本面。带 `v` 前缀的 `0.1` API 版本路径不是当前 API 形式，只允许出现在治理规则定义或 negative tests 中。

当前 public CLI 只有：

```text
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

当前 approved private local CLI 只有：

```text
qrp-private schema-inventory --project-root . --output <path>
qrp-private evidence run
private local verification harness
not a public CLI
not a public execution mode
not a public evaluate, predict, or report command
not simulated or real-money order execution
does not promise returns
does not output row-level private data
```

命名规则：

```text
API resource segment: kebab-case + plural
API path parameter: snake_case
ID value: snake_case
JSON field: snake_case
CLI group / command / option: kebab-case
Python module/function: snake_case
```

## 验证

API 相关变更必须至少说明：

- request/response contract
- error contract
- privacy impact
- tests
- backward compatibility decision

默认不保留旧字段兼容，除非用户批准兼容期。
