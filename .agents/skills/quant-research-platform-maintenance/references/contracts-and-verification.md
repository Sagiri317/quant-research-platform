# 合同与验证（Contracts and verification）

本文件用于 public API/CLI、contracts、artifacts、private safety、验证真实性相关任务。

## Public surface（公共接口面）

当前公共 API/CLI 只有：

```text
GET /api/0.1/catalog/factors
GET /api/0.1/catalog/labels
GET /api/0.1/research-datasets/{dataset_id}/requirements
qrp catalog factors
qrp catalog labels
qrp research-datasets requirements <dataset_id>
```

`qrp-private schema-inventory --project-root . --output <path>` 是窄 private local CLI，不是公共接口扩张。

新增任何 public API、CLI、registry、store、facade、artifact lifecycle API、execution mode、compatibility shim、legacy alias，都必须先停止并取得用户明确批准。

## Artifact truth（产物真实性）

- `PASS`：检查实际运行且通过。
- `FAIL`：检查实际运行且失败。
- `NOT_RUN`：没有运行证据。
- `BLOCKED`：环境、依赖、权限或输入缺失导致无法运行。

禁止把 smoke result 写成 full-horizon evidence；禁止把 partial compare 写成 official parity；禁止把未运行检查写成 `PASS`。

## Private safety（私有安全）

Public-safe evidence 只能包含聚合、脱敏、manifest、schema、统计摘要或路径安全信息。禁止包含 row-level private data、vendor markers、stock-code samples、private file names、credentials、tokens、本地绝对路径、raw tracebacks with private context。

## Verification ladder（验证阶梯）

1. 静态检查：path、public surface、private roots、blocked extensions。
2. 窄测试：改动 owner layer 对应 tests。
3. 集成测试：触及 contract/artifact/workflow/interface 时运行相关跨层 tests。
4. 私有数据验证：只有用户明确授权且在本地执行；结果只输出 public-safe 摘要。
5. Handoff：只打包 public-safe evidence，manifest 真实记录 `PASS/FAIL/NOT_RUN/BLOCKED`。

## Final report 最低字段

```text
changed files
checks actually run
checks not run
PASS/FAIL/NOT_RUN/BLOCKED
private-safety result
public-surface result
remaining risks
```
