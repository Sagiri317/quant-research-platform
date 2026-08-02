# Engineering Contract

本文只保存长期工程 invariant 与验证入口；字段、profile scope、schema 和 handoff layout 以 executable owner 为准。

## Python 与依赖

- runtime/dev/build dependency 与 Python 版本以 `pyproject.toml` 为源，`environment.yml` 提供同等或更严格的 conda 环境声明。
- `structure_audit.py` 校验 dependency、Python pin、Ruff target、自包含 tooling 和 machine ID 唯一性；它不替代 lockfile、行为测试或 package smoke。
- package smoke 使用 `python -m build --no-isolation --sdist --wheel`，随后只解压 wheel 并用 `PYTHONPATH` 做 import smoke，不安装 package。
- `pytest-xdist` 只用于可选耗时诊断；sequential `all` 才是完整 correctness truth。

## 验证与状态

验证状态只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`；每个状态必须绑定实际 command、scope 与 result。不同 gate 的 PASS 互不替代。

```bash
git diff --check
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py --repo-root . --json
conda run -n <environment.yml name> python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --governance-only --scan-scope active-docs --json
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile fast --run-dir .run/<purpose>/fast
conda run -n <environment.yml name> python tests/tools/run_profile.py --profile all --run-dir .run/<purpose>/all -- --durations=100
```

`tests/tools/run_profile.py` 是唯一正式 profile runner；marker 与 scope 不在文档复制。完整运行因环境或超时无法完成时必须如实写 `BLOCKED`。

## Owner 与 Lifecycle

- `governance_contract.py` 只拥有 active-doc、dependency/writer boundary、source-tree sensitive scan 与 package-safe path facts；普通名称、目录和实现形状不直接失败。
- 一个 logical operation 复用一个 immutable `RuntimeResourcePolicy`；各 owner 只声明并管理能真实诊断的 retained state，不以局部 payload 冒充总 RSS。
- `DatasetSession` 和 physical iterator 由 Storage 唯一拥有；scratch、exact key、manifest、fingerprint、compare 和 staged publication 不在 caller 层复制。
- research/reports 只构建 result/content；只有 Storage 与 Artifacts writer owners 写文件。cleanup 只处理能证明归属的路径，且保留 primary failure。
- fingerprint 只用于 quick reject；只有 quiescent inputs 上的 Storage exact compare 可证明 `MATCH`。

## JSON 与 Artifact

- QRP-owned JSON 使用 Contracts strict codec：strict UTF-8 顶层 object、duplicate-key/nonfinite 拒绝和 deterministic write。
- artifact owner 在 decode 后校验 exact `schema_id`、field set 与业务 invariant；generic codec 不复制各 artifact schema。
- dataset `spec.json`、manifest、logical dtype 与 physical Arrow schema 由 Storage current grammar 精确校验；diagnostic readability 不替代 normal consumability。
- run-level machine truth 是 `run.json`，人读摘要是 `summary.md`；output index 必须绑定 bundle 内真实 manifest 与文件。

## Handoff

`handoff_contract.py` 拥有 package/sidecar/task/validation schema，`receipt_contract.py` 拥有 receipt schema、normalize 与 compare，`verify_handoff_package.py` 拥有唯一 package load/validation flow，`replay_evidence.py` 执行 replay。

正式 builder 只接受 clean Git state，生成 bundle 后从 clone 派生 replay-backed receipts 与 validation summary，再由同一 package payload verifier 校验 schema、hash 和 share-safety。standalone verifier 复用该 flow 并执行完整 bundle replay；status-only、private/local-only 或不可复验证据不能支撑 machine `PASS`。
