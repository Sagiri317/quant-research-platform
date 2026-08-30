# 工程合同

本文只保存长期工程不变量；当前 branch、phase 和验证结果来自当前 Git 状态与本轮任务。

## 当前合同

QRP 使用 current-contract-only。Public machine contract、owner behavior、持久化 grammar 与 dependency direction 定义当前事实；被替换入口直接删除，不保留 alias、shim、compat reader、parallel implementation 或仅用于证明的 framework。

`pyproject.toml` 定义 runtime/dev/build dependency 与 Python 版本，`environment.yml` 提供同等或更严格的 conda 环境。新增依赖、public API/CLI 或 framework 需要明确批准。

## 资源生命周期

每个顶层 resource-bearing operation 在首次数据规模 read/decode/allocation 前只 admission 一次：host-local data-scale lock 保护一个 process-local active root；该 root 只捕获一次 OS available0，并固定 operation planning target（默认约为 `available0 * 70%`，与有效显式 cap 取较小者）和 root identity。70% 与派生出的 `memory_budget_bytes` 都是软规划参考，不是进程 RSS、真实 allocation 或逻辑 working set 的数学硬线。Lease ledger 用整数 bytes 只表达 owner 的保守粗估，以支持相加、转移和有界批次规划，不承诺字节级精确。只有粗估表明无法维持相对安全或 explicit cap 明确限制时才稳定失败；轻微估算/观测 overshoot 不在 70% 的精确点自动判死。Lease ledger 与 lexical lease state 随 owner 生命周期可变；borrowed baseline、native workspace 与验收 RSS 不重复计费。

Nested owner 继承相同 root identity；different-policy nesting/parallel root fail closed。Lease 以 lexical owner context 管理，跨 producer/consumer 边界转移同一 logical owner，不重复计费；owner context 退出时释放资源。主要 workflow stage 按顺序执行，前一 stage 释放 transient state 后，后一 stage 复用同一个 operation budget；只为当前 stage 中真正同时存活的 allocation 准入。所有数据规模 allocation 在保守准入后进行并保持有界，聚合持久化输出可以超过 RAM budget 而 exact 完成。

全部 raw、decoded、Arrow/Pandas、copy/index/sort/join、reducer/planner/model/native、output/writer/readback/spill/overlap state 共用该有界 budget；新 payload/workspace/output 只有一个 logical owner，准入不足则按 owner 语义 shrink、release、spill 或 fail closed。不建立 universal deep-size framework。

约 80% 是粗粒度的工程安全包络，不是逐字节硬线；小幅、稳定且可解释的 overshoot 由验收记录为风险供用户判断，不建立第二套分段阈值或更高的 magic threshold。仓库外 fresh-child 的 RSS/CPU/I/O 只用于校准和验收，不能成为 runtime 规划权威。

## Owner、identity 与发布

- Storage 拥有唯一 physical traversal/session、持久化 schema/dtype/PK、fingerprint、exact compare 和 staged publication。
- Fingerprint 只 quick reject；只有 quiescent exact compare 可以产生 `MATCH`。
- Research/Compute/Reports 构建结果或内容，不写文件；Storage 与 Artifacts 是物理写入 owner。
- Publication 必须 atomic 并保留 old-good；partial-run truth 不得被成功摘要覆盖。
- Cleanup 只处理 owner-qualified resources；active `BaseException` primary 不被 metrics、cleanup 或 secondary failure 覆盖。
- QRP-owned JSON 使用 strict UTF-8、top-level object、exact `schema_id`、duplicate-key/nonfinite rejection 与 deterministic write。

## 私有数据安全

LOCAL_ONLY 与 REVIEW_SAFE 分离。REVIEW_SAFE 内容不得包含私有行、secret、vendor raw marker、本地绝对路径、股票代码样本或 raw traceback。

## 直接验证

状态只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`，并绑定实际命令、范围与结果。环境名从 `environment.yml` 动态读取。

```text
git diff --check
conda run -n <env> ruff format --check .
conda run -n <env> ruff check . --no-cache
conda run -n <env> python -m pytest -q -o "addopts=--import-mode=importlib --strict-markers" tests
conda run -n <env> python -m build --no-isolation --sdist --outdir <sdist-run>
conda run -n <env> python -m build --no-isolation --wheel --outdir <wheel-run>
```

Package acceptance 从仓库外 clean export 开始：先构建 fresh sdist，再从解出的 sdist 构建 wheel，在仓库外 isolated environment 安装，最后从 empty CWD 做 `qrp`、`qrp-private`、API 与代表性 workflow smoke。editable checkout 或直接从源码路径导入不算通过。
