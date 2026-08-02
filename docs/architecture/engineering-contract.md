# Engineering Contract

本文只保存长期工程 invariant；当前 branch、phase 和验证结果来自 live Git 与本轮任务。

## Current contract

QRP 使用 current-contract-only。Public machine contract、owner behavior、persisted grammar 与 dependency direction 定义当前事实；被替换入口直接删除，不保留 alias、shim、compat reader、parallel implementation 或 proof-only framework。

`pyproject.toml` 定义 runtime/dev/build dependency 与 Python 版本，`environment.yml` 提供同等或更严格的 conda 环境。新增依赖、public API/CLI 或 framework 需要明确批准。

## Resource lifecycle

每个 top-level resource-bearing operation 在首次数据规模 read/decode/allocation 前只 admission 一次：有效 available memory 是 OS available memory 与适用 cgroup/container/job/process headroom 的最小值；operation budget 是该值 80% 向下取整与有效显式 cap 的最小值。Policy 与 budget 捕获后 immutable，pressure 只能缩小后续工作。

一个 process 默认只有一个 active root policy、execution lease 与 total reservation。Nested owner 继承相同 identity；different-policy nesting/parallel root fail closed。Copied context 不跨 thread/task 传递 authority；child 不越过 root lifetime。Resource-bearing child process 只能使用 parent 从 root budget 划分并显式传入的 cap；跨进程共享 reservation 未实现前保持 single-process execution。

全部 raw、decoded、Arrow/Pandas、copy/index/sort/join、reducer/planner/model/native、output/writer/readback/spill/overlap state 共用预算。每次数据规模 allocation 前 reserve conservative owner-local upper bound，planned reservation 不得超过 budget；不足则 shrink、release、spill 或 fail closed。不建立 universal deep-size framework，不为 probe 波动发明超配容忍线。

## Owner、identity 与 publication

- Storage 拥有唯一 physical traversal/session、persisted schema/dtype/PK、fingerprint、exact compare 和 staged publication。
- Fingerprint 只 quick reject；只有 quiescent exact compare 可以产生 `MATCH`。
- Research/Compute/Reports 构建结果或内容，不写文件；Storage 与 Artifacts 是 physical writer owners。
- Publication 必须 atomic 并保留 old-good；partial-run truth 不得被成功摘要覆盖。
- Cleanup 只处理 owner-qualified resources；active `BaseException` primary 不被 metrics、cleanup 或 secondary failure 覆盖。
- QRP-owned JSON 使用 strict UTF-8、top-level object、exact `schema_id`、duplicate-key/nonfinite rejection 与 deterministic write。

## Private safety

LOCAL_ONLY 与 REVIEW_SAFE 分离。REVIEW_SAFE 内容不得包含 private row、secret、vendor raw marker、本地绝对路径、股票代码样本或 raw traceback。

## Direct validation

状态只使用 `PASS`、`FAIL`、`NOT_RUN`、`BLOCKED`，并绑定实际 command、scope 与 result。环境名从 `environment.yml` 动态读取。

```text
git diff --check
conda run -n <env> ruff check .
conda run -n <env> python -m pytest -q -o "addopts=--import-mode=importlib --strict-markers" tests
conda run -n <env> python -m build --no-isolation --sdist --wheel --outdir <run-dir>
```

Build smoke 使用标准库解压 wheel，并以解压目录作为 `PYTHONPATH` import；不安装 package。
