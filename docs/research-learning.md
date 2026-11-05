# 策略学习、公式搜索与独立研究 agent

这些操作扩展既有数据、预测和账户计算。每项操作接收显式输入、参数与新输出路径，
CLI 只解析配置并调用同一个 Python 函数。操作之间由调用者连接，不自动重算历史产物。
[配置示例](../configs/research-learning/)中的路径、股票池、字段和日期需要替换为实际研究输入。

## 入口与职责

下表中的 Python 路径均相对于 `quant_research_platform`。

| CLI 命令 | Python 函数 | 输出 |
|---|---|---|
| `train-policy` | `research.policies.reinforcement.train_policy` | PPO 权重、观测归一化、训练事实与试验记录 |
| `backtest-policy` | `research.policies.reinforcement.backtest_policy` | 原生 `ledger/` 与 `summary.json` |
| `search-factors` | `research.factors.search.search_factors` | 完整候选成绩、选中公式与试验记录 |
| `apply-formula` | `research.factors.search.apply_formula` | 原生因子数据集 |
| `calibrate-uncertainty` | `research.predictions.uncertainty.calibrate_uncertainty` | 冻结残差半径 JSON |
| `apply-uncertainty` | `research.predictions.uncertainty.apply_uncertainty` | 原生区间与保守分数数据集 |
| `evaluate-uncertainty` | `research.predictions.uncertainty.evaluate_uncertainty` | 成熟样本覆盖率与区间宽度 |
| `fit-cost-model` | `research.policies.cost_aware.fit_cost_model` | PyTorch 权重、训练缩放与成本近似说明 |
| `backtest-cost-model` | `research.policies.cost_aware.backtest_cost_model` | 原生日频账本 |
| `research-agent` | `agents.runner.run_agent` | 每次行动、试验记录、产物与结论报告 |

例如：

```text
qrp train-policy --config configs/research-learning/train-policy-execution.json
qrp backtest-policy --config configs/research-learning/backtest-policy-execution.json
qrp search-factors --config configs/research-learning/search-factors.json
qrp apply-formula --config configs/research-learning/apply-formula.json
qrp research-agent --config configs/research-learning/research-agent.json
```

`agents/` 是研究操作的调用者，不属于研究算法。`research/policies/` 连接 Gymnasium、
Stable-Baselines3 PPO 和既有账户；`research/portfolios/daily.py` 与 `high_frequency.py`
负责推进交易和权益。账户不依赖 agent、Gymnasium 或 PPO。
顶层 `policy/` 负责资源关闭和证据分享规则，与学习策略分开。

## 强化学习：执行、库存与组合

`train-policy` 明确 `episodes`、`fit_cutoff`、`timesteps`、`seed`、`device`、
`n_envs`、`threads` 和 `ppo`。同次训练的 episode 使用相同任务、股票顺序及特征顺序。
每次 reset 从声明的训练 episode 中抽取，不采入随后评价窗口。
`n_envs=1` 使用单进程，更多环境使用 SB3 子进程；PPO 可能完成整个 rollout 后才停，
实际步数保存在 `policy.json.actual_timesteps`。PPO 的 `device` 沿用 SB3 行为，
支持 `cpu`、`cuda:0`、`auto`；CUDA 不可用时可回退 CPU，产物保存实际使用的 `device`。
LSTM、Transformer 和成本感知网络复用同一库的设备选择；XGBoost 使用自身的 CPU 回退。

每个 episode 指定 `kind`、`task`、`asset_ids`、`feature_columns`、有限 `window`、
`account`、`risk_aversion`、`unfinished_penalty_bps`。日期和精确时刻沿用现有窗口约定：
纯日期结束界包含全天，精确时间结束界包含该时刻。高频时刻使用原生 UTC 时钟。

| 任务 | 账户与动作 | 奖励 |
|---|---|---|
| `execution` | `kind=hft`；`order` 指定股票和有符号股数，正数买入、负数卖出；动作是剩余母单的提交比例 | 负的增量执行短缺与费用，除以母单到达中价名义金额；另扣剩余比例平方的时间积分和终局未完成惩罚 |
| `inventory` | `kind=hft`；每只股票的动作映射为 `maximum_inventory` 的比例 | 本步净值变动除以 episode 初始净值，再减 `risk_aversion × 本步收益²` |
| `portfolio` | `kind=daily` 或 `hft`；股票与现金的非负动作归一化为目标权重 | 同库存任务，以原生账本的净值变化计奖 |

执行短缺采用成交有符号金额减成交股数乘到达中价，再加真实费用；报告单位为 bps。
`unfinished_penalty_bps` 乘未完成比例后计入终局奖励，其他任务必须设为零。
母单限额在账户执行处约束：有延迟、部分成交或等待动作时也不能反向买回或超额执行。
该母单不改变其他股票的目标。

观测包含当时特征、持仓与可卖仓位、现金、剩余时间和母单剩余比例。
特征直接保留空值。普通神经模型和 PPO 的插补参数只在训练域拟合并冻结，不追加缺失指示列。日频只允许 factor/feature；
高频还允许原生盘口当前字段，不开放任意 context 列。上游仍须正确声明特征的可用时点；
列角色本身不能证明外部自制数据没有前视。

日频动作在当日决策之后进入原有下一会话成交；高频动作在严格后续报价受理，
保留延迟、盘口容量、买卖单位、费用、T+1、现金和权益处理。默认主动成交；显式提供
`passive` 与 `trades` 时使用[被动委托模型](a-share-hft.md#被动委托信号与执行基准)，按队列假设等待后续成交。
两种执行方式均不保证母单完成。日频策略目标与 benchmark-enhanced 优化器
是两个独立决策入口，不能同时指定。

训练先遍历有界 episode 的真实决策特征，以公共 `ModelMatrixScaler` 拟合中位数。
`backtest-policy` 冻结读取 `policy.zip`、`feature_scaler.pkl`、`normalization.pkl` 与 `policy.json`，
评价窗口必须晚于拟合截止，期间不更新插补或归一化。只加载可信本地模型产物。
基线可选：执行任务的 `immediate`、`twap`、`vwap`、`almgren_chriss`、`signal_ac`，库存的 `hold`，组合的
`hold`、`equal_weight`。TWAP/VWAP 给出截至当前时刻的累计目标，实际成交仍依赖后续报价。
VWAP 需预先声明 `volume_curve` 的 `as_of`、`fractions`、`cumulative_volume`；
两条曲线从 0 到 1，成交量累计比例单调，`as_of` 必须早于 episode。
末尾不会为了完成率虚构成交；应同时比较短缺、费用、完成率与净值。
AC 参数、公式和信号调整的近似边界见上述高频文档。

[执行示例](../configs/research-learning/train-policy-execution.json)、
[库存示例](../configs/research-learning/train-policy-inventory.json)和
[日频组合示例](../configs/research-learning/train-policy-portfolio.json)共用同一训练入口。
固定资产顺序决定网络的观测和动作位置，股票池变化需重新设计训练输入和模型。

## 自动因子搜索

`search-factors` 读取固定 `columns`、可选 `baseline_columns`、`label`、
`train_window`、`validation_window`、各自成熟截止、`budget`、`max_depth`、
滚动 `windows` 与 `seed`。标签须声明成熟时间。
随机搜索使用 `method=random`；PPO 搜索使用 `method=ppo` 并指定 `timesteps`，
可配置 `ppo` 和 `device`。候选数量与 PPO 步数分别有界，重复公式不重复计为新候选。

公式由 AST 白名单解释，不使用 `eval`。支持加减乘除、负号、有限常数、
`abs`、`sign`、`logabs`（`log1p(abs(x))`）、截面 `rank`，以及正整数窗口的
`delay`、`mean`、`std`。滚动窗口以输入观察行计数，要求完整窗口；有 `segment_id`
时按股票、日期和片段分组，不跨片段滚动。除零和不可表示结果保持缺失。
白名单外的函数调用、属性访问、负 lag 等不允许使用。

自适应提议只接收训练成绩；候选生成结束后才在验证窗选一次。
有基准列时，先在每日截面对候选排名回归基准排名与常数项，再计算残差的 RankIC；
这衡量相对于所选基准的增量排序关联，不是交易收益。
报告保存全部候选、未定义项、选中公式及 `validation_score`。
全体未定义时不产生胜者；合法负分仍参与选择。

`apply-formula` 读取冻结的搜索结果，在晚于选择截止的 `output_window` 发布
`discovered_factor`。`input_window` 可以包含更早的滚动预热历史，输出只保留请求窗口。
搜索和多轮 agent 调整都属于开发过程，验证成绩不能当作未经选型的测试证据。

## 不确定性估计

这里使用冻结的绝对残差 conformal 半径，不改变预测器。
校准输入为原生 PRED 与成熟标签，二者的目标列和单位须一致；
`block=row` 使用逐行残差，`block=timestamp` 使用每个截面的最大绝对残差。
对 n 个校准块取第 `ceil((n+1) × (1-alpha))` 个顺序统计量，样本不足时明确失败。
得到的区间是 `prediction_score ± radius`。

应用窗口必须晚于校准截止，且新 PRED 的模型事实与校准时一致。
`conservative_score = sign(score) × max(abs(score)-radius-cost, 0)`。
`cost` 与标签使用相同单位；只有标签为 `decimal_return` 时，0.001 才表示 10 bps。
标准化分数的成本不能直接按收益率解释。
输出为 factor/feature 体系可读取的原生数据集，应用不读取未来标签。
`evaluate-uncertainty` 另行读取成熟标签，报告配对行数、覆盖率和平均宽度。

理论覆盖依赖校准块与未来块的交换性；金融序列中的时间依赖和分布变化可能破坏该条件。
按 timestamp 分块保留截面关联，但不自动解决跨期依赖。实际覆盖需在随后窗口单独检验。

## 成本感知学习

`fit-cost-model` 用 PyTorch 训练共享资产打分网络及现金分量。
输入是固定股票池的完整、有成熟标签的连续面板；目标单位必须为 `decimal_return`。
`return_end_column` 明确各期经济结束点：同一截面结束点相同，前一期结束点等于
下一期决策时刻，最终结束点不晚于 `cutoff`。不能把重叠的 20 日标签当成逐期资产收益。
未知成熟或资产不齐时失败，不能通过删行拼接不连续时期。特征缺失由公共
`ModelMatrixScaler` 使用训练中位数填补；训练列全空时无法估计并明确失败。

插补与缩放只估计于本次训练域，保存为 `feature_scaler.pkl`，回测冻结复用。
每期以漂移后的持仓为起点，双边换手
`T = sum(abs(w - previous_weight))`，净收益近似为
`(1-cost_bps/10000 × T) × (1+w·r)-1`，目标为负平均净收益加收益方差惩罚。
这是连续权重和线性成本的训练近似，不包含离散股数、T+1、最低佣金或实际容量。
这些限制由 `backtest-cost-model` 的原生日频账户处理；训练目标与实际费用后净值需分别报告。
回测股票池及特征顺序必须一致，窗口晚于训练截止。

## 独立研究 agent

agent 通过 OpenAI `/responses` API 获取严格 JSON Schema 行动，客户端复用已安装的 HTTPX。
`llm` 明确以下六项；示例见 `configs/research-learning/research-agent.json`：

| 配置 | 示例值 | 含义 |
|---|---|---|
| `base_url` | `https://api.openai.com/v1` | API 根地址，客户端追加 `/responses` |
| `model` | `gpt-6-luna` | 本轮使用的模型，由调用者按实际项目权限选择 |
| `credential_env` | `QRP_RESEARCH_API_KEY` | 当前进程中保存密钥的环境变量名，不能填写密钥本身 |
| `max_output_tokens` | `128000` | 单次生成上限，包含内部推理和可见输出，并非要求生成这么多 token |
| `reasoning_effort` | `high` | 可改为 `none`、`low`、`medium`、`high`、`xhigh`、`max`；实际模型必须支持所选档位 |
| `timeout_seconds` | `300` | 单次网络等待超时，与本轮剩余时间取较小值 |

按 2026-11-05 的 [GPT-6 Luna 官方文档](https://developers.openai.com/api/docs/models/gpt-6-luna)，
上述模型最大输出为 128,000 tokens，支持表内六种推理强度。更换模型时须重新核对上限和支持档位。
请求固定使用标准服务档 `service_tier=default`、`text.verbosity=medium`、`store=false`，
不传 `temperature` / `top_p`，不自动重试或切换模型。
`store=false` 关闭响应保存检索功能，不等于零数据留存；每次请求显式发送本轮消息历史。
远程 API 要求 HTTPS；localhost 可用 HTTP。配置中不存密钥，容器不接收该环境变量。
Windows 用户环境变量更新后，已经打开的终端不会自动刷新；从新终端启动，或在当前
PowerShell 执行 `$env:QRP_RESEARCH_API_KEY = [Environment]::GetEnvironmentVariable('QRP_RESEARCH_API_KEY', 'User')`
后再运行。该命令只刷新当前进程，不打印密钥。
服务必须支持 Responses 和严格结构化输出；HTTP 失败、未完成响应、拒绝回答和非法行动均明确失败。
旧 Chat Completions 配置需补上推理强度和超时，并改用支持 Responses 的服务。

任务固定 `protocol.label`、训练/验证/测试窗、`costs` 和评价 `metric`。
工具模板可用 `protocol:label`、`protocol:train_window`、`protocol:costs.cost_bps`
引用这些事实，冲突直接拒绝。`tools` 声明允许的操作、基础配置、`vary` 的可选值和
`metrics` 的数值提取位置。模型只能选这些值，不能改窗口、截止、输出路径或固定成本。
行动的 `changes` 用参数路径和从 0 开始的候选编号选择，例如
`[{"path":"columns","choice":0}]` 选择该工具 `vary.columns` 的第一个候选；未列出的参数保持基础配置。
`bindings` 将参数路径映射到既有产物的相对成员；行动的 `inputs` 选择成功的证据 ID，
例如 `[{"path":"search","evidence":"e0001"}]`。路径不得重复。严格行动格式的所有字段都需返回，
未使用的列表为 `[]`，未使用的可空字段为 `null`。旧行动中的字典式 `changes` / `inputs`
已改为上述列表；候选值仍只保存于工具配置，不重复编码。输出位置始终由 runner 分配。

`budget` 显式限定 API 请求数、实验次数和 `wall_seconds`，到期不启动下一实验。
可信 CLI 和生成代码进程使用剩余时间作为超时，超时清理本次拥有的进程与容器。
HTTPX 的 timeout 限制连接和 I/O 等待，并非整个 HTTP 请求的绝对时钟；慢响应及清理
可能超过任务墙钟预算，返回后会再次检查截止。没有无限重试。
`max_output_tokens` 是单次响应上限，请求数上限不是精确费用上限。

`run_tool` 在宿主调用已声明的可信 QRP CLI。`run_python` 只在隔离容器内执行，
无 Docker 或容器失败时明确记录失败，绝不退回宿主执行生成代码。
输入只读挂载到 `/inputs/<alias>`，结果写到 `/outputs`；生成代码可调用镜像中的
QRP 读取器和数值库。sandbox 输入必须是开发数据集：所有行早于 test_start，
任何非空标签都有成熟时点且早于 test_start。含测试期的总目录不能直接挂载。

发给模型的内容限问题、固定协议、允许参数、输入列角色/单位/频率/行数以及显式
选定的数值汇总；不发送原始行、宿主路径或进程日志。汇总定义由调用者检查，
数值类型约束不等于自动隐私证明。生成代码产物只能通过允许的数据读取入口交给
可信操作，不能作为宿主 pickle、模型或策略加载。

每次行动保存为 `action-*.json`；每个 `e0001` 等实验目录保留配置、试验记录、
产物及本地私有日志。`report.json` 区分完成、预算耗尽及失败，结论引用成功证据 ID。
报告中的自然语言来自外部模型，需要研究者结合真实产物复核；失败、未定义或负结果
不能被文字结论改写为成功。最终测试由调用者在迭代结束后另行运行，不反馈给这轮 agent。

## 服务器隔离环境

`Dockerfile.research` 使用 [conda-forge 的 Miniforge 镜像](https://github.com/conda-forge/miniforge-images)，
以仓库 `environment.yml` 创建独立 Conda 环境。`.dockerignore` 仅放行源码、环境定义
和 Dockerfile，不复制 data、.run、SSH 信息或 API 凭据。源码注册到环境 site-packages，
因此 `python -I` 也可正常导入；不依赖可写工作目录或 `PYTHONPATH`。

在已经具备 Docker 的服务器中构建和验证：

```sh
docker build -f Dockerfile.research -t qrp-research:local .
docker run --rm --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --user 1000:1000 --tmpfs /tmp qrp-research:local python -I -c "import quant_research_platform, torch, gymnasium, stable_baselines3"
```

正式使用时通过 `--build-arg BASE_IMAGE=<已核验镜像摘要>` 固定基础镜像。
运行配置明确 CPU、内存、PID 上限和可选 `gpus`；容器 rootfs 只读、禁网、非 root，
不挂载 Docker socket 或其他宿主目录。Linux 非 root 调用者配置自己的 UID:GID，
以便写入输出；root 调用者仅给本次新建结果目录设置配置的所有者。
镜像的默认 UID 只是示例，runner 的 `sandbox.user` 决定实际 UID。
GPU 另需服务器驱动、运行时和镜像内匹配的 PyTorch 构建，`gpus` 不会自动补装它们。

宿主 agent 使用独立项目 Python 运行。多进程/数值线程安排由调用方在新进程导入
数值库前设置，例如 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`；示例资源值不是主机限制。
模型规模、并行度、训练预算和磁盘空间由真实实验决定。

验证分为小型合成数据的真实算法与原生持久化、控制器/传输测试和实际服务联调。
本地控制器测试不能替代真实 LLM API；构建文件和命令检查不能证明服务器已具备 Docker
或嵌套容器权限。服务器隔离、GPU、真实 API 和大规模研究结果必须在目标环境单独验收。
