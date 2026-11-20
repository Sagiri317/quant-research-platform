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
| `calibrate-amplitude` | `research.predictions.amplitude.calibrate_amplitude` | 逐证券、逐交易日的收益校准参数数据集 |
| `calibrate-uncertainty` | `research.predictions.uncertainty.calibrate_uncertainty` | 冻结残差半径 JSON |
| `apply-uncertainty` | `research.predictions.uncertainty.apply_uncertainty` | 原生区间与保守分数数据集 |
| `evaluate-uncertainty` | `research.predictions.uncertainty.evaluate_uncertainty` | 成熟样本覆盖率与区间宽度 |
| `fit-cost-model` | `research.policies.cost_aware.fit_cost_model` | PyTorch 权重、训练缩放与成本近似说明 |
| `backtest-cost-model` | `research.policies.cost_aware.backtest_cost_model` | 原生日频账本 |
| `agent` | `agents.runner.run_agent` | 每次行动、试验记录、产物与结论报告 |

例如：

```text
qrp train-policy --config configs/research-learning/train-policy-execution.json
qrp backtest-policy --config configs/research-learning/backtest-policy-execution.json
qrp search-factors --config configs/research-learning/search-factors.json
qrp apply-formula --config configs/research-learning/apply-formula.json
qrp agent --config configs/research-learning/agent.json
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
| `inventory` | `kind=hft`；每只股票的动作映射为 `maximum_weights × 当前净值 ÷ 当前可见价格` 的比例；已有超限持仓时以当前持仓为动作上界，允许保持 | 本步净值变动除以 episode 初始净值，再减 `risk_aversion × 本步收益²` |
| `portfolio` | `kind=daily` 或 `hft`；股票与现金的非负动作归一化为目标权重 | 同库存任务，以原生账本的净值变化计奖 |
| `dynamic_execution` | `kind=hft`；透明库存策略给出目标，PPO 只选等待、撤单、挂单或主动成交及缺口数量档 | 原生扣费后净值增量（初始净值的 bps），另扣交易时间内的目标偏离与库存暴露惩罚 |

执行短缺采用成交有符号金额减成交股数乘到达中价，再加真实费用；报告单位为 bps。
原 `implementation_shortfall_bps` 继续只计已成交部分；终局另存未完成量、末次已观察中价、
`opportunity_cost_bps` 和两者之和 `total_implementation_shortfall_bps`。未知终局观察保留未定义，
不使用日频收盘价替代，不虚构成交。配置和同母单 maker/taker 对照见[高频执行说明](a-share-hft.md#被动委托信号与执行基准)。
母单风险惩罚按每次实际成交更新剩余比例，以各段时长占整个执行窗口的比例加权；全部成交后不再计罚。`unfinished_penalty_bps` 乘未完成比例后计入终局奖励，其他任务必须设为零。
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
默认 `sample_role="out_of_sample"` 要求评价窗口晚于拟合截止，期间不更新插补或归一化。
显式 `sample_role="diagnostic"` 可用同一冻结模型回放训练期和验证期，账本保留诊断用途，
不能作为样本外证据。`checkpoint` 默认 `selected`，也可选择本次产物保存的 `final`。
只加载可信本地模型产物。
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

`statistics-portfolio` 可直接读取 `backtest-policy` 的 `ledger/` 和
`backtest-cost-model` 的输出目录。日频绩效、容量、Sharpe/CSCV 推断及显式基准、
对照、市场暴露和行情分段比较使用原生账本，不要求普通选股组合的 `candidate` 配置。
候选身份只由账户配置提供并写入账本资本：高频用可选的 `episode.account.candidate_id`，
省略时保留旧值 `learned_policy`；日频策略和成本模型直接使用已有的
`episode.account.candidate.candidate_id`，不再用 `learned_policy` 或 `cost_aware_allocator` 覆盖。
联合统计的每个候选须有不同身份，重复身份仍报错；旧日频配置无需新增字段，
但重跑后的身份改为其原先声明值，已保存账本不改写。统计不从账户配置
复制选股方法或选择事实。持仓归因仍须有实际匹配的候选元数据，
缺失时明确失败。高频账本按自身 `kind` 使用日终绩效口径，不与日频账本混合统计；
学习账本也不因缺少 `candidate` 而被误判为高频。

## PPO 检验所需记录

模型目录保留选中策略、与其同时冻结的观测归一化及训练插补参数；
最终策略与选中策略的采样步数或 SB3 优化 epoch 计数不同时，另存最终策略和对应归一化。
`policy.json.final_checkpoint` 指明最终文件，二者相同时共用同一份文件。
`actual_timesteps`、`actual_updates` 与验证记录中的 `selected_timesteps`、`selected_updates`
分别记录采样步数与 SB3 累计优化 epoch 计数；`updates` 不是 minibatch 的 `optimizer.step()` 次数。
同一采样步数的优化前、优化后策略可能不同。

SB3 训练日志保存于模型旁的 `.trial/`，验证事实保存在模型目录：

- `progress.csv` 保存策略损失、价值损失、熵、近似 KL、裁剪比例、价值解释度、学习率、优化 epoch 计数及 rollout/验证汇总；最后一次优化也写入。指标含义见 [SB3 Logger](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)。
- `training-env-*.monitor.csv` 保存各训练环境完成的 episode：原始累计奖励、步数、训练 episode 索引、起末净值、费用、经济奖励和惩罚。索引对应配置中的 episode 次序；这是随机探索轨迹，不等同于冻结策略的训练集评价。
- `policy.json.validation.evaluations` 保存每次验证的采样步数、优化 epoch 计数、逐 episode 奖励、长度及经济结果。`episode_metrics.episode_index` 对应 `validation_episodes`，不按不同长度 episode 的结束次序猜身份。

每次完成的验证还由同一个 SB3 logger 写入 `progress.csv` 的 `eval/episode_<索引>/...` 列，
`eval/n_updates` 标明该次优化 epoch 计数；后续训练或模型发布失败时，已写日志仍留在 trial 中。
成功模型中的验证记录是这些已计算事实的随模型快照，不另算一套指标。

`economic_reward` 与 `reward_penalty` 使用该任务的原始奖励单位：动态执行是初始净值的 bps，
库存/组合是初始净值的小数比例，母单执行是到达名义金额的小数比例；二者之差为累计奖励。
母单执行还保存完成率、已成交/未成交量、执行短缺、未成交机会成本及其缺失原因，
使低成交量或未完成母单造成的低短缺奖励可以识别；这些数值直接来自账户环境已有计算。
各任务的奖励不能跨单位直接比较。

过拟合判断优先比较同一冻结 checkpoint 在训练、历史验证和随后样本外窗口的逐日经济表现，
并使用相同费用、账户限制和基线。损失、KL、熵和价值解释度主要诊断优化状态，单独不能证明泛化。
训练奖励上升、验证收益下降是警示；验证还参与早停和选择，独立结论仍依赖随后评价。
多随机种子的独立重复与不同市场阶段检验用于区分研究效果与随机波动，参见
[SB3 评价建议](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)。

新增日志只影响之后启动的训练。已有冻结模型、预测和账本可补算的检验不要求重训；
未保存的历史梯度、逐轮损失或旧 checkpoint 无法事后恢复。重训产生新实验，不能冒充旧运行记录。
不默认保存全部 rollout 张量、每步梯度和每次更新的权重。

## ETF 动态库存基线

ETF预测驱动的下单研究采用两条路径：固定规则直接模拟执行，以及PPO学习执行。
两者共用目标库存、账户、容量、成交与费用实现；maker/taker是执行方式，不是另外两种预测策略。
仓库中的TWAP、VWAP、AC及信号AC是给定母单的执行基准，保留其独立用途，不作为本轮ETF的新增研究路线。

`backtest-policy` 的 `strategy="etf_inventory"` 使用同一个 `TradingEnv`、高频账户、
队列、成交与费用模块，不建立子账户或第二个回测器。
[动态 ETF 示例](../configs/research-learning/backtest-policy-etf-inventory.json)同时包含股票 ETF 与黄金 ETF。
`episode.prediction.paths` 选择已保存的滚动向前预测；先训练/验证、之后预测，
后续轮次可使用届时已成熟的历史样本，不额外要求独立最终留出。

默认 `prediction.horizon_seconds=30`、`decision_interval_seconds=3`、
`quote_visibility_delay_ms=1`，并与 PRED 原生目标定义、单位和时钟事实核对。
仅接受未变换的小数未来中价收益，分类方向/概率分数不能冒充收益预测。
账户回放不读取标签；可选历史校准在独立准备阶段读取当时已成熟的标签。PRED 的 timestamp 已是决策 T，不再加一次可见延迟；
决策估值与成本盘口只取事件时刻不晚于 T−可见延迟的报价，订单仍需经过独立的
`latency_ms`（默认 10ms）并在严格后续合格报价执行，不使用激活后的报价反推 T 的目标。
预测缺行或缺值不产生新信号目标，不当作零收益；持仓保留，原单依原有效期和撤单规则推进，
撤单确认前仍可能成交。完全没有预测的窗口也能保留实际账户与独立收盘恢复。

初始净资产由账户的 `initial_capital` 指定，包含开盘底仓市值。
初始使用资金半仓：`initial_weights` 给出占账户总初始资本的权重，单只为0.5，多只等分总计0.5。
账户按起始已知估值换算份额，复用交易单位向下取整，余款保留为现金；也可用 `initial_inventory` 明确给出份额，二者只能选一。
`maximum_weights` 是各证券占当前账户净值的金额比例上限，ETF 为 1，允许满仓。账户按当时可见价格换算份额，随实际净值和价格变化；直接策略、PPO 目标及其观察量共用这一事实。各证券上限可合计超过 1，最终仍受共享现金约束。初始持仓市值不得超过初始金额上限；价格上涨造成已有持仓超过较低上限时，允许保持或减仓，禁止继续增加。
实际成交受共享现金、库存/可卖量、买卖单位、盘口档位、参与率、部分成交及队列约束，
资金更多不保证成交更多。股票 ETF 当日买入不可当日卖出，黄金 ETF 沿原 T+0 规则。

每个有预测的决策事件重新计算目标 q。以实际库存 h、回测初始底仓 b、当时金额上限换算的份额 M（`maximum_weights × 当前净值 ÷ p`）、
T 时可见中价 p 和未来收益预测 μ，比较以下透明效用：

`p × μ × (q−b) − p × inventory_risk_bps/10000 × (q−b)²/(2M) − C(q−h) − E(b−q)`。

`inventory_parameters.inventory_risk_bps` 默认 0，即不另加库存风险惩罚；例如显式设 10
只是研究假设，不代表已校准或风险受控。候选包含实际库存、初始底仓、
预算边界，以及风险项为正时比例成本近似的四个二次驻点。再用候选实际交易量比较完整成本，
不构造通用优化框架，也不声称含最低佣金时求得全局最优。目标预算使用实际现金总额
（可用现金加订单冻结现金）和扣除 T+1、权益锁定后的实际可卖股数 `physical_sellable`，
由原生账户提供；挂单、在途和替换意图的冻结量不改变策略目标。候选成本始终按目标与
实际持仓的差额计算，部分成交后再使用更新的持仓、现金和可卖量。

C 默认复用 `PreparedTradeCost.stock_components`，按候选的实际增减数量及 T 时可见盘口
估计费用与不利价差，最低佣金不折成固定一手 bps。E 默认用同一可见状态估计回到底仓的成本，
只是未来退出成本假设，不是未来实际费用。maker 不预先把可能捕获的价差当作保证收益。
已有部分成交订单的最低佣金抵扣未进入候选估计，成本估计可偏保守；真实账本仍按原订单累计收费。
可选 `trade_cost_bps` / `exit_cost_bps` 明确覆盖对应决策成本假设，实际成交费用始终只扣一次。
成本与库存惩罚形成不交易区间，弱预测可能不动也可能恢复底仓，不写死一种行为。

北京时间 14:50 起，账户独立尝试恢复**回测初始底仓**，不依赖此时是否有预测，
也不继续接受当天新的信号或 PPO 动作。共享账户切换为主动恢复，旧 maker 先按原撤单延迟等待确认，确认前仍可成交；之后随真实报价继续尝试，PPO 的等待动作不能跳过此步骤。底仓份额跨日固定，不每日按净值重新配成50%，也不把前一日未恢复差额作为新底仓。T+1、资金或盘口不足时如实留下未完成及其账本事实。
30 秒仅是预测期限，没有到期检查或强制反向。账本分别保存预测期、决策间隔、目标、在途量、
实际成交时点与持仓轨迹，不把预测期限当作实际持有期。

maker/taker 使用同一目标算法与预测输入；比较时使用相同的显式成本覆盖可冻结决策成本假设，
或披露各自基于报价方式的成本估计。maker 增加原生 `passive` 和 `trades` 配置，
两者使用相同参与率和对手盘口档数，maker挂出量与taker单次成交量共用同一容量上限，
规则见[高频执行说明](a-share-hft.md#被动委托信号与执行基准)。直接、PPO训练/验证及冻结回放示例均使用10%参与率，
依据见[2026年盘口容量估算](../reports/ETF高频滚动预测报告.md#2026年单次委托容量估算)。这是首轮场景设定，
没有通过收益验证选成最优值；队列成交和收盘恢复仍不保证完成。

## ETF 历史预测幅度校准

执行研究配置的 `prediction_calibration` 默认为 `null`，省略也关闭。
开启时，直接策略和 PPO 共用每天开盘前估计、当天冻结的收益预测：

`校准后收益 μ = a + b × 原预测 s`。

逐证券使用此前 `lookback_sessions` 个明确交易会话内的样本外预测及其真实收益，
做带截距的普通最小二乘回归。建议起点为 20 个会话，与现有滚动节奏一致，并非已验证最优值。
样本不限定为曾成交的信号，也不再按上游研究池的 `eligible` 筛选。
预测必须晚于其模型拟合、选型和早停验证截止；标签须有效、有限，
且实际可用时点严格早于当天北京时间 09:30（UTC 01:30）。未知成熟时间不参与；
当天标签不参与当天估计，后续交易日可使用当时已成熟的历史结果。

会话轴来自显式交易日历，缺报价或缺预测不延长回看。历史日历不足指定长度、有效配对少于两条，
或预测全为常数时，当日参数及校准收益保留缺失并保存原因；不回退原预测或补零。
窗口中部分会话没有有效样本时仍使用固定窗口内的有效配对，保存实际样本数和有样本的会话数。
若整段准备范围没有任何历史预测，无法建立校准来源，准备操作会报错。
缺失校准收益沿既有缺预测规则处理，不产生新信号目标；已有订单和独立日终恢复继续按账户规则执行。
斜率允许负值，不裁剪成正数或统一缩小；实际有效性仍需回放检验。

`prediction_error_scale` 是该历史回归的残差均方根，分母为有效配对数，单位为小数收益。
它衡量历史残差幅度，不是胜率、均值标准误或置信区间；重叠的 30 秒标签不能视为独立样本。
开启校准后，动态执行 PPO 自动增加这一观测，插补与标准化继续只在 PPO 训练域估计。
原始预测保留为 `raw_prediction_score`，账户决策使用校准后的 `prediction_score`，
账本诊断同时保留原预测与历史误差尺度。原 PRED 文件不改写。

独立 `calibrate-amplitude --config <配置.json>` 接收 `inputs`、`predictions` 路径列表、`label`、
`asset_ids`、完整 `sessions` 日期列表、输出 `window`、`lookback_sessions` 和新 `output`。
原生日表保存截距、斜率、误差尺度、历史范围、实际样本数、资格与缺失原因；元数据保存来源和计算参数。
通用 `backtest-policy` / `train-policy` 通过 `episode.prediction.calibration` 指向这份日表。
缺少实际消费日的参数会报错，已计算但不可估计的日则保留缺失；不会悄悄借用前一天参数。

## ETF 预测驱动的 PPO 执行

`task="dynamic_execution"` 把目标库存与执行分开。`inventory_targets` 继续独占目标选择，
`episode.inventory_parameters` 使用上节同一透明效用；动态执行的目标成本采用可见主动侧报价，
不会因为 PPO 当步选择 maker 或 taker 而换目标。14:50 起账户独立尝试恢复回测初始底仓，
不再接收 PPO 的执行方式和数量决策。已有底仓即使因涨价超过较低金额上限也可保持；新增买入仍受当前金额上限约束，不能恢复时如实保留未完成。30 秒是预测期限，没有 30 秒到期回转；股票 ETF 的当日买入
继续受 T+1 约束，黄金 ETF 使用 T+0，恢复失败如实保留库存。

阶段接口沿用现有原生操作，不绑定数据集名称：

| 阶段 | 显式输入与参数 | 输出 |
|---|---|---|
| 上游预测 | 调用者选择滚动模型发布的 `prediction.paths`；声明预测期限、3 秒决策及报价可见延迟 | 原生 PRED；不由 PPO 重新训练预测模型 |
| 目标与执行环境 | `TradingEnv(episodes)`；episode 明确资产顺序、特征列、窗口、原始盘口、逐笔、日频账户事实、费用和交易参数 | `PolicyDecision`；动作经 `PolicyExecution(targets,modes,strategy_targets)` 进入原 `inventory_batches` |
| PPO 训练与历史验证 | `train_policy(config)`；独立 `episodes`、`validation_episodes`、`fit_cutoff`、PPO 参数、预算与 `early_stopping` | 最佳 `policy.zip`、配对 `normalization.pkl`、训练插补 `feature_scaler.pkl`、`policy.json` |
| 冻结回放 | `backtest_policy(config)`；明确 `policy`、新的 `episode` 和 `output` | 原生 `ledger/` 与 `summary.json` |

PRED 只检查本次使用的数值收益、单位与时钟定义，以及消费行晚于其模型拟合/选型截止的新关系。
账户回放不消费标签，不以 `data_id` 字面相等、整表范围或未使用字段准入。外部调用者可以换路径、
证券、窗口和数量参数；同一网络的资产与特征顺序仍须与训练一致。

每只 ETF 动作为 `MultiDiscrete(8)`：0 等待；1 撤单；2/3/4 以 maker 执行当前库存到策略目标
缺口的 25%/50%/100%；5/6/7 以 taker 执行相同比例。这些比例作用于目标缺口，不作用于盘口容量；
账户再按两种方式共用的对手盘容量限制本次数量及在途冻结，并应用整手、现金和可卖量约束。
小于一手的请求不虚构成交。等待保留原单；撤单保留确认前的冻结与可能成交；切换到 taker
须先完成原 maker 撤单，随后才在延迟已满的真实报价尝试一次 IOC。未成交量不会自动变成新的
主动单。maker 的部分成交沿原订单累计计费，撤后重下与主动 IOC 各自属于新订单。

状态复用所选原始盘口和已观察订单流，加上当前预测、实际持仓、可卖量、策略目标、
在途与待替换目标、原单/替换单方式、活单剩余量/队前量/订单龄/撤单余时、可用与冻结现金、窗口余时和交易时钟。
市场特征及估值只取 `T−quote_visibility_delay_ms` 之前的可见报价；本账户已经发生的订单事件
按原事件时钟更新，不把在途量当库存。没有预测时不补零收益，PPO 可等待或撤销旧单，
不提交新的信号订单。报价与本账户事件均未另加不存在的接收时间数据。

奖励单位为初始净值的 bps。经济项为 `10000 × (NAV_after−NAV_before)/NAV_initial`，
实际价差、部分成交、费用、持有损益和跨日权益均来自同一个原生账户。
`target_deviation_bps` 乘各资产 `((实际股数−策略目标)×估值/初始净值)²` 的交易时间积分；
`risk_aversion` 乘各资产实际市值占初始净值比例平方的同一积分。
时间积分采用北京时间 09:30—11:30、13:00—15:00 的共同观察时钟，后段覆盖收盘竞价时段，
以 14400 秒为一单位。只沿原生账户已证明的会话推进，排除午休、隔夜及非交易日；
窗口尾部不会延长到最后实际会话之后，同日盘中截止仍积分至给定时刻。
缺失目标不施加偏离惩罚。不重复扣成交费用，也不把捕获价差、订单数、撤单或成交量另计奖励。
账本 `inventory_target` 保存透明策略的完整目标，`execution_destination` 保存本次数量档的提交目标，
两者不会因部分执行而混淆。
`summary.json` 分别保存累计 `economic_reward_bps` 与 `reward_bps`，账户绩效始终从账本计算。

历史验证必须完全晚于训练、结束不晚于 `fit_cutoff`。特征中位数仅从训练 episode 的真实决策
拟合；每次验证同步当时训练观测统计，但验证自身不更新归一化。每个声明的验证 episode
都完整回放一次，以 episode 奖励总和的等权均值选最佳 checkpoint，同时保存该时点的训练
归一化。`early_stopping={eval_freq,patience,min_evals}` 使用 SB3 的 `EvalCallback` 和
`StopTrainingOnNoModelImprovement`：经过最少评估次数后，连续无提升次数**超过** patience
才结束整个 `learn`。`eval_freq` 按总环境步数折算，末次梯度更新也评价；
已早停且权重及归一化未再更新时不重复回放。`policy.json.validation`
保存评估成绩、选中步数与是否早停，`training_decisions` 保存实际首末决策和数量。
`ppo.target_kl` 只限制单轮策略更新，不代替整段训练早停。

本轮对 510300、510500、159915、518880 分别使用模型预测。每次回放只指定一只证券、一个预测来源和一个账户。示例以 518880 为例，初始资金 100 万元，初始半仓，14:50 起尽量恢复原有份额。调用者提供本次窗口实际发布的预测路径。[ETF 账户配置](../configs/high-frequency/backtest-policy-etf.json)

`backtest-policy` 的 `immediate` 按预测与成交成本选择目标并尝试主动成交。`hold` 从相同的份额和现金开始持有。两次回放分别写入明确指定的新目录，再由 `analyze-inventory` 比较两本账的净值差、费用和成交情况。PPO 训练或历史预测幅度校准各有独立入口，可按研究需要调用。

示例依赖服务器的结构化账户事实 `data/interim/etf-execution/*`，包括真实上市/市场/交易状态、
权益与 ETF 费用；这些事实必须先由已有结构化来源准备。股票费率和缺失上市/停牌信息
不能当作 ETF 事实或正常状态。ETF 高频数据与逐行产物保留服务器；本地测试使用独立合成
价格、成交和费用手算，不把测试结果冒称实际市场收益。

实现依据为 [PPO 原论文](https://arxiv.org/abs/1707.06347)、
[SB3 PPO 参数](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、
[历史评价与早停回调](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)、
[VecNormalize](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize)
及 [Gymnasium Env API](https://gymnasium.farama.org/api/env/)。本项目的市场环境与奖励是执行研究
适配，论文基准结果不构成本项目策略收益证据。

## 股票高频全市场组合

`strategy="stock_portfolio"`、`episode.kind="hft"`、`task="portfolio"` 接入同一账户。
[配置示例](../configs/research-learning/backtest-policy-stock-portfolio.json)默认使用低频
`build_simple_target_weights` 的前 20% 等权选股；`portfolio_parameters.top_quantile`
可调整分位，`construction="quantile_float_cap"` 使用同一低频市值加权实现。
输入仍是共同决策 T 上的数值 PRED，默认每 3 秒、预测未来 30 秒；不把预测期限当持有期。
示例初始净资产为 10 亿元，由账户参数指定，包含已有持仓，单账户跨日持有且不恢复底仓。
示例中的股票、数量上限和参与率只是可替换假设，不是全市场规模或成交能力结论。

缺预测保留 NaN 目标与旧单。简单组合先保留缺预测持仓、尚未冻结的旧买单资金和
T+1 不可卖部分，再分配剩余预算；数量上限截断后的剩余资金留现金。
真实股数、订单现金冻结、盘口/队列容量、最低佣金、费用与 T+1 始终由原账户决定。
独立 ST / 退整维护沿原账户执行，不依赖新的预测。
没有指数权重时不构造替代 benchmark，也不称指数增强；既有低频指数增强入口不变。

可显式选择 `construction="absolute_risk"`，`risk_limits` 使用 `AbsoluteRiskConfig`：
单票、行业、风格、总波动率和换手上限均默认 `null`，表示未设置，不表示风险已受控。
与相对风险优化复用同一个优化器，不需要 benchmark 或日 ADV，
算法见 [A 股组合优化器](a-share.md#520-组合优化器)。
输入可卖股数与账户按 `maximum_weights` 派生的当前份额上限在求解内变成边界，不在求解后裁剪并假称仍满足风险。
总波动率沿风险模型的年化单位，只作独立约束，不与 30 秒 alpha 直接相加。
返回的总波动率不是 tracking error；未提供完整风险事实时诊断保持未定义，
只有依赖缺失风险的策略计算受阻，已发布的 raw PRED 不会被抹除。
默认决策成本复用原费用模块的比例费用及可见不利价差导数，**不含最低佣金**；
`trade_cost_bps` 可显式替换该凸成本近似，真实账本仍按原规则扣费一次。

市值权重和完整风险可通过可选 `portfolio_context` 源配置接入 Python 与 CLI：

```json
{
  "portfolio_context": {
    "inputs": {
      "daily_risk": "data/daily_risk_pit",
      "daily_returns": "data/daily_returns_pit"
    },
    "risk_exposure_input": {
      "data_id": "daily_risk",
      "available_at_column": "risk_available_at"
    },
    "daily_return_input": {"data_id": "daily_returns"},
    "stock_pool": {
      "stock_master": "data/stock_master_pit",
      "status": "data/stock_status_pit",
      "available_at_column": "published_at",
      "trading_calendar": "data/trading_calendar"
    }
  }
}
```

`stock_pool` 复用共同采样的同一配置与模块。日收益源必须是完整研究市场，不能裁成持仓；
其 timestamp 表示收益结束的日会话，listing_date 是该终点的上市生命，包含
period_return、is_trading、实际区间起点 return_start_at 和真实成熟时刻 return_available_at。
`daily_return_input` 可用 column、start_at_column、available_at_column 映射已有列；
已有标签成熟列声明会直接复用，无需复制时间字段。不能把日日期午夜当收益已可知。
缺失起点、成熟时刻或同生命证明时，只令相应收益不可用于拟合。

每个历史起点使用共享 PIT 暴露和股票池资格，先冻结完整截面的预处理，再屏蔽未成熟或跨生命收益；
收盘起点不要求处于连续竞价，但仍遵守真实交易日、ST/PT/退整和 IPO 资格。
日历缺收益行仍推进该会话的年龄；明确休市不增加该证券的特质风险会话数。
普通新日仅追加，过去收益晚到才重放模型当前保留的真实会话，复用已冻结的设计，
不重放无关全历史、不拿日频方差缩放成 30 秒方差。模型定义继承风险源或显式 risk_model，保存到账本配置。
原始前沿读取、标准化、WLS、协方差及特质风险继续分别复用现有模块，没有第二套风险引擎。

默认等权不打开这些输入；市值组合不打开日收益源。首次有需要的有效预测才建立 provider。
若历史不足或所需风险事实未知，该次依赖风险的新目标为空，原账户、旧单及独立退出继续，
不把未知风险填零，也不吞掉结构、数值或求解错误。

Python 仍可用 `backtest_policy(config, portfolio_context=provider)` 注入已有实现；
不可同时声明 native 源配置和另一个 provider。
`provider(decision)` 返回 `as_of`、按 asset_id 索引的 `assets` DataFrame，
以及可选 `factor_covariance`、`industry_columns`、`style_columns`。
市值列为 `float_market_cap`，个股风险列为 `specific_variance`；
可选 `trade_capacity_weight` 表示本次剩余容量，不能重复拿日 ADV 当作每 3 秒新容量。
`as_of ≤ T` 仅检查快照时点，不能证明历史无前视：调用方须用
`prepare_asof_risk_sessions` 的前一日且已发布、同生命周期暴露，沿完整日会话调用
`FactorRiskState.observe_return_session`、`current_exposures`、`estimate`；不得以持仓小截面
拟合全市场风险，也不得接事后 attribution 的未来端点。
账本保存构造方式和配置，summary 保留最后一次优化目标的风险诊断。
约束针对所求目标，不能保证部分成交或继续执行的旧单形成的实际持仓已满足；
实际路径风险仍需按账本另行核验，不作零风险或已达标承诺。

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
搜索和多轮 Agent 调整都属于假设或参数选择过程，验证成绩不能当作未经选型的测试证据。

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
校准和评价以已发布的预测或区间为样本；标签表只提供目标值和成熟时点，不按其中的 `eligible` 再筛一次。

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
`model.json.learning_curve` 保存每次更新前的全训练样本目标值；最终净收益与换手单独保存。
原生回测账本保留实际使用的完整 episode、模型来源与设备，支持按同一账户规则复核。
每期以漂移后的持仓为起点，双边换手
`T = sum(abs(w - previous_weight))`，净收益近似为
`(1-cost_bps/10000 × T) × (1+w·r)-1`，目标为负平均净收益加收益方差惩罚。
这是连续权重和线性成本的训练近似，不包含离散股数、T+1、最低佣金或实际容量。
这些限制由 `backtest-cost-model` 的原生日频账户处理；训练目标与实际费用后净值需分别报告。
回测股票池及特征顺序必须一致，窗口晚于训练截止。

## 独立研究 Agent

使用 `qrp agent` 启动或恢复量化研究会话。完整的配置、工具、交互、记忆和报告约定见 [Agent 文档](agent.md)；示例为 `configs/research-learning/agent.json`。

## 服务器隔离环境

本节用于显式配置的容器 `python` 工具；默认 `qrp agent` 的 `terminal` 在宿主机执行，不受这些容器限制，见 [Agent 权限边界](agent.md#本地命令网络与复杂文件)。

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
