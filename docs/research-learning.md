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
| `inventory` | `kind=hft`；每只股票的动作映射为 `maximum_inventory` 的比例 | 本步净值变动除以 episode 初始净值，再减 `risk_aversion × 本步收益²` |
| `portfolio` | `kind=daily` 或 `hft`；股票与现金的非负动作归一化为目标权重 | 同库存任务，以原生账本的净值变化计奖 |

执行短缺采用成交有符号金额减成交股数乘到达中价，再加真实费用；报告单位为 bps。
原 `implementation_shortfall_bps` 继续只计已成交部分；终局另存未完成量、末次已观察中价、
`opportunity_cost_bps` 和两者之和 `total_implementation_shortfall_bps`。未知终局观察保留未定义，
不使用日频收盘价替代，不虚构成交。配置和同母单 maker/taker 对照见[高频执行说明](a-share-hft.md#被动委托信号与执行基准)。
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

## ETF 动态库存基线

`backtest-policy` 的 `strategy="etf_inventory"` 使用同一个 `TradingEnv`、高频账户、
队列、成交与费用模块，不建立子账户或第二个回测器。
[动态 ETF 示例](../configs/research-learning/backtest-policy-etf-inventory.json)同时包含股票 ETF 与黄金 ETF。
`episode.prediction.paths` 选择已保存的滚动向前预测；先训练/验证、之后预测，
后续轮次可使用届时已成熟的历史样本，不额外要求独立最终留出。

默认 `prediction.horizon_seconds=30`、`decision_interval_seconds=3`、
`quote_visibility_delay_ms=1`，并与 PRED 原生目标定义、单位和时钟事实核对。
仅接受未变换的小数未来中价收益，分类方向/概率分数不能冒充收益预测。
策略不读取未来标签。PRED 的 timestamp 已是决策 T，不再加一次可见延迟；
决策估值与成本盘口只取事件时刻不晚于 T−可见延迟的报价，订单仍需经过独立的
`latency_ms`（默认 10ms）并在严格后续合格报价执行，不使用激活后的报价反推 T 的目标。
预测缺行或缺值不产生新信号目标，不当作零收益；持仓保留，原单依原有效期和撤单规则推进，
撤单确认前仍可能成交。完全没有预测的窗口也能保留实际账户与独立收盘恢复。

每只 ETF 初始净资产默认 100 万元，`episode.capital_scale` 默认 1；
总初始资本为 ETF 数量 × 100 万 × scale，**包括开盘底仓市值，不是额外赠送底仓**。
`initial_inventory` 与 `maximum_inventory` 仍须明确给出股数；增加 scale 不自动改数量上限。
实际成交受共享现金、库存/可卖量、买卖单位、盘口档位、参与率、部分成交及队列约束，
资金更多不保证成交更多。股票 ETF 当日买入不可当日卖出，黄金 ETF 沿原 T+0 规则。

每个有预测的决策事件重新计算目标 q。以实际库存 h、当日开盘底仓 b、最大库存 M、
T 时可见中价 p 和未来收益预测 μ，比较以下透明效用：

`p × μ × (q−b) − p × inventory_risk_bps/10000 × (q−b)²/(2M) − C(q−h) − E(b−q)`。

`inventory_parameters.inventory_risk_bps` 默认 0，即不另加库存风险惩罚；例如显式设 10
只是研究假设，不代表已校准或风险受控。候选包含实际库存、开盘底仓、活单完成后库存、
预算边界，以及风险项为正时比例成本近似的四个二次驻点。再用候选实际交易量比较完整成本，
不构造通用优化框架，也不声称含最低佣金时求得全局最优。活单未成交量只影响候选和可用预算，
不能当成实际持仓；撤掉未成交买单不被误计成卖出。

C 默认复用 `PreparedTradeCost.stock_components`，按候选的实际增减数量及 T 时可见盘口
估计费用与不利价差，最低佣金不折成固定一手 bps。E 默认用同一可见状态估计回到底仓的成本，
只是未来退出成本假设，不是未来实际费用。maker 不预先把可能捕获的价差当作保证收益。
已有部分成交订单的最低佣金抵扣未进入候选估计，成本估计可偏保守；真实账本仍按原订单累计收费。
可选 `trade_cost_bps` / `exit_cost_bps` 明确覆盖对应决策成本假设，实际成交费用始终只扣一次。
成本与库存惩罚形成不交易区间，弱预测可能不动也可能恢复底仓，不写死一种行为。

北京时间 14:50 起，账户独立尝试恢复**当日开盘底仓**，不依赖此时是否有预测，
也不继续接受当天新的信号目标。T+1、资金或盘口不足时如实留下未完成及其账本事实。
30 秒仅是预测期限，没有到期检查或强制反向。账本分别保存预测期、决策间隔、目标、在途量、
实际成交时点与持仓轨迹，不把预测期限当作实际持有期。

maker/taker 使用同一目标算法与预测输入；比较时使用相同的显式成本覆盖可冻结决策成本假设，
或披露各自基于报价方式的成本估计。maker 增加原生 `passive` 和 `trades` 配置，
两者使用相同参与率可作为对照，但不强行配平成交量，也不保证队列成交或收盘恢复完成。

## 股票高频全市场组合

`strategy="stock_portfolio"`、`episode.kind="hft"`、`task="portfolio"` 接入同一账户。
[配置示例](../configs/research-learning/backtest-policy-stock-portfolio.json)默认使用低频
`build_simple_target_weights` 的前 20% 等权选股；`portfolio_parameters.top_quantile`
可调整分位，`construction="quantile_float_cap"` 使用同一低频市值加权实现。
输入仍是共同决策 T 上的数值 PRED，默认每 3 秒、预测未来 30 秒；不把预测期限当持有期。
初始净资产为 10 亿元 × `capital_scale`，包含已有持仓，单账户跨日持有且不恢复底仓。
示例中的股票、数量上限和参与率只是可替换假设，不是全市场规模或成交能力结论。

缺预测保留 NaN 目标与旧单。简单组合先保留缺预测持仓、尚未冻结的旧买单资金和
T+1 不可卖部分，再分配剩余预算；数量上限截断后的剩余资金留现金。
真实股数、订单现金冻结、盘口/队列容量、最低佣金、费用与 T+1 始终由原账户决定。
独立 ST / 退整维护沿原账户执行，不依赖新的预测。
没有指数权重时不构造替代 benchmark，也不称指数增强；既有低频指数增强入口不变。

可显式选择 `construction="absolute_risk"`，`risk_limits` 使用 `AbsoluteRiskConfig`：
单票、行业、风格、总波动率和换手上限均默认 `null`，表示未设置，不表示风险已受控。
与相对风险优化复用同一个 SLSQP 核，不需要 benchmark 或日 ADV。
输入可卖股数与 `maximum_inventory` 在求解内变成边界，不在求解后裁剪并假称仍满足风险。
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

## 独立研究 Agent

使用 `qrp agent` 启动或恢复量化研究会话。完整的配置、工具、交互、记忆、冻结评价和报告约定见 [Agent 文档](agent.md)；示例为 `configs/research-learning/agent.json`。

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
