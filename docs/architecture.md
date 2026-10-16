# 架构与工程合同

## agent 与计算代码

agent 选择研究问题、实际输入路径、参数、时间窗、操作顺序、新输出位置及重算范围。
代码只执行明确请求的计算，并核验本次科学资格。来源准备、组装、fit、predict、
evaluate/score/select、explain、calibrate、backtest、statistics 和 plot 互相独立。

不存在 DAG/planner、阶段阶梯、自动补上游、源码失效传播、自动复用搜索、latest 指针、
请求 hash 身份、强制研究 run 或整研究 COMPLETE 门槛。相同输入可在新位置重新计算；
旧产物不覆盖、不改写。agent 漏判影响或选错旧结果时，系统不保证自动发现。

## Python 与 CLI

公开 Python 函数留在实际计算 owner；薄 CLI 严格读 JSON 后直接调用。

- `sources.prepare.prepare_source(config)`：指定原始源与新规范表位置
- `markets.prepare.prepare(config)`：指定已准备表或原生源，生成一个市场数据表
- `research.datasets.assembler.assemble(config)`：明确表、因子、标签和 feature_recipe
- `research.models.production.fit(config)`：一次真实拟合及 append-only trial；选择后的最终重新拟合仍用该函数
- `research.predictions.runner.predict(config)`：读取保存模型，按保存列顺序做新预测；`artifacts.outputs.prediction.read_predictions(path)` / `iter_predictions(path)` 只读已有预测
- `research.models.explanations.explain(config)`：明确冻结模型、解释方法及样本，保存独立解释数值
- `research.models.trial_selection.score(config)` / `select(config)`：完整有序候选的成熟验证评分及明确规则选择
- `research.portfolios.production.calibrate(config)` / `backtest(config)`：A 股监督校准与实际账户计算
- `markets.implementations.crypto.portfolio.calibrate_maker(config)` / `backtest_taker(config)` / `backtest_maker(config)`：各自独立数学和事件时钟
- `research.evaluations.runner.evaluate(config)`：显式信号、标签、时间窗和成熟截止评估
- `research.statistics_production.model_statistics(config)`：显式模型与预测集合统计
- `research.statistics_production.factor_statistics(config)`：显式因子与标签统计
- `research.portfolios.diagnostics.portfolio_statistics(config)` / `crypto_statistics(config)`：显式账本和比较集合
- `reports.plots.plot(config)`：已算数值到新图片

独立阶段的 config 使用 JSON 值（包括路径字符串和窗口对象）；底层 Python 存储 API
另可直接接收 Path、DataSpec、TimeWindow。复杂选择、窗口、风险、执行及候选用严格 JSON。每个字段只有一处真值，不合并 profile、
环境变量、CLI override 或旧运行配置。固定默认值记录解析后的实际值。
未知字段、重复 JSON key、非法类型/非有限数、空必要列、重复成员和非法窗口早失败。
完整 parameter_candidates 与 base hyperparameters 不能同时声明。

CLI 非零退出区别非法请求(2)、执行失败(1)、必需输入缺失(3)；数值输出自己的
UNDEFINED/PARTIAL/NOT_APPLICABLE 保留原语义，不统一包装成研究状态机。
数据质量问题只影响必要域，其他独立研究可继续；无法合法计算的结果如实失败，不能造数。

## 科学 owner

来源 owner 接纳真实格式、列投影、单位、原始请求和来源证据。
reader 只解码实际选择成员/列；计算 owner 核公式、PIT、成熟、采样、配对与覆盖。
writer 一次核验本产物的 schema、跨批 PK、覆盖和科学关系，再原子发布。
无需逐层重读同一批输入，但新增 join、域和账本关系必须真实检查。

训练、选型评价与事后评价是窗口的用途，不是样本的永久身份。walk-forward 每折明确
拟合窗口、决策截止、预测窗口及标签成熟截止；早期预测样本在以后标签成熟后可进入拟合。
冻结最终评价另存真实冻结事实，评价窗口这个名称本身不证明未参与研究选择。
拟合的 scaler、rank方向/权重、监督校准只在允许历史估计，之后冻结。
固定规则的当期截面裁剪/中性化/zscore/rank 在完整且当时可用 eligible 截面计算，
不能按物理 chunk 分开，也不能改成历史训练截面。
标签不能插补；派生目标成熟时钟先规范化为真实时间，再取参与标签的最晚可用时点。
任何参与标签的时钟未知，该依赖截面的派生成熟时间也未知；其他合法截面仍可计算。预测不等待未来标签。

G 需要协议要求的完整验证历史；Q 只用决策前成熟历史。测试指标不参与选型。
候选失败/合法未定义不等于缺候选，任何 defined 值包括负值都优于 undefined，
全 undefined 按预声明顺序选择。最终重新拟合保存真实 fit_windows/cutoff，离散窗不填平。
测试评估核模型、选择、参数、预处理和必要校准实际冻结记录；不能靠布尔 is_oos 放行。

科研选择参考在 `research-rosters/`，只用于人工/agent 研究设计，不由运行代码加载为默认值。
模型算法能力、数学默认值与正式合同仍只有代码中的唯一 owner。

## 明确路径与发布

每次输入明确路径。native metadata 只记录实际输入、参数/seed、列序、窗口用途、必要资格、
成员和输出范围，不套跨阶段 envelope，不用软件摘要或 schema hash 查缓存。
实际代码/环境可在本次执行材料中记录一次，dirty标记不能冒充未提交源码的精确内容。

一个模型目录保存 `predictor.pkl` 与 `model.json`；`model.json` 保存实际模型 metadata、
`facts` 和 `trial_id`；trial 是独立实验记录，不是模型加载通行证。`research.models.persisted.load_model(path)`
只加载可信本地模型并核自身完成状态及必要拟合事实，不依赖外部 trial 日志，不触发训练；
不得加载不可信 pickle。
PRED 的 `prediction.parquet` 保存键及 `prediction_score` / `prediction_rank_score`，
标签与成熟列在评分/评估时从明确数据输入读取。score/select 各写自己的 JSON；
预测元数据保存后续消费必需的拟合与信号事实；评价、统计、回测读取已有预测时不加载 predictor。
消费者核实际需要的选型和校准时点，不要求单个全研究 MODEL 容器或全部 sibling 成员。

正式产物不原地改写；半写目录不能冒充发布成功。新输出拒绝覆盖。
失败和中断保留真实 trial；attempted 不补写成功。显式复用已完成片段必须核实际科学范围，
不接受 staging 或失败片段，也不改写旧 trial。

普通读取支持指定文件成员、列、年份集合、日期/时间窗口及有限字段条件，保持顺序及 dtype。
不连续年份按集合筛选；年份与窗口同时声明取交集。日期结束日包括全天，精确时间区间闭合，
有时区时刻规范为 UTC；不按运行机器时区解释无时区字段。裸 Parquet 必须有本次必要 role/units/availability 依据。
已明确声明的输入、模型、评价 market_id 必须相容；不从名字或路径猜市场，也不要求原本缺失的市场元数据。
跨市场迁移目前没有独立声明操作，不能静默把 A 股输入发布成 Crypto 模型或评价事实。
支持列只用于检查，不自动进入模型 X。完整声明 schema 中的 canonical flag 即使被单独选择，
仍自动读取其声明的底层结构列；普通外部列名的前缀本身不建立这些依赖。
日频闭区间的结束日含全天，纳秒时间保精度；Crypto观察覆盖[start,stop)与流量(t−100ms,t]各自保留。
缩输出不缩数学回看、成熟端点或完整截面。
时序准备使用完整声明 PANEL 键/生命周期轴确定真实会话，文件、年份、行筛选排除的报价保持未知，不能压紧交易时钟。
最终只发布选中 PANEL 信号键与输出窗口的交集，ON/ID、普通标签、执行许可和滚动因子采用同一规则。
完整轴只额外读取键和上市生命周期，不用未选价格补值。市场模型保留完整声明市场域：
被选择排除的成员不是已知不合格成员，当前市场收益及下一会话依赖的前日市值权重按未知处理，
已有滚动状态在后续完整窗口恢复；真实休市仍按原有市场生命周期处理。

普通读不承诺dev/inode/mtime/ctime/size快照、读中替换探测、全目录SHA、未消费分区footer检查，
也不保证全输入树瞬时一致。前提是产物不可变、同次计算不并发覆盖输入。
真实解码失败、路径约束、成员缺失及实际消费科学错误仍失败。关闭所有所属资源，保留主异常。
来源接纳、传输验收和显式比较各按实际目的保留证据，不恢复全运行审计。
隐式读取原生数据时，复用已打开 footer 的原始行数核对所选物理分区；缺片或未声明的
非空分区不能静默变成少样本。显式 files 子集不要求凑齐全分区，未选年份不检查。
缺失 part 在同一物理分区内的具体时间未知时，该分区保守拒绝；可显式选择完好成员隔离。
该计数检查不提供同计数内容替换检测，也不增加成员身份、快照或全树扫描。

## 执行、资金与统计

市场会话、滚动窗口、候选计算和账户推进可在一次调用中批量循环，不要求agent逐日逐行调度。
性能优先共享必要读取和批量核；不建设配额/租约/长期缓存/调度平台。
Tardis保全原源集合、顺序、单位、档深和重建状态，临时缓冲在该次实际计算结束及时释放；
未证等价前不裁历史前缀。

保留严格本市场20观察ADV、生命周期和权益精确单位；缺成交价不能产生交易或费用。
普通缺价已有持仓只用同生命周期可信锚点暂估并披露，不造现金。冻结持仓仍占风险与预算。
有限金额不要求比例可定义，零NAV比例保UNDEFINED，费用/PnL直接取账本。
Crypto资金费预测与实际结算分离，真实缺口不补零，独立episode不拼NAV/Sharpe。

统计保固定候选族、完整区间、真实有效count和未定义原因，不删除困难成员求显著。
探索性选型的推断限制以实际研究事实声明。DSR/FDR只覆盖本次声明集合，不代表全部人工探索校正。
Fama–MacBeth精确非零转float可下溢为DEFINED 0，保当前表示口径，不宣称全字段均能分辨。

## 展示与检查

HTTP只读，并绑定项目根；不新增计算服务，查询不暗读raw。
图片直接消费指定正式数值，研究文字由研究者/agent写。
未定义留空、真实断段不连线；负权益不能截断；抽样图不能重算正式指标。
Matplotlib与中文字体只在绘图时加载，缺字体明确报错。

测试按责任与真实路径维护。每组删除或迁移先明确实际保护的规则、必要性及新 owner 的覆盖。
退休 DAG/hash/run/快照测试删除，必要科学负例迁至直接 API、发布、
原生回读和实际消费路径。控制面 mock 不替代真实模型/账本往返。
主键、枚举、整数单位、列序与精确时间坐标按其离散合同比较；金额、收益、统计估计及数值解
按经济量纲、算法误差和既有容差比较，不将独立浮点计算的逐位一致作为通用准入。
同一持久化值的无损往返与稳定数值反例仍须验证；不能用宽泛容差掩盖真实权益、排名或样本变化。
先定向、再格式/lint/全量测试及独立审查，真实研究回放另按完成情况说明。

## 数据准备的显式选择

`prepare` 必填 `market_id, data_id, inputs, time_window, output_root`。
`inputs` 是本次操作实际消费的 data ID 到路径映射；源档案入口额外给
`source` 的实际 `root` 与 `archive_paths`；来源种类由被选择的原生清单解释。
Bybit/OKX normalized 来源另用 `source={root, archives}`，不套用原生事件归档字段。
仅支持对应计算确实消费的 `parameters`、`factor_columns`、`base_panel`。
财报 PIT 的因子选择缩小必需源集合，未消费的参数直接拒绝。

`assemble` 必填 `market_id, dataset_id, inputs, time_window, output_root`；
可明确选择 `factor_columns, label_columns, feature_recipe, base_input_data_id`。
A 股的 feature recipe 不自动训练变换。Crypto 的 `sparse_input_data_ids`
只允许明确选定的衍生品、现货、外部辅助输入，不能把基表或标签改成稀疏。
标签成熟时刻和辅助信息可用时刻保留为真实 context，不进入模型输入 X。

模型选择的 `score` 配置明确列出完整 `validation_windows`；该字段表示本次选型评价窗口，
不永久定义样本身份。`select` 明确 `refit_windows` 与 `refit_cutoff`；最终拟合只消费这些
显式窗口内截至拟合截止已成熟的样本，允许加入之前未使用的新日期，不自动填平离散窗间空档。
`test_window` 仅在声明冻结最终评价时给出。模型、预测、选择记录由实际消费者核验。

因子和模型统计的 `dataset` 直接指向实际数据；native 输入读取自身 `spec.json`，
裸 Parquet 额外给 `input_spec`（与模型输入相同的 DataSpec 字段）。不要求内置 dataset ID。
`evaluation.factors` 明确选择统计列，`primary_label` 指定标签；自定义成熟列优先复用输入 DataSpec 的
`label_availability_columns`，已有显式声明须与输入科学事实一致。PK、资格、成熟及加权统计所需列
由统计 owner 加入读取，未选因子不进入计算。

## 并行与数值验收

支持并行的模型通过已有 `hyperparameters.n_jobs` 控制；`-1` 使用当前有效 CPU 容量，
按物理核、进程亲和与 cgroup 配额解析，不硬编码某台服务器。未指定时保留单任务默认值。
agent 可让独立候选/折分别运行；外层多任务与模型、BLAS/OpenMP、Arrow 内层线程应共同分配，
避免每层各自使用全部核。单账户事件链、市场共同状态和完整日期截面不能按物理年份任意拆开。

预测使用模型实际支持的线程设置，不为浮点累计逐位一致强制随机森林串行。
并行加法可能有舍入差；测试按求和误差界与任务量纲验证，极端抵消不能用普通尺度的相对误差解释。
固定种子不代表不同并行归约必然逐位相同；离散样本、参数、列序和可用时点不得随执行方式改变。

性能比较固定输入、任务量、科学参数和 seed，分别记录内核与端到端墙钟、吞吐及真实扫描量。
不通过少算候选、裁掉回看、放宽有效域或缺失规则来声称提速。没有代表性测量的优化只报告实现，
不报告推测的提速倍数；合成测试通过不等于私有全量研究运行完成。


### 特征选择与缺失值

`assemble` 的 `label_columns=[]` 生成无标签预测输入，不要求标签产物。
`feature_transform` 可选择 `clip`（`none`、`mad`、`quantile_1_99`）、
`industry_neutralize`、`size_neutralize`、`center`、`scale` 与 `impute_missing`。
命名 recipe 只提供默认组合，元数据保存解析后的实际参数。执行顺序固定为裁剪、
中性化、中心化、总体标准差缩放；关闭的步骤不要求其支持列。

辅助规模、行业、资格字段由所选操作推导，仅作为工作列，不擅自加入模型 X。
规模因子不对自己做规模回归。中心化保留原量纲，除标准差后为无量纲。
中心化加缩放沿用常量输出 0 的既定约定；仅缩放遇零方差为未定义，保留缺失和诊断，
不偷偷采用单位缩放。仅中心化的单观测结果为 0。

`impute_missing=True` 保留命名 recipe 的技术零填补；只在截面变换可定义时填普通原始缺失，
`feature_observed_*=0` 记录其不是实际观测。零不总是均值，也不能直接参与实际观测统计。
设为 False 保留 NaN 和指示列；标签始终不填补。输入角色、模型有序 X 与输出选择是不同集合。
观测/结构伴随列由完整声明 schema 解析，再补入物理投影。声明的 observed 列缺失或无效仍拒绝；
外部普通 feature_x 未声明伴随关系时不凭名字索取 feature_observed_x。

`fit` 与 `score` 的 `label_unit` 可从输入 `column_units` 复用；缺声明时才需明确提供，
两处同时声明必须一致。模型保存实际成熟列及最终训练样本的最大成熟时刻，核其不晚于拟合截止。
通用相关和期间回归不为单位未知另设准入；其结果不冒称已知经济单位或风险溢价。
模型 facts 必须保存 input_units，只包含实际使用的有序 X 中已声明单位；空对象表示均未知。
预测在调用估计器前拒绝这些列的显式单位冲突，不自动换算，不重开训练输入。
任一侧未知仍是未知；未消费列的单位、文件路径和 data_id 不形成模型身份绑定。


## 选择计算与数值产物

`prepare.factor_columns` 选择实际计算的因子，`prepare.label_columns` 选择标签值列；
标签的前瞻期限由已有定义推导，对应可用时间列自动输出。省略选择沿用该表的默认集合。
依赖表、列、回看及前瞻由所选计算决定，不为未选标签读满最长前瞻，不为现金比率强求市值表。
输入的 `years`、`time_window`、`files` 与数学窗口取交集；显式裁掉必要历史时结果按真实缺失处理，
不绕过调用者的选择。年分区读取在发现文件前裁剪年份，仍验证实际选择路径。

因子统计可分别选择 RankIC、Fama–MacBeth；组合统计可分别选择绩效、基准、归因、容量、
Sharpe 推断、CSCV 和组间比较。未选分析不读取其专用输入，也不生成占位表。
这些数值结果默认作为探索性研究输出；完整研究是否保留了独立最终评价，由研究设计与真实
决策记录判断，不能凭 `test` 或 `global` 名称自动授予统计资格。

PRED 只保存下游实际需要的 recipe、训练域、成熟/拟合/选择截止及必要标定，不复制 scaler、
特征重要性或依赖包版本。拟合状态属于模型产物。已有预测可直接用于下游；不要求模型文件仍在。

## 共同因子域、分层与持久化评价复用

`evaluate` 的 `sample_scope` 明确选择 `own` 或 `common`，默认 `own`。
共同域/分层扩展限 A 股 cross_sectional、eligible 域中的 factor/feature 信号；预测信号和 Crypto 不使用这套截面分层合同。
`comparison_inputs` 只描述真实输入选择器与其参与共同域的因子/特征列，不用研究昵称证明配置相同。
共同域逐标签核对精确资产/日期、资格、真实 observed 值及各自成熟时点；只在最终共同域内要求标签值一致。
原有特征值不重做中性化，完整焦点截面的信号变换先于共同域掩码；peer 缺日期保留空共同域诊断。

可选 `strata` 明确主标签及行业、规模、流动性或预测年龄分层。
先在焦点 eligible 且主标签成熟的总体定义层，再施加共同域，不能在收缩后的样本重新排名。
缺失层与真实行业字符串分开；可定位的非法数值只使相应分类未知，另存 invalid 计数。
年龄字面闭区间之间的小数未分配不等于真正缺失。
日评价、覆盖和分层由同一个 EVAL writer 发布，正式 selected-reader 与流式消费者共用其实际语义校验。
持久化日表按 timestamp 非递减；同一天可跨批次延续，但主键不能重复。键校验只保留当前日期，避免全历史键集合。
预测年龄读取 median_forecast_age_days_fy1（原列单位 days），须选用该原始 factor 或声明由它生成的 feature，
freshness 选择 90 或 180 天；闭区间为 0–7、8–30、31–60、61–90，
180 天配置另含 91–180。7.5 等区间间隙单列未分配计数，负值、无穷及超范围年龄为非法分类，NaN 才计真正缺失。

`factor_statistics(evaluation_inputs=[...])` 显式消费已发布 EVAL 日表并可切窗，不重读 DATASET、重算 IC 或触发 FMB。
`dataset` 计算路径与持久化输入路径互斥；原有 FMB 仍由实际数据计算 owner 承担。
`stratified_ic` 使用本次声明的 HAC/FDR 参数，常用研究配方为 lag=20、BY、alpha=0.05；它们不是不可改的阶段版本。
校正家族一次覆盖本次明确输入的 scope、信号、层和 IC/RankIC，不能逐层校正后拼接或冒称覆盖全部历史探索。
两份 own/common 归约核焦点实际输入、信号、单位、成熟、窗口与分层事实，不以 evaluation_id/data_id 昵称放行。

## 研究比较的科学绑定

G/Q 模型比较核实际选择组和目标语义，不能把任意季度 STAT 填进 `global_statistics` 当作全局组。
`model_losses.target_domain_id` 在本次已有逐日遍历中记录实际参与统计的成熟目标样本：排序资产键与标签值。
它与资产域、计数共同保证配对差值使用同一目标，不是整个数据集身份、缓存键或后续输入改写探测。
不同路径、data_id、分区与未消费因子可继续独立使用；成熟列名字或截止不同但实际目标域相同不必误拒绝。
已知单位冲突仍拒绝，未知不推断为已知。
组合的组间比较同时核原生 candidate.selection_group 与固定构造事实，不能由候选 ID 前缀授予研究角色；
相同预测或相同现金轨迹本身不是错误。

## 任意组合的可选基准比较

`portfolio_statistics` 与 `crypto_statistics` 省略 `analyses` 时均只计算 `performance`。
其他分析必须显式选择；未选分析不读取其专用输入。组合构造的 `benchmark_id` 只属于构造与持仓归因，
通用比较使用独立的 `comparison_benchmarks`，简单组合、增强组合和 Crypto 均可选多个参考序列。
旧 `benchmark` 分析和日表中的重复宽列由以下长表替代，不保留平行比较入口。

- `benchmark_comparison`：配对区间收益、收益差、财富比与相对风险指标。
- `market_exposure`：完整配对时间轴上的市场回归，显式 `hac_lag`、`confidence`、`minimum_observations`。
- `regime_comparison`：按显式 `threshold` 区分参考市场区间收益，输出描述性统计。
- `control_comparison`：读取明确提供的已有对照收益，不自动重跑回测。

每个参考项声明 `reference_id`、`input`、`representation`、`value_column`。科学事实 `valuation_currency`、
`timestamp_basis`、`return_basis`、`cost_basis` 必须可解析得到，可显式声明或复用输入 DataSpec/manifest.parameters 的事实；冲突拒绝。`endpoint_level` 使用 `timestamp_column`；
`interval_simple_return` 使用 `start_column`、`end_column`，收益单位由数据规格或显式 `return_unit` 给出。
`timestamp_basis` 为 trading_day_open/trading_day_close/utc；`return_basis` 为 price/gross_total_return/net_total_return/strategy_net/constant_cash；
`cost_basis` 为 frictionless/execution_costs/zero_cash；区间收益单位为 decimal 或 percent。
trading_day_open/close 的时间列保存午夜会话日期标签，名称说明估值相位；utc 才保存真实 UTC 时刻，不能混用 ON/ID 事件成熟时刻。
文件、年份、时间及行列筛选复用原输入选择器；混合多个未筛选序列不能当作一条指数。
`comparison_controls` 使用同一描述结构，明确策略净收益与执行成本口径。

两个市场共用 `portfolio_benchmark_intervals`、`portfolio_benchmark_metrics`、
`portfolio_market_exposure`、`portfolio_regime_comparison` 长表，保存 candidate、execution、episode、fold、reference 身份。
区间两端时钟、币种和收益口径必须相容。价格指数和全收益指数分别声明，不暗中替代或换汇。
局部缺点、非正点位及无穷值使依赖区间未知，计划区间和观测计数仍保留；不能填零或丢弃后冒称完整覆盖。
缺某个参考源只影响其比较，已有组合绩效继续保留。

市场回归仅在完整合格配对日轴上计算系数与推断，缺口时保留原因及计数，不把有缺口的日期压紧后套 HAC。
回归一次以本次候选、参考、episode、fold 的截距与 beta 为 BY 校正家族，保存计划/有效假设数。
A 股比较按 252 日年化；Crypto 比较默认 365 日、至少 2 个完整 UTC 日，可由现有 statistics 字段覆盖并保存实际参数。
未提供无风险收益时截距叫 `market_return_intercept`；请求无风险序列但缺失时不能退回零利率，constant_cash 不能冒充无风险输入。
按当期实现的市场收益划分环境属于 `ex_post_descriptive`，预先选阈值也不使它成为当期可交易信号。
这些结果帮助区分市场暴露和相对表现，不单凭正收益或显著截距证明因子因果有效。
对照序列只有收益时只作描述比较，不能声称已证明股票池、成本、调仓等研究设计一致。

## 冻结模型解释

Python `research.models.explanations.explain(config)` 与 CLI `qrp explain --config ...` 共享入口。
解释读取现有 MODEL，不拟合、不改 scaler、不重新选择模型。输出独立 EXPLAIN，预测、组合和统计都不依赖它。
`explain` 可选依赖组提供 SHAP；只选模型原生重要性时不需要 SHAP 或输入数据。

必需字段是 `model`、`output`、非空 `methods`。只有 `native_importance` 时不传数据选项。
贡献方法一次选择一个：`affine`、`linear_shap`、`rank`、`native_tree`、`tree_shap_path`、`tree_shap_interventional`；
还可同时选择 `native_importance`、`permutation`。方法须与真实模型匹配，不能用另一种解释静默替代。
有逐行贡献或置换时另传 `inputs`、有限 `window`、`sample_role`、`selection`、`output_space`。
`selection` 明确 `max_rows`、`max_dates`、`root_seed`，可选 `asset_ids`；`max_dates=null` 表示全窗口。
先保留选中时间的完整截面，再抽展示/评分行，rank 不在展示子集重新排名。
解释输入不支持 `row_filters`，资产展示筛选通过 `selection.asset_ids`，避免无意改动 rank 截面。

`output_space` 为 `model_output`、`raw_margin` 或 `calibrated_label`，按真实模型能力使用。
非恒等目标变换的原生树贡献只解释 raw margin，不把它标成收益。
`linear_shap`、`tree_shap_interventional` 必须明确 `background`：inputs、window、max_rows、root_seed。
背景逐行受真实拟合窗口、精确拟合截止和训练资产域约束，但标为历史参考分布，不冒称还原全部训练样本。
抽样使用有界 reservoir；其余方法不接收未使用的背景选项。

贡献表保存 timestamp、asset_id、feature、input_value、contribution；预测表保存基值、真实输出、预测分数、
加和残差与容差及逐行状态。不可表示或强抵消导致解释精度丢失时保留有效预测，贡献标 UNDEFINED 与原因，
不能用巨大项幅度容差伪装解释成功。summary 给出有符号/绝对均值、绝对占比、排序与覆盖，另可按 month/quarter 汇总。
贡献汇总先逐日取截面均值，再对日期等权，不按展示行数跨日期加权。
原生重要性表记录真实方法和单位，不能统一叫 SHAP，更不能自动叫样本外有效性。

置换配置明确 `metrics`（ic/rankic/mse）、成熟 `cutoff`、`groups`、`repeats`、`root_seed`、`axis`。
`within_timestamp` 在日期截面内置换；`temporal_blocks` 明确 episode_column、block_size、step_seconds，
要求单资产连续窗口、max_dates=null，只支持 ic/mse，不跨独立 episode 或缺口重排。
同组列一起置换；原始与置换后的评分键保持一致。rank 的 MSE 只在 calibrated_label 尺度计算。
重复结果与均值/标准差均保存，未成熟标签不参与，测试集结果不反馈为自动调参。

`explanation_summary`、`explanation_waterfall` 只消费 EXPLAIN，支持 PNG/SVG/PDF 与中文字体。
前者选择贡献绝对均值或 native_importance、period 和 top_n；后者精确选择 timestamp、asset_id，
保留有符号其余贡献、存储残差与真实输出。无定义解释展示原因，不画假零。
