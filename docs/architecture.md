# 架构与工程合同

## agent 与计算代码

agent 选择研究问题、实际输入路径、参数、时间窗、操作顺序、新输出位置及重算范围。
代码只执行明确请求的计算，并核验本次科学资格。来源准备、组装、fit、predict、
evaluate/score/select、calibrate、backtest、statistics 和 plot 互相独立。

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
支持列只用于检查，不自动进入模型 X。完整声明 schema 中的 canonical flag 即使被单独选择，
仍自动读取其声明的底层结构列；普通外部列名的前缀本身不建立这些依赖。
日频闭区间的结束日含全天，纳秒时间保精度；Crypto观察覆盖[start,stop)与流量(t−100ms,t]各自保留。
缩输出不缩数学回看、成熟端点或完整截面。

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
