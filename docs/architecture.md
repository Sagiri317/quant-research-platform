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

正式研究默认使用声明训练窗口内的全部合法样本，A 股与 Crypto 一致：先按真实可用时点、
标签成熟条件及所选输入的缺失规则确定资格，不按机器内存、物理批次或模型拟合耗时自动抽样。
`ModelSpec.max_train_rows` 默认 `null`，示例研究配置保持不限制；性能优化不得暗中改变训练样本。
只有调用者明确要求改变样本范围时才设置正整数上限，它属于模型科学配置而非运行资源参数：
按时间倒序保留不超过上限的最近完整 timestamp 组，不随机抽行、不切开同一股票截面或多币时点，
也不在最新一组放不下时改用更早样本。训练窗口、样本权重衰减与行数上限是不同定义。
拟合按声明窗口逐窗读取，不扫描窗口间隙；窗口按实际时间边界有序且互不重叠，
纯日期结束界包含当日全天。精确时间/资产主键由 reader 唯一核验；更宽的源主键
仍须由拟合 owner 核验时间/资产轴唯一性。

训练、选型评价与事后评价是窗口的用途，不是样本的永久身份。walk-forward 每折明确
拟合窗口、决策截止、预测窗口及标签成熟截止；早期预测样本在以后标签成熟后可进入拟合。
不登记留出集、查看历史或最终评价资格；评价窗口的名称本身不证明未参与研究选择。
拟合的 scaler、rank方向/权重、监督校准只在允许历史估计，之后冻结。
固定规则的当期截面裁剪/中性化/zscore/rank 在完整且当时可用 eligible 截面计算，
不能按物理 chunk 分开，也不能改成历史训练截面。
标签不能插补；派生目标成熟时钟先规范化为真实时间，再取参与标签的最晚可用时点。
任何参与标签的时钟未知，该依赖截面的派生成熟时间也未知；其他合法截面仍可计算。预测不等待未来标签。
A 股当前 wd 价格收益标签保留固定会话的经济开盘端点；复权调整来源依赖当日收盘，
普通 forward-return 的 `available_at` 因此为退出会话 Asia/Shanghai 15:00（现代 UTC 07:00）。
ON/ID 的 pre-entry、ID、ON/endpoint 分别在 t+1、t+h、t+h+1 会话收盘达到可用门槛。
经济开盘事件仍为 09:30；此 EOD 日频研究约定不等同于已核实供应商实际发布时点。

G 使用所声明的完整验证历史；Q 只用决策前成熟历史。选型只消费本次明确提供的验证指标。
候选失败/合法未定义不等于缺候选，任何 defined 值包括负值都优于 undefined，
全 undefined 不产生胜者；只有定义的指标相等时使用预声明顺序。最终重新拟合保存真实 fit_windows/cutoff，离散窗不填平。
预测、评估和回测核实际拟合截止、标签成熟及必要校准时点，不要求预先登记未来窗口。

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
标签与成熟列在评分/评估时从明确数据输入读取，不重开 PRED 的历史特征输入。
EVAL 按当前输入实际键与保存预测配对，使用当前资格与标签成熟域；不以历史模型输入重算预测资格，
缺预测或缺成熟标签的行不计入对应配对样本，空共同域保留未定义成绩。显式请求的因子/特征仍须提供。
score 按实际配对键记录当前成熟标签域、缺预测/缺标签行数及共同样本数；空共同域保留未定义成绩。
score/select 各写自己的 JSON；
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

原生 Binance/Tardis 来源同样不以 SHA、状态或请求描述作为普通读取准入；
manifest 摘要只描述输入，前缀游标关闭不触发未消费数据解码或哈希。
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
通用序列图在同一坐标轴共享显示缩放与单位，全空序列仍保留真实日期或类别轴，
缺失说明合并展示，不画假零数据。
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
原生 manifest 的 `inputs` 保留本次输入路径及显式 `files/years/time_window/row_filters/columns`
等选择；省略的选择器不额外限域。`parameters.input_columns` 记录实际消费列（含必要依赖），
输入级时间窗口与总体 `time_window` 取交集，不建立下游身份绑定或自动重算。
A 股的 feature recipe 不自动训练变换；未标准化的派生特征保留实际选中源列的单位，
源单位未知则继续未知。Crypto 的 `sparse_input_data_ids` 可明确选择非基表输入，
包括标签；缺失标签保留 NULL，依赖标签的训练或评价各自筛选合法样本，基表仍须完整。
标签成熟时刻和辅助信息可用时刻保留为真实 context，不进入模型输入 X。

模型选择的 `score` 配置明确列出完整 `validation_windows`；该字段表示本次选型评价窗口，
不永久定义样本身份；标签只读取所选离散窗与本次评分窗的交集，不检验未消费空档。`select` 明确 `refit_windows` 与 `refit_cutoff`；最终拟合只消费这些
显式窗口内截至拟合截止已成熟的样本，允许加入之前未使用的新日期，不自动填平离散窗间空档。
模型与选型不声明 `test_window`。预测直接使用调用者选择的未来窗口，包含开放结束窗口；
实际预测开始必须晚于拟合截止，后续评价、回测不重新接纳模型的训练域。

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
`size_column` 默认 `log_total_market_cap`，可显式选择 `log_float_market_cap`；
依赖投影、计算和持久化语义均使用同一选择，关闭规模中性化不要求辅助规模列。
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

模型统计的 `model_losses` 分析同时保存逐日预测描述表，字段与单位见
[模型预测描述指标](data.md#模型预测描述指标)。Head 20% 先由全部有限预测固定成员并纳入阈值并列，
再核标签成熟；分组评价也先按预测分组。最高组没有可评价标签时保留零观察组，
不把下一组替作最高组计算 spread。描述指标不依赖 IC 是否有定义。

多重检验主方案为 BY，BH 为辅助；模型统计 bootstrap 默认 2000 次，显式 repetitions 覆盖原值并存入元数据。
HAC、bootstrap、模型联合检验、FMB 和分层推断不把已知缺失日期压紧：
实际输入日期轴上缺少有效统计量时推断记 `incomplete_session_clock`，有效值的描述均值仍保留。
完全未出现在输入中的交易日仍依赖上游完整市场日历事实，不能用普通工作日索引冒充交易日历。

## 共同因子域、分层与持久化评价复用

`evaluate` 的 `sample_scope` 明确选择 `own` 或 `common`，默认 `own`。
共同域/分层扩展限 A 股 cross_sectional、eligible 域中的 factor/feature 信号；预测信号和 Crypto 不使用这套截面分层合同。
`comparison_inputs` 只描述真实输入选择器与其参与共同域的因子/特征列，不用研究昵称证明配置相同。
共同域逐标签核对精确资产/日期、资格、真实 observed 值及各自成熟时点；只比较最终共同域内同目标、同单位的标签。
独立浮点标签允许最多 8 个局部 float64 可表示步长的舍入差异；步长取两值较大绝对值向零的相邻间距，
保留标签量纲，覆盖次正规数，双零间距为零。不设宽泛绝对容差；超出舍入边界的真实目标变化仍拒绝。
整数标签在成熟掩码、缺键对齐后仍保留精确整数表示，不能转浮点而吞掉一个整数单位的差异。
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
预定家族中未定义的假设仍参与校正数量，内部以 p=1 占位；对外 p/q 保持 NULL 及原原因。
`defined_hypothesis_count` 仅披露可定义项数，不用它缩小校正家族。
两份 own/common 归约核声明信号、目标、单位、成熟截止及分层事实，并分别核本次选中窗口在其已发布范围内。
统计只归约所选日表，不要求两份 EVAL 的外围窗口或原始内容逐位相同，不生成或核验输入内容摘要。

## 研究比较的科学绑定

G/Q 模型比较核实际选择组和目标语义，不能把任意季度 STAT 填进 `global_statistics` 当作全局组。
模型比较直接消费已有 RankIC 损失，核实际日期、家族、资产键域和观测计数的配对，以及声明的目标名、角色和单位。
调用者负责提供意图相同的目标；不散列或重新验证历史连续标签向量，也不声称已证明数值目标逐项相同。
独立重算产生的标签浮点差异或秩变化不阻断比较，实际损失差值如实进入统计。
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
trading_day_open/close 的时间列保存午夜会话日期标签，名称说明估值相位；utc 才保存真实 UTC 时刻，不能混用 ON/ID 经济事件或标签可用时刻。
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

## 随机种子约定

`model.random_seed` 省略时在配置解析中规范化为 42；显式合法种子原样传入后端，
不自动按模型、折或季度重派生。解释及随机统计的根种子同样在各自配置解析中默认 42；
显式 null 不等于省略。实际解析种子随参数/模型产物保存，不修改进程全局随机状态。
解释日期、行、背景通过既有 `derive_seed` 按用途独立派生；置换按 axis、group、repeat
及 segment 样本键派生，不包含 model_id、worker 身份或执行顺序。
匹配比较要求相同根种子、数据域、样本资格和抽样/置换配置；不能仅凭相同 seed 声称支持集相同。
既有统计派生规则与显式非默认种子保留，不承诺跨环境逐位一致。

## 冻结模型解释

Python `research.models.explanations.explain(config)` 与 CLI `qrp explain --config ...` 共享入口。
解释读取现有 MODEL，不拟合、不改 scaler、不重新选择模型。输出独立 EXPLAIN，预测、组合和统计都不依赖它。
`explain` 可选依赖组提供 SHAP；只选模型原生重要性时不需要 SHAP 或输入数据。

必需字段是 `model`、`output`、非空 `methods`。只有 `native_importance` 时不传数据选项。
贡献方法一次选择一个：`affine`、`linear_shap`、`rank`、`native_tree`、`tree_shap_path`、`tree_shap_interventional`；
还可同时选择 `native_importance`、`permutation`。方法须与真实模型匹配，不能用另一种解释静默替代。
有逐行贡献或置换时另传 `inputs`、有限 `window`、`sample_role`、`selection`、`output_space`。
`selection` 明确 `max_rows`、`max_dates`，可选 `asset_ids`；`root_seed` 省略时为 42，显式值覆盖默认。
`max_dates=null` 表示全窗口。
先保留选中时间的完整截面，再抽展示/评分行，rank 不在展示子集重新排名。
解释输入不支持 `row_filters`，资产展示筛选通过 `selection.asset_ids`，避免无意改动 rank 截面。

`output_space` 为 `model_output`、`raw_margin` 或 `calibrated_label`，按真实模型能力使用。
非恒等目标变换的原生树贡献只解释 raw margin，不把它标成收益。
`linear_shap`、`tree_shap_interventional` 必须明确 `background`：inputs、window、max_rows；
background 的 `root_seed` 省略时为 42，显式值覆盖默认。
背景逐行受真实拟合窗口、精确拟合截止和训练资产域约束，但标为历史参考分布，不冒称还原全部训练样本。
抽样使用有界 reservoir；其余方法不接收未使用的背景选项。

贡献表保存 timestamp、asset_id、feature、input_value、contribution；预测表保存基值、真实输出、预测分数、
加和残差与容差及逐行状态。不可表示或强抵消导致解释精度丢失时保留有效预测，贡献标 UNDEFINED 与原因，
不能用巨大项幅度容差伪装解释成功。summary 给出有符号/绝对均值、绝对占比、排序与覆盖，另可按 month/quarter 汇总。
贡献汇总先逐日取截面均值，再对日期等权，不按展示行数跨日期加权。
原生重要性表记录真实方法和单位，不能统一叫 SHAP，更不能自动叫样本外有效性。

置换配置明确 `metrics`（ic/rankic/mse）、成熟 `cutoff`、`groups`、`repeats`、`axis`；
`root_seed` 省略时为 42，显式值覆盖默认。
`within_timestamp` 在日期截面内置换；`temporal_blocks` 明确 episode_column、block_size、step_seconds，
要求单资产连续窗口、max_dates=null，只支持 ic/mse，不跨独立 episode 或缺口重排。
同组列一起置换；原始与置换后的评分键保持一致。rank 的 MSE 只在 calibrated_label 尺度计算。
模型选型、统计和置换解释复用目标定义/单位比较；定义从训练实际消费的 spec/label_semantics 冻结，随 MODEL→PRED 保存。
未知或不一致只使平方损失未定义，合法 IC/RankIC 继续计算；score 的候选 `mse_reason` 与解释行 `reason` 保存准确原因。
定义只包含源标签、transform、clip 和所选源标签语义，不绑定路径、hash 或无关 spec 字段。
within_timestamp 的 MSE 先逐日算均值，再日期等权；temporal_blocks 保持整条成熟时间序列一个评分 period。
重复结果与均值/标准差均保存，未成熟标签不参与，测试集结果不反馈为自动调参。

`explanation_summary`、`explanation_waterfall` 只消费 EXPLAIN，支持 PNG/SVG/PDF 与中文字体。
前者选择贡献绝对均值或 native_importance、period 和 top_n；后者精确选择 timestamp、asset_id，
保留有符号其余贡献、存储残差与真实输出。无定义解释展示原因，不画假零。

### 输入实例与已有特征复用

`assemble.inputs` 的键是本次输入实例名，可以为同一原生 `data_id` 的不同目录分别命名。
原生 `data_id` 及科学声明保留，不要求为横向拼接改写原件身份。`base_input_data_id`
指本次基表实例，`sparse_input_data_ids` 同样引用实例名。相同输出列仍要求唯一生产者；
公共行情、资格、行业等辅助列通过
输入 `columns` 投影选定权威来源，不能按输入顺序覆盖。唯一可用依赖仍可自动补入。

`feature_columns` 选择已经物化的特征，省略时取输入实际投影中的 FEATURE 列；
可用 `factor_columns=[]` 不选原始因子，这些已有列不重新变换。
已声明的 observed/missing 伴随列、单位、标签成熟事实与原变换说明随复用保留，
再次组装或评价不能把它们当作未经处理的原始值。省略配方及变换参数时，
已有完整目标组输出的因子直接复用，仅为未处理的新增因子生成默认特征；
显式配方/参数要求的新生成特征与复用列同名时明确失败。
组装的基表及每个辅助输入各自只扫描一次选定文件。辅助输入的物理分区可按资产排列，
因此先物化本次已投影、已裁窗的辅助行，再按真实时点稳定排序并对齐完整基表截面；
内存占用与本次选定的辅助输入行相关，不为每个基表块重扫辅助年度文件。
基表仍沿用完整截面迭代器的时序校验。辅助排序不修改时间精度或主键。
strict/sparse 主键覆盖仍在整个请求窗口校验，
包含基表前、间隙及末尾的辅助多余键；失败或关闭时关闭所有输入迭代器。
解析后的输入实例、原生 data ID、实际消费列、filter/context、recipe 和目标列写入参数记录。
当前命名目录仍可提供省略字段的既有便利默认，实际解析结果才是本次计算事实。

### 按消费成员读取解释与校准

`read_explanation(path, members=[...])` 只读取选定成员；省略时保持完整读取。
summary/native 重要性图不依赖未消费的贡献明细，waterfall 读取预测及贡献，仍核对键、
状态与加和关系。请求 predictions 或 contributions 会成对读取二者，不能借投影绕过该关系。
发布时仍验证完整解释产物。账本中的可选最终 alpha 校准仅在实际访问时读取；
净值图和基础绩效不依赖这个无关成员，账本现金、成交与权益勾稽保留。

`temporal_blocks.step_seconds` 允许可精确表示为正整数纳秒的秒数，例如 0.1、1.5 或 1e-9；
连续性按整数纳秒差检查，不把微小缺口截断成完整会话。需要舍入到纳秒的数值仍拒绝。
候选评分中不可表示的辅助 MSE 保存为未定义，合法 IC/RankIC 保留；这不改变
选型指标全未定义时不产生胜者。明确请求的解释 MSE 仍遵守其数值可定义合同。

所有已经带明确时区的时刻都转换为 UTC，包括跨夏令时 offset 的输入；
未声明解释依据的 naive/aware 混合仍拒绝，不猜测机器时区。

### 模型统计的独立分析

`statistics-model.analyses` 可选 `model_losses`、`prediction_hac`、
`prediction_bootstrap`、`model_comparison`、`spa_reality_check`、
`stepm_superior_models`、`model_confidence_set`。省略保持既有整套默认。
仅所选分析要求其专用参数和输入：损失表不要求比较基线或 bootstrap；MCS 不因其它
比较方法而要求基线。模型比较仍采用既定 HAC 与 bootstrap 组合，不改变损失或阈值。
需要随机分析的模型统计配置将省略的 `root_seed` 规范化为 42，显式值覆盖默认；
非随机分析不因此要求随机参数。paired group 比较另外需要自己的 bootstrap 参数，并使用其声明的 seed。

模型与组合 G/Q 比较将声明的 confidence 同时用于 HAC 和 bootstrap 区间。
所请求分析的 lag、bootstrap 与置信度参数先按合法类型及范围验证，不因空数据或退化推断跳过。
仅请求 prediction_hac 时不构建未消费的共同 RankIC、损失域哈希与损失表。

HAC 的 status/reason 与 bootstrap_status/bootstrap_reason 分开保存；某项推断未定义
不会删除另一个合法结果或真实描述均值。零观测不能伪造均值。缺失日期保留在完整输入会话时钟中；
时间序列缺少任何日期的合法观测时，推断为 incomplete_session_clock，不能压紧缺口。
FDR、选型指标及正式研究门槛仍按实际科学合同判定。

省略 sample_scope 的 own 评价与显式 own 采用相同日事实路径。EVAL 生成时负责实际键、
共同域和成熟规则；STAT 复用时来源路径只作出处，祖先 DATASET 不再可用也能归约。
调用者负责选择同一研究问题的 EVAL；STAT 不通过原始资产或标签内容身份替调用者选输入。
所选日表的精确主键、日期顺序、状态和必要计数仍由原有 reader/归约 owner 校验。
已有特征继承的变换说明同样参与必要科学核验与预测年龄分层。

模型统计窗口完全没有输入观测时，明确报告 `statistics_model_observations_empty`，不发布空产物；
已有预测却缺输入覆盖时仍先报告覆盖不一致。有实际日期、但标签未成熟或估计未定义，
继续保留相应未定义结果，不能把“无输入”与“有输入但不可估计”混为一谈。
