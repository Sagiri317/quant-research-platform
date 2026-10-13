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

复杂选择、窗口、风险、执行及候选用严格 JSON。每个字段只有一处真值，不合并 profile、
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

训练集 train、验证集 validation、测试集 test 是用途。每折训练和验证及最终测试窗独立给出。
拟合的 scaler、rank方向/权重、监督校准只在允许历史估计，之后冻结。
固定规则的当期截面裁剪/中性化/zscore/rank 在完整且当时可用 eligible 截面计算，
不能按物理 chunk 分开，也不能改成历史训练截面。
标签不能插补；目标成熟时钟是实际参与标签中最晚可用时点。预测不等待未来标签。

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
`facts` 和 `trial_id`，完成 trial 在显式记录的路径中。`research.models.persisted.load_model(path)`
只加载可信本地模型并核状态及完成 trial，不触发训练；不得加载不可信 pickle。
PRED 的 `prediction.parquet` 保存键及 `prediction_score` / `prediction_rank_score`，
标签与成熟列在评分/评估时从明确数据输入读取。score/select 各写自己的 JSON；
消费者核本次需要的模型、PRED、选择和校准关系，不要求单个全研究 MODEL 容器或全部 sibling 成员。

正式产物不原地改写；半写目录不能冒充发布成功。新输出拒绝覆盖。
失败和中断保留真实 trial；attempted 不补写成功。显式复用已完成片段必须核实际科学范围，
不接受 staging 或失败片段，也不改写旧 trial。

普通读取支持指定列与窗口，保持顺序及dtype。裸 Parquet 必须有本次必要 role/units/availability依据。
支持列只用于检查，不自动进入模型X；只选flag时仍需真实底层结构列。
日频闭区间的结束日含全天，纳秒时间保精度；Crypto观察覆盖[start,stop)与流量(t−100ms,t]各自保留。
缩输出不缩数学回看、成熟端点或完整截面。

普通读不承诺dev/inode/mtime/ctime/size快照、读中替换探测、全目录SHA、未消费分区footer检查，
也不保证全输入树瞬时一致。前提是产物不可变、同次计算不并发覆盖输入。
真实解码失败、路径约束、成员缺失及实际消费科学错误仍失败。关闭所有所属资源，保留主异常。
来源接纳、传输验收和显式比较各按实际目的保留证据，不恢复全运行审计。

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

测试按责任与真实路径维护。退休DAG/hash/run/快照测试删除，必要科学负例迁至直接API、发布、
原生回读和实际消费路径。控制面mock不替代真实模型/账本往返。
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

模型选择的 `score` 配置明确列出完整 `validation_windows`；`select` 明确
`refit_windows` 与 `refit_cutoff`。最终拟合只消费明确选定的训练和验证窗口并集内的合法范围，
Crypto 不连续窗口之间的空档不补齐。模型、预测、选择记录由实际消费者核验。

因子和模型统计的 `dataset` 直接指向实际数据；native 输入读取自身 `spec.json`，
裸 Parquet 额外给 `input_spec`（与模型输入相同的 DataSpec 字段）。不要求内置 dataset ID。
`evaluation.factors` 明确选择统计列，`primary_label` 指定标签；自定义成熟列通过
`label_availability_columns` 给出标签到可用时间列的映射。PK、资格、成熟及加权统计所需列
由统计 owner 加入读取，未选因子不进入计算。
