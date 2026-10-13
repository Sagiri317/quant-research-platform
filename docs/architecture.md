# 架构与工程合同

本文件定义共用工程不变量；市场经济语义见 a-share.md、crypto.md，数据位置见 data.md。
Git、代码、测试和运行记录提供事实；路线图和历史 PASS 不代替被验收代码的实现证明。

## 唯一责任与调用链

来源/市场事实 → 因子/标签/掩码/上下文 → 研究数据集 → 评估/模型/预测 →
组合/诊断 → 产物/报告。数据流不是导入依赖关系，不允许反向引用制造循环。

| owner | 唯一职责 |
|---|---|
| contracts/ | IDs、schema/dtype/PK、不可变合同、时间窗口、校验；不依赖高层实现 |
| sources/ | 供应商摄取、解码、字段映射、原始/中间层及来源身份 |
| markets/implementations/ | 日历、市场规则、有效行情、掩码、市场执行与成本机制 |
| policy/ | 共享安全规则与结果汇总 |
| profiles/ | 协议、目录、生产器及市场/来源的组合根 |
| catalog/ | 从各 owner 定义派生目录视图，不复制规格 |
| research/ | 因子、标签、研究协议、模型、预测、组合与统计语义 |
| compute/ | 数值核心，不持有市场会话、不写报告 |
| storage/ | 物理 IO、清单、遍历/会话、持久化身份、精确比较与发布原语 |
| runtime/ | 通用时间批次、CPU 容量与操作耗时观测 |
| artifacts/ | 产物校验与持久化 |
| reports/ | 纯报告内容 |
| workflows/ | 唯一目标 DAG 编排、发布和部分真实状态 |
| interfaces/ | 已批准的 CLI/API 路由与适配 |

research 可以依赖 contracts、storage、runtime；市场/data ID仅从各市场的轻量 `ids.py` owner导入，
不得依赖 sources 或 markets/implementations 的执行、来源、生产器实现。
research/protocols 持有配置选择校验、窗口、冻结及统计研究语义，profiles 只组合目录与来源/生产器绑定。
开发到留出统一复用冻结 target/reference：市场 protocol 指定冻结模型/校准及资格，workflow 负责 DAG 和引用发布。
留出创建新的预测、评估、账本、统计、报告与正常运行记录，不重新训练或校准。
允许 artifacts→reports，禁止 reports→artifacts。
可移植模型算法 ID、依赖、输入种类及并行类别由 contracts 定义；研究选择与参数由显式配置提供；
拟合、预测、缩放、排名、重要性和验证由 research 持有。字段名使用 model_algorithm，
算法集合使用 model_algorithm_ids；只保留唯一合同入口，不保留兼容支路。

规格构造及外部 JSON 读取验证合同，内部存储消费已构造的 frozen 规格时只做类型分派；
不承诺检测调用者通过 `object.__setattr__` 绕过冻结机制后伪造的活动对象。
持久化 schema、字段、PK、身份及非法输入仍由边界 owner 严格验证。
业务 metadata 和模型参数继续接受一般 Mapping，并在构造时复制、校验与冻结；
外部 JSON 仍只接受严格 JSON 对象，不增加受信哨兵或绕过独立构造验证的路径。

factor 是信号来源，feature 是模型输入，label 是监督目标，prediction 是模型输出，
signal 是评估/组合使用的排序、权重或交易意图，evaluation 是研究表现，portfolio 是构建与收益路径。
多资产同一时点采用 cross_sectional；单资产 HFT 跨时间采用 single_asset_time_series。
后者以有界在线矩计算 Pearson；RankIC/分组价差在此评估域明确不适用。
评估 signal ID 和来源列不得覆盖 `timestamp`、`asset_id`、`market_id` 结构列；
配置绑定时由 EvaluationSpec 拒绝该命名冲突。

## 研究配置

可编辑 JSON 位于 `configs/research/a_share/` 与 `configs/research/crypto/`。
因子库、评估选择和模型特征是三个不同集合；配置分别声明评估因子、标签、A股主统计标签、
每个模型的输入、算法和参数、使用的组合参数，以及 windows.development/windows.holdout。
research run 只通过显式研究配置选择 profile、scenario 和研究窗口，不接受独立覆盖参数，避免双重选择来源。
配置窗口必须位于协议访问边界内；选择窗口不会授予留出访问权。CORE6、PV25、Crypto baseline 是普通示例配置，
运行入口不隐式选择它们，不动态展开“全部因子”。各阶段复用同一 DAG。
profiles 把确定的配置绑定到现有 catalog/protocol/producer，完整配置保存在 protocol 与 run 记录。
Crypto 的所选标签并列评估，模型消费各自声明的标签；不接纳无科学消费者的 `evaluation.primary_label`。
每个 target 仅绑定真正影响其计算的规范合同；workflow、业务 envelope、trial、writer 和 reader
消费同一合同投影。只改变执行模式或未消费的费率，不改变模型、预测或校准的身份；
真实输入、标签、成熟截止或参数变化仍使相关目标失效。不在各层复制字段排除名单。
A 股因子及 G/Q 预测统计不绑定组合候选列表或主组合选择；组合统计仍绑定这些真实消费配置。
A 股权益事件只作为组合目标的可选原生来源：缺源绑定明确的 ABSENT 身份，
来源出现后仅改变组合及下游统计身份；存在但不合法的来源阻塞消费方，不能降为缺源。
缺源状态在执行或复用前后重新核对；市场会话只读取 DAG 明确绑定的事件实例，不回退扫描目录。
组合市场会话独立于 PANEL 生产模块，未来仅会话逻辑变更不改变未消费它的上游软件身份；
首次职责迁移仍按真实模块内容及组合根更新身份，不伪造旧软件 hash。

Crypto 的训练、评估资产由每个模型的显式范围决定，不由模型名称推断。
执行仍限定 Binance BTC/ETH USDT perpetual。模型各自保留完整预测和有效样本数；
直接比较在同一时点、同一资产主键的共同有效 OOS 样本上重算，不能只对齐日期。
经济账本不因模型样本交集而裁剪。只选择一个模型时，比较表为空，联合检验明确为 UNDEFINED。

配置字段必须有真实消费者。A股沿用季度选择 owner，允许按顺序列出
`parameter_candidates`；Crypto 沿用每个显式模型一组 `hyperparameters` 的 trial，
不同参数实验写成不同模型/研究配置，不接受会被忽略的 `parameter_candidates`。
Crypto 的训练、验证和OOS时段必须由 `research_plan` 声明；A股不接受该字段，时间访问范围来自
`windows`，季度选择规则仍由既有 A股 protocol 管理。

研究配置显式选择一次运行的停止边界，不自动推进下一阶段：

| stage | 请求的终端 |
|---|---|
| `dataset` | 已选数据集；保留模型与组合声明以确定数据规格，不执行模型或组合 |
| `evaluation` | 因子评估；A 股另发布独立因子统计，模型与组合列表为空 |
| `models` | 模型及验证比较；A 股包括 G/Q 预测统计，组合列表为空 |
| `research` | 所选完整研究目标；只在该阶段允许冻结后的 holdout |

报告按请求阶段消费证据；`dataset` 保留的组合声明不构成 PORT/STAT 请求，
未请求部分显示 `UNDEFINED/not_requested`，不报缺失目标。

每个阶段可分别运行多个具名配置并比较其报告。程序保留原有模型推荐、验证 IC 和风险诊断；
用户或获授权 agent 决定一个或多个下一阶段配置，分别调用同一入口。不自动合成模型，
不展开配置笛卡尔积，不人为引入近似并列阈值。已显式选择 `research` 表示授权当前市场
protocol/access_window 内的完整 DAG。Crypto development 的终端包含模型、评估与所选校准产物；
冻结 test 的预测、经济账本和组合统计仅属于合法冻结后的 holdout 图。

下一阶段配置可附 `stage_selection`，字段为 `selected_by`（`user` 或 `agent`）、非空 `reason`、
非空 `sources`；每个来源只有 `run_id`、`target_id`、`instance_identity`。入口核对来源运行完成、
同 profile/scenario/windows、严格更早阶段、development 访问、终端目标以及原生产物完整身份。
这记录选择依据；实际计算输入仍由目标 DAG 的精确依赖身份决定，不能用来源引用冒充输入复用。
完整选择保存在冻结配置/run 和报告中；所选配置产生的新训练仍遵守各自成熟截止。
A 股开发期看过阶段结果后形成的选择注明 `development_stage_selection_exploratory`，
不把选后的开发期联合推断当作预先确定的检验。holdout 不参与选择，仍需匹配已完成完整开发运行。
完整选择记录进入冻结 run 和报告；计算身份只绑定 `sources` 等真实科学依赖与选择带来的推断资格，
不把 `reason` 或 `selected_by` 的说明性变更作为重算原因。来源引用仍按原合同核验。

报告重建从指定 run 记录中的冻结 JSON 重新绑定完整目录，不沿用调用者上一次裁剪的选择。
配置绑定由既有市场 protocol/catalog owner 完成；artifact reader 只消费绑定后的选择与持久化证据。
模型、预测、组合和稳健性分析未被请求时，
相应部分记为 `UNDEFINED / not_requested`，不制造缺产物 blocker。
A股与 Crypto 的评估报告均从同一个验证精确身份的 artifact reader 读取真实评估汇总；
A股因子评估和 `${scenario}_factor_statistics` 不依赖拟合模型；后者独立发布因子诊断及
等权/总市值加权 Fama–MacBeth。模型统计仅持有预测和模型比较，不复制因子统计。

配置文件不存运行产物。研究数据、模型和报告使用 data 分层与 instances 身份目录；
研究运行记录为 `data/runs/research/<run_id>.json`，trial 为 `data/runs/trials/<run_id>/`。

## 计划、身份与持久化

所有来源、数据集、评估、模型、预测、组合和统计由同一计算 DAG 展开依赖。
报告在计算 DAG 之后，从指定不可变 run 的已落盘引用单独发布：先执行 `research run`，
再执行 `report build --run-id`；通用目标构建不接受 report。
报告代码仅汇总计算、统计与来源证据，保留适用范围、推断资格、状态和缺失原因；
`report.json` 与 `report.md` 不自动撰写文字性研究结论。具体研究的分析与结论由研究者
另行撰写并引用确切运行与报告身份；研究图片只读展示已发布证据，不替代正式统计。
业务目录、规格与生产器绑定分别有唯一来源，不能另存并行规划器或手工注册副本。
公开 A 股窄窗口构建在同一规划器内区分目标的输出实例与历史依赖实例；
因子按既有 lookback 定义使用 `DATA_START` 至请求末端的依赖包络，首次冷构建须物化该包络，
随后生产者只扫描一次交易日轴，各输出批次读取必要的历史会话。知识截至与标签成熟末端不扩展。
同时请求 PANEL 与因子时，PANEL 的公开实例仍仅发布请求窗口，因子谱系绑定另一个精确历史实例。
公开 `targets`、`target_results` 与运行引用仍按逻辑 `target_id` 唯一投影；内部失败按消费方
绑定的实例传播并计入整体状态，不能因公开投影而隐藏。`research run` 使用研究配置声明的窗口定义。
`--rebuild` 仅重建本次明确请求的终端输出实例；已发布且经现有 reader 校验的精确匹配上游
直接 REUSE，包括同一逻辑目标仅为下游提供的历史窗口实例。缺失的上游正常构建，身份变化
仍沿依赖图传播；无效上游明确阻塞，不能因发布过就跳过身份或内容校验。
明确请求的重建逐实例执行精确比较；`rebuild_evidence` 每条同时记录 `target_id`、被比较的
`instance_identity`、`candidate_token` 与 `comparison_status`，消费者不得按逻辑 ID 覆盖同目标的多条比较。
辅助源路径由已有 producer binding 绑定，规划、生产与复用验证读取同一来源根；
数据计算参数仍属于 processed 规格，路径不进入共享证据，原始内容身份进入谱系。
数据问题只阻塞依赖子图；已发布的其他组合和产物保持可见。
运行尝试与目标真实状态来自持久化记录，不能将部分输出概括成完整成功。
CLI 先输出原有结构化结果，再将顶层 `COMPLETE`、`FAILED`、`BLOCKED` 映射为退出码
0、1、3；参数及输入错误继续返回 2。计划与指定状态查询遵循相同映射，没有顶层终态的
列表查询正常返回 0。失败与阻塞的 JSON 内容不因非零退出码而省略。
模型拟合前先在现有 trial ledger 发布 attempted 事件，结束后追加同 attempt 的不可变终态事件；
读取者核对不变身份后投影当前状态。硬退出留下 attempted，不能宣称从未开始或已完成；
不自动重试、不改写旧事件，也不因此解锁 HOLDOUT_LOCKED。
A 股模型 owner 在每季全部家族和配置完成原生科学、参数、trial、PK 与覆盖验收，
且上游读取快照通过核对后，使用同一 semantic artifact/原子目录机制发布季度分片。
分片绑定完整 MODEL 请求的精确身份，只由该 owner 接续同一身份的后续请求；
已完成 trial、日志和未发布 `.a-*` 目录不能替代分片验收。新请求保留旧分片原来的
run_id/experiment_id 引用，仅为实际新增拟合写入新 trial，不建立第二份账本。
恢复只流式读取已发布预测成员和元数据，不保存训练矩阵；分片与完整 MODEL 各自保留
不可变预测成员。完整季度矩阵、全部选择与最终 refit 通过既有完整验收后才发布 MODEL，
季度分片不成为 DAG 的完整 MODEL，也不单独授予预测、统计或留出资格。
ExperimentPlan.stage 是 selection/refit 阶段的唯一事实，TrialRecord 通过 plan/protocol_hash 绑定。
数据角色统一为 train、validation、test；development 是访问阶段，不是第四角色。
同一请求、同一折或最终 refit 阶段内，相同截止、输入列、标签成熟域、资格过滤、行数控制、
输入类型和时间权重共享训练域及其一次性预处理；A 股与 Crypto 均遵循该规则，
Crypto 还按训练资产集合区分域，不合并 refit 的两个离散窗口。预处理在 trial 启动后首次求值；估计器各自持有
矩阵、标签和权重副本，冻结 scaler 保持一致。切换折或阶段释放共享域，不建立持久化缓存。
原始行的格式、有限值、资格和成熟事实由 collector 统一验证一次；被过滤掉的非法行仍拒绝。

selection fold 只声明互不重叠的 train/validation，Plan 唯一声明锁定 test。
refit 使用空 folds 与显式 fit_windows/fit_cutoff、selection_group 及来源选择集合：
A 股最终拟合为一个合法历史窗，Crypto 保留最后 fold 的两个真实窗口。
任何拟合/选择/监督校准均按自己的 cutoff 验证 label maturity；test 不参与冻结状态生成。
模型候选预测和经济使用由既有模型 artifact owner 持久化：
同一成员保留候选自身合法预测及共同评分资格，评分域不得削去经济预测行。
模型包遵循可信发布者的不可变产物合同。Crypto reader 核对成员内容身份、冻结合同、
trial 引用及验证 PK/覆盖，不重算持久化验证预测的 IC/MSE；模型状态在实际加载并消费时
由同一 fitting owner 校验。协同改写成员和 manifest 后建立的新读取会话不享有更强的
评分一致性或提前状态加载保证；普通成员损坏和旧读取快照变化仍须拒绝。

同一 TrialRecorder/存储机制分别记录选择与真实 refit；refit 使用实际 resolved spec、seed、
单候选单 fold 及 budget=1，并绑定父 selection 语义身份。refit 不复用选择预算或伪造验证成绩。
TrialRecord.metrics 统一承载阶段实际指标；selection 保留验证指标，refit 记录真实 n_train。
选型统计只消费 selection，不能按引用位置猜阶段或把 refit 计为第二次选型。
与合同不兼容的机器可读派生产物须从已验证上游重建，不保留字段别名或双写；raw、归档和历史报告保留。

数据集身份包含 schema、dtype、PK、逻辑窗口、来源谱系和精确内容。
`fingerprint_dataset` 独立提供既有指纹 API；`compare_datasets` 直接在一次读取中
核对 schema、dtype、PK、精确值，以及 schema owner 允许的实例业务元数据，
包括来源/输入身份、common_manifest 和 producer_diagnostics 中的覆盖事实。
行相同但业务元数据不同返回 METADATA_MISMATCH；分块、文件布局和运行观测不参与业务比较。
不额外计算逻辑指纹。比较摘要不再返回 `content_hash`；
持久化 manifest 的原始成员内容身份保持不变。只有精确比较才能给出 MATCH。
软件、协议、种子、后端、上游以及 trial/fold/asset/refit/季度谱系按现有合同绑定；
持久化模型读取不重新发现主机参数。数据集身份不受下游模型或组合调参污染。影响数据计算的参数属于 ProcessedDataSpec，
以严格 JSON 冻结并进入持久化规格及身份，例如辅助状态的 max_age_ms；改变 age 上限不能复用旧结果。
默认/正式/定制选择若消费相同语义定义，复用同一身份。

研究运行记录保留整个生产包及九个数值/模型后端版本的精确软件身份；
各市场的产物身份绑定实际 producer、source callback 和所用 writer/reader 的代码依赖，
从真实 callable/partial 展开包内静态模块导入闭包（含父包、函数内导入），不维护另一份文件注册表。
模块内容按完整模块保守失效；workflows/research.py 科学 owner 与 profile 组合根单独绑定
完整内容，不递归聚合入口的所有导入。同一科学模块内仍保守失效其消费者，不自动判断任意
代码行是否数学等价。workflows/operations.py 统一 plan/data/dataset/research/report 操作入口，
只负责请求寿命、观测和运行记录；来源、输入资格、模型预检、冻结、窗口、目标与 producer
绑定均由 research.py 准备，实际生产、持久化和复用验证也留在该科学 owner。
科学 owner 不反向导入操作外壳，不保留旧入口转发；操作外壳变化只进入全包运行 provenance。
来源父包只作轻量初始化；Crypto 原生格式选择由普通模块 sources/market.py 承接，
A 股来源导入父包时不会隐式绑定未消费的 Binance/Tardis 代码。
`runtime.resources` 与 `runtime.execution_metrics` 仅提供性能批次默认值、实际 CPU 容量与操作观测，
其代码变更保留在运行 provenance，不使科学产物失效；科学公式不得放入这两个 owner。
分帧、来源解码、数值计算、合同、存储与校验代码仍进入产物依赖。
统计、组合和模型的原生 reader 返回已经验证的业务封装，workflow 直接核对其目标身份，
不在同一调用边界先重复扫描一次通用封装；完整消费会话先验证内容，退出时核对捕获的
manifest、成员集合、文件及目录身份，新消费会话重新验证内容；主异常保持优先。
同一research调用的冻结预检、计划和执行共享已验证的原生对象及快照；模型同时绑定其外部trial成员。
复用前后检查全部捕获身份，读取期间变更仍失败；不把这一请求内借用变为跨请求缓存。
PRED 无论本请求新建还是复用，均先经同一全域原生内容、PK 与模型谱系准入；
STAT/PORT 只借用请求内已验证的文件快照，每个消费者分别构建选行与匹配状态。
消费者读取前后及退出时复核快照，退出释放匹配状态；不共享已匹配标记，不用行数或存在性替代准入。
请求内借用减少重复全域验证，但未声明 Parquet 列统计时，选行不保证跳过物理文件扫描。
指定 run 的报告发现只读取全体报告 manifest 并核对结构及快照，随后由原生 reader 校验
被选报告正文。无关 run 的正文损坏不阻塞指定报告；无效 manifest 和发现期间身份变化仍拒绝。
报告和阶段接续引用数据集时，复用原生 DatasetSession 核对成员集合、内容身份、
schema、footer 与计数，不仅信任 manifest 中的历史摘要，也不重复逐行物化。
请求模型损坏或缺失时，该目标显示 `FAILED/model_reference_invalid`，仍保留其他有效证据；
不能退回通用封装校验，把语义非法的模型重新标为可用。
完整且经原生 reader 验证的精确匹配节点直接 REUSE；真实输入、规格、窗口、计算依赖或上游身份
变化仅沿对应 DAG 子图传播。留出复用资格比较精确科学产物身份、冻结输入及窗口，
不因单独的操作 provenance 变化宣称科学证据发生修正；冻结模型、未访问留出与失败边界保持原合同。
A 股 simple ledger 的模型身份由预测上游绑定，组合统计由各账本上游绑定；两者不重复绑定整个冻结运行凭据。
增强账本另绑定该候选实际消费的开发 alpha 校准来源实例身份；不绑定其他候选或 BUILT/REUSED、run_id 等操作字段。
完整开发来源、授权与运行 provenance 继续留在冻结记录和报告，冻结配置的完整匹配要求保持。

研究 run_id 是不可变尝试标识。查询、报告与研究图片 只消费指定运行的确切产物引用，
核对模型、预测、组合、报告之间的身份。历史报告读取保存的 Markdown，不重新训练或重渲染。
预测按持久化 PK 对齐；生产者和统计、组合消费者从同一模型输入有效性 owner 派生必需域，域内缺失、重复或错位稳定失败。
QRP 自有 JSON 使用严格 UTF-8、顶层对象和精确 schema_id，拒绝重复键及非有限数并确定性写入。
自有接口、规格和 schema 不维护版本号；当前字段合同与精确内容、软件依赖身份共同决定可用性。
合同变更后重建不匹配的机器产物，不设置旧格式兼容路径；真实依赖版本和来源数据修订身份保留。
控制 manifest、plan、trial、原始响应和 CSV 记录不设人为总字节、列数、嵌套深度或整数位数额度。
Python 自身的解析/递归限制不被全局修改；循环、非法类型、截断和非有限内部 JSON 仍明确拒绝。
研究数据、股票代码、供应商信息和数值计数不按隐私误拒；共享边界继续拒绝凭据与本机绝对路径。读取按捕获的实际长度检查增长或截断，
只接受 manifest 声明的成员，并核对完整身份与终态。
trial 记录由 storage 的唯一读写实现验证，消费者直接取得普通记录，不设预算作用域。

## 执行与性能

研究任务的并行度、执行顺序和重试由推进任务的 agent 根据机器状态及产物依赖决定。
仓库不实现内存额度、租约、资源根、主机独占锁或替代调度器。
内存耗尽后只复用已经完整校验和原子发布的产物；未发布工作不构成可续跑结果。
分段处理须保持回看窗口、标签可用时点和完整横截面，不能改变科学计算范围。

性能实现优先使用原生 Arrow/Pandas 操作、列投影、谓词下推、批量计算和增量写入；
行批次用于吞吐，不按可用内存计算准入。时间批次只为保证计算需要的完整时间片。
不保留只为内存估算而设的逐行扫描、复制、溢写、作用域和测试。
CPU 参数结合物理核、当前进程的 affinity 和实际 bandwidth quota 解析；Linux读取sched_getaffinity及cgroup，Windows读取原生进程cpu_affinity，避免无效过量线程；
模型持久化记录实际使用的参数。操作观测仅在目标边界记录耗时、CPU、I/O 与并行情况，
不采集内存、不参与执行准入。

确有访问或查询需求的临时文件使用标准 `tempfile`，由启动环境的
`TMPDIR`／`TEMP`／`TMP` 选择位置；不增加另一套 scratch 配置。
原子发布的事务暂存保留在目标同父目录，以保证同文件系统发布。
存储层只统一数据合同、精确身份与发布语义；不保留纯转发入口、空生命周期包装，
同一规则不在生产者、写入者、读取者之间维护多份实现。

## 发布与失败

不执行磁盘预留或写入容量准入；空间不足由实际 write、flush 或 fsync 的原生 I/O 明确失败。
事务使用同父目录 UUID 暂存及平台原生不覆盖移动，
或归属明确的替换。成功才原子发布，失败保留原有已验证产物；
只清理本次操作能证明创建的资源。
实例发现器忽略原子目录 owner 的合法 `.a-` 加十二位小写十六进制暂存目录；
别名、同名普通文件和其他非法正式成员仍拒绝，读取间变化检查保持有效。
活动主 BaseException 不得被清理、验证、指标或释放次异常覆盖，不能依赖 GC 完成责任清理。
必需模型不可通过静默备用后端、伪零值或改为可选依赖绕过。

## 数值与统计结果

数值确定性以固定种子、实际后端/线程及既有精确或容差合同为准，不承诺普遍跨平台逐比特一致。
直接收益观测的零方差按实际数值一致性判定；NAV派生收益的风险统计使用由浮点NAV及比率运算传播的数值分辨率，不改写账本或收益。不可分辨风险时风险调整指标为UNDEFINED，固定候选宇宙不因退化候选而减员。
组合账本允许非负净值及保留数量的零估值权益；金额损益按持仓价值变化与真实现金流归约。
A 股调整单位及必要的权益换算估值保留精确有理数；账本精确数值字段以带类型标记的
规范文本持久化：`q:十六进制分子/十六进制分母` 为最简分数，`f:float.hex` 为原二进制浮点数，
Crypto 原有浮点数量与估值解码后仍为浮点。非规范、未约分或非有限编码拒绝。
非零累计成交数量无法表示为有限非零浮点数时，金额账继续保留，`fill_quantity` 未定义且
`fill_quantity_reason=nonfinite_result`；不让辅助数量汇总破坏有效现金/NAV。
有限金额的收益比率或初始资本归一化不可表示时同样明确为 `UNDEFINED/nonfinite_result`；
金额净值不因此失败。资本收益汇总用 `capital_return_reason`，配对表的两组期末财富
各自保留 `*_terminal_wealth_reason`，不能以缺失的描述比率否定其他可识别的配对推断。
权益事件与成交分表记录于同一本账，事件 ID 全账本精确唯一；新字段和成员合同使旧格式
机器产物必须按新身份重建，不读旧格式兼容分支。
区间期初 NAV 大于零时，跌至零的收益率为 -100%；期初 NAV 为零时，
`portfolio_daily` 的收益率、成本拖累及换手率为缺失，`return_status=UNDEFINED`、
`return_reason=nonpositive_start_nav`；期初 NAV 正但收益比率不可表示时原因为 `nonfinite_result`。
期后现金权重仅在分母为正且比率可表示时定义。
零估值恢复不以零分母制造收益率；净值曲线、初始资本收益及回撤直接从金额账推导，
不跨未定义区间连乘收益率。完整候选与日期网格保留，涉及未定义收益的风险统计、
配对比较及联合稳健性如实未定义，不通过删去破产区间或候选来恢复统计资格。


IC聚合分别判断标准差和ICIR的最终可表示性：非零标准差若舍入为零或溢出，该标准差为UNDEFINED/nonfinite_result，不标记zero_variance；只要均值/标准差的无量纲比值本身可表示，ICIR仍可发布。真实零方差保持未定义。
CSCV 每个候选以同一二进制尺度计算 mean/std/ratio，避免可表示有限结果被中间溢出改变排序/PBO；误差区间端点舍入相等时精确复核交集，不额外添加容差。真实常量、并列及样本不足仍保持既有 UNDEFINED 语义。

## 验证

一项能力的证明链为 owner→真实生产调用→持久化效果→系统性不变量检查→与被验收代码一致的可执行证据。
检查量化因果、批准能力、单一 owner、数据精确性、故障生命周期、
身份复现、私有边界、CLI/API 与研究图片。关注实际影响，不另造验证框架。
定向检查优先；默认全量命令是 python -m pytest -q。测试夹具错误与生产错误区分记录。
验收使用独立的、已验证容量与挂载类型的任务临时目录，并在新进程导入数值库前固定原生线程环境。
性能对照使用相同测试、相同线程设置和独立进程，记录 call 时间及真实产物身份；
改变 `TMPDIR` 也可能改变 pytest fixture 产物位置，不能把总差异直接归因某个计算或 I/O 调用。
内存文件系统仅在任务规模和内存余量允许时使用。

pyproject.toml 定义依赖/Python合同，
environment.yml 提供对应 Conda 环境；不复制包版本表。
验收命令、提交、测试数量和豁免仅记 .run/，不在长期合同中追加验收流水账。
