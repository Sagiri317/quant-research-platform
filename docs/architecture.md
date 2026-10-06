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
留出创建新的预测、评估、账本、统计、报告与 holdout_evidence，不重新训练或校准。
允许 artifacts→reports，禁止 reports→artifacts。
可移植模型算法 ID、依赖、输入种类及并行类别由 contracts 定义；研究选择与参数由显式配置提供；
拟合、预测、缩放、排名、重要性和验证由 research 持有。字段名使用 model_algorithm，
算法集合使用 model_algorithm_ids；只保留唯一合同入口，不保留兼容支路。

factor 是信号来源，feature 是模型输入，label 是监督目标，prediction 是模型输出，
signal 是评估/组合使用的排序、权重或交易意图，evaluation 是研究表现，portfolio 是构建与收益路径。
多资产同一时点采用 cross_sectional；单资产 HFT 跨时间采用 single_asset_time_series。
后者以有界在线矩计算 Pearson；尚未有精确有界实现的 RankIC/分组价差明确不适用。

## 研究配置

可编辑 JSON 位于 `configs/research/a_share/` 与 `configs/research/crypto/`。
因子库、评估选择和模型特征是三个不同集合；配置分别声明评估因子、标签与主统计标签、
每个模型的输入、算法和参数、使用的组合参数，以及 windows.development/windows.holdout。
research run 只通过显式研究配置选择 profile、scenario 和研究窗口，不接受独立覆盖参数，避免双重选择来源。
配置窗口必须位于协议访问边界内；选择窗口不会授予留出访问权。CORE6、PV25、Crypto baseline 是普通示例配置，
运行入口不隐式选择它们，不动态展开“全部因子”。仅评估和完整研究复用同一 DAG。
profiles 把确定的配置绑定到现有 catalog/protocol/producer，完整配置保存在 protocol 与 run 记录。
每个 target 仅绑定真正影响其计算的规范合同；workflow、业务 envelope、trial、writer 和 reader
消费同一合同投影。只改变执行模式或未消费的费率，不改变模型、预测或校准的身份；
真实输入、标签、成熟截止或参数变化仍使相关目标失效。不在各层复制字段排除名单。

Crypto 的训练、评估资产由每个模型的显式范围决定，不由模型名称推断。
执行仍限定 Binance BTC/ETH USDT perpetual。模型各自保留完整预测和有效样本数；
直接比较在同一时点、同一资产主键的共同有效 OOS 样本上重算，不能只对齐日期。
经济账本不因模型样本交集而裁剪。只选择一个模型时，比较表为空，联合检验明确为 UNDEFINED。

配置字段必须有真实消费者。A股沿用季度选择 owner，允许按顺序列出
`parameter_candidates`；Crypto 沿用每个显式模型一组 `hyperparameters` 的 trial，
不同参数实验写成不同模型/研究配置，不接受会被忽略的 `parameter_candidates`。
Crypto 的训练、验证和OOS时段必须由 `research_plan` 声明；A股不接受该字段，时间访问范围来自
`windows`，季度选择规则仍由既有 A股 protocol 管理。

研究配置的 `stage=evaluation` 通过同一 DAG 只生成已选因子评估。
报告重建从指定 run 记录中的冻结 JSON 重新绑定完整目录，不沿用调用者上一次裁剪的选择。
配置绑定由既有市场 protocol/catalog owner 完成；artifact reader 只消费绑定后的选择与持久化证据。
模型、预测、组合和稳健性分析未被请求时，
相应部分记为 `UNDEFINED / not_requested`，不制造缺产物 blocker。
A股与 Crypto 的评估报告均从同一个验证精确身份的 artifact reader 读取真实评估汇总；
A股因子评估报告不依赖模型统计产物。

配置文件不存运行产物。研究数据、模型和报告使用 data 分层与 instances 身份目录；
研究运行记录为 `data/runs/research/<run_id>.json`，trial 为 `data/runs/trials/<run_id>/`。

## 计划、身份与持久化

所有来源、数据集、评估、模型、预测、组合、统计和报告由同一目标 DAG 展开依赖。
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
ExperimentPlan.stage 是 selection/refit 阶段的唯一事实，TrialRecord 通过 plan/protocol_hash 绑定。
数据角色统一为 train、validation、test；development 是访问阶段，不是第四角色。
selection fold 只声明互不重叠的 train/validation，Plan 唯一声明锁定 test。
refit 使用空 folds 与显式 fit_windows/fit_cutoff、selection_group 及来源选择集合：
A 股最终拟合为一个合法历史窗，Crypto 保留最后 fold 的两个真实窗口。
任何拟合/选择/监督校准均按自己的 cutoff 验证 label maturity；test 不参与冻结状态生成。
模型候选预测和经济使用由既有模型 artifact owner 持久化：
同一成员保留候选自身合法预测及共同评分资格，评分域不得削去经济预测行。

同一 TrialRecorder/存储机制分别记录选择与真实 refit；refit 使用实际 resolved spec、seed、
单候选单 fold 及 budget=1，并绑定父 selection 语义身份。refit 不复用选择预算或伪造验证成绩。
TrialRecord.metrics 统一承载阶段实际指标；selection 保留验证指标，refit 记录真实 n_train。
选型统计只消费 selection，不能按引用位置猜阶段或把 refit 计为第二次选型。
与合同不兼容的机器可读派生产物须从已验证上游重建，不保留字段别名或双写；raw、归档和历史报告保留。

数据集身份包含 schema、dtype、PK、逻辑窗口、来源谱系和精确内容。
指纹仅快速拒绝；只有精确比较才能给出 MATCH。
软件、协议、种子、后端、上游以及 trial/fold/asset/refit/季度谱系按现有合同绑定；
持久化模型读取不重新发现主机参数。数据集身份不受下游模型或组合调参污染。影响数据计算的参数属于 ProcessedDataSpec，
以严格 JSON 冻结并进入持久化规格及身份，例如辅助状态的 max_age_ms；改变 age 上限不能复用旧结果。
默认/正式/定制选择若消费相同语义定义，复用同一身份。

研究运行记录保留整个生产包及九个数值/模型后端版本的精确软件身份；
各市场的产物身份绑定实际 producer、source callback 和所用 writer/reader 的代码依赖，
从真实 callable/partial 展开包内静态模块导入闭包（含父包、函数内导入），不维护另一份文件注册表。
模块内容按完整模块保守失效；workflow 与 profile 组合根单独绑定完整内容，不递归聚合入口的所有导入。
`runtime.resources` 与 `runtime.execution_metrics` 仅提供性能批次默认值、实际 CPU/磁盘容量与操作观测，
其代码变更保留在运行 provenance，不使科学产物失效；科学公式不得放入这两个 owner。
分帧、来源解码、数值计算、合同、存储与校验代码仍进入产物依赖。
完整且经原生 reader 验证的精确匹配节点直接 REUSE；真实输入、规格、窗口、计算依赖或上游身份
变化仅沿对应 DAG 子图传播。留出复用资格比较精确科学产物身份、冻结输入及窗口，
不因单独的操作 provenance 变化宣称科学证据发生修正；冻结模型、未访问留出与失败边界保持原合同。

研究 run_id 是不可变尝试标识。查询、报告与 Notebook 只消费指定运行的确切产物引用，
核对模型、预测、组合、报告之间的身份。历史报告读取保存的 Markdown，不重新训练或重渲染。
预测按持久化 PK 对齐；生产者和统计、组合消费者从同一模型输入有效性 owner 派生必需域，域内缺失、重复或错位稳定失败。
QRP 自有 JSON 使用严格 UTF-8、顶层对象和精确 schema_id，拒绝重复键及非有限数并确定性写入。
自有接口、规格和 schema 不维护版本号；当前字段合同与精确内容、软件依赖身份共同决定可用性。
合同变更后重建不匹配的机器产物，不设置旧格式兼容路径；真实依赖版本和来源数据修订身份保留。
控制 manifest、plan 和 trial 事件保留 1 MiB 序列化格式边界，包含末尾换行；
这不是内存额度。读取只接受 manifest 声明的成员，并核对完整身份与终态。
trial 记录由 storage 的唯一读写实现验证，消费者直接取得普通记录，不设预算作用域。

## 执行与性能

研究任务的并行度、执行顺序和重试由推进任务的 agent 根据机器状态及产物依赖决定。
仓库不实现内存额度、租约、资源根、主机独占锁或替代调度器。
内存耗尽后只复用已经完整校验和原子发布的产物；未发布工作不构成可续跑结果。
分段处理须保持回看窗口、标签可用时点和完整横截面，不能改变科学计算范围。

性能实现优先使用原生 Arrow/Pandas 操作、列投影、谓词下推、批量计算和增量写入；
行批次用于吞吐，不按可用内存计算准入。时间批次只为保证计算需要的完整时间片。
不保留只为内存估算而设的逐行扫描、复制、溢写、作用域和测试。
CPU 参数结合物理核、affinity 和实际 bandwidth quota 解析，避免无效过量线程；
模型持久化记录实际使用的参数。操作观测仅在目标边界记录耗时、CPU、I/O 与并行情况，
不采集内存、不参与执行准入。

确有访问或查询需求的临时文件使用标准 `tempfile`，由启动环境的
`TMPDIR`／`TEMP`／`TMP` 选择位置；不增加另一套 scratch 配置。
原子发布的事务暂存保留在目标同父目录，以保证同文件系统发布。
存储层只统一数据合同、精确身份与发布语义；不保留纯转发入口、空生命周期包装，
同一规则不在生产者、写入者、读取者之间维护多份实现。

## 发布与失败

已知写入字节量时执行磁盘预检；Parquet 编码大小由实际写入确定，空间不足由原生 I/O 明确失败。
事务使用同父目录 UUID 暂存及平台原生不覆盖移动，
或归属明确的替换。成功才原子发布，失败保留原有已验证产物；
只清理本次操作能证明创建的资源。
活动主 BaseException 不得被清理、验证、指标或释放次异常覆盖，不能依赖 GC 完成责任清理。
必需模型不可通过静默备用后端、伪零值或改为可选依赖绕过。

## 数值与统计结果

数值确定性以固定种子、实际后端/线程及既有精确或容差合同为准，不承诺普遍跨平台逐比特一致。
直接收益观测的零方差按实际数值一致性判定；NAV派生收益的风险统计使用由浮点NAV及比率运算传播的数值分辨率，不改写账本或收益。不可分辨风险时风险调整指标为UNDEFINED，固定候选宇宙不因退化候选而减员。

IC聚合的真实零方差与不可表示尺度分开：非零标准差若舍入为零，标准差及ICIR保留未定义，原因使用既有nonfinite_result，不标记zero_variance。
CSCV 每个候选以同一二进制尺度计算 mean/std/ratio，避免可表示有限结果被中间溢出改变排序/PBO；误差区间端点舍入相等时精确复核交集，不额外添加容差。真实常量、并列及样本不足仍保持既有 UNDEFINED 语义。

## 验证

一项能力的证明链为 owner→真实生产调用→持久化效果→系统性不变量检查→与被验收代码一致的可执行证据。
检查量化因果、批准能力、单一 owner、数据精确性、故障生命周期、
身份复现、私有边界、CLI/API/Notebook。关注实际影响，不另造验证框架。
定向检查优先；默认全量命令是 python -m pytest -q。测试夹具错误与生产错误区分记录。
验收使用独立的、已验证容量与挂载类型的任务临时目录，并在新进程导入数值库前固定原生线程环境。
性能对照使用相同测试、相同线程设置和独立进程，记录 call 时间及真实产物身份；
改变 `TMPDIR` 也可能改变 pytest fixture 产物位置，不能把总差异直接归因某个计算或 I/O 调用。
内存文件系统仅在任务规模和内存余量允许时使用。

pyproject.toml 定义依赖/Python合同，
environment.yml 提供对应 Conda 环境；不复制包版本表。
验收命令、提交、测试数量和豁免仅记 .run/，不在长期合同中追加验收流水账。
