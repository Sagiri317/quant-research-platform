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
| runtime/ | 主机锁、可用量基线、工作集规划、租约及资源观测 |
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
A股与 Crypto 的评估报告均从同一个有资源边界的 artifact reader 读取真实评估汇总；
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
重建逐实例执行精确比较；`rebuild_evidence` 每条同时记录 `target_id`、被比较的
`instance_identity`、`candidate_token` 与 `comparison_status`，消费者不得按逻辑 ID 覆盖同目标的多条比较。
辅助源路径由已有 producer binding 绑定，规划、生产与复用验证读取同一来源根；
数据计算参数仍属于 processed 规格，路径不进入共享证据，原始内容身份进入谱系。
数据问题只阻塞依赖子图；已发布的其他组合和产物保持可见。
运行尝试与目标真实状态来自持久化记录，不能将部分输出概括成完整成功。
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

研究 run_id 是不可变尝试标识。查询、报告与 Notebook 只消费指定运行的确切产物引用，
核对模型、预测、组合、报告之间的身份。历史报告读取保存的 Markdown，不重新训练或重渲染。
预测按持久化 PK 对齐；生产者和统计、组合消费者从同一模型输入有效性 owner 派生必需域，域内缺失、重复或错位稳定失败。
QRP 自有 JSON 使用严格 UTF-8、顶层对象和精确 schema_id，拒绝重复键及非有限数并确定性写入。
控制 manifest 保留 1 MiB 边界；报告和 run 证据按实际序列化大小预留 JSON 工作集，写入及精确身份重读使用同一资源准入。语义产物成员由 manifest 字节预算约束，重读只接受已声明成员，不另设固定文件数量上限。
trial ledger 通过唯一的 storage 读取作用域持有累计目录、JSON 解码、typed 记录和返回值预算，消费者结束才释放。
plan 和事件在发布前按最终 UTF-8 字节数（含末尾换行）验证同一 1 MiB 读写边界；根目录逐项验证，不先物化全部成员。
追加事件同时计入原 ledger、输出序列化与仍存活模型，保持完整历史身份和终态验证；不以单文件上限替代累计准入。

## 一个资源边界

托管数据规模操作先取得同主机、同操作系统账户的本地锁，只捕获一次 available0；
Linux 锁位于固定 /tmp、按有效运行账户 UID 区分；Windows 锁位于账户主目录。
任务 TMPDIR/TEMP/TMP 不改变锁域。不同账户不共享预算，跨账户协调不属于本地研究合同；
规划目标为 min(available0×70%, explicit cap)。70% 和 memory_budget_bytes 是软规划参考，
不是 RSS、真实分配或逻辑工作集的精确硬线，不建立第二百分比阈值。
bytes 租约只使保守粗估可相加；在读取、解码、分配前选择有界批次。
仅在保守估算无法维持相对安全或显式限制不允许时稳定失败。
聚合持久化输出可以大于内存规划目标而精确完成。

操作内 SQLite 索引与 dataset inventory 使用标准 `tempfile.gettempdir()`，由启动环境的
`TMPDIR`／`TEMP`／`TMP` 选择临时文件系统；不增加 QRP 专属 scratch 配置，也不硬编码挂载点。
运行和验收入口在启动新 Python 进程前，显式核对任务临时目录的真实挂载、非别名路径、
可写性及可用容量，并确认 Python 实际选中的临时目录等于指定目录；失败即停止，不依赖
标准库的候选目录回退。共享文件系统上的小文件与 SQLite journal 延迟须单独测量，不能由
目录名推断本地磁盘性能。两个索引 owner 保留创建及增长前的容量准入和结束清理。
持久化 artifact 的事务暂存仍在目标同父目录，不能为了临时存储提速迁走原子发布边界。
若选用内存文件系统，还须将预计 scratch 占用与进程工作集一起核对可用内存；它不是无限溢写空间。

嵌套 owner 继承同一根，拒绝冲突策略或并行资源根。
输入已有 DataFrame/ndarray 为借用基线；新载荷、原生工作区和输出各有一个逻辑归属，
在生产/消费边界转移，不重复计费。阶段顺序执行，释放后复用预算，责任上下文负责关闭。
所有解码、索引、排序、模型、输出、溢写及重叠状态按实际同时存活工作集准入。

默认单进程加原生线程，只允许一层主要并行；不叠加 Python 进程池或调度器。
Linux runtime 读取标准 cgroup v2 的实际成员及可见父层，取宿主可用量与有限
memory.max-memory.current 的较小值；没有 cgroup v2 时使用宿主模式。
CPU 结合物理核、affinity 和 bandwidth quota，合同层只解析线程参数。
Windows 保留宿主资源与原生文件锁。仓库外 RSS/CPU/IO 是校准证据，不参与重复计费。
性能先用列投影、分区/谓词下推、有界列批次与增量输出；不以缩小逻辑研究范围证明优化。

## 发布与失败

事务内进行保守磁盘预检，使用同父目录 UUID 暂存及平台原生不覆盖移动，
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
检查量化因果、批准能力、单一 owner、数据精确性、故障生命周期、资源、
身份复现、私有边界、CLI/API/Notebook。关注实际影响，不另造验证框架。
定向检查优先；默认全量命令是 python -m pytest -q。测试夹具错误与生产错误区分记录。
验收使用独立的、已验证容量与挂载类型的任务临时目录，并在新进程导入数值库前固定原生线程环境。
性能对照使用相同测试、相同线程设置和独立进程，记录 call 时间及真实产物身份；
改变 `TMPDIR` 也可能改变 pytest fixture 产物位置，不能把总差异直接归因某个 SQLite 调用。
内存文件系统仅在任务规模和内存余量允许时使用。

pyproject.toml 定义依赖/Python合同，
environment.yml 提供对应 Conda 环境；不复制包版本表。
验收命令、提交、测试数量和豁免仅记 .run/，不在长期合同中追加验收流水账。
