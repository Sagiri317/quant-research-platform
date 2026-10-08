# 数据布局与来源边界

本文件只记录结构与规则，不保存私有行、原始供应商载荷、凭据或本机路径。

## 原始来源

所有来源共用 data/raw，各 owner 只在 data/raw/<source_id> 内递归发现和验证输入。
来源家族由供应商/表标识及结构证明，不从目录名猜测，不探测备用根。

### CSMAR

CSMAR 提供 A 股市场、指数、行业和分析师事实；其财务只作验证。

现有 `sources/csmar/field_mappings.py` 声明支持的表和必需字段，
`sources/csmar/interim.py` 统一执行类型、枚举、日期及单位规范化。
供应商说明书中的同名字段只证明该版字段定义；输入还须绑定实际表标识、原件哈希和
适用域，不因此承诺任意导出版本、未知字段或历史完整性。股数、市值、价格、成交量/额
及百分比换算见 [A 股价格、收益与市值真值](a-share.md#55-价格收益与市值真值)。
未知枚举在对应 owner 明确失败；缺失哨兵只按已经核实的表/字段规则处理，
不能将某字段的 `-9666` 定义推广到其他列。`index_info.base_point` 的既有特殊值
规则属于项目明确的适配约定，不能据此宣称供应商全部字段使用同一缺失值。
财务验证输入不能替代正式来源的首次披露或历史版本证明。
CSMAR 固定投影所需研究字段，不提供可选列 override；合法额外列及长 CSV 记录不受人为额度限制，
必需字段、重复列、严格 CSV 语法、源内容身份与读取期间变更校验仍保留。

CSMAR `index_membership` 是供应商指数成分事件，包含不同市场及证券类型，不能将所有
成分强制解释为六位 A 股股票代码。`constituent_stock_code` 保留供应商代码字符串；
`constituent_asset_id` 是原码、`market_code`、`security_type` 三项规范 JSON 的内容哈希，
用于区分同指数、同日期的跨市场或证券类型同码事件。缺失市场/类型仍保留缺失事实，
在该来源身份中表示为 null，不补造交易所。它不表示可交易 A 股身份。已有基准消费者
使用的 `index_weights` 仍按其 A 股代码合同处理。
CSMAR `index_weights.Weight` 与 `index_market.Idxtrd08` 的原始缺失值在源规范化表中保留未知，
不补零或删行；非缺失值仍须有限，权重不得为负，百分比只除以 100 一次。两列继续使用
`float64`，缺失表示为 NaN，Parquet 写为 NULL。源表可保留部分事实，不表示完整基准快照；
基准消费者仍要求所选基准权重非空、严格为正且完整，正式全收益绩效仍要求有效开盘序列。

### Choice

Choice 是现有正式财务 PIT 的来源绑定；原生响应先不可变保存，再由已有 owner 离线处理。
Raw 响应状态以 `status` 为唯一事实，不再持久化重复布尔状态；原始请求、响应字节及哈希保留。
资产负债表、利润表和现金流量表分别接纳，只阻塞实际依赖缺失表的因子。
绑定配置按已选因子和组合风险暴露的字段依赖裁剪报表输入；仍要求真实披露与日历，
不造空表、不以零填补缺失财务值。Raw CTR 接受官方 `IndexCode` 选项，
指数权重请求仍须保留查询日期与原始失败响应；服务错误不等于空权重。
[官方接口与指标资料](https://quantapi.eastmoney.com/AIGuide/Install?from=client)
对沪深一般企业财务 CSS 指标明确使用 `Type`：1 为合并报表，2 为母公司报表，
3 为调整后的合并报表，4 为调整后的母公司报表。现有正式消费者只接受 `Type=1/3`。
CTR 财务报表的 `ReportType` 则按 1、2、3、4 分别表示合并、调整合并、母公司、
调整母公司，不能将其数字直接用于 CSS。Raw 原样保留请求；已知财务 CSS 指标若带
`ReportType`，正式 source identity 与 interim 消费明确拒绝，不默认为 `Type=1`，
不增加 CTR 到 CSS 的兼容映射。

`sources/choice/fields.py` 是三张报表共 14 项一般企业指标的唯一字段映射。
官方指标资料将这些数值列声明为元。按 CNY 消费的依据是境内上市一般企业的
编报财务报表域：[财会〔2006〕3 号](https://kjs.mof.gov.cn/zhengcefabu/200805/t20080522_33653.htm)
规定上市公司自 2007 年起执行企业会计准则，
[《企业会计准则第 19 号》第四、十五条](https://kjs.mof.gov.cn/zt/kjzzss/kuaijizhunzeshishi/200806/t20080618_46229.htm)
要求编报报表折算为人民币，记账本位币可以不同。这里使用报表列报币种，
不能只从“元”推定，也不移用其他接口的币种参数。现有适配器用这些指标、
CSS 类型及境内证券代码限定消费域，不执行汇率换算；境外原币报表、其他报表
类型及未知单位不属于这一合同。该适用域依据不代替逐期快照与版本资格。
`EndDate` 只按官方定义表示截止日期，参数存在或请求成功不证明历史版本可回溯。
`STMTACTDATE` 表示定期报告实际披露日期，`ACTUALISSUINGDATE` 排除招股披露，
`ITEMVALUECHANGEDATE` 表示财报科目数据变更日期；三者不相互冒充，也不单独证明
对应数值当日已公开。每个被消费的版本必须由同一请求坐标和不可变原件证明。
日期字符串在现有日期 owner 归一为语义坐标，原始请求与字节身份保留。
日期型已公开信息在下一市场交易日可用；这是项目的保守研究规则，
不是供应商对日内可见性或全部历史修订的保证。规则与修订传播见 [A 股](a-share.md#53-pit-会计与可用时点)。
缺少真实版本、含糊的同日冲突或覆盖不足只阻塞依赖该来源的目标；
不能补造旧原件的时间、版本或完成字段，也不等待无关的全供应商历史保证才消费已合格范围。

### Binance

Binance 原生录制 reader 仅消费录制清单声明的 gzip JSONL 成员；
历史归档、辅助状态与资金费按各自显式清单和身份绑定。不提供独立 JSONL 封装或分钟线适配入口；详细来源合同见 [Crypto](crypto.md)。

### Tardis

Tardis 历史下载与导入按连续切片范围运行，保留 Binance 官方实时采集；
实现就绪状态与输入资格分别记录，不能把下载完成写成研究链已接入。
公开免费数据可直接请求；需要 API key 而没有授权时明确权限不足，不自动注册或购买。
Binance Futures 原生历史归档的不可变原始字节保存在
`data/raw/tardis/binance-futures/<archive-id>/`；清单绑定供应商、
交易场所、UTC 起始日期与连续切片范围、原生频道及频率、文件路径与字节哈希、
稳定行序和可证明的缺口/有效区间。归档和研究均不限制为一天；原件文件名中的日期仅用于寻址。
原生行是供应商 UTC 前缀与 JSON，空行表示供应商断线，因此压缩原件使用
`tardis__binance-futures__YYYY-MM-DD__minutes-HHHH-HHHH.txt.gz`，不称为 JSONL。
原生压缩字节不改写，清单、缺口与观察区间旁文件各有精确哈希。
行序属于归档身份，不能冒充本机录制的原始 capture_seq。
归档完整性、价格表示资格、盘口连续覆盖和研究完成是不同检查，不能相互替代。
单次研究默认消费一份有明确来源身份的输入，不自动拼接另一来源补洞。
同一市场的录制与历史归档复用既有 L2、PANEL、因子、标签、模型和账本责任模块。

Tardis native continuity 保留原字节，包括 depth 重复/过期更新。
显式多归档集合逐原件绑定精确身份，按实际覆盖跨度及请求窗口路由。
原件的本地 source_sequence/event/segment/observation 可重新起算；
统一 source owner 为持久化 observation/segment 标识加入原件身份，并保留每段真实 gap。
按窗口选取的 funding 原件分别验证实际 coverage，不以跨月包络推断未知结算。

下载校验只推进 depth update-id 高水位，旧更新不倒退高水位、不制造新的连续性；
既有 Binance book engine 负责丢弃不适用的更新。
断连关闭观察区间，后续更新使用新的观察身份；前向 sequence gap 仍形成 gap。
aggTrade ID 非单调仍明确失败，并保存频道、slice offset、行号和前后 ID 的诊断注记，
不输出原始价格、数量或完整 payload。
前向 ID 跳号按来源 adapter 合同处理：Tardis 原生导入记录 gap，官方录制 reader
仅验证单调性。[Binance 聚合成交流说明](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market#aggregate-trade-streams)
定义聚合 ID；该定义本身不证明订阅消息逐 ID 加一；[Tardis 来源说明](https://docs.tardis.dev/historical-data-details/binance-futures)
也不能替代该保证。adapter 严格度差异不构成已证实漏消息，也不授权自动放宽 gap 合同。

来源适用域按 [Tardis 历史频道说明](https://docs.tardis.dev/historical-data-details/binance-futures)
和原件清单中的频道/频率共同决定。只给出变更日期而没有精确时刻的深度、markPrice
过渡日由现有 source owner 拒绝，不能推定在 UTC 零点切换。
[OKX 面值调整公告](https://www.okx.com/zh-hans/help/adjustment-of-face-value-for-usdt-margined-perpetual-swap-futures-trading)
给出的调整半小时区间保持为不确定域；区间外才使用已经证明的合约乘数。
[Tardis 数量说明](https://docs.tardis.dev/faq/data)中的 `amount` 可能是合约数、基础资产或
报价金额，须按场所、品种、时段和实际单位归一；当前接口字段名不能证明全历史单位。
[历史合约 metadata](https://docs.tardis.dev/api/instruments-metadata-api) 的字段和变更记录
也只能在其声明的保证范围内使用，不能把一个字段的历史保证推广到所有字段。
这些限制保留不确定域的局部失败，不否定域外已通过原件、单位、身份及覆盖校验的输入。

Binance Spot 和期货辅助状态的原生归档由配置显式选择，格式和消费边界见 [Crypto](crypto.md)。
Bybit 与 OKX normalized 来源在 `data/raw/tardis/<exchange>/` 下按配置选择归档，
其中 exchange 为 `bybit` 或 `okex-swap`；每份归档保存不可变 `data.csv.gz` 与 `manifest.json`。
清单绑定场所、品种、数据类型、日期、观测范围及压缩原件哈希；只证明观测行，不证明原生消息连续性。

### 规范原始输入与归档

完全重复指字节相同。规范暂存仅保留一份，在保留文件附属元数据记录来源名称与哈希。
同表更新导出成为新的活动候选，保留覆盖/结构差异证据，不为修复导出改写原始字节。
无法满足来源 adapter 合同的已知家族标为 KEEP_RAW_ONLY；
只有无法证明表标识时才用 unclassified。两者均不以备用路径进入正式生产。

每个保留 CSV 有一份规范 UTF-8 .metadata.json，记录结构、计数、日期、来源、哈希和质量事实。
它是操作元数据，不是第二数据集真值。原始切换先暂存、读回核验再替换，失败保留旧数据。
已验证必需字段的派生投影可作为唯一活动规范输入；保留原件于 data/archive/<source_id>/<dataset>，
该目录不参与 raw 扫描。元数据分别记录原件观察日、未知的原生导出日及真实投影导出日，
并以原件、转换脚本和投影哈希证明来源；不能把未合规的原文叙述标为结构通过。
独立历史费率 CSV 是 QRP 研究参考输入；其精确原始内容保持，
独立外部来源核验状态与实现是否可消费分别说明，不制造第二份规范副本。

### 公开行业参考来源

公开行业快照属于 `reference` 来源，由现有 reference interim owner 读取
`data/raw/reference/a_share_industry_snapshots/metadata.json`。元数据声明机构对应的官方页面、
标准、观察日、公开日、快照身份、候选 JSON/原始附件/发布页面逻辑文件名及全部来源文件 SHA256；
候选逐行保留资产源代码、公司原名、分类原字段和原件页码或表/行定位。文件使用共同的有界读取和精确身份
owner，提交前重核完整源身份；缺少已声明文件、身份变化或结构不合法不能以空列表替代，
失败保留已验证旧产物。公开源与 CSMAR 原件分别保留，不伪造供应商字段或修改原始附件。

`a_share_industry_snapshots` 是正式 PANEL 的显式 DAG 输入，source_id 为 reference。
Interim 的 observation_date 保留观察期，publication_date 保留页面公开日，timestamp
为日期型公开日之后的次日下界；按角色窗口末端裁剪，保留窗口起点前的历史状态。
PANEL 在真实交易日轴上实施可用日规则。分类词典、历史资产身份、未知屏障与混合标准
消费仍由现有 A 股 owner 统一裁定，
见 [历史行业](a-share.md#543-历史行业)。元数据或候选完整不等于覆盖区间已获研究资格。

## 持久化与就绪

数据层分为 raw、interim、processed、research、artifacts、runs，由 storage 布局派生路径。
每个 QRP 数据集有 spec.json 和 _manifest/manifest.json，
`spec.json` 是列、角色、主键、分区、特征配方等静态合同的唯一持久化来源；
manifest 只保存运行事实和 `spec_identity`，读取投影从规格派生静态字段。
schema_id 分别为 qrp-dataset-spec 与 qrp-dataset-manifest；研究运行记录使用 `data/runs/research/<run_id>.json`，Binance 录制清单使用录制根目录下的 `run.json`。
数据集 manifest 的 `content_hash` 按规范物理顺序绑定全部成员路径与原始字节摘要；
新会话在暴露数据前有界核对全部成员，不能把同大小、同统计的替换文件当作旧实例。
未保存该内容证明的历史实例不静默补证或改写；需要从已验证上游重建新的实例。
物理 IO、发布和精确比较见 [架构合同](architecture.md)。

raw 存在、供应商接口成功、字段齐全、范围内 PIT 适用、质量合格及正式研究验收是不同事实。
正式就绪要求 owner、真实消费者、持久化输出与适用语义检查；
记录每个缺口只影响的目标，不能把缺失解释成零或完整覆盖。
Choice 的实际披露、修订及历史版本约定见 [A 股](a-share.md)。
标签访问与开发末端规则见 [A 股研究数据集语义边界](a-share.md#512-研究数据集语义边界)。

## 本地操作与交付

采集入口、计划和 SDK 操作资料整理在 .run/acquisition/；
本地供应商说明书、字段参考资料和历史采集缺口清单保存在 `data/reference/`，不属于 raw 数据发现根，且不进入 Git；历史缺口清单不构成数据覆盖证明。
代码所需的已核实单位、字段口径及页码写入对应现有市场合同；
CSMAR 股数、市值、价格与成交量/额单位见 [A 股价格、收益与市值真值](a-share.md#55-价格收益与市值真值)。
库存和账户额度留在本机操作记录，不进入长期文档。
研究原始数据仍放 data/raw，不能因整理证据删除有效原始来源。
交付副本按需由仓库文档生成，不维护第二份正文。
SSH、激活信息、账户状态、供应商原始文档/表格均不进入 Git 或网页版交付。

Git 只保存代码、合同、结构定义、文档、合成测试和安全模板。
私有原始与生成行数据、供应商载荷、秘密、本地绝对路径、股票代码样本及原始回溯不提交。
共享检查只输出结构、计数、哈希、状态、逻辑路径及不含私有行的摘要；
在配置/进程/错误边界先选择安全字段，不能输出完整可能含令牌的内容。
