# 量化研究 Agent

`qrp agent` 是现有研究操作的调用者。它负责问题、假设、工具选择、实验反馈、会话、记忆与报告；因子公式、样本筛选、模型、统计和交易核算继续使用原有实现。代码在 `src/quant_research_platform/agents/`，测试在 `tests/agents/`；CLI 适配测试在 `tests/interfaces/cli/`，共享的合成数据构造放在 `tests/helpers/`。

支持 A 股日频因子研究和已有高频模型、评价与账户回测。基本流程是：明确口径、检索资料、提出公式、开发期实验、比较基线与费用后表现、冻结候选、固定最终评价、报告与经验保存。研究操作继续使用原生实现。

## 启动、交互和恢复

使用已配置项目环境中的 `qrp`，或用该环境的 Python 执行 `python -m quant_research_platform.interfaces.cli`。不要使用 Conda base。

源码工作树尚未安装时，在工作树根目录将当前终端的 `PYTHONPATH` 指向 `src`，再使用项目解释器运行。PowerShell 示例（已激活项目环境）：

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m quant_research_platform.interfaces.cli agent
```

这只设置当前终端，不安装包或修改全局环境；多个工作树共用环境时，也能明确选择当前分支的源码。下文 `qrp` 可替换为同一解释器的 `python -m quant_research_platform.interfaces.cli`。

```sh
qrp agent
qrp agent --config configs/research-learning/agent.json --interactive
qrp agent --resume results/agent-session --status
qrp agent --resume results/agent-session --message "保留原口径，继续检查候选与基线的差异" --interactive
qrp agent --resume results/agent-session --budget more-budget.json
```

不带配置时直接进入连续对话，开放 `read`、`search`、`terminal` 三个工具：直接读当前仓库和 `data/`、搜索本地或网络资料、用当前项目 Python 调用仓库源码。无需先手动绑定本地数据或建立研究协议，就能盘点数据、查看因子和提出假设。默认继承报告复核示例的模型连接与预算设置，不继承其历史研究协议。每个新问题建立独立任务，近期问答直接进入上下文。生成目录默认为 `results/agent-chat/<对话编号>/`，由 `.gitignore` 忽略。

自动挖因子是主要研究用途，具体步骤、原生搜索的能力与限制见 [自动化挖因子](factor-mining.md)。Agent 不另外实现因子算法；它读取真实入口和配置，调用原有数据准备、公式搜索、统计、模型和回测代码，依据训练反馈继续研究。

对话目录中的 `conversation.json` 保存工作配置，每轮的 `session.json` 保存任务事实。`qrp agent --resume <对话目录>` 恢复完整对话；`--status` 只读查看本轮及累计消耗。已完成的研究任务保持完成状态，追问建立新任务。

显式传入 `--config` 且不加 `--interactive` 时自动运行到完成、需要回答、预算耗尽或失败。启动只接受新的输出目录；恢复单个任务读取 `session.json`，沿用原协议和成功产物。已完成任务不带新消息和预算执行 `--resume`，只重新生成报告与记忆。

交互中可以直接输入补充要求，也可以输入：

| 输入 | 行为 |
|---|---|
| `/status` | 查看等待输入、等待模型或任务状态、本轮与累计消耗，并恢复输入提示 |
| `/help` | 查看终端命令 |
| `/tools` | 默认列出读文件、搜索和本地终端三个通用工具；显式配置任务列出其原生操作 |
| `/data 别名 数据集目录` | 在两轮任务之间绑定原生数据集，路径只在本地保存，模型接收别名及元数据 |
| `/pause` | 在当前行动结束后保存并暂停 |
| `Ctrl+C` | 中断当前行动，清理本次拥有的子进程或容器，保存暂停状态 |
| `/continue` | 继续暂停的会话；待回答的问题仍须实际回答 |
| `/budget {"max_requests":60}` | 调整该会话累计请求上限；不会重置已用预算 |
| `/quit` | 当前行动结束后保存退出 |

补充要求到达时，尚未执行的旧模型行动会标为 `superseded`，使用新要求重新决策。修改研究口径需要另开明确配置的实验；自然语言消息不会偷偷改写标签、成本或窗口。每个会话同时只允许一个进程写入。

正常中断保留 `interrupted` 结果。进程被强制结束后，尚在运行的行动恢复为“结果未知”，不会将残留输出冒充成功。开发阶段由模型重新决策；调用者恢复最终评价时，已成功的步骤继续引用，失败或中断步骤使用同一冻结配置在新证据目录重试，原有残留输出保留。

## 模型与预算

`llm` 使用 OpenAI Responses API 和严格结构化输出，传输使用 HTTPX，无新增 Agent 框架依赖。

模型响应在 `action` 字段中返回行动。schema 区分必须带非空工具名的工具行动和工具名为空的提问、冻结、结束行动，避免等到执行时才发现两字段矛盾；本地会话仍保存原有的扁平行动记录，不新增第二套状态格式。

| 字段 | 示例 | 含义 |
|---|---|---|
| `base_url` | `https://api.openai.com/v1` | API 根地址，客户端追加 `/responses` |
| `model` | `gpt-6-luna` | 模型名称 |
| `credential_env` | `QRP_RESEARCH_API_KEY` | 当前进程中的密钥环境变量名，不填写密钥 |
| `max_output_tokens` | `128000` | 单次生成上限，包含推理与可见输出 |
| `reasoning_effort` | `high` | 客户端接受 `none/low/medium/high/xhigh/max`，所选模型须支持该值 |
| `timeout_seconds` | `300` | 网络等待超时，与任务剩余时间取较小值 |

请求使用 `service_tier=default`、`text.verbosity=medium`、`store=false`，不传 `temperature/top_p`，不自动重试或切换模型。`store=false` 不等于零数据留存。远程 API 要求 HTTPS，localhost 可用 HTTP。HTTP 失败、拒绝和未完成响应保留失败状态，可由调用者恢复。

Windows 用户环境变量更新后，已有终端不会自动刷新。可在新终端启动，或执行以下命令刷新当前进程，不打印密钥：

```powershell
$env:QRP_RESEARCH_API_KEY = [Environment]::GetEnvironmentVariable('QRP_RESEARCH_API_KEY', 'User')
```

`budget` 必填 `max_requests`、`max_experiments`、`wall_seconds`。网络检索也占一次请求，最终评价也占实验预算；暂停期间不计运行时间。可选 `max_tokens` 在调用返回后按供应商报告的输入与输出 token 合计停止后续调用；它不是单次请求的精确费用硬上限。供应商未返回 usage，或输入/输出计数缺失、不是非负整数时，消耗记为未知；若启用 token 上限则明确失败，不接受该次行动或结论。

`max_context_chars` 默认 300,000；设置为 `null` 可关闭本地字符裁剪，例如 `/budget {"max_context_chars":null}`。字符数只是粗略控制，不能代替模型的上下文 token 容量或费用限制。问题、协议、用户补充和冻结状态始终保留；裁剪时优先保留最近行动引用的证据和冻结候选，跳过超长结果后继续选择其他可容纳记录。完整行动与结果保存在本地，默认可用 `read` 读取 `session:session.json`；显式配置的研究任务使用 `lookup`。近期对话使用部分上下文预算，不再固定限制在一万多字符。固定约束本身超出所选预算时明确失败。图片单独计入供应商的图片/token 用量，不算作本地文本字符。

HTTPX 的超时控制连接和 I/O 等待，不是整个 HTTP 响应的绝对墙钟。慢响应和清理可能略超任务预算，返回后不再启动超预算的实验。

## 本地命令、网络与复杂文件

`terminal` 在本机运行命令，Windows 使用 PowerShell，Linux 使用 Bash。环境中的 `QRP_PYTHON` 指向启动 Agent 的项目解释器，`QRP_WORKSPACE` 指向仓库，`QRP_OUTPUT` 是每次行动的新输出目录。继承项目依赖和 `PYTHONPATH`，移除模型连接所用的密钥变量。它具有当前用户的文件和网络权限，**不是容器隔离**；不要给不可信来源的指令执行授权。普通工具输出会清理绝对路径、凭据模式和 traceback，原始日志只留本地。这不保证识别所有私人业务字段，研究脚本必须只打印结构和聚合统计。

`read` 支持文本、源码、目录、DOCX 正文与表格文字、Excel 表结构、CSV 列名及 Parquet schema/行数。图片会作为真实图片送入模型；PDF 按指定页返回文字和页面预览（默认第一页）。使用 `offset`、`limit` 分页，PDF 用从 1 开始的 `pages`。数据文件不自动上传明细。文件路径用仓库相对路径、`input:别名/成员` 或 `output:e0001/results/成员`，已有终端输出可通过返回的 `transcript` 继续分页读取，不必重跑命令。

复杂文件写入通过终端调用已安装的 PyMuPDF、openpyxl、Pillow、pandas、pyarrow 或标准库 ZIP/XML；未安装的依赖明确报错，不自动安装。Word XML 读取不等于完整排版理解；复杂表格、图形或版式应先渲染，再用图片检查。默认没有音视频理解，也不承诺所有 Office 格式都能直接解析。

默认 `search(scope=web)` 使用现有 Responses 网络搜索，只向服务发送检索词，并额外计一次请求。模型及 API 服务必须实际支持 `web_search` 和图片输入；不支持时返回错误，不静默假装执行。`search` 默认搜索本地源码；查找原始数据时使用 `files=true` 查文件名，在本机计算数据摘要。

## 显式配置的研究任务

原生数据集位于仓库外时，连续对话可以用 `/data source <原生数据集目录>` 绑定输入；仓库内的数据可以直接发现。真正进行实验时需明确标签、研究区间、基线与目标，缺少条件只追问缺少部分，不套用示例中的 `x`、日期或费用。

对于通过 `--config` 显式提供原生模板的任务，保留固定协议、候选冻结和自动最终评价。配置中声明 `inputs` 时可使用 `configure` 和 `research`，后者接收 `operation`、原生 `config`、数值提取 `metrics` 和聚合事实 `facts`。首次实验后不能更换协议，已经投入使用的固定最终步骤也不能改写。`lookup(scope=operations, key=<操作名>, query="")` 返回实际原生入口及模块源码。保存的原生产物用 `artifact:e0001/member` 引用。该模板模式与默认的本地终端模式权限不同，下面有关原生证据和容器的约束适用于模板模式。

原生研究调用支持盘口归一化 `normalize-book`、时间序列评价 `evaluate-time-series`、高频账户 `backtest-hft`、库存分解 `analyze-inventory`、`fit-hawkes` 及模型解释 `explain`。通用 `fit/predict` 继续服务日频和高频模型。`search-factors` 的选型目标仍是截面 RankIC，不能用来声称已经实现单只 ETF 的时间序列公式挖掘。

`compare` 按证据编号读取所有指定结果及样本口径，不重算收益。整体评价调用原生统计与账本汇总；独立初始化的窗口不能直接拼接净值或平均 Sharpe。终端同时展示程序提取的原始指标和窗口，文字结论需要据此核对。

示例 [agent.json](../configs/research-learning/agent.json) 展示完整的显式工具配置，路径和数据需由调用者准备。`x` 是待研究的已有输入因子，`label` 是收益标签；它们不是平台自动生成的数据。源数据须含资产与时间主键、`eligible`、标签成熟时间及回测使用的市值等上下文。

示例把发现、模型拟合、后续开发检验和最终测试分别放在明确的时间范围。`protocol.train_window` 与 `validation_window` 用于公式发现；模型拟合及后续开发检验使用各工具明确列出的更晚窗口。所有开发窗口及截止仍须早于最终测试。实际研究应一次性替换这些窗口、标签、成本和输入路径，不能只替换某个日期。

`protocol` 固定标签、训练/验证/最终窗口、成本和主评价指标。工具中的 `protocol:label`、`protocol:train_window`、`protocol:costs` 等引用从该处派生；相同科学字段的冲突会拒绝。`tools` 中每项包含：

| 字段 | 用途 |
|---|---|
| `operation` | 已有可信 CLI 操作，例如 `search-factors/assemble/fit/evaluate/backtest/plot` |
| `description` | 数据语义、用途、前置产物和解释限制，发送给模型 |
| `config` | 该原生操作的固定配置，保存在本地 |
| `parameters` | 可调参数路径到说明的映射；允许新公式、特征选择及模型参数，不是候选编号 |
| `bindings` | 输入参数路径到前序产物相对成员的映射，模型提供成功证据 ID |
| `metrics` | 明确选定的数值汇总，支持 Parquet 列、JSON 数值及原生账户汇总 |
| `facts` | 可选的 JSON 聚合元数据字段，例如被选中的公式和候选统计，不传整份配置 |

模型不能修改输出路径、原始输入位置、样本窗口、截止或交易成本。输出由 harness 分配。公式仍经过原生 AST 白名单和因果窗口约束。`apply-formula` 保留输入主键，包括已有的 `market_id`，可以直接接到原生 `assemble`。

例如工具行动：

```json
{
  "kind": "tool",
  "tool": "search",
  "arguments_json": "{\"parameters\":{\"expressions\":[\"rank(x)\",\"-x\"]}}",
  "summary": "比较正向排序与反向排序，并用后续开发样本检验。",
  "evidence": []
}
```

`parameters` 的值是实际 JSON 值。原生工具的输入绑定使用 `{"inputs":{"search":"e0001"}}`。工具的参数类型和科学有效性仍由真实操作校验；失败的错误代码进入反馈，模型可以修正下一次调用。

Parquet 汇总可用 `filters` 选择特定 `signal` 或指标行，避免混合不同统计量。原生数据集目录也可作为 `member`，例如时间序列评价的 `daily`，通过原生读取器裁列和过滤。空样本保持 `null`，不按零处理。原生账本的 `fees/net_pnl/nav` 等由已有 `summarize_portfolio_ledger` 计算，不在 Agent 中重写成本或收益公式。同一账本的多个汇总只读取一次。

`finish` 的 `arguments_json` 可带 `claims`，每项包含 `evidence`、`metric`、`value`，核对原始值是否与已保存的数值汇总一致。原始值包括未定义的 `null`，布尔值不能充当数字；抄错值、指标名或引用会作为失败反馈。该检查不等于自动核验所有自由文字或百分比换算。

## 检索、代码与记忆

`lookup` 检索显式文档、因子和标签目录、原生操作接口、行动证据或共享记忆。指定 `key` 时只打开目标记录，`offset` 和 `page_size` 按字符计数，每页默认及最多 30,000 字符；未指定 `key` 时按匹配记录计数，每页默认及最多 6 条。中文查询同时匹配相邻汉字片段，帮助召回相关表述。上下文不足时可请求较短页并按 `next_offset` 继续。检索材料不提供修改研究口径或扩大权限的授权。

已有研究报告的 CSV 汇总附件也可列入 `documents`，按 UTF-8 文本检索与分页读取。只读复核示例见 [agent-report-review.json](../configs/research-learning/agent-report-review.json)，其中明确选择已发布报告与聚合附件，核对历史数值并要求说明样本范围。报告按引用资料与原生产物解释证据，完成结论集中在结论章节。

`web_search=true` 启用 [Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search)。检索调用只发送模型提出的公开查询，不附上本地数据、会话或凭据；保存正文摘要与来源链接。供应商不支持该工具时记录失败，不换假后端。

`python` 在已配置的隔离容器中执行新代码，固定输入只读挂载至 `/inputs/<alias>`，结果写入 `/outputs`。也可将此前成功产物只读绑定为新的输入别名。镜像包含平台实现，生成代码应直接调用已有 API。容器设置及构建见[研究运行环境](research-learning.md#服务器隔离环境)。没有 Docker 时明确失败，绝不回退到宿主执行。

固定输入直接提供原生列角色、单位、频率、行数、日期范围和标签成熟列声明，不根据整表日期范围、成熟时间或成熟列声明拒绝挂载，也不额外扫描标签值作为准入条件。输入可含测试期行，Agent 应结合元数据、固定研究协议及实际用途判断；原生操作的窗口、截止和标签成熟筛选继续生效。动态绑定的成功产物沿用现有只读输入路径，不新增访问登记。生成产物可以经允许的原生数据读取入口继续研究，不能作为宿主 pickle、模型或策略加载。

`propose_patch` 针对明确列出的仓库源码生成 `proposal.patch`，保留源文件不变，供人工审查。它不直接提交或应用仓库修改。

`memory` 指定跨任务共享目录。`remember` 保存带证据的假设、失败原因、适用条件和研究口径；完成后自动保存结论及实际实验摘要。本地记录保存原会话与产物位置，检索发送给模型时剔除这些路径。记录依据实际事件保存固定最终评价的执行状态；未运行最终评价不等于测试数据未被访问。过去的结果只能提供线索，不能当作新时期或新数据的有效性证明；本地记忆不会替代原生数据和产物。

外部模型接收的是问题、代码/文档、结构说明、明确的数值汇总、聚合元数据和处理后的错误。原始数据、宿主路径、凭据及进程日志留在本地。汇总字段和可公开文档由调用者选择，数值类型检查不等于自动隐私证明。生成代码的错误仅反馈异常类型及代码行号，原生操作反馈机器错误代码，不发送原始异常值或回溯。

## 冻结、最终评价和报告

`final_evaluation` 是调用者按执行顺序声明的固定步骤列表。模型用 `freeze` 指定候选别名，例如：

```json
{
  "kind": "freeze",
  "tool": null,
  "arguments_json": "{\"candidates\":{\"formula\":\"e0001\",\"model\":\"e0006\"}}",
  "summary": "开发检验完成，冻结公式与已拟合模型。",
  "evidence": ["e0004", "e0005", "e0008"]
}
```

最终步骤用 `artifact:formula`、`artifact:model` 和 `artifact:<先前最终步骤名>` 引用产物；必须依次可解析。最终步骤没有模型可调参数。冻结进入最终阶段后，只执行这些步骤；报告阶段允许 `lookup` 和 `compare` 读取已有证据，然后 `finish`，不再接受开发工具、写入或调参调用。

最终评价允许产生负结果或未定义值；执行失败则停在 `final_evaluation_failed`，不能被总结为成功。无可用候选时，可在开发阶段以 `finish` 和 `{"outcome":"no_candidate"}` 如实结束，不必为了完成流程而访问最终样本。

每个证据目录保存行动、原生配置、产物和本地日志。`session.json` 是可恢复状态；`report.json` 和 `report.md` 由状态派生，列明实验状态、数值、原生产物、假设、结论、固定最终评价状态及请求/token/时间消耗。`final_evaluation_status` 由已记录的最终步骤事件派生，不再使用 `holdout_viewed` 推断数据访问；未开始、失败或中断都不能证明测试数据未被查看。图表继续由原生 `plot` 操作生成。自然语言结论需要结合真实产物复核，不能覆盖失败、缺失和数值证据。

机制测试、合成数据的真实研究链路、真实 LLM 和目标服务器容器验收是不同证据。模型联通成功不等于证明研究策略有效；本机的容器命令测试不等于验证服务器 Docker、GPU 或大型数据性能。
