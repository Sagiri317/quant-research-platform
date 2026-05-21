import { type FormEvent } from 'react'

import { FieldLabel, SelectField, TextField } from './components'
import { CreateRunGuide } from './CreateRunGuide'
import { CsvPreviewPanel } from './CsvPreviewPanel'
import { CsvUploadField } from './CsvUploadField'
import {
  DATA_STAGE_OPTIONS,
  DIRECTION_OPTIONS,
  ENGINE_OPTIONS,
  KIND_OPTIONS,
  PORTFOLIO_MODE_OPTIONS,
  WEIGHT_METHOD_OPTIONS,
  type CreateFormState,
  type CsvPreview,
} from './runForm'

export function CreateRunPanel({
  createError,
  csvFileName,
  csvPreview,
  form,
  onFileChange,
  onFormChange,
  onSubmit,
  submitting,
}: {
  createError: string
  csvFileName: string | undefined
  csvPreview: CsvPreview | null
  form: CreateFormState
  onFileChange: (file: File | null) => void | Promise<void>
  onFormChange: (form: CreateFormState) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  submitting: boolean
}) {
  const update = (field: keyof CreateFormState, value: string | boolean) => {
    onFormChange({ ...form, [field]: value })
  }
  const toggleBreakdown = (value: string) => {
    const current = new Set(
      form.time_breakdowns
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    )
    if (current.has(value)) {
      current.delete(value)
    } else {
      current.add(value)
    }
    onFormChange({ ...form, time_breakdowns: Array.from(current).join(',') })
  }
  const selectedBreakdowns = new Set(
    form.time_breakdowns
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean),
  )
  const columnOptions = csvPreview?.columns ?? []
  const toggleColumn = (field: 'feature_columns' | 'label_columns', value: string) => {
    const current = new Set(
      form[field]
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean),
    )
    if (current.has(value)) {
      current.delete(value)
    } else {
      current.add(value)
    }
    onFormChange({ ...form, [field]: Array.from(current).join(',') })
  }
  return (
    <form className="create-run" onSubmit={onSubmit}>
      <section className="create-hero" aria-label="新建回测说明">
        <div>
          <span className="eyebrow">开发辅助入口</span>
          <h2>开发用 CSV 上传</h2>
          <p>
            上传本地 CSV，完成列映射和组合参数后生成可复查的 run。平台只读取你选择的文件，
            不下载外部数据，也不连接外部 API。
          </p>
        </div>
        <div className="create-steps" aria-label="新建回测步骤">
          <span>1 上传面板</span>
          <span>2 映射列名</span>
          <span>3 生成结果文件</span>
        </div>
      </section>

      <CreateRunGuide />

      <section className="form-grid" aria-label="回测输入">
        <div className="form-section-title wide">
          <h3>基础信息</h3>
          <p>这些字段用于给结果文件记录来源和运行身份。默认值通常可以直接使用。</p>
        </div>
        <TextField
          fieldName="run_id"
          helperText="每次回测的唯一 ID；重复 ID 会覆盖同名输出目录。"
          label="运行编号"
          onChange={(value) => update('run_id', value)}
          required
          value={form.run_id}
        />
        <TextField
          fieldName="project"
          helperText="用于记录这次回测来自哪个项目或研究入口。"
          label="项目名"
          onChange={(value) => update('project', value)}
          required
          value={form.project}
        />
        <SelectField
          fieldName="kind"
          helperText="当前上传入口按因子研究处理。"
          label="研究类型"
          onChange={(value) => update('kind', value)}
          options={KIND_OPTIONS}
          required
          value={form.kind}
        />
        <TextField
          fieldName="domain"
          helperText="例如 user_upload、crypto_perp、a_share，用来区分研究场景。"
          label="市场域"
          onChange={(value) => update('domain', value)}
          required
          value={form.domain}
        />
        <SelectField
          fieldName="data_stage"
          helperText="网页上传固定为 uploaded_panel，表示用户上传的本地面板。"
          label="数据阶段"
          onChange={(value) => update('data_stage', value)}
          options={DATA_STAGE_OPTIONS}
          required
          value={form.data_stage}
        />
        <label className="field checkbox-field">
          <FieldLabel fieldName="uses_external_data" label="外部数据" required />
          <input
            checked={form.uses_external_data}
            onChange={(event) => update('uses_external_data', event.target.checked)}
            type="checkbox"
          />
          <span className="field-help">用户上传 CSV 默认视为外部输入；平台不会下载数据或连接 API。</span>
        </label>
        <div className="form-section-title wide">
          <h3>CSV 列映射</h3>
          <p>上传 CSV 后可以从列名里选择。因子列可填写多个，用英文逗号分隔。</p>
        </div>
        <SelectField
          fieldName="timestamp_column"
          helperText="每行所属时间点，例如 timestamp 或 date。"
          label="时间列"
          onChange={(value) => update('timestamp_column', value)}
          options={columnOptions}
          required
          value={form.timestamp_column}
        />
        <SelectField
          fieldName="asset_column"
          helperText="资产代码列，例如 asset、symbol、ticker。"
          label="资产列"
          onChange={(value) => update('asset_column', value)}
          options={columnOptions}
          required
          value={form.asset_column}
        />
        <ColumnMultiSelect
          columns={columnOptions}
          fieldName="feature_columns"
          helperText="要测试的信号列；多个因子用英文逗号分隔。"
          label="因子列"
          onChange={(value) => update('feature_columns', value)}
          onToggle={(value) => toggleColumn('feature_columns', value)}
          required
          value={form.feature_columns}
        />
        <ColumnMultiSelect
          columns={columnOptions}
          fieldName="label_columns"
          helperText="用于 IC、RankIC 和分组诊断；可以多选多个 label。"
          label="标签列"
          onChange={(value) => update('label_columns', value)}
          onToggle={(value) => toggleColumn('label_columns', value)}
          required
          value={form.label_columns}
        />
        <SelectField
          fieldName="pnl_return_column"
          helperText="用于真实回测收益曲线的可交易收益列，默认 forward_return_1d。"
          label="PnL收益列"
          onChange={(value) => update('pnl_return_column', value)}
          options={columnOptions}
          required
          value={form.pnl_return_column}
        />
        <TextField
          fieldName="segment_columns"
          helperText="可选，用于按市场阶段、行业等分组评估；多个用英文逗号分隔。"
          label="分段列"
          onChange={(value) => update('segment_columns', value)}
          value={form.segment_columns}
        />
        <div className="field wide">
          <FieldLabel fieldName="time_breakdowns" label="时间分解" />
          <div className="chip-selector" aria-label="时间分解 time_breakdowns">
            {[
              ['year', '年度'],
              ['quarter', '季度'],
              ['month', '月度'],
            ].map(([value, label]) => (
              <label
                className={
                  selectedBreakdowns.has(value) ? 'chip-option selected' : 'chip-option'
                }
                key={value}
              >
                <input
                  checked={selectedBreakdowns.has(value)}
                  onChange={() => toggleBreakdown(value)}
                  type="checkbox"
                />
                <span>{label}</span>
                <small>{value}</small>
              </label>
            ))}
          </div>
          <span className="field-help">
            可选；勾选后额外输出 by-year / by-quarter / by-month 汇总。
          </span>
        </div>
        <div className="form-section-title wide">
          <h3>回测设置</h3>
          <p>默认设置适合先跑通流程；只有明确知道含义时再调整权重和分位数。</p>
        </div>
        <TextField
          fieldName="n_groups"
          helperText="用于分组收益和分层检验，常用 5 或 10。"
          label="分组数"
          onChange={(value) => update('n_groups', value)}
          required
          value={form.n_groups}
        />
        <SelectField
          fieldName="engine"
          helperText="默认 DataFrame；Tensor 为实验性 opt-in 引擎，输出 artifact 契约保持一致。"
          label="计算引擎"
          onChange={(value) => update('engine', value)}
          options={ENGINE_OPTIONS}
          required
          value={form.engine}
        />
        <SelectField
          fieldName="weight_method"
          helperText="组合权重的生成方式；不确定时用等权分位。"
          label="权重方法"
          onChange={(value) => update('weight_method', value)}
          options={WEIGHT_METHOD_OPTIONS}
          required
          value={form.weight_method}
        />
        <SelectField
          fieldName="portfolio_mode"
          helperText="A 股等不能做空场景选只做多；多空组合会同时构造多头和空头。"
          label="组合模式"
          onChange={(value) => update('portfolio_mode', value)}
          options={PORTFOLIO_MODE_OPTIONS}
          required
          value={form.portfolio_mode}
        />
        <SelectField
          fieldName="direction"
          helperText="定义因子值高的一侧是多头，还是因子值低的一侧是多头。"
          label="信号方向"
          onChange={(value) => update('direction', value)}
          options={DIRECTION_OPTIONS}
          required
          value={form.direction}
        />
        <TextField
          fieldName="top_quantile"
          helperText="进入多头组合的比例，例如 0.2 表示前 20%。"
          label="多头分位"
          onChange={(value) => update('top_quantile', value)}
          required
          value={form.top_quantile}
        />
        <TextField
          fieldName="bottom_quantile"
          helperText="进入空头组合的比例，例如 0.2 表示后 20%。"
          label="空头分位"
          onChange={(value) => update('bottom_quantile', value)}
          required
          value={form.bottom_quantile}
        />
        <TextField
          fieldName="cost_bps"
          helperText="单边成本，单位 bps；0 表示不扣交易成本。"
          label="成本 bps"
          onChange={(value) => update('cost_bps', value)}
          value={form.cost_bps}
        />
        <div className="form-section-title wide">
          <h3>高级设置</h3>
          <p>这些字段不是跑通回测的必要项，通常可以留空。</p>
        </div>
        <SelectField
          allowEmpty
          fieldName="risk_column"
          helperText="仅在按风险缩放权重时需要，且数值必须为正。"
          label="风险列"
          onChange={(value) => update('risk_column', value)}
          options={columnOptions}
          value={form.risk_column}
        />
        <SelectField
          allowEmpty
          fieldName="portfolio_feature"
          helperText="可选；留空时使用第一个因子列生成组合。"
          label="组合因子"
          onChange={(value) => update('portfolio_feature', value)}
          options={columnOptions}
          value={form.portfolio_feature}
        />
        <TextField
          fieldName="tensor_max_memory_mb"
          helperText="仅 Tensor 引擎使用；留空自动记录 full strategy，填小值会记录 chunked strategy。"
          label="Tensor 内存 MB"
          onChange={(value) => update('tensor_max_memory_mb', value)}
          value={form.tensor_max_memory_mb}
        />
        <label className="field wide">
          <FieldLabel fieldName="boundary" label="边界说明" />
          <textarea
            aria-label="边界说明 boundary"
            onChange={(event) => update('boundary', event.target.value)}
            value={form.boundary}
          />
          <span className="field-help">可补充数据来源、样本边界或实验说明。</span>
        </label>
        <CsvUploadField csvFileName={csvFileName} onFileChange={onFileChange} />
        <CsvPreviewPanel csvPreview={csvPreview} />
      </section>
      {createError ? <div className="alert">{createError}</div> : null}
      <div className="form-actions">
        <button className="submit-run" disabled={submitting} type="submit">
          {submitting ? '生成中...' : '生成回测'}
        </button>
      </div>
    </form>
  )
}

function ColumnMultiSelect({
  columns,
  fieldName,
  helperText,
  label,
  onChange,
  onToggle,
  required,
  value,
}: {
  columns: string[]
  fieldName: string
  helperText: string
  label: string
  onChange: (value: string) => void
  onToggle: (value: string) => void
  required?: boolean
  value: string
}) {
  const selected = new Set(
    value
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean),
  )
  return (
    <div className="field wide">
      <FieldLabel fieldName={fieldName} label={label} required={required} />
      <input
        aria-label={`${label} ${fieldName}`}
        onChange={(event) => onChange(event.target.value)}
        type="text"
        value={value}
      />
      {columns.length > 0 ? (
        <div className="chip-selector compact" aria-label={`${label}候选列`}>
          {columns.map((column) => (
            <label
              className={selected.has(column) ? 'chip-option selected' : 'chip-option'}
              key={column}
            >
              <input
                checked={selected.has(column)}
                onChange={() => onToggle(column)}
                type="checkbox"
              />
              <span>{column}</span>
            </label>
          ))}
        </div>
      ) : null}
      <span className="field-help">{helperText}</span>
    </div>
  )
}
