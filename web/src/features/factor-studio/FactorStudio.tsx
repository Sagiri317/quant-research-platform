import { useEffect, useState } from 'react'
import { listFactorTemplates, type FactorTemplate } from '../../shared/api/api'
import type { FactorDraft } from '../../shared/factorDraft'

export function FactorStudio({
  draft,
  onDraftChange,
  onOpenPipeline,
}: {
  draft: FactorDraft
  onDraftChange: (draft: FactorDraft) => void
  onOpenPipeline: () => void
}) {
  const [templates, setTemplates] = useState<FactorTemplate[]>([])
  const activeTemplate =
    templates.find((template) => template.id === draft.templateId) ?? null

  useEffect(() => {
    let cancelled = false
    listFactorTemplates('a_share')
      .then((loaded) => {
        if (!cancelled && Array.isArray(loaded) && loaded.length > 0) {
          setTemplates(loaded)
        }
      })
      .catch(() => {
        // API server 未运行时不伪造 registry 之外的模板。
      })
    return () => {
      cancelled = true
    }
  }, [])

  const selectTemplate = (template: FactorTemplate) => {
    onDraftChange({
      dependencies: template.dependencies ?? [],
      expression: template.expression,
      templateId: template.id,
      templateName: template.name,
    })
  }

  return (
    <section className="studio-grid" aria-label="因子工作台">
      <div className="studio-hero">
        <div>
          <span className="eyebrow">因子工作台</span>
          <h2>研究因子配置</h2>
        </div>
        <div className="market-strip" aria-label="市场支持">
          <span className="market enabled">A 股</span>
          <span className="market planned">加密资产预留</span>
          <button className="primary-action" onClick={onOpenPipeline} type="button">
            进入流程配置
          </button>
        </div>
      </div>

      <div className="studio-panel">
        <div className="form-section-title">
          <h3>因子模板</h3>
          <p>后端 registry 返回的 A 股模板可直接修改，再进入流程配置。</p>
        </div>
        <div className="template-grid">
          {templates.map((template) => (
            <button
              className={
                template.id === activeTemplate?.id ? 'template active' : 'template'
              }
              key={template.id}
              onClick={() => selectTemplate(template)}
              type="button"
            >
              <strong>{template.name}</strong>
              <span>{template.expression}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="studio-panel">
        <label className="field-label" htmlFor="factor-expression">
          因子表达式
        </label>
        <textarea
          className="expression-editor"
          id="factor-expression"
          onChange={(event) =>
            onDraftChange({ ...draft, expression: event.target.value })
          }
          value={draft.expression}
        />
        <div className="dependency-row" aria-label="依赖字段">
          {(draft.dependencies.length > 0
            ? draft.dependencies
            : activeTemplate?.dependencies ?? []
          ).map((dependency) => (
            <span key={dependency}>{dependency}</span>
          ))}
        </div>
      </div>

      <div className="studio-panel">
        <div className="form-section-title">
          <h3>配置校验</h3>
          <p>A 股字段已可用；加密资产模板目前仍保持预留。</p>
        </div>
        <div className="checkpoint-list">
          <span className="checkpoint ready">解析器就绪</span>
          <span className="checkpoint ready">依赖字段已识别</span>
          <span className="checkpoint planned">加密资产未启用</span>
        </div>
      </div>
    </section>
  )
}
