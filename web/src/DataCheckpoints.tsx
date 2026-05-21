const CHECKPOINTS = [
  ['run_config.yaml', 'ready'],
  ['warnings.json', 'ready'],
  ['中间层摘要', 'pending'],
  ['面板摘要', 'pending'],
  ['因子摘要', 'pending'],
  ['评估清单', 'pending'],
]

export function DataCheckpoints() {
  return (
    <section className="studio-grid" aria-label="数据检查点">
      <div className="studio-hero">
        <div>
          <span className="eyebrow">数据检查点</span>
          <h2>流程证据交接</h2>
        </div>
        <span className="market planned">快照骨架</span>
      </div>
      <div className="studio-panel">
        <div className="form-section-title">
          <h3>运行快照</h3>
          <p>reports/private_real/pipeline_runs/a_share_reversal_smoke</p>
        </div>
        <div className="checkpoint-list">
          {CHECKPOINTS.map(([label, status]) => (
            <span className={`checkpoint ${status}`} key={label}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
